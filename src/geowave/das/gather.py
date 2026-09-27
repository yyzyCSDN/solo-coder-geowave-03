from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .geometry import FiberGeometry
from .gauge import as_strain_rate, apply_gauge, GaugeResult
from .qc import QualityTable, assess_channels
from .resample import (SpatialResampling, ChannelProvenance, resample_channels,
                       resample_time)
from .sensitivity import (DirectionalResult, directional_sensitivity,
                          apply_directional)


@dataclass
class DasGather:
    """A channel×time DAS matrix that never forgets it lives on a fiber.

    ``geometry`` maps every channel to a 3-D position, segment, arc length and
    tangent. ``spatial_resampling`` carries the mixing matrix back to the
    original channels whenever a spatial resampling has happened; ``parent``
    chains to the previous gather, so any channel can be traced to the raw
    channels that produced it.
    """
    data: np.ndarray
    geometry: FiberGeometry
    dt_s: float
    unit: str = 'strain-rate'
    gauge_length_m: float | None = None
    history: tuple[str, ...] = ()
    parent: 'DasGather | None' = None
    spatial_resampling: SpatialResampling | None = None
    directional: DirectionalResult | None = None
    quality: QualityTable | None = None

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError('DAS gather needs a (channel, time) matrix')
        if self.data.shape[0] != self.geometry.n_channels:
            raise ValueError('channel count does not match fiber geometry')
        if self.dt_s <= 0:
            raise ValueError('sample interval must be positive')

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_samples) * self.dt_s

    @property
    def channel_ids(self) -> tuple[str, ...]:
        return self.geometry.channel_ids

    def channel_trace(self, channel_id: str) -> np.ndarray:
        return self.data[self.geometry.index_of(channel_id)]

    def origins_of(self, channel_id: str) -> tuple[tuple[str, float], ...]:
        """Original-channel contributions of a possibly resampled channel.

        Walks the parent chain: a spatial resampling contributes its mixing
        weights, every other operation passes channel identity through.
        """
        if self.spatial_resampling is not None:
            direct = self.spatial_resampling.origins_of(channel_id)
        else:
            direct = ((channel_id, 1.0),)
        if self.parent is None:
            return direct
        merged: dict[str, float] = {}
        for parent_id, weight in direct:
            for origin_id, origin_weight in self.parent.origins_of(parent_id):
                merged[origin_id] = merged.get(origin_id, 0.0) + weight * origin_weight
        return tuple(sorted(merged.items()))

    def trace_to_origin(self, channel_id: str) -> list[str]:
        """Processing lineage, newest first, ending at the original channel set."""
        chain = [self._label(channel_id)]
        gather: DasGather = self
        current = channel_id
        while gather.parent is not None:
            if gather.spatial_resampling is not None:
                direct = gather.spatial_resampling.origins_of(current)
                current = max(direct, key=lambda sw: sw[1])[0]
            gather = gather.parent
            chain.append(gather._label(current))
        return chain

    def _label(self, channel_id: str) -> str:
        return f'{self.unit}:{channel_id}'


# --------------------------------------------------------------- construction
def build_das_gather(raw_data, geometry: FiberGeometry, dt_s: float, *,
                     input_kind: str = 'strain-rate',
                     phase_to_strain_scale: float = 1.0) -> DasGather:
    """Entry point for raw interrogator channels on a curved fiber."""
    data = as_strain_rate(raw_data, dt_s, input_kind=input_kind,
                          phase_to_strain_scale=phase_to_strain_scale)
    return DasGather(data, geometry, dt_s,
                     history=(f'build(input={input_kind})',))


# ----------------------------------------------------------------- operators
def reconstruct_gauge(gather: DasGather, gauge_length_m: float, *,
                      min_coverage: float = 0.5) -> DasGather:
    """Rebuild channels by trapezoid gauge integration of strain rate."""
    result: GaugeResult = apply_gauge(
        gather.data, gather.geometry, gauge_length_m,
        min_coverage=min_coverage)
    n_partial = sum(1 for w in result.windows if w.partial and w.covered_fraction >= min_coverage)
    return DasGather(
        result.data, gather.geometry, gather.dt_s,
        unit=gather.unit, gauge_length_m=gauge_length_m,
        history=gather.history + (f'gauge(L={gauge_length_m:g}m,partial={n_partial})',),
        parent=gather)


def correct_sensitivity(gather: DasGather, *, source_xyz=None, directions=None,
                        mode: str = 'cos2', action: str = 'correct',
                        min_sensitivity: float = 0.05) -> DasGather:
    """Apply directional sensitivity along the curved fiber to reconstructed channels."""
    sens, angle = directional_sensitivity(
        gather.geometry, directions=directions, source_xyz=source_xyz, mode=mode)
    result = apply_directional(gather.data, sens, angle, mode=action,
                               min_sensitivity=min_sensitivity)
    n_weak = int(result.weak_mask.sum())
    return DasGather(
        result.data, gather.geometry, gather.dt_s,
        unit=gather.unit, gauge_length_m=gather.gauge_length_m,
        history=gather.history + (
            f'directional(mode={mode},action={action},weak={n_weak})',),
        parent=gather,
        directional=result, quality=gather.quality)


def mark_quality(gather: DasGather, **kwargs) -> DasGather:
    """Attach dead / clipped / low-signal / reversed / near-break channel marks."""
    table = assess_channels(gather.data, gather.geometry, **kwargs)
    n_low = len(table.low_quality_ids())
    return DasGather(
        gather.data, gather.geometry, gather.dt_s,
        unit=gather.unit, gauge_length_m=gather.gauge_length_m,
        history=gather.history + (
            f'quality(low={n_low},reversed={len(table.reversed_ids())},'
            f'breaks={len(table.breaks)},zones={len(table.low_quality_zones)})',),
        parent=gather,
        directional=gather.directional, quality=table)


def resample_spatial(gather: DasGather, spacing_m: float, *,
                     method: str = 'linear', min_coverage: float = 0.5,
                     new_id_prefix: str = 'das') -> DasGather:
    """Resample channels on an even spatial grid, retaining channel provenance."""
    result = resample_channels(
        gather.data, gather.geometry, spacing_m, method=method,
        min_coverage=min_coverage, new_id_prefix=new_id_prefix)
    return DasGather(
        result.data, result.geometry, gather.dt_s,
        unit=gather.unit, gauge_length_m=gather.gauge_length_m,
        history=gather.history + (
            f'resample-space(dx={spacing_m:g}m,method={method},'
            f'n={result.geometry.n_channels})',),
        parent=gather, spatial_resampling=result.resampling)


def resample_temporal(gather: DasGather, new_dt_s: float,
                      max_denominator: int = 1000) -> DasGather:
    """Resample along time; the channel-to-position mapping is untouched."""
    data, dt = resample_time(gather.data, gather.dt_s, new_dt_s, max_denominator)
    return DasGather(
        data, gather.geometry, dt,
        unit=gather.unit, gauge_length_m=gather.gauge_length_m,
        history=gather.history + (f'resample-time(dt={new_dt_s:g}s)',),
        parent=gather,
        directional=gather.directional, quality=gather.quality)


# ------------------------------------------------------------------ pipeline
@dataclass
class DasProcessConfig:
    input_kind: str = 'strain-rate'
    phase_to_strain_scale: float = 1.0
    gauge_length_m: float | None = 10.0
    spatial_spacing_m: float | None = None
    spatial_method: str = 'linear'
    new_dt_s: float | None = None
    sensitivity_mode: str = 'cos2'
    sensitivity_action: str = 'correct'
    min_sensitivity: float = 0.05
    source_xyz: tuple[float, ...] | None = None
    min_gauge_coverage: float = 0.5
    assess_quality: bool = True


def process_das(raw_data, geometry: FiberGeometry, dt_s: float,
                config: DasProcessConfig | None = None, **overrides) -> DasGather:
    """Standard curved-fiber DAS entry pipeline.

    Order: unit conversion → gauge integration → even spatial resampling
    (original channels stay traceable) → time resampling → directional
    sensitivity → quality marking (breaks, reversed sections, low-quality zones).
    """
    cfg = config or DasProcessConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f'unknown DAS option {key!r}')
        setattr(cfg, key, value)

    gather = build_das_gather(
        raw_data, geometry, dt_s, input_kind=cfg.input_kind,
        phase_to_strain_scale=cfg.phase_to_strain_scale)
    if cfg.gauge_length_m is not None:
        gather = reconstruct_gauge(gather, cfg.gauge_length_m,
                                   min_coverage=cfg.min_gauge_coverage)
    if cfg.spatial_spacing_m is not None:
        gather = resample_spatial(gather, cfg.spatial_spacing_m,
                                  method=cfg.spatial_method,
                                  min_coverage=cfg.min_gauge_coverage)
    if cfg.new_dt_s is not None:
        gather = resample_temporal(gather, cfg.new_dt_s)
    if cfg.source_xyz is not None:
        gather = correct_sensitivity(
            gather, source_xyz=tuple(cfg.source_xyz),
            mode=cfg.sensitivity_mode, action=cfg.sensitivity_action,
            min_sensitivity=cfg.min_sensitivity)
    if cfg.assess_quality:
        gather = mark_quality(gather)
    return gather
