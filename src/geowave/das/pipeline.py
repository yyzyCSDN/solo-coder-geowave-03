from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .fiber import FiberPath, ChannelGeometry
from .gather import DASGather, load_strain_rate
from .quality import (QualityAssessment, QualityFlag, assess_quality)
from .sensitivity import (ReconstructionOperator, gauge_average_operator,
                          gauge_difference_operator)
from .resample import resample_time, SpatialLineage, lineage_from_operator


@dataclass
class DASConfig:
    """Configuration of the curved-fibre DAS entry chain."""

    gauge_m: float = 10.0
    mode: str = 'average'                       # 'average' | 'difference'
    fiber_gap_m: float | None = None
    reference_azimuth_deg: float | None = None
    reverse_cos_threshold: float = 0.0
    rms_floor: float | None = None
    rms_ceiling: float | None = None
    snr_db_floor: float | None = None
    bad_flags_for_reconstruction: int = int(QualityFlag.DEAD_FIBER |
                                            QualityFlag.DEAD_CHANNEL |
                                            QualityFlag.SATURATED)
    output_spacing_m: float | None = None
    output_dt_s: float | None = None
    resample_gap_tolerance_m: float | None = None
    drop_invalid_targets: bool = False


@dataclass
class DASProduct:
    """Result bundle: reconstructed gather, QC and full provenance."""

    gather: DASGather
    target_geometry: ChannelGeometry
    quality: QualityAssessment
    operator: ReconstructionOperator
    output_flags: np.ndarray
    lineage: SpatialLineage | None = None
    config: DASConfig | None = None

    @property
    def fiber(self) -> FiberPath:
        return self.target_geometry.fiber

    def trace_output(self, target_index: int) -> dict:
        """Source channels contributing to one reconstructed channel."""
        row = self.operator.rows[target_index]
        g = self.operator.source_geometry
        return {
            'target_id': self.target_geometry.channel_ids[target_index],
            'arc_s': row.center_s,
            'world_position': self.fiber.point_at(row.center_s).tolist(),
            'status': row.status.value,
            'sources': [
                {'channel_id': g.channel_ids[int(j)],
                 'arc_s': float(g.channel_s[int(j)]),
                 'weight': float(w)}
                for j, w in zip(row.indices, row.weights)],
        }

    def trace_world_position(self, channel_id: str):
        """Channel id -> world position (original source id resolved via lineage)."""
        target_ids = self.target_geometry.channel_ids
        if channel_id in target_ids:
            k = target_ids.index(channel_id)
            s = float(self.target_geometry.channel_s[k])
            return self.fiber.point_at(s)
        if self.lineage is not None:
            src = self.lineage.source_geometry
            if channel_id in src.channel_ids:
                s = float(src.channel_s[src.channel_ids.index(channel_id)])
                return self.fiber.point_at(s)
        raise KeyError(f'unknown channel id {channel_id!r}')


def process_das(data, fiber: FiberPath, channel_s: np.ndarray, dt_s: float,
                config: DASConfig | None = None, *,
                start_s: float = 0.0,
                units: str = 'strain_rate',
                name: str = 'das',
                channel_ids: list[str] | None = None) -> DASProduct:
    """Curved-fibre DAS processing entry.

    Steps
    -----
    1. bind raw strain-rate channels to fibre arc positions (never station ids);
    2. QC: broken fibre, reversed/back-tracked cable sections, low-quality zones;
    3. reconstruct channels by arc-length gauge integration (strain rate) or
       gauge difference (axial field), honouring cable tangents/gaps;
    4. optionally resample to regular spacing / sampling rate while keeping the
       original-channel -> world-position lineage.
    """
    config = config or DASConfig()
    channel_s = np.asarray(channel_s, dtype=float)
    geometry = ChannelGeometry(channel_s, fiber, list(channel_ids) if channel_ids else [])
    gather = load_strain_rate(data, geometry, dt_s, start_s, units, name)

    quality = assess_quality(
        gather,
        fiber_gap_m=config.fiber_gap_m,
        reference_azimuth_deg=config.reference_azimuth_deg,
        reverse_cos_threshold=config.reverse_cos_threshold,
        rms_floor=config.rms_floor,
        rms_ceiling=config.rms_ceiling,
        snr_db_floor=config.snr_db_floor)

    forbidden = (quality.flags & config.bad_flags_for_reconstruction) != 0

    # Reconstruction target: same channel positions unless spatial resampling
    # is requested, in which case the gauge operator builds directly on the
    # regular grid and the linear lineage is attached afterwards.
    if config.output_spacing_m is not None:
        target_s = np.arange(channel_s[0], channel_s[-1] + 1e-9,
                             config.output_spacing_m)
        # label each output channel after its nearest original channel
        nearest = np.searchsorted(channel_s, target_s)
        nearest = np.clip(nearest, 0, channel_s.size - 1)
        target_ids = [geometry.channel_ids[i] + '#rg' for i in nearest]
    else:
        target_s = channel_s.copy()
        target_ids = list(geometry.channel_ids)
    target_geometry = ChannelGeometry(target_s, fiber, target_ids)

    builder = (gauge_average_operator if config.mode == 'average' else
               gauge_difference_operator if config.mode == 'difference' else None)
    if builder is None:
        raise ValueError("config.mode must be 'average' or 'difference'")
    operator = builder(geometry, target_geometry, config.gauge_m,
                       gap_tolerance_m=config.fiber_gap_m,
                       forbidden=forbidden)
    rebuilt = operator.apply(gather.data)
    output_flags = operator.output_flags(quality.flags)
    for k, row in enumerate(operator.rows):
        if not row.valid:
            output_flags[k] |= int(QualityFlag.DEAD_FIBER)

    lineage: SpatialLineage | None = None
    if config.output_spacing_m is not None:
        # The gauge operator already reconstructs on the regular target grid;
        # wrap its rows as lineage so original-channel provenance is retained.
        lineage = lineage_from_operator(operator, config.bad_flags_for_reconstruction)
        if config.drop_invalid_targets:
            keep = np.array([r.valid for r in operator.rows])
            kept_geometry = ChannelGeometry(
                target_s[keep], fiber,
                [target_geometry.channel_ids[i] for i in np.flatnonzero(keep)])
            rebuilt = rebuilt[keep]
            output_flags = output_flags[keep]
            operator = _filter_operator(operator, keep, kept_geometry)
            target_geometry = kept_geometry
            lineage = lineage_from_operator(operator, config.bad_flags_for_reconstruction)

    out_gather = gather.copy(rebuilt, geometry=target_geometry,
                             tags={**gather.tags,
                                   'gauge_m': f'{config.gauge_m:g}',
                                   'reconstruction': config.mode})

    if config.output_dt_s is not None:
        out_gather = resample_time(out_gather, config.output_dt_s)

    if config.output_dt_s is not None:
        out_gather = resample_time(out_gather, config.output_dt_s)

    return DASProduct(gather=out_gather, target_geometry=target_geometry,
                      quality=quality, operator=operator,
                      output_flags=output_flags, lineage=lineage, config=config)


def _filter_operator(operator: ReconstructionOperator, keep: np.ndarray,
                     target_geometry: ChannelGeometry | None = None
                     ) -> ReconstructionOperator:
    rows = [r for r, k in zip(operator.rows, keep) if k]
    for new_i, row in enumerate(rows):
        row.index = new_i
    return ReconstructionOperator(rows, operator.source_geometry,
                                  target_geometry or operator.target_geometry,
                                  operator.gauge_m, operator.kind,
                                  operator.gap_tolerance_m, operator.forbidden)
