from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction

from .fiber import ChannelGeometry
from .gather import DASGather
from .sensitivity import (ReconstructionOperator, OperatorRow, RowStatus,
                          _gauge_weights)


@dataclass
class SpatialSample:
    """Exact provenance of one resampled spatial channel."""

    target_index: int
    arc_s: float
    source_indices: np.ndarray          # original channel rows
    weights: np.ndarray                 # interpolation weights (sum to 1)
    status: RowStatus

    def channel_ids(self, geometry: ChannelGeometry) -> list[str]:
        return [geometry.channel_ids[i] for i in self.source_indices]


@dataclass
class SpatialLineage:
    """Original-channel -> spatial-position traceability after resampling."""

    source_geometry: ChannelGeometry
    target_geometry: ChannelGeometry
    samples: list[SpatialSample]
    forbidden_flags: int

    def location_of(self, target_index: int) -> tuple[float, np.ndarray]:
        smp = self.samples[target_index]
        return smp.arc_s, self.target_geometry.fiber.point_at(smp.arc_s)

    def source_channels_of(self, target_index: int) -> list[tuple[str, float, float]]:
        smp = self.samples[target_index]
        g = self.source_geometry
        return [(g.channel_ids[i], float(g.channel_s[i]), float(w))
                for i, w in zip(smp.source_indices, smp.weights)]

    def target_indices_from_source(self, source_index: int) -> np.ndarray:
        return np.array([k for k, smp in enumerate(self.samples)
                         if source_index in set(smp.source_indices.tolist())], dtype=int)


def _linear_operator(source: ChannelGeometry, target_s: np.ndarray,
                     gap_tolerance_m: float, forbidden: np.ndarray
                     ) -> list[OperatorRow]:
    rows = []
    s = source.channel_s
    for k, cs in enumerate(target_s):
        if cs < s[0] - 1e-12 or cs > s[-1] + 1e-12:
            rows.append(OperatorRow(k, float(cs), np.empty(0, int),
                                    np.empty(0, float), RowStatus.OUT_OF_RANGE))
            continue
        j = int(np.searchsorted(s, cs, side='right') - 1)
        j = max(0, min(j, s.size - 1))
        if abs(cs - s[j]) < 1e-12:
            idx, w, status = np.array([j]), np.array([1.0]), RowStatus.OK
            if forbidden.size and forbidden[j]:
                status = RowStatus.FLAGGED_SOURCE
        elif j + 1 >= s.size:
            idx, w, status = np.empty(0, int), np.empty(0, float), RowStatus.OUT_OF_RANGE
        else:
            if forbidden.size and (forbidden[j] or forbidden[j + 1]):
                idx, w, status = np.empty(0, int), np.empty(0, float), RowStatus.FLAGGED_SOURCE
            elif s[j + 1] - s[j] > gap_tolerance_m:
                idx, w, status = np.empty(0, int), np.empty(0, float), RowStatus.GAP_CROSSING
            else:
                f = (cs - s[j]) / (s[j + 1] - s[j])
                idx, w, status = np.array([j, j + 1]), np.array([1.0 - f, f]), RowStatus.OK
        rows.append(OperatorRow(k, float(cs), idx, w, status))
    return rows


def resample_spatial(gather: DASGather, spacing_m: float | None = None,
                     target_s: np.ndarray | None = None,
                     gap_tolerance_m: float | None = None,
                     forbidden: np.ndarray | None = None,
                     drop_invalid: bool = False) -> tuple[DASGather, SpatialLineage]:
    """Resample channels onto a regular arc-length grid.

    Interpolation is purely local (two bracketing channels); broken-fibre
    gaps, out-of-range targets and ``forbidden`` channels are never interpolated
    across -- those targets come back NaN (or are dropped). Every target keeps
    an explicit record of its source channels and weights, so original-channel
    -> world-position traceability survives resampling.
    """
    s = gather.geometry.channel_s
    if target_s is None:
        if spacing_m is None:
            raise ValueError('provide spacing_m or explicit target_s')
        if spacing_m <= 0:
            raise ValueError('spacing must be positive')
        target_s = np.arange(s[0], s[-1] + 1e-9, spacing_m)
    target_s = np.asarray(target_s, dtype=float)
    if gap_tolerance_m is None:
        gap_tolerance_m = 3.0 * float(np.median(np.diff(s)))
    forbidden = np.zeros(gather.n_channels, dtype=bool) if forbidden is None else np.asarray(forbidden, dtype=bool)

    rows = _linear_operator(gather.geometry, target_s, gap_tolerance_m, forbidden)
    target_geom = ChannelGeometry(target_s, gather.geometry.fiber)
    op = ReconstructionOperator(rows, gather.geometry, target_geom,
                                gauge_m=float(np.median(np.diff(s))) if spacing_m is None else spacing_m,
                                kind='linear_resample',
                                gap_tolerance_m=gap_tolerance_m, forbidden=forbidden)
    out_data = op.apply(gather.data)

    samples = [SpatialSample(r.index, r.center_s, r.indices, r.weights, r.status)
               for r in rows]
    lineage = SpatialLineage(gather.geometry, target_geom, samples,
                             forbidden_flags=0)

    if drop_invalid:
        keep = np.array([r.valid for r in rows])
        out_data = out_data[keep]
        target_geom = ChannelGeometry(target_s[keep], gather.geometry.fiber)
        samples = [sm for sm, k in zip(samples, keep) if k]
        lineage = SpatialLineage(gather.geometry, target_geom, samples, 0)

    out = gather.copy(out_data, geometry=target_geom,
                      tags={**gather.tags, 'spatial_resample': f'{spacing_m or "custom"} m'})
    return out, lineage


def lineage_from_operator(operator: ReconstructionOperator,
                          forbidden_flags: int = 0) -> SpatialLineage:
    """Expose source-channel provenance recorded in a reconstruction operator."""
    samples = [SpatialSample(r.index, r.center_s, r.indices, r.weights, r.status)
               for r in operator.rows]
    return SpatialLineage(operator.source_geometry, operator.target_geometry,
                          samples, forbidden_flags)


def resample_time(gather: DASGather, new_dt_s: float,
                  max_denominator: int = 1000) -> DASGather:
    """Temporal anti-alias polyphase resample; channel geometry is untouched."""
    if new_dt_s <= 0:
        raise ValueError('positive new_dt_s required')
    ratio = Fraction(gather.dt_s / new_dt_s).limit_denominator(max_denominator)
    out = resample_poly(gather.data, ratio.numerator, ratio.denominator, axis=1)
    return gather.copy(out, dt_s=new_dt_s,
                       tags={**gather.tags,
                             'time_resample': f'{gather.dt_s:g}->{new_dt_s:g}'})
