from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy import sparse

from .fiber import ChannelGeometry


class RowStatus(str, Enum):
    OK = 'ok'
    GAP_CROSSING = 'gap-crossing'
    OUT_OF_RANGE = 'out-of-range'
    FLAGGED_SOURCE = 'flagged-source'


@dataclass
class OperatorRow:
    """One reconstructed channel as a linear combination of source channels.

    ``indices`` / ``weights`` are the source rows and their arc-length-aware
    coefficients, so every output channel is fully traceable to its inputs.
    """

    index: int
    center_s: float
    indices: np.ndarray
    weights: np.ndarray
    status: RowStatus = RowStatus.OK

    @property
    def valid(self) -> bool:
        return self.status is RowStatus.OK and self.indices.size > 0


@dataclass
class ReconstructionOperator:
    """Sparse ``(n_target, n_source)`` operator with lineage and diagnostics."""

    rows: list[OperatorRow]
    source_geometry: ChannelGeometry
    target_geometry: ChannelGeometry
    gauge_m: float
    kind: str
    gap_tolerance_m: float
    forbidden: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def matrix(self) -> sparse.csr_matrix:
        n_src = self.source_geometry.n_channels
        n_tgt = len(self.rows)
        ii, jj, ww = [], [], []
        for out_i, row in enumerate(self.rows):
            for j, w in zip(row.indices, row.weights):
                ii.append(out_i)
                jj.append(int(j))
                ww.append(float(w))
        m = sparse.csr_matrix((ww, (ii, jj)), shape=(n_tgt, n_src))
        m.sum_duplicates()
        return m

    @property
    def statuses(self) -> list[RowStatus]:
        return [r.status for r in self.rows]

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the operator; invalid target rows are returned as NaN."""
        data = np.asarray(data, dtype=float)
        out = np.full((len(self.rows), data.shape[1]), np.nan, dtype=float)
        m = self.matrix
        valid = np.array([r.valid for r in self.rows])
        if valid.any():
            out[valid] = m[valid] @ data
        return out

    def output_flags(self, source_flags: np.ndarray) -> np.ndarray:
        """OR-merge input flags that actually contribute to each output row."""
        source_flags = np.asarray(source_flags)
        out = np.zeros(len(self.rows), dtype=source_flags.dtype)
        for i, row in enumerate(self.rows):
            if row.indices.size:
                out[i] = np.bitwise_or.reduce(source_flags[row.indices])
        return out


# ---------------------------------------------------------------------------
# weight construction along arc length
# ---------------------------------------------------------------------------

def _endpoints(center_s: float, gauge_m: float) -> tuple[float, float]:
    if gauge_m <= 0:
        raise ValueError('gauge length must be positive')
    return center_s - 0.5 * gauge_m, center_s + 0.5 * gauge_m


def _gauge_weights(source_s: np.ndarray, a: float, b: float,
                   gap_tolerance_m: float, forbidden: np.ndarray,
                   s_min: float, s_max: float) -> tuple[np.ndarray, np.ndarray, RowStatus]:
    """Arc-length trapezoidal weights for a single integration window."""
    if a < s_min - 1e-9 or b > s_max + 1e-9:
        return np.empty(0, int), np.empty(0, float), RowStatus.OUT_OF_RANGE

    # Source knots strictly inside (a, b); endpoints are handled as fractional
    # samples so the window geometry is exact even if gauge/channel pitches
    # do not align.
    inside = np.flatnonzero((source_s > a + 1e-12) & (source_s < b - 1e-12))
    knots = list(inside)

    if forbidden.size:
        for j in knots:
            if forbidden[j]:
                return np.empty(0, int), np.empty(0, float), RowStatus.FLAGGED_SOURCE
        # any forbidden sample inside [a, b] whose cell is covered is rejected;
        # for a gauge difference the endpoints are point reads, so boundaries count
        j_lo = int(np.searchsorted(source_s, a, side='left'))
        j_hi = int(np.searchsorted(source_s, b, side='right')) - 1
        for j in range(max(0, j_lo - 1), min(source_s.size, j_hi + 1)):
            if forbidden[j] and a - 1e-12 <= source_s[j] <= b + 1e-12:
                return np.empty(0, int), np.empty(0, float), RowStatus.FLAGGED_SOURCE

    # gap check: consecutive contributing knots must not straddle a gap larger
    # than the tolerance (e.g. broken fibre / missing section).
    boundary = list({a, b})
    check_s = np.array(boundary + [float(source_s[j]) for j in knots])
    check_s.sort()
    idx = np.searchsorted(source_s, check_s, side='right') - 1
    idx = np.clip(idx, 0, source_s.size - 2)
    for k in np.unique(idx):
        seg = check_s[idx == k]
        if seg.size and source_s[k + 1] - source_s[k] > gap_tolerance_m:
            covered = min(seg[-1], source_s[k + 1]) - max(seg[0], source_s[k])
            if covered > 1e-9:
                return np.empty(0, int), np.empty(0, float), RowStatus.GAP_CROSSING

    # ordered sample positions in the window. A fractional endpoint a (or b)
    # is inserted as a knot between its bracketing samples: the sample on each
    # side contributes only the fraction of the sample cell covered, so the
    # window geometry is exact even if gauge/channel pitches do not align.
    points: list[tuple[float, int, float]] = [(float(source_s[j]), int(j), 1.0)
                                              for j in knots]
    for endpoint in (a, b):
        j = int(np.searchsorted(source_s, endpoint, side='right') - 1)
        j = max(0, min(j, source_s.size - 1))
        if abs(endpoint - source_s[j]) < 1e-12:
            points.append((endpoint, j, 1.0))
        elif j + 1 < source_s.size:
            f = (endpoint - source_s[j]) / (source_s[j + 1] - source_s[j])
            points.append((endpoint, j, f))
            points.append((endpoint, j + 1, 1.0 - f))
    points.sort(key=lambda q: q[0])

    weights = np.zeros(source_s.size, dtype=float)
    for left, right in zip(points[:-1], points[1:]):
        ds = right[0] - left[0]
        if ds <= 1e-12:
            continue
        # trapezoidal half-cell weight to each bracketing sample
        weights[left[1]] += 0.5 * ds * left[2]
        weights[right[1]] += 0.5 * ds * right[2]
    used = np.flatnonzero(weights)
    total = float(weights[used].sum())
    if total <= 0.0:
        return np.empty(0, int), np.empty(0, float), RowStatus.GAP_CROSSING
    return used, weights[used] / total, RowStatus.OK


def gauge_average_operator(source: ChannelGeometry, target: ChannelGeometry,
                           gauge_m: float, gap_tolerance_m: float | None = None,
                           forbidden: np.ndarray | None = None) -> ReconstructionOperator:
    """Boxcar average of a strain-rate field over each target gauge interval."""
    if gap_tolerance_m is None:
        gap_tolerance_m = gauge_m
    forbidden = np.zeros(source.n_channels, dtype=bool) if forbidden is None else np.asarray(forbidden, dtype=bool)
    rows = []
    s = source.channel_s
    for k, cs in enumerate(target.channel_s):
        a, b = _endpoints(float(cs), gauge_m)
        idx, w, status = _gauge_weights(s, a, b, gap_tolerance_m, forbidden,
                                        float(s[0]), float(s[-1]))
        rows.append(OperatorRow(k, float(cs), idx, w, status))
    return ReconstructionOperator(rows, source, target, gauge_m,
                                  'gauge_average', gap_tolerance_m, forbidden)


def gauge_difference_operator(source: ChannelGeometry, target: ChannelGeometry,
                              gauge_m: float, gap_tolerance_m: float | None = None,
                              forbidden: np.ndarray | None = None) -> ReconstructionOperator:
    """DAS difference operator acting on an axial displacement/velocity field.

    Returns ``(u(s+L/2) - u(s-L/2)) / L``; this is how gauge integration is
    applied to a point-measured axial field (or how a new gauge length is built
    from oversampled strain-rate samples after a second integration).
    """
    if gap_tolerance_m is None:
        gap_tolerance_m = gauge_m
    forbidden = np.zeros(source.n_channels, dtype=bool) if forbidden is None else np.asarray(forbidden, dtype=bool)
    rows = []
    s = source.channel_s
    for k, cs in enumerate(target.channel_s):
        a, b = _endpoints(float(cs), gauge_m)
        idx_a, w_a, st_a = _gauge_weights(s, a - 1e-9, a + 1e-9, gap_tolerance_m,
                                          forbidden, float(s[0]), float(s[-1]))
        idx_b, w_b, st_b = _gauge_weights(s, b - 1e-9, b + 1e-9, gap_tolerance_m,
                                          forbidden, float(s[0]), float(s[-1]))
        status = RowStatus.OK if st_a is RowStatus.OK and st_b is RowStatus.OK else (
            st_a if st_a is not RowStatus.OK else st_b)
        if status is RowStatus.OK:
            idx = np.concatenate([idx_b, idx_a])
            w = np.concatenate([w_b / gauge_m, -w_a / gauge_m])
            keep = np.abs(w) > 1e-15
            idx, w = idx[keep], w[keep]
        else:
            idx, w = np.empty(0, int), np.empty(0, float)
        rows.append(OperatorRow(k, float(cs), idx, w, status))
    return ReconstructionOperator(rows, source, target, gauge_m,
                                  'gauge_difference', gap_tolerance_m, forbidden)


def reconstruct_channels(gather, target_geometry: ChannelGeometry, gauge_m: float,
                         mode: str = 'average', gap_tolerance_m: float | None = None,
                         forbidden: np.ndarray | None = None):
    """Rebuild strain-rate channels on a curved fibre with an explicit gauge.

    Parameters
    ----------
    mode:
        ``'average'`` integrates the source strain-rate field over the gauge
        interval (re-gauge an oversampled DAS record); ``'difference'`` applies
        the gauge-difference operator to an axial field.
    """
    from .gather import DASGather
    builder = (gauge_average_operator if mode == 'average' else
               gauge_difference_operator if mode == 'difference' else None)
    if builder is None:
        raise ValueError("mode must be 'average' or 'difference'")
    op = builder(gather.geometry, target_geometry, gauge_m, gap_tolerance_m, forbidden)
    return gather.copy(op.apply(gather.data), geometry=target_geometry,
                       tags={**gather.tags, 'gauge_m': f'{gauge_m:g}',
                             'reconstruction': mode}), op


# ---------------------------------------------------------------------------
# directional sensitivity along the curved cable
# ---------------------------------------------------------------------------

def axial_projection(fiber, s, polarization: np.ndarray) -> np.ndarray:
    """``t(s) . p`` projection of a polarisation vector or per-channel field."""
    t = fiber.tangent_at(s)
    p = np.asarray(polarization, dtype=float)
    if p.ndim == 1:
        if p.size != t.shape[1]:
            raise ValueError('polarization dimensionality does not match fiber')
        return t @ p
    if p.shape != t.shape:
        raise ValueError('polarization field must have one tangent vector per channel')
    return np.einsum('ij,ij->i', t, p)


def directional_sensitivity_p(fiber, s, slowness_azimuth_deg: float,
                              takeoff_deg: float = 0.0) -> np.ndarray:
    """DAS P-wave amplitude factor ``(t . n)^2`` for a plane P arrival."""
    a = np.radians(slowness_azimuth_deg)
    th = np.radians(takeoff_deg)
    dim = fiber.dimension
    if dim == 2:
        n = np.array([np.sin(a), np.cos(a)])
    else:
        n = np.array([np.sin(th) * np.sin(a), np.sin(th) * np.cos(a), np.cos(th)])
    t = fiber.tangent_at(s)
    p = t @ n
    return p * p


def directional_sensitivity_s(fiber, s, slowness_azimuth_deg: float,
                              polarization_azimuth_deg: float,
                              takeoff_deg: float = 0.0,
                              polarization_takeoff_deg: float = 90.0) -> np.ndarray:
    """DAS S-wave amplitude factor ``(t . n)(t . e)``.

    ``n`` is the slowness direction and ``e`` the (transverse) polarisation.
    Channels near a blind zone (``t`` perpendicular to either vector) read ~0.
    """
    def vec(az_deg, to_deg):
        az, to = np.radians(az_deg), np.radians(to_deg)
        if fiber.dimension == 2:
            return np.array([np.sin(az), np.cos(az)])
        return np.array([np.sin(to) * np.sin(az), np.sin(to) * np.cos(az), np.cos(to)])

    n = vec(slowness_azimuth_deg, takeoff_deg)
    e = vec(polarization_azimuth_deg, polarization_takeoff_deg)
    t = fiber.tangent_at(s)
    return (t @ n) * (t @ e)


def axial_field(three_component: np.ndarray, fiber, s) -> np.ndarray:
    """Project a 3-C particle field ``(..., 2|3, n_time)`` onto cable tangent."""
    t = fiber.tangent_at(s)
    three_component = np.asarray(three_component, dtype=float)
    if three_component.shape[0] != s.size:
        raise ValueError('3C field must have one row per arc position')
    if three_component.shape[1] != t.shape[1]:
        raise ValueError('component count does not match fiber dimension')
    return np.einsum('ij,ijk->ik', t, three_component)
