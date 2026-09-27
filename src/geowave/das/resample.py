from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction
from .geometry import FiberGeometry, FiberBreak


@dataclass(frozen=True)
class ChannelProvenance:
    """How one output channel was built from original fiber channels."""
    new_channel_id: str
    segment: int
    s_m: float
    x: float
    y: float
    z: float
    sources: tuple[tuple[str, float], ...]  # (original channel id, weight), weights sum to 1

    def dominant_source(self) -> str:
        return max(self.sources, key=lambda sw: sw[1])[0]


@dataclass(frozen=True)
class SpatialResampling:
    """Traceability record of a channel-space resampling run."""
    method: str
    provenance: tuple[ChannelProvenance, ...]
    weights: np.ndarray  # (n_new, n_original) normalised mixing matrix

    def origins_of(self, new_channel_id: str) -> tuple[tuple[str, float], ...]:
        for p in self.provenance:
            if p.new_channel_id == new_channel_id:
                return p.sources
        raise KeyError(new_channel_id)

    def position_of(self, new_channel_id: str) -> tuple[float, float, float]:
        for p in self.provenance:
            if p.new_channel_id == new_channel_id:
                return (p.x, p.y, p.z)
        raise KeyError(new_channel_id)


@dataclass(frozen=True)
class ResampledDas:
    data: np.ndarray
    geometry: FiberGeometry
    resampling: SpatialResampling


def target_grid(geometry: FiberGeometry, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced arc positions covering every segment of the fiber."""
    if spacing_m <= 0:
        raise ValueError('spacing must be positive')
    centers = []
    segments = []
    for extent in (geometry.segment_extent(s) for s in range(geometry.segment_count)):
        if extent is None:
            continue
        seg, _, _, s0, s1 = extent
        length = s1 - s0
        if length <= 0:
            centers.append(s0)
        else:
            n = max(2, int(np.ceil(length / spacing_m)) + 1)
            centers.extend(np.linspace(s0, s1, n))
        segments.extend([seg] * (len(centers) - len(segments)))
    return np.asarray(centers, dtype=float), np.asarray(segments, dtype=int)


def resample_channels(data: np.ndarray, geometry: FiberGeometry, spacing_m: float,
                      *, method: str = 'linear', centers_s_m=None, segments=None,
                      min_coverage: float = 0.5, new_id_prefix: str = 'das') -> ResampledDas:
    """Resample channels onto an even spatial grid, keeping full provenance.

    Every output channel records which original channels (and with what
    weights) built it, so the original-channel-to-position relation survives
    resampling. Mixing never crosses a break: each output channel draws only
    from its own fiber segment.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] != geometry.n_channels:
        raise ValueError('data must be (n_channels, n_time) matching the geometry')
    if centers_s_m is None:
        centers_s_m, segments = target_grid(geometry, spacing_m)
    centers_s_m = np.asarray(centers_s_m, dtype=float)
    segments = np.asarray(segments, dtype=int)

    half = spacing_m / 2.0
    n_new = centers_s_m.size
    n_old = geometry.n_channels
    weights = np.zeros((n_new, n_old))
    provenance = []
    points = np.full((n_new, 3), np.nan)
    tangents = np.zeros((n_new, 3))
    out_segments = np.full(n_new, -1, dtype=int)
    out_s = np.full(n_new, np.nan)

    for k in range(n_new):
        seg = int(segments[k])
        center = float(centers_s_m[k])
        if method == 'linear':
            pair = geometry.interp_pair(seg, center)
            rows = () if pair is None else pair
        elif method == 'boxcar':
            win = geometry.boxcar_weights(seg, center, half)
            total = win.weights.sum()
            rows = tuple(zip(win.indices.tolist(), (win.weights / total).tolist())) \
                if total > 0 and win.covered_m >= min_coverage * spacing_m else ()
        else:
            raise ValueError("method must be 'linear' or 'boxcar'")
        located = geometry.position_at(seg, center)
        if located is not None:
            points[k], tangents[k] = located
            out_segments[k] = seg
            out_s[k] = center
        total = sum(w for _, w in rows)
        sources = tuple((geometry.channel_ids[i], w / total) for i, w in rows) \
            if total > 0 else ()
        for i, w in rows:
            if total > 0:
                weights[k, i] = w / total
        provenance.append(ChannelProvenance(
            f'{new_id_prefix}-{k:05d}', seg, center,
            float(points[k, 0]), float(points[k, 1]), float(points[k, 2]), sources))

    out = np.full((n_new, x.shape[1]), np.nan)
    for k in range(n_new):
        ind = np.nonzero(weights[k])[0]
        if ind.size == 0:
            continue
        block = x[ind, :]
        ok = np.isfinite(block)
        w = weights[k, ind]
        wsum = w @ ok
        acc = w @ np.where(ok, block, 0.0)
        good = wsum > 1e-12
        out[k, good] = acc[good] / wsum[good]

    new_geometry = _resampled_geometry(geometry, provenance, points, out_segments,
                                       out_s, tangents)
    record = SpatialResampling(method, tuple(provenance), weights)
    return ResampledDas(out, new_geometry, record)


def _resampled_geometry(old: FiberGeometry, provenance, points, segments, s_m,
                        tangents) -> FiberGeometry:
    ids = [p.new_channel_id for p in provenance]
    breaks = []
    for k in range(len(provenance) - 1):
        if segments[k] >= 0 and segments[k + 1] >= 0 and segments[k + 1] != segments[k]:
            breaks.append(FiberBreak(k, float('nan'), 'segment-boundary'))
    return FiberGeometry._assemble(ids, points, segments, s_m, tangents, breaks,
                                   old.reference_direction_xy)


def resample_time(data: np.ndarray, dt_s: float, new_dt_s: float,
                  max_denominator: int = 1000) -> tuple[np.ndarray, float]:
    """Polyphase resampling along time; channel provenance is unaffected."""
    if dt_s <= 0 or new_dt_s <= 0:
        raise ValueError('positive dt required')
    ratio = Fraction(dt_s / new_dt_s).limit_denominator(max_denominator)
    x = np.asarray(data, dtype=float)
    out = np.stack([resample_poly(row, ratio.numerator, ratio.denominator)
                    for row in x]) if x.ndim == 2 else \
        resample_poly(x, ratio.numerator, ratio.denominator)
    return out, float(new_dt_s)
