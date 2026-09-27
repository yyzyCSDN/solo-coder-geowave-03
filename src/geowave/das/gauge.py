from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .geometry import FiberGeometry


@dataclass(frozen=True)
class GaugeWindow:
    """One output channel reconstructed by integrating strain rate over a gauge."""
    segment: int
    center_s_m: float
    lo_s_m: float
    hi_s_m: float
    indices: np.ndarray
    weights: np.ndarray  # normalised trapezoid weights, sum to 1
    covered_fraction: float
    partial: bool


@dataclass(frozen=True)
class GaugeResult:
    data: np.ndarray          # (n_windows, n_time) gauge-averaged strain rate
    windows: tuple[GaugeWindow, ...]
    centers_s_m: np.ndarray
    segments: np.ndarray


def as_strain_rate(data: np.ndarray, dt_s: float, *, input_kind: str = 'strain-rate',
                   phase_to_strain_scale: float = 1.0) -> np.ndarray:
    """Bring interrogator output to strain-rate units before any spatial operator.

    - 'strain-rate' : returned unchanged (already dε/dt per channel).
    - 'strain'      : differentiated along time with ``np.gradient``.
    - 'phase'       : raw interferometric phase, scaled by ``phase_to_strain_scale``
                      (strain per radian, set from interrogator constants) and
                      then differentiated along time.
    """
    if dt_s <= 0:
        raise ValueError('positive dt required')
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError('DAS data must be a (channel, time) matrix')
    if input_kind == 'strain-rate':
        return x
    if input_kind == 'strain':
        return np.gradient(x, dt_s, axis=1)
    if input_kind == 'phase':
        return np.gradient(x * phase_to_strain_scale, dt_s, axis=1)
    raise ValueError(f"unknown input kind '{input_kind}'")


def gauge_windows(geometry: FiberGeometry, gauge_length_m: float,
                  centers_s_m=None, segments=None) -> tuple[GaugeWindow, ...]:
    """Integration windows of ``gauge_length_m`` centred on the requested positions.

    Windows never cross a break: each window lives inside one fiber segment.
    """
    if gauge_length_m <= 0:
        raise ValueError('gauge length must be positive')
    half = gauge_length_m / 2.0
    if centers_s_m is None:
        centers_s_m = geometry.arclength_m[geometry.valid]
        segments = geometry.segments[geometry.valid]
    centers_s_m = np.asarray(centers_s_m, dtype=float)
    if segments is None:
        raise ValueError('segments must be given with explicit centers')
    segments = np.asarray(segments, dtype=int)
    if segments.shape != centers_s_m.shape:
        raise ValueError('centers and segments must align')
    windows = []
    for seg, center in zip(segments, centers_s_m):
        win = geometry.boxcar_weights(int(seg), float(center), half)
        total = win.weights.sum()
        weights = win.weights / total if total > 0 else win.weights
        extent = geometry.segment_extent(int(seg))
        requested = gauge_length_m
        windows.append(GaugeWindow(
            int(seg), float(center), win.lo, win.hi, win.indices, weights,
            win.covered_m / requested if requested else 1.0,
            not win.complete and extent is not None))
    return tuple(windows)


def apply_gauge(data: np.ndarray, geometry: FiberGeometry, gauge_length_m: float,
                centers_s_m=None, segments=None, min_coverage: float = 0.5) -> GaugeResult:
    """Reconstruct channels by gauge-length integration of strain rate.

    For every requested centre the strain-rate matrix is integrated along the
    fiber with trapezoid weights and normalised by the covered gauge length,
    yielding the mean strain rate over the gauge. Windows whose covered length
    falls below ``min_coverage`` of the requested gauge produce NaN rows.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] != geometry.n_channels:
        raise ValueError('data must be (n_channels, n_time) matching the geometry')
    windows = gauge_windows(geometry, gauge_length_m, centers_s_m, segments)
    out = np.full((len(windows), x.shape[1]), np.nan)
    for k, win in enumerate(windows):
        if win.indices.size == 0 or win.covered_fraction < min_coverage:
            continue
        block = x[win.indices, :]
        ok = np.isfinite(block)
        wsum = win.weights @ ok
        acc = win.weights @ np.where(ok, block, 0.0)
        good = wsum > 1e-12
        out[k, good] = acc[good] / wsum[good]
    return GaugeResult(out, windows,
                       np.array([w.center_s_m for w in windows]),
                       np.array([w.segment for w in windows], dtype=int))
