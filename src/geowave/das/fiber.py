from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class FiberPath:
    """Piecewise-linear cable geometry parameterised by arc length.

    ``points`` are world-coordinate vertices (2D or 3D) and ``arc_length`` is
    the cumulative along-fibre distance at every vertex. Channels never carry
    station coordinates directly; they reference an arc-length position on the
    fibre instead, so back-tracked and curved sections stay unambiguous.
    """

    points: np.ndarray
    arc_length: np.ndarray

    def __post_init__(self):
        pts = np.asarray(self.points, dtype=float)
        s = np.asarray(self.arc_length, dtype=float)
        if pts.ndim != 2 or pts.shape[1] not in (2, 3) or pts.shape[0] < 2:
            raise ValueError('fiber needs at least two 2D/3D vertices')
        if s.shape != (pts.shape[0],):
            raise ValueError('arc_length length must match vertices')
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if np.any(seg <= 0.0):
            raise ValueError('duplicate consecutive vertices produce zero-length fiber segments')
        if np.any(np.diff(s) <= 0.0):
            raise ValueError('arc length must be strictly increasing')
        if abs(s[-1] - seg.sum()) > 1e-9 * max(1.0, seg.sum()):
            raise ValueError('arc_length is not consistent with vertex spacing')
        object.__setattr__(self, 'points', pts)
        object.__setattr__(self, 'arc_length', s)

    @classmethod
    def from_vertices(cls, x, y, z=None) -> 'FiberPath':
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape or x.ndim != 1:
            raise ValueError('x and y must be equally sized 1D arrays')
        cols = [x, y]
        if z is not None:
            z = np.asarray(z, dtype=float)
            if z.shape != x.shape:
                raise ValueError('z must match x/y length')
            cols.append(z)
        pts = np.column_stack(cols)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        return cls(pts, s)

    @property
    def dimension(self) -> int:
        return int(self.points.shape[1])

    @property
    def length_m(self) -> float:
        return float(self.arc_length[-1])

    @property
    def segment_tangents(self) -> np.ndarray:
        seg = np.diff(self.points, axis=0)
        return seg / np.linalg.norm(seg, axis=1, keepdims=True)

    @property
    def vertex_tangents(self) -> np.ndarray:
        """Unit tangents centred on vertices (natural for per-channel lookups)."""
        seg = np.diff(self.points, axis=0)
        seg = seg / np.linalg.norm(seg, axis=1, keepdims=True)
        out = np.empty_like(self.points)
        out[0] = seg[0]
        out[-1] = seg[-1]
        mid = seg[:-1] + seg[1:]
        n = np.linalg.norm(mid, axis=1, keepdims=True)
        n = np.where(n == 0.0, 1.0, n)
        out[1:-1] = mid / n
        return out

    def _segments_for(self, s) -> tuple[np.ndarray, np.ndarray]:
        s_arr = np.asarray(s, dtype=float)
        scalar = s_arr.ndim == 0
        s_arr = np.atleast_1d(s_arr)
        if np.any(s_arr < 0.0) or np.any(s_arr > self.length_m):
            raise ValueError('arc position outside fiber extent')
        idx = np.searchsorted(self.arc_length, s_arr, side='right') - 1
        idx = np.clip(idx, 0, self.points.shape[0] - 2)
        s0 = self.arc_length[idx]
        s1 = self.arc_length[idx + 1]
        frac = np.clip((s_arr - s0) / (s1 - s0), 0.0, 1.0)
        if scalar:
            return int(idx[0]), float(frac[0])
        return idx, frac

    def point_at(self, s) -> np.ndarray:
        idx, frac = self._segments_for(s)
        if np.isscalar(idx):
            return self.points[idx] + frac * (self.points[idx + 1] - self.points[idx])
        p0 = self.points[idx]
        p1 = self.points[idx + 1]
        return p0 + frac[:, None] * (p1 - p0)

    def tangent_at(self, s) -> np.ndarray:
        """Outward tangent of the segment containing each arc position."""
        idx, _ = self._segments_for(s)
        if np.isscalar(idx):
            return self.segment_tangents[idx].copy()
        return self.segment_tangents[idx]

    def azimuth_at(self, s) -> np.ndarray:
        """Survey azimuth in degrees (atan2 of east/north), same convention as LocalFrame."""
        t = self.tangent_at(s)
        ang = np.degrees(np.arctan2(t[..., 0], t[..., 1]))
        return (ang + 360.0) % 360.0


@dataclass
class ChannelGeometry:
    """Mapping between instrument channel order and physical fibre location."""

    channel_s: np.ndarray
    fiber: FiberPath
    channel_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        s = np.asarray(self.channel_s, dtype=float)
        if s.ndim != 1 or s.size < 1:
            raise ValueError('channel_s must be a 1D array with at least one channel')
        if s.size > 1 and np.any(np.diff(s) <= 0.0):
            raise ValueError('channel arc positions must be strictly increasing')
        if np.any(s < 0.0) or np.any(s > self.fiber.length_m):
            raise ValueError('channel outside fiber extent')
        self.channel_s = s
        if not self.channel_ids:
            self.channel_ids = [f'ch-{i:05d}' for i in range(s.size)]
        elif len(self.channel_ids) != s.size:
            raise ValueError('channel_ids length mismatch')

    @classmethod
    def from_arc_positions(cls, fiber: FiberPath, channel_s, ids=None) -> 'ChannelGeometry':
        return cls(np.asarray(channel_s, dtype=float), fiber, list(ids) if ids is not None else [])

    @classmethod
    def from_coordinates(cls, x, y, z=None, ids=None) -> 'ChannelGeometry':
        """Build geometry when every channel is a measured vertex of the cable."""
        fiber = FiberPath.from_vertices(x, y, z)
        n = len(np.asarray(x, dtype=float))
        s = fiber.arc_length.copy()
        return cls(s, fiber, list(ids) if ids is not None else [f'ch-{i:05d}' for i in range(n)])

    @property
    def n_channels(self) -> int:
        return int(self.channel_s.size)

    @property
    def positions(self) -> np.ndarray:
        return self.fiber.point_at(self.channel_s)

    @property
    def tangents(self) -> np.ndarray:
        return self.fiber.tangent_at(self.channel_s)

    @property
    def azimuths_deg(self) -> np.ndarray:
        return self.fiber.azimuth_at(self.channel_s)

    def arc_gaps(self) -> np.ndarray:
        return np.diff(self.channel_s)

    def large_gaps(self, gap_m: float) -> np.ndarray:
        """Indices ``i`` (1-based downstream channel) where the incoming gap exceeds ``gap_m``."""
        gaps = self.arc_gaps()
        return np.flatnonzero(gaps > gap_m) + 1


def detect_reversed(geometry: ChannelGeometry,
                    reference_azimuth_deg: float | None = None,
                    cos_threshold: float = 0.0) -> np.ndarray:
    """Flag channels whose cable tangent runs against the reference lay direction.

    A U-turn / back-tracked cable section has ``t . t_ref < cos_threshold``; the
    default threshold (0.0) flags anything more than 90 degrees off the first
    channel's tangent.
    """
    t = geometry.tangents
    if reference_azimuth_deg is None:
        ref = t[0]
    else:
        a = np.radians(reference_azimuth_deg)
        ref = np.array([np.sin(a), np.cos(a)] + [0.0] * (t.shape[1] - 2))
        ref = ref / np.linalg.norm(ref)
    return (t @ ref) < cos_threshold


def hairpin_indices(geometry: ChannelGeometry, cos_threshold: float = 0.0) -> np.ndarray:
    """Channel indices where the heading relative to the previous channel reverses."""
    t = geometry.tangents
    dots = np.einsum('ij,ij->i', t[1:], t[:-1])
    return np.flatnonzero(dots < cos_threshold) + 1
