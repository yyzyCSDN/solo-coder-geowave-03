from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np

_TOL = 1e-9


@dataclass(frozen=True)
class FiberBreak:
    """A discontinuity in the interrogated fiber, located after ``after_index``."""
    after_index: int
    gap_m: float
    reason: str  # 'gap' | 'declared' | 'nan-geometry'


@dataclass(frozen=True)
class ReversedSection:
    """Contiguous channels whose tangent points against the fiber reference direction."""
    start_index: int
    stop_index: int  # inclusive
    mean_angle_deg: float


@dataclass(frozen=True)
class FiberTurn:
    """A sharp plan-view change of fiber direction (e.g. a hairpin)."""
    index: int
    angle_deg: float


@dataclass(frozen=True)
class _Window:
    indices: np.ndarray
    weights: np.ndarray  # trapezoid integration weights, in metres
    lo: float
    hi: float
    covered_m: float
    complete: bool


class FiberGeometry:
    """Channel-to-space mapping for a fiber that may be curved, broken or reversed.

    Channels are never treated as ordinary receiver points: every channel carries a
    3-D position, a segment label (segments are separated by breaks), an arc length
    measured inside its segment and a unit tangent vector used for gauge integration
    and directional sensitivity.
    """

    def __init__(self, channel_ids, points, valid, segments, s_m, tangent,
                 breaks, reversed_sections, turns, reference_xy):
        ids = tuple(str(v) for v in channel_ids)
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError('fiber points must have shape (n, 3)')
        n = pts.shape[0]
        if len(ids) != n:
            raise ValueError('one channel id per fiber point required')
        if len(set(ids)) != n:
            raise ValueError('channel ids must be unique')
        self._ids = ids
        self._index = {cid: i for i, cid in enumerate(ids)}
        self._points = pts
        self._valid = np.asarray(valid, dtype=bool)
        self._segments = np.asarray(segments, dtype=int)
        self._s = np.asarray(s_m, dtype=float)
        self._tangent = np.asarray(tangent, dtype=float)
        self._breaks = tuple(breaks)
        self._reversed_sections = tuple(reversed_sections)
        self._turns = tuple(turns)
        self._reference_xy = np.asarray(reference_xy, dtype=float)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, channel_ids, x, y, z=None, *, declared_breaks=(),
              max_step_m=None, reversal_threshold_deg: float = 150.0,
              turn_threshold_deg: float = 60.0, reference_direction=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.zeros_like(x) if z is None else np.asarray(z, dtype=float)
        if not (x.shape == y.shape == z.shape and x.ndim == 1):
            raise ValueError('x, y and z must be equally sized 1-D arrays')
        n = x.size
        if n == 0:
            raise ValueError('at least one fiber channel required')
        points = np.column_stack([x, y, z])
        valid = np.isfinite(points).all(axis=1)

        segments, breaks = cls._segment(n, points, valid, set(declared_breaks), max_step_m)
        s_m = cls._arclength(n, points, valid, segments)
        tangent = cls._tangents(points, valid, segments)
        ref_xy = cls._reference_xy(reference_direction, tangent, valid)
        reversals, turns = cls._direction_marks(
            tangent, segments, valid, ref_xy, reversal_threshold_deg, turn_threshold_deg)
        return cls(channel_ids, points, valid, segments, s_m, tangent,
                   breaks, reversals, turns, ref_xy)

    @classmethod
    def _assemble(cls, channel_ids, points, segments, s_m, tangent, breaks,
                  reference_xy, reversal_threshold_deg=150.0, turn_threshold_deg=60.0):
        points = np.asarray(points, dtype=float)
        valid = np.isfinite(points).all(axis=1)
        reversals, turns = cls._direction_marks(
            tangent, segments, valid, np.asarray(reference_xy, dtype=float),
            reversal_threshold_deg, turn_threshold_deg)
        return cls(channel_ids, points, valid, segments, s_m, tangent,
                   breaks, reversals, turns, reference_xy)

    @staticmethod
    def _segment(n, points, valid, declared, max_step_m):
        segments = -np.ones(n, dtype=int)
        cuts: dict[int, FiberBreak] = {}
        seg = 0
        if valid[0]:
            segments[0] = 0
        for i in range(1, n):
            reason = None
            if not valid[i - 1] or not valid[i]:
                reason = 'nan-geometry'
            elif i - 1 in declared:
                reason = 'declared'
            elif max_step_m is not None:
                step = float(np.linalg.norm(points[i] - points[i - 1]))
                if step > max_step_m:
                    reason = 'gap'
            if reason is not None:
                if reason == 'nan-geometry':
                    gap = float('nan')
                else:
                    gap = float(np.linalg.norm(points[i] - points[i - 1]))
                cuts.setdefault(i - 1, FiberBreak(i - 1, gap, reason))
                seg += 1
            if valid[i]:
                segments[i] = seg
        return segments, tuple(cuts.values())

    @staticmethod
    def _arclength(n, points, valid, segments):
        s_m = np.full(n, np.nan)
        for _, idx in FiberGeometry._runs(valid, segments):
            p = points[idx]
            steps = np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1)]
            s_m[idx] = np.cumsum(steps)
        return s_m

    @staticmethod
    def _tangents(points, valid, segments):
        tangent = np.zeros_like(points)
        for _, idx in FiberGeometry._runs(valid, segments):
            p = points[idx]
            t = np.zeros_like(p)
            m = idx.size
            if m == 2:
                d = p[1] - p[0]
                unit = d / np.linalg.norm(d)
                t[0] = t[1] = unit
            elif m > 2:
                central = p[2:] - p[:-2]
                norms = np.linalg.norm(central, axis=1)
                ok = norms > _TOL
                t[1:-1][ok] = central[ok] / norms[ok, None]
                t[0] = _unit(p[1] - p[0])
                t[-1] = _unit(p[-1] - p[-2])
                for k in range(m):
                    if np.linalg.norm(t[k]) <= _TOL:
                        if k > 0 and np.linalg.norm(t[k - 1]) > _TOL:
                            t[k] = t[k - 1]
                        elif k + 1 < m:
                            t[k] = _unit(p[k + 1] - p[k])
            tangent[idx] = t
        return tangent

    @staticmethod
    def _reference_xy(reference_direction, tangent, valid):
        if reference_direction is None:
            first = int(np.argmax(valid))
            ref = tangent[first].copy()
        else:
            ref = np.asarray(reference_direction, dtype=float)
            if ref.shape not in ((2,), (3,)):
                raise ValueError('reference direction must be 2-D or 3-D')
            if ref.shape[0] == 2:
                ref = np.r_[ref, 0.0]
        ref_xy = ref[:2]
        norm = np.linalg.norm(ref_xy)
        if norm <= _TOL:
            raise ValueError('reference direction has no plan-view component')
        return ref_xy / norm

    @staticmethod
    def _direction_marks(tangent, segments, valid, reference_xy,
                         reversal_threshold_deg, turn_threshold_deg):
        n = tangent.shape[0]
        xy = tangent[:, :2]
        xy_norm = np.linalg.norm(xy, axis=1)
        usable = valid & (xy_norm > _TOL)
        xy_hat = np.zeros_like(xy)
        xy_hat[usable] = xy[usable] / xy_norm[usable, None]
        cosang = xy_hat @ reference_xy
        angles = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        angles[~usable] = np.nan

        reversals: list[ReversedSection] = []
        run: list[int] = []
        for i in range(n):
            hot = usable[i] and angles[i] > reversal_threshold_deg
            same_segment = run and segments[i] == segments[run[-1]]
            if hot and (not run or same_segment):
                run.append(i)
            else:
                if run:
                    reversals.append(_make_reversed(run, angles))
                    run = []
                if hot:
                    run = [i]
        if run:
            reversals.append(_make_reversed(run, angles))

        turns: list[FiberTurn] = []
        for _, idx in FiberGeometry._runs(valid, segments):
            for a, b in zip(idx[:-1], idx[1:]):
                if usable[a] and usable[b]:
                    c = float(np.clip(np.dot(xy_hat[a], xy_hat[b]), -1.0, 1.0))
                    ang = float(np.degrees(np.arccos(c)))
                    if ang > turn_threshold_deg:
                        turns.append(FiberTurn(int(b), ang))
        return tuple(reversals), tuple(turns)

    @staticmethod
    def _runs(valid, segments):
        runs = []
        start = None
        seg_at = None
        for i in range(valid.size):
            if valid[i]:
                if start is None or segments[i] != seg_at:
                    if start is not None:
                        runs.append((seg_at, np.arange(start, i)))
                    start, seg_at = i, int(segments[i])
            elif start is not None:
                runs.append((seg_at, np.arange(start, i)))
                start = None
        if start is not None:
            runs.append((seg_at, np.arange(start, valid.size)))
        return runs

    # ------------------------------------------------------------- properties
    @property
    def n_channels(self) -> int:
        return self._points.shape[0]

    @property
    def channel_ids(self) -> tuple[str, ...]:
        return self._ids

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def x(self) -> np.ndarray:
        return self._points[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self._points[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self._points[:, 2]

    @property
    def valid(self) -> np.ndarray:
        return self._valid

    @property
    def segments(self) -> np.ndarray:
        return self._segments

    @property
    def arclength_m(self) -> np.ndarray:
        """Segment-local arc length; resets to zero after every break."""
        return self._s

    @property
    def tangent(self) -> np.ndarray:
        return self._tangent

    @property
    def reference_direction_xy(self) -> np.ndarray:
        return self._reference_xy

    @property
    def breaks(self) -> tuple[FiberBreak, ...]:
        return self._breaks

    @property
    def reversed_sections(self) -> tuple[ReversedSection, ...]:
        return self._reversed_sections

    @property
    def turns(self) -> tuple[FiberTurn, ...]:
        return self._turns

    @property
    def segment_count(self) -> int:
        return int(self._segments[self._valid].max() + 1) if self._valid.any() else 0

    # ----------------------------------------------------------------- lookup
    def index_of(self, channel_id: str) -> int:
        return self._index[str(channel_id)]

    def location(self, channel_id: str) -> tuple[float, float, float]:
        return tuple(float(v) for v in self._points[self.index_of(channel_id)])

    def segment_of(self, channel_id: str) -> int:
        return int(self._segments[self.index_of(channel_id)])

    def is_reversed(self, channel_id: str) -> bool:
        i = self.index_of(channel_id)
        return any(sec.start_index <= i <= sec.stop_index for sec in self._reversed_sections)

    def channel_mask_reversed(self) -> np.ndarray:
        mask = np.zeros(self.n_channels, dtype=bool)
        for sec in self._reversed_sections:
            mask[sec.start_index:sec.stop_index + 1] = True
        return mask

    def channel_mask_turns(self) -> np.ndarray:
        mask = np.zeros(self.n_channels, dtype=bool)
        for turn in self._turns:
            mask[turn.index] = True
        return mask

    def channel_mask_segment_ends(self) -> np.ndarray:
        """Channels immediately adjacent to a break or a segment boundary."""
        mask = np.zeros(self.n_channels, dtype=bool)
        for _, idx in self._runs(self._valid, self._segments):
            mask[idx[0]] = True
            mask[idx[-1]] = True
        return mask

    # ------------------------------------------------------- spatial operators
    def boxcar_weights(self, segment: int, center_s: float, half_width_m: float,
                       tol: float = _TOL) -> _Window:
        """Trapezoid weights of a [s-h, s+h] integration window inside one segment."""
        run = self._run_arrays(segment)
        if run is None:
            return _Window(np.empty(0, int), np.empty(0), center_s, center_s, 0.0, False)
        idx, ss = run
        if idx.size == 1:
            complete = half_width_m <= tol
            w = np.ones(1) if complete else np.empty(0)
            ind = idx if complete else np.empty(0, int)
            return _Window(ind, w, ss[0], ss[0], 0.0, complete)
        lo = max(center_s - half_width_m, ss[0])
        hi = min(center_s + half_width_m, ss[-1])
        weights: dict[int, float] = {}
        covered = 0.0
        j0 = max(0, int(np.searchsorted(ss, lo, side='right')) - 1)
        j1 = min(idx.size - 1, int(np.searchsorted(ss, hi, side='left')))
        for j in range(j0, j1):
            a = max(ss[j], lo)
            b = min(ss[j + 1], hi)
            if b <= a + tol:
                continue
            d = ss[j + 1] - ss[j]
            weights[idx[j]] = weights.get(idx[j], 0.0) + (b - a) * (2 * ss[j + 1] - a - b) / (2 * d)
            weights[idx[j + 1]] = weights.get(idx[j + 1], 0.0) + (b - a) * (a + b - 2 * ss[j]) / (2 * d)
            covered += b - a
        ind = np.array(sorted(weights), dtype=int)
        w = np.array([weights[i] for i in ind], dtype=float)
        requested = 2.0 * half_width_m
        complete = covered >= requested - tol * max(1.0, requested)
        return _Window(ind, w, lo, hi, float(covered), bool(complete))

    def interp_pair(self, segment: int, s_value: float):
        """Linear interpolation weights at an arc position inside one segment."""
        run = self._run_arrays(segment)
        if run is None:
            return None
        idx, ss = run
        if s_value < ss[0] - _TOL or s_value > ss[-1] + _TOL:
            return None
        if idx.size == 1:
            return ((int(idx[0]), 1.0),)
        j = int(np.searchsorted(ss, s_value, side='right')) - 1
        j = min(max(j, 0), idx.size - 2)
        if abs(ss[j + 1] - ss[j]) <= _TOL:
            return ((int(idx[j]), 1.0),)
        if s_value >= ss[-1] - _TOL:
            return ((int(idx[-1]), 1.0),)
        f = (s_value - ss[j]) / (ss[j + 1] - ss[j])
        return ((int(idx[j]), 1.0 - f), (int(idx[j + 1]), f))

    def position_at(self, segment: int, s_value: float):
        pair = self.interp_pair(segment, s_value)
        if pair is None:
            return None
        point = sum(w * self._points[i] for i, w in pair)
        tangent = _unit(sum(w * self._tangent[i] for i, w in pair))
        return point, tangent

    def segment_extent(self, segment: int):
        run = self._run_arrays(segment)
        if run is None:
            return None
        idx, ss = run
        return int(segment), int(idx[0]), int(idx[-1]), float(ss[0]), float(ss[-1])

    def _run_arrays(self, segment: int):
        for seg, idx in self._runs(self._valid, self._segments):
            if seg == segment:
                return idx, self._s[idx]
        return None


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > _TOL else np.zeros_like(vector)


def _make_reversed(run, angles):
    mean_angle = float(np.nanmean([angles[i] for i in run]))
    return ReversedSection(int(run[0]), int(run[-1]), mean_angle)


def load_fiber_geometry(path: str | Path, **build_kwargs) -> FiberGeometry:
    """Read a channel geometry CSV (columns: channel_id?, x, y, z?)."""
    with Path(path).open(newline='', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('fiber geometry file is empty')
    fields = set(rows[0])
    if not {'x', 'y'}.issubset(fields):
        raise ValueError("fiber geometry requires 'x' and 'y' columns")
    ids = [row['channel_id'] for row in rows] if 'channel_id' in fields else \
        [str(i) for i in range(len(rows))]
    x = [float(row['x']) for row in rows]
    y = [float(row['y']) for row in rows]
    z = [float(row['z']) for row in rows] if 'z' in fields else None
    return FiberGeometry.build(ids, x, y, z, **build_kwargs)
