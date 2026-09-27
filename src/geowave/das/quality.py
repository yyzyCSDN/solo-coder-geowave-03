from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntFlag
import numpy as np

from .fiber import ChannelGeometry, detect_reversed
from .gather import DASGather


class QualityFlag(IntFlag):
    OK = 0
    DEAD_FIBER = 1          # broken fibre: gap beyond tolerance / no signal beyond
    DEAD_CHANNEL = 2        # zero variance, all-NaN or constant trace
    SATURATED = 4           # clipped against a digitiser rail
    REVERSED = 8            # cable section runs opposite to reference lay
    LOW_QUALITY = 16        # weak / noisy trace (SNR or RMS threshold)

    @property
    def label(self) -> str:
        return {0: 'ok', 1: 'dead-fiber', 2: 'dead-channel', 4: 'saturated',
                8: 'reversed', 16: 'low-quality'}.get(int(self), 'mixed')


@dataclass
class Segment:
    kind: str
    start_channel: int
    stop_channel: int           # inclusive
    detail: str = ''

    @property
    def length_channels(self) -> int:
        return self.stop_channel - self.start_channel + 1


@dataclass
class ChannelQC:
    index: int
    channel_id: str
    arc_s: float
    flags: QualityFlag
    rms: float
    std: float
    saturated_fraction: float


@dataclass
class QualityAssessment:
    flags: np.ndarray                           # QualityFlag per channel
    channel_qc: list[ChannelQC]
    segments: list[Segment]
    metrics: dict[str, np.ndarray] = field(default_factory=dict)

    def mask(self, flag: QualityFlag) -> np.ndarray:
        return (self.flags & int(flag)) != 0

    def channels(self, flag: QualityFlag) -> np.ndarray:
        return np.flatnonzero((self.flags & int(flag)) != 0)

    def merged_mask(self, *flags: QualityFlag) -> np.ndarray:
        value = 0
        for f in flags:
            value |= int(f)
        return (self.flags & value) != 0

    def good_mask(self, treat_low_quality_bad: bool = False) -> np.ndarray:
        bad = int(QualityFlag.DEAD_FIBER | QualityFlag.DEAD_CHANNEL |
                  QualityFlag.SATURATED | QualityFlag.REVERSED)
        if treat_low_quality_bad:
            bad |= int(QualityFlag.LOW_QUALITY)
        return (self.flags & bad) == 0


def contiguous_segments(mask: np.ndarray, kind: str, detail: str = '') -> list[Segment]:
    mask = np.asarray(mask, dtype=bool)
    out = []
    i = 0
    n = mask.size
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append(Segment(kind, i, j, detail))
            i = j + 1
        else:
            i += 1
    return out


def assess_quality(gather: DASGather,
                   fiber_gap_m: float | None = None,
                   reference_azimuth_deg: float | None = None,
                   reverse_cos_threshold: float = 0.0,
                   rms_floor: float | None = None,
                   rms_ceiling: float | None = None,
                   snr_db_floor: float | None = None,
                   signal_band_hz: tuple[float, float] | None = (1.0, 40.0),
                   saturation_quantile: float = 0.999,
                   saturated_fraction_limit: float = 0.02) -> QualityAssessment:
    """Flag broken fibre, reversed/back-tracked sections and poor traces.

    Parameters
    ----------
    fiber_gap_m:
        channel spacing above which the cable is treated as broken; defaults to
        3x median spacing. All channels downstream of a break (no fibre beyond
        it) are flagged ``DEAD_FIBER``; the break-facing channel is flagged as
        well by ``fiber_gap_m`` bookkeeping in :func:`break_segments`.
    rms_floor / rms_ceiling:
        absolute RMS envelopes; when omitted, data-driven limits are used
        (below 5th / above 99.5th percentile by 0.125x and 8x).
    snr_db_floor:
        optional in-band / out-of-band SNR gate.
    """
    g = gather.geometry
    data = gather.data
    n = gather.n_channels

    finite = np.isfinite(data)
    std = np.nanstd(data, axis=1)
    rms = np.sqrt(np.nanmean(data * np.where(finite, data, 0.0), axis=1))
    rms = np.where(np.isfinite(rms), rms, 0.0)

    dead = np.all(~finite, axis=1) | (std <= 1e-12 * (np.nanmax(std) + 1e-30))

    # saturation: many samples piled up at the channel extrema
    sat_frac = np.zeros(n)
    for i in range(n):
        row = data[i, finite[i]]
        if row.size:
            hi = np.quantile(row, saturation_quantile)
            lo = np.quantile(row, 1.0 - saturation_quantile)
            span = hi - lo
            if span > 0:
                sat_frac[i] = float(np.mean((row >= hi - 1e-9 * span) |
                                            (row <= lo + 1e-9 * span)))
    saturated = sat_frac > saturated_fraction_limit

    # low quality: per-channel statistics against data-driven envelopes
    positive = rms[rms > 0]
    if positive.size:
        ref_lo = float(np.quantile(positive, 0.05))
        ref_hi = float(np.quantile(positive, 0.995))
        floor = rms_floor if rms_floor is not None else 0.125 * ref_lo
        ceiling = rms_ceiling if rms_ceiling is not None else 8.0 * ref_hi
    else:
        floor = rms_floor or 0.0
        ceiling = rms_ceiling or np.inf
    low_q = ((rms < floor) | (rms > ceiling) | dead)

    if snr_db_floor is not None:
        snr = _band_snr_db(data, gather.dt_s, signal_band_hz)
        low_q |= snr < snr_db_floor
    else:
        snr = np.full(n, np.nan)
    # geometry-driven flags
    reversed_mask = detect_reversed(g, reference_azimuth_deg, reverse_cos_threshold)
    break_after, dead_fiber = _dead_fiber_mask(g, fiber_gap_m)

    flags = np.zeros(n, dtype=np.int16)
    flags[dead_fiber] |= int(QualityFlag.DEAD_FIBER)
    flags[dead] |= int(QualityFlag.DEAD_CHANNEL)
    flags[saturated & ~dead_fiber] |= int(QualityFlag.SATURATED)
    flags[reversed_mask] |= int(QualityFlag.REVERSED)
    weak = (rms < floor) | (rms > ceiling)
    flags[weak & ~dead_fiber & ~dead & ~saturated] |= int(QualityFlag.LOW_QUALITY)
    if snr_db_floor is not None:
        flags[(snr < snr_db_floor) & ~dead_fiber & ~dead & ~saturated] |= int(
            QualityFlag.LOW_QUALITY)

    segments: list[Segment] = []
    if break_after:
        first = break_after[0]
        gap = g.arc_gaps()[first - 1]
        segments.append(Segment('dead-fiber', first, n - 1,
                                f'break after channel {first - 1}: gap {gap:.1f} m'))
    segments += contiguous_segments(reversed_mask, 'reversed',
                                    'tangent opposite to reference lay direction')
    low_quality_seg = low_q & ~dead_fiber & ~dead & ~saturated
    segments += contiguous_segments(low_quality_seg, 'low-quality',
                                    'below SNR/RMS acceptance envelope')
    segments += contiguous_segments(dead & ~dead_fiber, 'dead-channel',
                                    'constant or fully non-finite trace')
    segments += contiguous_segments(saturated & ~dead_fiber, 'saturated',
                                    'clipped / rail-piled samples')
    segments.sort(key=lambda s_: s_.start_channel)

    qc = [ChannelQC(i, g.channel_ids[i], float(g.channel_s[i]),
                    QualityFlag(int(flags[i])), float(rms[i]), float(std[i]),
                    float(sat_frac[i]))
          for i in range(n)]
    return QualityAssessment(flags, qc, segments,
                             metrics={'rms': rms, 'std': std,
                                      'saturated_fraction': sat_frac, 'snr_db': snr})


def break_segments(geometry: ChannelGeometry, fiber_gap_m: float | None = None
                   ) -> list[tuple[int, int, float]]:
    """Return ``(channel_before, channel_after, gap_m)`` for each fibre break."""
    gaps = geometry.arc_gaps()
    if fiber_gap_m is None:
        fiber_gap_m = 3.0 * float(np.median(gaps))
    idx = np.flatnonzero(gaps > fiber_gap_m)
    return [(int(i), int(i + 1), float(gaps[i])) for i in idx]


def _dead_fiber_mask(geometry: ChannelGeometry, fiber_gap_m: float | None = None
                     ) -> tuple[list[int], np.ndarray]:
    """A break kills every channel on the far side (instrument ordering)."""
    n = geometry.n_channels
    mask = np.zeros(n, dtype=bool)
    breaks = [after for _, after, _ in break_segments(geometry, fiber_gap_m)]
    for after in breaks:
        mask[after:] = True
    return breaks, mask


def _band_snr_db(data: np.ndarray, dt_s: float,
                 band_hz: tuple[float, float]) -> np.ndarray:
    try:
        from scipy.signal import butter, sosfilt
    except Exception:  # pragma: no cover - scipy is a project dependency
        return np.full(data.shape[0], np.nan)
    nyq = 0.5 / dt_s
    lo, hi = band_hz
    hi = min(hi, 0.95 * nyq)
    if lo >= hi:
        return np.full(data.shape[0], np.nan)
    sos = butter(4, [lo / nyq, hi / nyq], btype='band', output='sos')
    n = data.shape[0]
    out = np.full(n, -np.inf)
    for i in range(n):
        x = data[i]
        if not np.all(np.isfinite(x)) or np.std(x) <= 0:
            continue
        xf = sosfilt(sos, x)
        p_sig = float(np.mean(xf ** 2))
        p_tot = float(np.mean(x ** 2))
        p_noise = max(p_tot - p_sig, 1e-300)
        out[i] = 10.0 * np.log10(max(p_sig, 1e-300) / p_noise)
    return out


def apply_flags(gather: DASGather, assessment: QualityAssessment,
                flags: QualityFlag = QualityFlag.DEAD_FIBER | QualityFlag.DEAD_CHANNEL,
                fill: float = np.nan) -> DASGather:
    """Return a copy with flagged channels zeroed/NaN-filled (kept as rows)."""
    mask = (assessment.flags & int(flags)) != 0
    out = gather.data.copy()
    out[mask] = fill
    return gather.copy(out, tags={**gather.tags, 'flagged': QualityFlag(int(flags)).label})
