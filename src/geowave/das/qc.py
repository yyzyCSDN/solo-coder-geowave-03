from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .geometry import FiberGeometry


@dataclass(frozen=True)
class ChannelQuality:
    channel_id: str
    index: int
    nan_fraction: float
    rms: float
    dead: bool
    clipped: bool
    low_signal: bool
    reversed: bool
    near_break: bool
    low_quality: bool


@dataclass(frozen=True)
class QualityZone:
    """A contiguous run of low-quality channels along the fiber."""
    start_index: int
    stop_index: int  # inclusive
    channel_ids: tuple[str, ...]


@dataclass(frozen=True)
class QualityTable:
    channels: tuple[ChannelQuality, ...]
    low_quality_zones: tuple[QualityZone, ...]
    reversed_sections: tuple
    breaks: tuple

    def low_quality_ids(self) -> tuple[str, ...]:
        return tuple(c.channel_id for c in self.channels if c.low_quality)

    def reversed_ids(self) -> tuple[str, ...]:
        return tuple(c.channel_id for c in self.channels if c.reversed)

    def channel_flags(self) -> dict[str, tuple[str, ...]]:
        flags: dict[str, tuple[str, ...]] = {}
        for c in self.channels:
            tags = []
            if c.dead:
                tags.append('dead')
            if c.clipped:
                tags.append('clipped')
            if c.low_signal:
                tags.append('low-signal')
            if c.reversed:
                tags.append('reversed')
            if c.near_break:
                tags.append('near-break')
            if c.low_quality:
                tags.append('low-quality')
            flags[c.channel_id] = tuple(tags)
        return flags


def assess_channels(data: np.ndarray, geometry: FiberGeometry, *,
                    dead_std: float = 1e-9, clip_fraction: float = 0.01,
                    low_signal_ratio: float = 0.05, max_nan_fraction: float = 0.5,
                    zone_gap: int = 1) -> QualityTable:
    """Flag dead, clipped, weak, reversed and break-adjacent channels.

    Low-quality zones are contiguous runs of flagged channels, allowing gaps of
    up to ``zone_gap`` good channels inside a run so single recovered channels
    do not split a bad zone.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] != geometry.n_channels:
        raise ValueError('data must be (n_channels, n_time) matching the geometry')
    n = x.shape[0]
    finite = np.isfinite(x)
    nan_fraction = 1.0 - finite.mean(axis=1)
    safe = np.where(finite, x, 0.0)
    count = np.maximum(finite.sum(axis=1), 1)
    rms = np.sqrt((safe ** 2).sum(axis=1) / count)
    median_rms = float(np.median(rms[rms > 0])) if np.any(rms > 0) else 0.0

    reversed_mask = geometry.channel_mask_reversed()
    near_break = geometry.channel_mask_segment_ends()
    channels = []
    low_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        row = x[i][finite[i]]
        dead = bool(row.size == 0 or float(row.std()) <= dead_std)
        clipped = False
        if row.size:
            peak = float(np.abs(row).max())
            clipped = peak > 0 and float((np.abs(row) >= 0.999 * peak).mean()) > clip_fraction
        low_signal = bool(median_rms > 0 and 0 < rms[i] < low_signal_ratio * median_rms)
        low = bool(dead or clipped or low_signal or nan_fraction[i] > max_nan_fraction)
        low_mask[i] = low
        channels.append(ChannelQuality(
            geometry.channel_ids[i], i, float(nan_fraction[i]), float(rms[i]),
            dead, clipped, low_signal, bool(reversed_mask[i]),
            bool(near_break[i]), low))
    zones = _zones(low_mask, geometry.channel_ids, zone_gap)
    return QualityTable(tuple(channels), zones, geometry.reversed_sections, geometry.breaks)


def _zones(low_mask: np.ndarray, ids: tuple[str, ...], zone_gap: int) -> tuple[QualityZone, ...]:
    zones = []
    start = None
    last_low = None
    gap = 0
    for i, low in enumerate(low_mask):
        if low:
            if start is None:
                start = i
            last_low = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > zone_gap:
                zones.append(QualityZone(start, last_low, tuple(ids[start:last_low + 1])))
                start = None
    if start is not None:
        zones.append(QualityZone(start, last_low, tuple(ids[start:last_low + 1])))
    return tuple(zones)
