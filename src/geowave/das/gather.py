from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .fiber import ChannelGeometry


@dataclass
class DASGather:
    """Strain-rate DAS gather keyed to fibre arc position.

    ``data`` is laid out ``(n_channels, n_time)``. Unlike a geophone gather,
    channel index is only an instrument order: every physical question must go
    through ``geometry.channel_s`` / ``geometry.positions``.
    """

    data: np.ndarray
    geometry: ChannelGeometry
    dt_s: float
    start_s: float = 0.0
    units: str = 'strain_rate'
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        data = np.asarray(self.data, dtype=float)
        if data.ndim != 2:
            raise ValueError('DAS data must have shape (n_channels, n_time)')
        if data.shape[0] != self.geometry.n_channels:
            raise ValueError('data channel count does not match geometry')
        if self.dt_s <= 0.0:
            raise ValueError('sample interval must be positive')
        self.data = data

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def times(self) -> np.ndarray:
        return self.start_s + np.arange(self.n_samples) * self.dt_s

    @property
    def duration_s(self) -> float:
        return self.start_s + (self.n_samples - 1) * self.dt_s

    def channel(self, index: int) -> np.ndarray:
        return self.data[index]

    def channel_at(self, s: float) -> np.ndarray:
        """Exact lookup of the channel whose arc position is ``s``."""
        matches = np.flatnonzero(np.isclose(self.geometry.channel_s, s))
        if matches.size != 1:
            raise KeyError(f'no unique channel at arc {s:g} m')
        return self.data[int(matches[0])]

    def copy(self, data: np.ndarray, **overrides) -> 'DASGather':
        kwargs = {'geometry': self.geometry, 'dt_s': self.dt_s,
                  'start_s': self.start_s, 'units': self.units,
                  'tags': dict(self.tags)}
        kwargs.update(overrides)
        return DASGather(np.asarray(data, dtype=float), **kwargs)


def load_strain_rate(data, geometry: ChannelGeometry, dt_s: float,
                     start_s: float = 0.0, units: str = 'strain_rate',
                     name: str = 'das') -> DASGather:
    """Processing entry point for raw strain-rate traces.

    File readers decode into ``(data, geometry, dt_s)`` and hand off here so all
    downstream code consumes a validated :class:`DASGather`.
    """
    return DASGather(data, geometry, dt_s, start_s, units,
                     tags={'name': name, 'entry': 'load_strain_rate'})
