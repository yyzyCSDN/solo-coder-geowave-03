from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

from .fiber import FiberPath, ChannelGeometry
from .gather import load_strain_rate, DASGather

_FIBER_COLUMNS = [('x', ('x', 'easting', 'east', 'lon', 'longitude')),
                  ('y', ('y', 'northing', 'north', 'lat', 'latitude')),
                  ('z', ('z', 'elevation', 'depth', 'elev'))]


def _pick_column(fieldnames, aliases):
    for alias in aliases:
        for name in fieldnames:
            if name.strip().lower() == alias:
                return name
    return None


def read_fiber_csv(path: str | Path) -> FiberPath:
    """Read a cable survey CSV (x, y[, z] columns, header required)."""
    with Path(path).open(newline='', encoding='utf8') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        picks = [_pick_column(fields, aliases) for _, aliases in _FIBER_COLUMNS]
        if picks[0] is None or picks[1] is None:
            raise ValueError('fiber CSV needs x/easting and y/northing columns')
        cols = [[], [], []]
        for row in reader:
            cols[0].append(float(row[picks[0]]))
            cols[1].append(float(row[picks[1]]))
            if picks[2] is not None and row.get(picks[2]) not in (None, ''):
                cols[2].append(float(row[picks[2]]))
    if cols[2] and len(cols[2]) != len(cols[0]):
        raise ValueError('z column is partially populated')
    return FiberPath.from_vertices(cols[0], cols[1], np.array(cols[2]) if cols[2] else None)


def read_channel_csv(path: str | Path, fiber: FiberPath) -> ChannelGeometry:
    """Read a channel map CSV with ``channel_id, arc_s`` (metres) columns."""
    ids, s_vals = [], []
    with Path(path).open(newline='', encoding='utf8') as f:
        reader = csv.DictReader(f)
        fields = {n.strip().lower(): n for n in (reader.fieldnames or [])}
        s_key = next((fields[k] for k in ('arc_s', 's', 'distance_m', 'arc_length', 'chainage')
                      if k in fields), None)
        id_key = next((fields[k] for k in ('channel_id', 'id', 'channel') if k in fields), None)
        if s_key is None:
            raise ValueError('channel CSV needs an arc_s / distance_m column')
        for i, row in enumerate(reader):
            s_vals.append(float(row[s_key]))
            ids.append(row[id_key].strip() if id_key is not None else f'ch-{i:05d}')
    return ChannelGeometry(np.asarray(s_vals, dtype=float), fiber, ids)


def read_strain_rate_npy(path: str | Path, geometry: ChannelGeometry,
                         dt_s: float, start_s: float = 0.0,
                         units: str = 'strain_rate', name: str = 'das') -> DASGather:
    """Load a ``(n_channels, n_time)`` strain-rate array from ``.npy``/``.npz``."""
    p = Path(path)
    if p.suffix == '.npz':
        with np.load(p) as z:
            key = next((k for k in ('data', 'strain_rate') if k in z.files), z.files[0])
            data = z[key]
    else:
        data = np.load(p)
    return load_strain_rate(data, geometry, dt_s, start_s, units, name)


def save_strain_rate_npy(path: str | Path, gather: DASGather) -> None:
    np.save(path, gather.data)
