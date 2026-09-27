from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .geometry import FiberGeometry

_EPS = 1e-12


@dataclass(frozen=True)
class DirectionalResult:
    data: np.ndarray            # corrected or flagged data, same shape as input
    sensitivity: np.ndarray     # per-channel |response| factor actually used
    angle_deg: np.ndarray       # plan-view angle between fiber tangent and wave direction
    weak_mask: np.ndarray       # channels whose response is too weak to trust


def incidence_directions(geometry: FiberGeometry, source_xyz) -> np.ndarray:
    """Unit vectors from each channel towards a source, in 3-D."""
    src = np.asarray(source_xyz, dtype=float)
    if src.shape == (2,):
        src = np.r_[src, 0.0]
    if src.shape != (3,):
        raise ValueError('source position must be 2-D or 3-D')
    vec = src[None, :] - geometry.points
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return np.where(norm > _EPS, vec / np.maximum(norm, _EPS), 0.0)


def directional_sensitivity(geometry: FiberGeometry, directions=None,
                            source_xyz=None, mode: str = 'cos2') -> tuple[np.ndarray, np.ndarray]:
    """Directional response of each channel of a curved fiber.

    ``mode='cos2'`` is the axial strain response of a straight gauge to a
    P-wave, proportional to cos²θ; ``mode='cos'`` keeps the signed projection
    (polarity-preserving). Returns (sensitivity, angle_deg).
    """
    if mode not in ('cos2', 'cos'):
        raise ValueError("mode must be 'cos2' or 'cos'")
    if directions is None:
        if source_xyz is None:
            raise ValueError('directions or source_xyz required')
        directions = incidence_directions(geometry, source_xyz)
    directions = np.asarray(directions, dtype=float)
    if directions.shape == (3,):
        directions = np.tile(directions, (geometry.n_channels, 1))
    if directions.shape != (geometry.n_channels, 3):
        raise ValueError('directions must be (3,) or (n_channels, 3)')
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit = np.where(norms > _EPS, directions / np.maximum(norms, _EPS), 0.0)
    proj = np.sum(geometry.tangent * unit, axis=1)
    angle = np.degrees(np.arccos(np.clip(np.abs(proj), 0.0, 1.0)))
    sensitivity = proj ** 2 if mode == 'cos2' else np.abs(proj)
    return sensitivity, angle


def apply_directional(data: np.ndarray, sensitivity: np.ndarray, angle_deg: np.ndarray,
                      *, mode: str = 'correct', min_sensitivity: float = 0.05):
    """Account for directional sensitivity on reconstructed channels.

    - 'correct' : divide by the sensitivity to recover broadside-comparable
      amplitudes; channels below ``min_sensitivity`` become NaN and are
      reported in ``weak_mask``.
    - 'flag'    : data pass through unchanged, only the mask is reported.
    - 'none'    : no-op, sensitivity and angles are still attached.
    """
    x = np.asarray(data, dtype=float)
    sens = np.asarray(sensitivity, dtype=float)
    if x.shape[0] != sens.shape[0]:
        raise ValueError('sensitivity must match the channel count')
    weak = ~(np.isfinite(sens)) | (sens < min_sensitivity)
    if mode == 'correct':
        safe = np.where(weak, np.nan, sens)
        out = x / safe[:, None]
    elif mode in ('flag', 'none'):
        out = x.copy()
    else:
        raise ValueError("mode must be 'correct', 'flag' or 'none'")
    return DirectionalResult(out, sens, np.asarray(angle_deg, dtype=float), weak)
