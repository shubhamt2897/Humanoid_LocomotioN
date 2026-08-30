"""Small batched math helpers shared across the env and reward modules."""

from __future__ import annotations

import numpy as np


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` (..., 3) from world frame into the local frame described by ``quat`` (..., 4, wxyz)."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    qvec = np.stack([x, y, z], axis=-1)
    a = vec * (2.0 * w[..., None] ** 2 - 1.0)
    b = np.cross(qvec, vec) * (2.0 * w[..., None])
    c = qvec * (np.sum(qvec * vec, axis=-1, keepdims=True) * 2.0)
    return a - b + c
