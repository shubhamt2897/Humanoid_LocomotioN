"""Procedural terrain generation for the domain-randomization curriculum.

Generates 2D Perlin noise heightfields and writes them directly into a MuJoCo model's
``hfield_data`` buffer, so terrain can be re-randomized every episode without recompiling.
"""

from __future__ import annotations

import numpy as np


def _fade(t: np.ndarray) -> np.ndarray:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def perlin_noise_2d(shape: tuple[int, int], res: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Generate a single octave of 2D Perlin noise on a ``shape`` grid with ``res`` gradient cells.

    Both dimensions of ``shape`` must be divisible by the corresponding entry in ``res``.
    Returns values roughly in ``[-1, 1]``.
    """
    d = (shape[0] // res[0], shape[1] // res[1])
    angles = rng.uniform(0, 2 * np.pi, (res[0] + 1, res[1] + 1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    gradients = gradients.repeat(d[0], axis=0).repeat(d[1], axis=1)

    g00 = gradients[: -d[0], : -d[1]]
    g10 = gradients[d[0] :, : -d[1]]
    g01 = gradients[: -d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]

    grid_x, grid_y = np.mgrid[0 : res[0] : 1 / d[0], 0 : res[1] : 1 / d[1]]
    grid_x, grid_y = grid_x % 1, grid_y % 1

    n00 = g00[..., 0] * grid_x + g00[..., 1] * grid_y
    n10 = g10[..., 0] * (grid_x - 1) + g10[..., 1] * grid_y
    n01 = g01[..., 0] * grid_x + g01[..., 1] * (grid_y - 1)
    n11 = g11[..., 0] * (grid_x - 1) + g11[..., 1] * (grid_y - 1)

    t = _fade(grid_x)
    n0 = n00 * (1 - t) + n10 * t
    n1 = n01 * (1 - t) + n11 * t
    return np.sqrt(2) * ((1 - _fade(grid_y)) * n0 + _fade(grid_y) * n1)


def generate_rolling_terrain(nrow: int, ncol: int, rng: np.random.Generator, res: int = 4) -> np.ndarray:
    """Generate a smooth, bounded [0, 1] heightfield of "rolling bumps" for the given grid size.

    ``res`` controls bump frequency (higher = bumpier). ``nrow``/``ncol`` must be divisible by ``res``;
    callers should size the hfield grid accordingly (e.g. 128x128 with res in {2, 4, 8}).
    """
    noise = perlin_noise_2d((nrow, ncol), (res, res), rng)
    noise -= noise.min()
    peak = noise.max()
    if peak > 1e-8:
        noise /= peak
    return noise


def apply_terrain_to_model(model, hfield_name: str, rng: np.random.Generator, res: int = 4) -> None:
    """Regenerate the named heightfield's data in-place on a (per-env) MjModel."""
    import mujoco

    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name)
    nrow = int(model.hfield_nrow[hfield_id])
    ncol = int(model.hfield_ncol[hfield_id])
    adr = int(model.hfield_adr[hfield_id])
    heights = generate_rolling_terrain(nrow, ncol, rng, res=res)
    model.hfield_data[adr : adr + nrow * ncol] = heights.reshape(-1)


def flatten_terrain(model, hfield_name: str) -> None:
    """Zero out the named heightfield -- a flat floor, used while terrain DR is disabled."""
    import mujoco

    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name)
    nrow = int(model.hfield_nrow[hfield_id])
    ncol = int(model.hfield_ncol[hfield_id])
    adr = int(model.hfield_adr[hfield_id])
    model.hfield_data[adr : adr + nrow * ncol] = 0.0


def terrain_height_at_center(model, hfield_name: str) -> float:
    """World-frame ground height directly under the hfield's local (0, 0) -- i.e. under a robot
    spawned at the geom's xy origin. Assumes the hfield geom itself has no xy/z position offset
    (true for ``assets/g1/scene_train.xml``'s "floor" geom).
    """
    import mujoco

    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name)
    nrow = int(model.hfield_nrow[hfield_id])
    ncol = int(model.hfield_ncol[hfield_id])
    adr = int(model.hfield_adr[hfield_id])
    elevation_z = float(model.hfield_size[hfield_id, 2])
    center_idx = (nrow // 2) * ncol + (ncol // 2)
    return elevation_z * float(model.hfield_data[adr + center_idx])
