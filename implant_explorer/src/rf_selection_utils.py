from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def build_ball_offsets(radius_vox: int) -> np.ndarray:
    """Return integer xyz offsets inside a 3D ball of `radius_vox`."""
    r = int(radius_vox)
    offsets = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if (dx * dx + dy * dy + dz * dz) <= (r * r):
                    offsets.append((dx, dy, dz))
    return np.asarray(offsets, dtype=np.int16)


def collect_local_rf_neighborhood(
    seed_coord: Sequence[int],
    offsets_xyz: np.ndarray,
    *,
    ecc_map: np.ndarray,
    polar_map: np.ndarray,
    r2_map: np.ndarray,
    area_volume: np.ndarray,
    visible_area_indices: Iterable[int],
    r2_threshold: float,
    max_points: int = 250,
    sz_map: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Sample a local RF neighborhood around seed voxel in voxel coordinates.

    Returns a dict with:
      - ecc, pol, sz: float arrays for selected neighborhood
      - coords: (N, 3) int coords
      - seed_ecc, seed_pol: nearest retained sample to original seed (or None)
      - seed_coord: that retained seed coordinate (or None)
    """
    seed = np.asarray(seed_coord, dtype=np.int32).reshape(1, 3)
    if offsets_xyz.size == 0:
        return {
            "ecc": np.array([], dtype=np.float32),
            "pol": np.array([], dtype=np.float32),
            "sz": np.array([], dtype=np.float32),
            "coords": np.empty((0, 3), dtype=np.int32),
            "seed_ecc": None,
            "seed_pol": None,
            "seed_coord": None,
        }

    coords = seed + offsets_xyz.astype(np.int32)
    shape = area_volume.shape

    in_bounds = (
        (coords[:, 0] >= 0) & (coords[:, 0] < shape[0]) &
        (coords[:, 1] >= 0) & (coords[:, 1] < shape[1]) &
        (coords[:, 2] >= 0) & (coords[:, 2] < shape[2])
    )
    coords = coords[in_bounds]
    if coords.shape[0] == 0:
        return {
            "ecc": np.array([], dtype=np.float32),
            "pol": np.array([], dtype=np.float32),
            "sz": np.array([], dtype=np.float32),
            "coords": np.empty((0, 3), dtype=np.int32),
            "seed_ecc": None,
            "seed_pol": None,
            "seed_coord": None,
        }

    visible = set(int(v) for v in visible_area_indices)
    if visible:
        areas = area_volume[coords[:, 0], coords[:, 1], coords[:, 2]]
        coords = coords[np.isin(areas, list(visible))]
        if coords.shape[0] == 0:
            return {
                "ecc": np.array([], dtype=np.float32),
                "pol": np.array([], dtype=np.float32),
                "sz": np.array([], dtype=np.float32),
                "coords": np.empty((0, 3), dtype=np.int32),
                "seed_ecc": None,
                "seed_pol": None,
                "seed_coord": None,
            }

    ecc = ecc_map[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
    pol = polar_map[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
    r2 = r2_map[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
    if sz_map is not None:
        sz = sz_map[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
    else:
        sz = np.full(coords.shape[0], np.nan, dtype=np.float32)

    valid = (
        np.isfinite(ecc) &
        np.isfinite(pol) &
        (ecc > 0) &
        (r2 >= float(r2_threshold))
    )
    coords = coords[valid]
    ecc = ecc[valid]
    pol = pol[valid]
    sz = sz[valid]

    if coords.shape[0] == 0:
        return {
            "ecc": np.array([], dtype=np.float32),
            "pol": np.array([], dtype=np.float32),
            "sz": np.array([], dtype=np.float32),
            "coords": np.empty((0, 3), dtype=np.int32),
            "seed_ecc": None,
            "seed_pol": None,
            "seed_coord": None,
        }

    seed_xyz = seed.reshape(3)
    dist2 = np.sum((coords - seed_xyz) ** 2, axis=1)
    order = np.argsort(dist2)
    if int(max_points) > 0 and order.size > int(max_points):
        order = order[: int(max_points)]

    coords = coords[order]
    ecc = ecc[order]
    pol = pol[order]
    sz = sz[order]
    dist2 = dist2[order]

    seed_idx = int(np.argmin(dist2))
    return {
        "ecc": ecc,
        "pol": pol,
        "sz": sz,
        "coords": coords.astype(np.int32),
        "seed_ecc": float(ecc[seed_idx]),
        "seed_pol": float(pol[seed_idx]),
        "seed_coord": tuple(int(v) for v in coords[seed_idx].tolist()),
    }
