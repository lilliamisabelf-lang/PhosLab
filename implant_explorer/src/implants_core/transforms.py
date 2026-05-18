"""Shared implant geometry transforms and soft validation helpers."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


def _rot_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    rx_m = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rx), -np.sin(rx)],
            [0.0, np.sin(rx), np.cos(rx)],
        ],
        dtype=np.float64,
    )
    ry_m = np.array(
        [
            [np.cos(ry), 0.0, np.sin(ry)],
            [0.0, 1.0, 0.0],
            [-np.sin(ry), 0.0, np.cos(ry)],
        ],
        dtype=np.float64,
    )
    rz_m = np.array(
        [
            [np.cos(rz), -np.sin(rz), 0.0],
            [np.sin(rz), np.cos(rz), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rx_m @ ry_m @ rz_m


def _mat_translate(dx: float, dy: float, dz: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[0, 3], m[1, 3], m[2, 3] = dx, dy, dz
    return m


def _mat_scale_uniform(scale: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = float(scale)
    m[1, 1] = float(scale)
    m[2, 2] = float(scale)
    return m


def _mat_mirror_x(enabled: bool) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    if enabled:
        m[0, 0] = -1.0
    return m


def _mat_rot4(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = _rot_matrix(rx_deg, ry_deg, rz_deg)
    return m


def compose_transform_matrix(
    *,
    tx_mm: float = 0.0,
    ty_mm: float = 0.0,
    tz_mm: float = 0.0,
    rx_deg: float = 0.0,
    ry_deg: float = 0.0,
    rz_deg: float = 0.0,
    scale: float = 1.0,
    mirror_x: bool = False,
    pivot_mm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compose 4x4 matrix: p' = T * Pivot * R * S * Mirror * Pivot^-1 * p."""
    pivot = np.asarray(pivot_mm if pivot_mm is not None else [0.0, 0.0, 0.0], dtype=np.float64)
    to_pivot = _mat_translate(*pivot)
    from_pivot = _mat_translate(*(-pivot))
    t_m = _mat_translate(tx_mm, ty_mm, tz_mm)
    r_m = _mat_rot4(rx_deg, ry_deg, rz_deg)
    s_m = _mat_scale_uniform(scale)
    m_m = _mat_mirror_x(mirror_x)
    return t_m @ to_pivot @ r_m @ s_m @ m_m @ from_pivot


def apply_transform_to_contacts(
    contacts_mm: np.ndarray,
    *,
    transform: Optional[Any] = None,
    tx_mm: float = 0.0,
    ty_mm: float = 0.0,
    tz_mm: float = 0.0,
    rx_deg: float = 0.0,
    ry_deg: float = 0.0,
    rz_deg: float = 0.0,
    scale: float = 1.0,
    mirror_x: bool = False,
    pivot_mm: Optional[np.ndarray] = None,
    selection_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply a rigid-like transform (+ uniform scale + optional mirror) to contacts."""
    pts = np.asarray(contacts_mm, dtype=np.float64)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("contacts_mm must have shape (N, 3)")

    if transform is not None:
        if hasattr(transform, "tx_mm"):
            tx_mm = float(getattr(transform, "tx_mm"))
            ty_mm = float(getattr(transform, "ty_mm"))
            tz_mm = float(getattr(transform, "tz_mm"))
            rx_deg = float(getattr(transform, "rx_deg"))
            ry_deg = float(getattr(transform, "ry_deg"))
            rz_deg = float(getattr(transform, "rz_deg"))
            scale = float(getattr(transform, "scale"))
            mirror_x = bool(getattr(transform, "mirror_x"))
        else:
            tx_mm = float(transform.get("tx_mm", tx_mm))
            ty_mm = float(transform.get("ty_mm", ty_mm))
            tz_mm = float(transform.get("tz_mm", tz_mm))
            rx_deg = float(transform.get("rx_deg", rx_deg))
            ry_deg = float(transform.get("ry_deg", ry_deg))
            rz_deg = float(transform.get("rz_deg", rz_deg))
            scale = float(transform.get("scale", scale))
            mirror_x = bool(transform.get("mirror_x", mirror_x))

    if pivot_mm is None:
        pivot_mm = pts.mean(axis=0)
    matrix = compose_transform_matrix(
        tx_mm=tx_mm,
        ty_mm=ty_mm,
        tz_mm=tz_mm,
        rx_deg=rx_deg,
        ry_deg=ry_deg,
        rz_deg=rz_deg,
        scale=scale,
        mirror_x=mirror_x,
        pivot_mm=np.asarray(pivot_mm, dtype=np.float64),
    )

    def _apply(arr: np.ndarray) -> np.ndarray:
        homog = np.concatenate([arr, np.ones((arr.shape[0], 1), dtype=np.float64)], axis=1)
        return (homog @ matrix.T)[:, :3]

    if selection_mask is None:
        return _apply(pts).astype(np.float32)

    mask = np.asarray(selection_mask, dtype=bool)
    if mask.shape[0] != pts.shape[0]:
        raise ValueError("selection_mask length must match contacts")
    out = pts.copy()
    out[mask] = _apply(pts[mask])
    return out.astype(np.float32)


def rotate_contacts(
    contacts_mm: np.ndarray,
    rx_deg: float = 0.0,
    ry_deg: float = 0.0,
    rz_deg: float = 0.0,
    pivot_mm: Optional[np.ndarray] = None,
) -> np.ndarray:
    return apply_transform_to_contacts(
        contacts_mm,
        rx_deg=rx_deg,
        ry_deg=ry_deg,
        rz_deg=rz_deg,
        pivot_mm=pivot_mm,
    )


def translate_contacts(
    contacts_mm: np.ndarray,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
) -> np.ndarray:
    return apply_transform_to_contacts(contacts_mm, tx_mm=dx, ty_mm=dy, tz_mm=dz)


def contacts_to_voxels(
    contacts_mm: np.ndarray,
    voxel_size_mm: float | np.ndarray,
    origin_vox: Optional[np.ndarray] = None,
) -> np.ndarray:
    vs = np.asarray(voxel_size_mm, dtype=np.float64)
    vox = np.asarray(contacts_mm, dtype=np.float64) / vs
    if origin_vox is not None:
        vox = vox + np.asarray(origin_vox, dtype=np.float64)
    return np.round(vox).astype(np.int32)


def validate_contacts_soft(
    contacts_mm: np.ndarray,
    *,
    scale: Optional[float] = None,
    min_spacing_mm: float = 0.01,
    recommended_scale: Tuple[float, float] = (0.5, 2.0),
    volume_shape: Optional[Sequence[int]] = None,
    voxel_size_mm: Optional[float | np.ndarray] = None,
) -> List[str]:
    """Return non-blocking warnings for likely non-physical or unusable states."""
    warnings: List[str] = []
    pts = np.asarray(contacts_mm, dtype=np.float64)
    if pts.size == 0:
        warnings.append("No contacts available.")
        return warnings

    if pts.shape[0] >= 2:
        diffs = pts[:, None, :] - pts[None, :, :]
        dist = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_d = float(dist.min())
        if min_d < min_spacing_mm:
            warnings.append(
                f"Minimum contact spacing is very small ({min_d:.4f} mm < {min_spacing_mm:.4f} mm)."
            )

    if scale is not None:
        lo, hi = recommended_scale
        if scale < lo or scale > hi:
            warnings.append(
                f"Scale {scale:.3f} is outside recommended range [{lo:.3f}, {hi:.3f}]."
            )

    if volume_shape is not None and voxel_size_mm is not None:
        vox = contacts_to_voxels(pts.astype(np.float32), voxel_size_mm)
        shp = np.asarray(volume_shape, dtype=int)
        inb = (
            (vox[:, 0] >= 0)
            & (vox[:, 0] < shp[0])
            & (vox[:, 1] >= 0)
            & (vox[:, 1] < shp[1])
            & (vox[:, 2] >= 0)
            & (vox[:, 2] < shp[2])
        )
        out_n = int((~inb).sum())
        if out_n > 0:
            warnings.append(f"{out_n} contacts are outside the volume bounds.")

    return warnings


__all__ = [
    "compose_transform_matrix",
    "apply_transform_to_contacts",
    "rotate_contacts",
    "translate_contacts",
    "validate_contacts_soft",
    "contacts_to_voxels",
]
