"""
implants_core.constraints – Validation engine
==============================================
Validates ``ImplantSpec`` (design-only checks) and
``ImplantInstance`` (placement-on-anatomy checks).

Each constraint is a callable returning a ``ConstraintResult``.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .spec import ConstraintResult, ImplantSpec, ImplantInstance


# ---------------------------------------------------------------------------
# Spec-level constraints (no anatomy needed)
# ---------------------------------------------------------------------------

def _check_min_contact_spacing(spec: ImplantSpec, min_spacing_mm: float = 0.010) -> ConstraintResult:
    """Block if any two contacts are closer than *min_spacing_mm*."""
    pts = spec.contacts_local
    if pts.shape[0] < 2:
        return ConstraintResult("min_contact_spacing", "ok",
                                margin=float("inf"), message="< 2 contacts")
    # pairwise distances (only upper triangle)
    diffs = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = float(dists.min())
    margin = min_dist - min_spacing_mm
    if margin < 0:
        return ConstraintResult(
            "min_contact_spacing", "block", margin=margin,
            message=f"Min contact spacing {min_dist*1000:.1f} µm < {min_spacing_mm*1000:.0f} µm limit",
            suggested_fix="Increase contact pitch or remove overlapping sites",
        )
    if margin < min_spacing_mm * 0.5:
        return ConstraintResult(
            "min_contact_spacing", "warn", margin=margin,
            message=f"Min contact spacing {min_dist*1000:.1f} µm is close to {min_spacing_mm*1000:.0f} µm limit",
        )
    return ConstraintResult("min_contact_spacing", "ok", margin=margin,
                            message=f"Min spacing {min_dist*1000:.1f} µm")


def _check_max_contacts(spec: ImplantSpec, max_contacts: int = 10000) -> ConstraintResult:
    """Warn if contact count is very high (performance)."""
    n = spec.total_contacts()
    margin = float(max_contacts - n)
    if n > max_contacts:
        return ConstraintResult(
            "max_contacts", "warn", margin=margin,
            message=f"{n} contacts exceeds {max_contacts} (may be slow)",
            suggested_fix="Reduce contact count or use LOD rendering",
        )
    return ConstraintResult("max_contacts", "ok", margin=margin,
                            message=f"{n} contacts")


def _check_sane_dimensions(spec: ImplantSpec) -> ConstraintResult:
    """Warn if footprint or depth is unusually large."""
    w, h = spec.footprint_mm()
    d_min, d_max = spec.depth_range_mm()
    depth = d_max - d_min
    issues = []
    if w > 50 or h > 50:
        issues.append(f"Footprint {w:.1f}×{h:.1f} mm is very large")
    if depth > 30:
        issues.append(f"Depth range {depth:.1f} mm is very large")
    if issues:
        return ConstraintResult(
            "sane_dimensions", "warn", margin=0.0,
            message="; ".join(issues),
            suggested_fix="Verify units are in mm",
        )
    return ConstraintResult("sane_dimensions", "ok",
                            message=f"Footprint {w:.1f}×{h:.1f} mm, depth {depth:.1f} mm")


def _check_shank_lengths(spec: ImplantSpec) -> ConstraintResult:
    """Warn if any shank is unreasonably long or has zero length."""
    if not spec.shanks:
        return ConstraintResult("shank_lengths", "ok", message="No shanks")
    lengths = [s["length_mm"] for s in spec.shanks]
    max_l = max(lengths)
    min_l = min(lengths)
    if min_l <= 0:
        return ConstraintResult(
            "shank_lengths", "block", margin=-min_l,
            message=f"Shank length ≤ 0 detected ({min_l:.3f} mm)",
            suggested_fix="Set all shank lengths > 0",
        )
    if max_l > 25.0:
        return ConstraintResult(
            "shank_lengths", "warn", margin=25.0 - max_l,
            message=f"Max shank length {max_l:.1f} mm exceeds 25 mm",
        )
    return ConstraintResult("shank_lengths", "ok",
                            message=f"Shanks {min_l:.1f}–{max_l:.1f} mm")


def validate_spec(spec: ImplantSpec) -> List[ConstraintResult]:
    """Run all design-only checks on an ``ImplantSpec``."""
    return [
        _check_min_contact_spacing(spec),
        _check_max_contacts(spec),
        _check_sane_dimensions(spec),
        _check_shank_lengths(spec),
    ]


# ---------------------------------------------------------------------------
# Instance-level constraints (needs anatomy)
# ---------------------------------------------------------------------------

def _check_max_depth(
    instance: ImplantInstance,
    anatomy_volume: Optional[np.ndarray] = None,
    voxel_size: tuple = (1.0, 1.0, 1.0),
    max_depth_mm: float = 10.0,
) -> ConstraintResult:
    """Warn if any contact extends beyond *max_depth_mm* from the surface."""
    # simple proxy: check if z-range of contacts_world exceeds threshold
    if instance.contacts_world.shape[0] == 0:
        return ConstraintResult("max_depth", "ok", message="No contacts placed")
    zs = instance.contacts_world[:, 2]
    depth_range = float(zs.max() - zs.min())
    margin = max_depth_mm - depth_range
    if margin < 0:
        return ConstraintResult(
            "max_depth", "warn", margin=margin,
            message=f"Contact depth range {depth_range:.1f} mm > {max_depth_mm} mm",
            suggested_fix="Reduce shank length or adjust placement depth",
        )
    return ConstraintResult("max_depth", "ok", margin=margin,
                            message=f"Depth range {depth_range:.1f} mm")


def _check_angle_from_normal(
    instance: ImplantInstance,
    max_angle_deg: float = 45.0,
) -> ConstraintResult:
    """Warn if implant Z-axis deviates from surface normal beyond threshold."""
    T = instance.transform_world_from_implant
    z_axis = T[:3, 2]  # implant Z in world
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        return ConstraintResult("angle_from_normal", "block",
                                message="Degenerate transform")
    # angle with world Z (proxy for surface normal identity)
    cos_angle = abs(z_axis[2] / z_norm)
    angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))
    margin = max_angle_deg - angle_deg
    if margin < 0:
        return ConstraintResult(
            "angle_from_normal", "warn", margin=margin,
            message=f"Implant tilted {angle_deg:.1f}° from normal (limit {max_angle_deg}°)",
            suggested_fix="Re-snap to surface or reduce tilt",
        )
    return ConstraintResult("angle_from_normal", "ok", margin=margin,
                            message=f"Tilt {angle_deg:.1f}° (limit {max_angle_deg}°)")


def _check_no_go_zone(
    instance: ImplantInstance,
    no_go_mask: Optional[np.ndarray] = None,
    voxel_size: tuple = (1.0, 1.0, 1.0),
) -> ConstraintResult:
    """Block if any contact lands in a no-go zone."""
    if no_go_mask is None:
        return ConstraintResult("no_go_zone", "ok", message="No no-go mask provided")
    pts = instance.contacts_world
    if pts.shape[0] == 0:
        return ConstraintResult("no_go_zone", "ok", message="No contacts")
    vs = np.asarray(voxel_size, dtype=np.float64)
    voxels = np.round(pts / vs).astype(np.int32)
    shape = np.array(no_go_mask.shape)
    in_bounds = np.all((voxels >= 0) & (voxels < shape), axis=1)
    in_no_go = 0
    for i in np.where(in_bounds)[0]:
        v = voxels[i]
        if no_go_mask[v[0], v[1], v[2]]:
            in_no_go += 1
    if in_no_go > 0:
        return ConstraintResult(
            "no_go_zone", "block", margin=-float(in_no_go),
            message=f"{in_no_go} contacts in no-go zone",
            suggested_fix="Move implant away from excluded region",
        )
    return ConstraintResult("no_go_zone", "ok", message="No contacts in no-go zone")


def _check_footprint_vs_surface(
    instance: ImplantInstance,
    max_footprint_mm: float = 20.0,
) -> ConstraintResult:
    """Warn if world-space footprint is very large."""
    if instance.contacts_world.shape[0] == 0:
        return ConstraintResult("footprint", "ok", message="No contacts")
    xy = instance.contacts_world[:, :2]
    w = float(xy[:, 0].max() - xy[:, 0].min())
    h = float(xy[:, 1].max() - xy[:, 1].min())
    diag = np.sqrt(w**2 + h**2)
    margin = max_footprint_mm - diag
    if margin < 0:
        return ConstraintResult(
            "footprint", "warn", margin=margin,
            message=f"Footprint diagonal {diag:.1f} mm > {max_footprint_mm} mm",
        )
    return ConstraintResult("footprint", "ok", margin=margin,
                            message=f"Footprint {w:.1f}×{h:.1f} mm")


def validate_instance(
    instance: ImplantInstance,
    *,
    anatomy_volume: Optional[np.ndarray] = None,
    area_volume: Optional[np.ndarray] = None,
    voxel_size: tuple = (1.0, 1.0, 1.0),
    no_go_mask: Optional[np.ndarray] = None,
    max_depth_mm: float = 10.0,
    max_angle_deg: float = 45.0,
    max_footprint_mm: float = 20.0,
) -> List[ConstraintResult]:
    """Run placement checks on an ``ImplantInstance``."""
    results = validate_spec(instance.spec)
    results.extend([
        _check_max_depth(instance, anatomy_volume, voxel_size, max_depth_mm),
        _check_angle_from_normal(instance, max_angle_deg),
        _check_no_go_zone(instance, no_go_mask, voxel_size),
        _check_footprint_vs_surface(instance, max_footprint_mm),
    ])
    return results
