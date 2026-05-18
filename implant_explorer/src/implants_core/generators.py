"""
implants_core.generators – Parametric implant generators
========================================================
Each function returns an ``ImplantSpec`` with fully populated
``contacts_local``, ``contact_normals_local``, ``trajectories_local``,
and a ``shanks`` list for backward-compatible per-site editing.

All coordinates in millimetres. Contact z = 0 is the shank tip.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .spec import ImplantSpec


# ---------------------------------------------------------------------------
# Utah / pedestal-style MEA
# ---------------------------------------------------------------------------

def generate_utah(
    rows: int = 10,
    cols: int = 10,
    pitch_mm: float = 0.4,
    shank_length_mm: float = 1.5,
    tip_angle_deg: float = 25.0,
    contact_diameter_um: float = 20.0,
    depth_map: Optional[np.ndarray] = None,
    site_mask: Optional[np.ndarray] = None,
    name: str = "Utah Array",
) -> ImplantSpec:
    """
    Maskable Utah grid with optional per-site depth map.

    Parameters
    ----------
    rows, cols : grid dimensions
    pitch_mm : centre-to-centre spacing (uniform X and Y)
    shank_length_mm : default shank length (overridden per-site by *depth_map*)
    tip_angle_deg : shank tip angle (metadata)
    contact_diameter_um : contact diameter (metadata)
    depth_map : (rows, cols) float array of per-site shank lengths in mm.
                ``None`` → uniform ``shank_length_mm``.
    site_mask : (rows, cols) bool array.  ``True`` = active site.
                ``None`` → all sites active.
    name : design name stored in the spec
    """
    if site_mask is None:
        site_mask = np.ones((rows, cols), dtype=bool)
    else:
        site_mask = np.asarray(site_mask, dtype=bool)
        if site_mask.shape != (rows, cols):
            raise ValueError(f"site_mask shape {site_mask.shape} != ({rows}, {cols})")

    if depth_map is None:
        depth_map = np.full((rows, cols), shank_length_mm, dtype=np.float64)
    else:
        depth_map = np.asarray(depth_map, dtype=np.float64)
        if depth_map.shape != (rows, cols):
            raise ValueError(f"depth_map shape {depth_map.shape} != ({rows}, {cols})")

    contacts = []
    normals = []
    shanks = []
    trajectories = []
    sid = 0

    for r in range(rows):
        for c in range(cols):
            if not site_mask[r, c]:
                continue
            x = c * pitch_mm
            y = r * pitch_mm
            length = float(depth_map[r, c])

            # Utah local frame: footprint on X-Y plane, insertion axis along Z.
            contacts.append([x, y, 0.0])
            normals.append([0.0, 0.0, -1.0])

            shanks.append({
                "id": sid,
                "x_mm": x,
                "y_mm": y,
                "length_mm": length,
                "tip_angle_deg": tip_angle_deg,
                # Keep per-contact z_mm aligned with the insertion/depth axis.
                "contacts": [{"id": 0, "z_mm": 0.0, "diameter_um": contact_diameter_um}],
            })

            # trajectory: entry (z=length, surface side) -> tip (z=0)
            trajectories.append(np.array([[x, y, length], [x, y, 0.0]]))
            sid += 1

    contacts_arr = np.array(contacts, dtype=np.float64) if contacts else np.empty((0, 3), dtype=np.float64)
    normals_arr = np.array(normals, dtype=np.float64) if normals else np.empty((0, 3), dtype=np.float64)

    params: Dict[str, Any] = {
        "rows": rows,
        "cols": cols,
        "pitch_mm": pitch_mm,
        "shank_length_mm": shank_length_mm,
        "tip_angle_deg": tip_angle_deg,
        "contact_diameter_um": contact_diameter_um,
    }
    if site_mask is not None and not np.all(site_mask):
        params["site_mask"] = site_mask.tolist()
    if depth_map is not None and not np.allclose(depth_map[site_mask], shank_length_mm):
        params["depth_map"] = depth_map.tolist()

    return ImplantSpec(
        name=name,
        family="utah",
        params=params,
        contacts_local=contacts_arr,
        contact_normals_local=normals_arr,
        trajectories_local=trajectories,
        shanks=shanks,
        materials_meta={"contact_material": "Pt"},
    )


# ---------------------------------------------------------------------------
# Thread bundle (Neuralink-style)
# ---------------------------------------------------------------------------

def generate_thread_bundle(
    n_threads: int = 64,
    contacts_per_thread: int = 16,
    contact_spacing_mm: float = 0.050,
    thread_length_mm: float = 4.0,
    entry_points: Optional[np.ndarray] = None,
    directions: Optional[np.ndarray] = None,
    hub_radius_mm: float = 3.0,
    layout: str = "circular",
    contact_diameter_um: float = 12.0,
    name: str = "Thread Bundle",
) -> ImplantSpec:
    """
    Straight-trajectory thread bundle.

    Parameters
    ----------
    n_threads : number of threads
    contacts_per_thread : electrodes per thread
    contact_spacing_mm : inter-contact distance along thread
    thread_length_mm : insertion depth of each thread
    entry_points : (n_threads, 3) array of thread entry positions.
                   ``None`` → auto-generated from *layout*.
    directions : (n_threads, 3) unit vectors for each thread.
                 ``None`` → all point along -Z.
    hub_radius_mm : radius for auto-generated circular/grid layout
    layout : ``"circular"`` | ``"grid"`` — used when *entry_points* is None
    contact_diameter_um : metadata
    name : design name
    """
    # --- entry points ---
    if entry_points is not None:
        entry_points = np.asarray(entry_points, dtype=np.float64)
        if entry_points.shape != (n_threads, 3):
            raise ValueError(f"entry_points shape {entry_points.shape} != ({n_threads}, 3)")
    else:
        entry_points = _auto_thread_layout(n_threads, hub_radius_mm, layout)

    # --- directions ---
    if directions is not None:
        directions = np.asarray(directions, dtype=np.float64)
        # normalise
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        directions = directions / norms
    else:
        # Thread local frame: entry footprint on X-Z plane, insertion axis along +Y.
        directions = np.tile([0.0, 1.0, 0.0], (n_threads, 1))

    contacts = []
    normals = []
    shanks = []
    trajectories = []

    for ti in range(n_threads):
        ep = entry_points[ti]
        d = directions[ti]
        thread_contacts = []

        # contacts distributed from tip toward entry
        for ci in range(contacts_per_thread):
            # tip is at depth = thread_length_mm along direction
            depth = thread_length_mm - ci * contact_spacing_mm
            if depth < 0:
                depth = 0.0
            pos = ep + d * depth
            contacts.append(pos)
            normals.append(-d)  # normal points outward from cortex
            thread_contacts.append({
                "id": ci,
                "z_mm": depth,
                "diameter_um": contact_diameter_um,
            })

        shanks.append({
            "id": ti,
            "x_mm": float(ep[0]),
            "y_mm": float(ep[1]),
            "length_mm": thread_length_mm,
            "tip_angle_deg": 0.0,  # threads have no tip angle
            "contacts": thread_contacts,
        })

        # trajectory: entry → tip
        tip = ep + d * thread_length_mm
        trajectories.append(np.stack([ep, tip]))

    contacts_arr = np.array(contacts, dtype=np.float64) if contacts else np.empty((0, 3), dtype=np.float64)
    normals_arr = np.array(normals, dtype=np.float64) if normals else np.empty((0, 3), dtype=np.float64)

    params: Dict[str, Any] = {
        "n_threads": n_threads,
        "contacts_per_thread": contacts_per_thread,
        "contact_spacing_mm": contact_spacing_mm,
        "thread_length_mm": thread_length_mm,
        "hub_radius_mm": hub_radius_mm,
        "layout": layout,
        "contact_diameter_um": contact_diameter_um,
    }
    if entry_points is not None:
        params["entry_points"] = entry_points.tolist()

    return ImplantSpec(
        name=name,
        family="thread",
        params=params,
        contacts_local=contacts_arr,
        contact_normals_local=normals_arr,
        trajectories_local=trajectories,
        shanks=shanks,
        materials_meta={"contact_material": "Pt"},
    )


def _auto_thread_layout(n: int, radius: float, layout: str) -> np.ndarray:
    """Generate entry point positions for *n* threads."""
    pts = np.zeros((n, 3), dtype=np.float64)
    if layout == "circular":
        if n == 1:
            return pts
        # concentric rings — fill from center outward
        placed = 0
        ring = 0
        while placed < n:
            if ring == 0:
                pts[placed] = [0.0, 0.0, 0.0]
                placed += 1
                ring += 1
                continue
            r = radius * ring / max(1, int(np.ceil(np.sqrt(n / np.pi))))
            circumference = 2 * np.pi * r
            n_on_ring = min(n - placed, max(6, int(circumference / (radius * 0.3))))
            for i in range(n_on_ring):
                if placed >= n:
                    break
                theta = 2 * np.pi * i / n_on_ring
                pts[placed] = [r * np.cos(theta), 0.0, r * np.sin(theta)]
                placed += 1
            ring += 1
    elif layout == "grid":
        side = int(np.ceil(np.sqrt(n)))
        spacing = (2.0 * radius) / max(1, side - 1) if side > 1 else 0.0
        placed = 0
        for r in range(side):
            for c in range(side):
                if placed >= n:
                    break
                pts[placed] = [c * spacing - radius, 0.0, r * spacing - radius]
                placed += 1
    else:
        raise ValueError(f"Unknown layout '{layout}'. Use 'circular' or 'grid'.")
    return pts


# ---------------------------------------------------------------------------
# Multi-shank comb (Michigan / Neuropixels / NHP SPIKE style)
# ---------------------------------------------------------------------------

def generate_multishank(
    n_shanks: int = 8,
    contacts_per_shank: int = 128,
    contact_pitch_mm: float = 0.025,
    shank_pitch_mm: float = 0.250,
    shank_lengths: Optional[List[float]] = None,
    default_shank_length_mm: float = 6.0,
    shank_width_um: float = 70.0,
    arrangement: str = "single_row",
    stagger_mode: str = "none",
    stagger_mm: float = 0.0,
    tip_angle_deg: float = 20.0,
    contact_diameter_um: float = 12.0,
    name: str = "Multi-Shank Comb",
) -> ImplantSpec:
    """
    Multi-shank probe with staggering support.

    Parameters
    ----------
    n_shanks : number of shanks
    contacts_per_shank : electrodes per shank
    contact_pitch_mm : inter-contact spacing along shank
    shank_pitch_mm : inter-shank spacing (X direction)
    shank_lengths : per-shank lengths (mm). ``None`` → uniform.
    default_shank_length_mm : used when *shank_lengths* is None
    shank_width_um : shank width (metadata)
    arrangement : ``"single_row"`` | ``"checkerboard"``
    stagger_mode : ``"none"`` | ``"linear"`` | ``"alternating"`` | ``"custom"``
    stagger_mm : amount of stagger per shank (for linear/alternating)
    tip_angle_deg : metadata
    contact_diameter_um : metadata
    name : design name
    """
    # --- per-shank lengths ---
    if shank_lengths is not None:
        lengths = list(shank_lengths)
        if len(lengths) != n_shanks:
            raise ValueError(f"len(shank_lengths)={len(lengths)} != n_shanks={n_shanks}")
    else:
        lengths = _apply_stagger(n_shanks, default_shank_length_mm, stagger_mode, stagger_mm)

    contacts = []
    normals = []
    shanks = []
    trajectories = []

    for si in range(n_shanks):
        x = si * shank_pitch_mm
        length = lengths[si]
        shank_contacts = []

        for ci in range(contacts_per_shank):
            z = ci * contact_pitch_mm  # 0 = tip
            if z > length:
                z = length  # clamp to shank length

            # checkerboard: alternate contacts offset in Y by half-pitch
            y_offset = 0.0
            if arrangement == "checkerboard" and ci % 2 == 1:
                y_offset = contact_pitch_mm * 0.5

            pos = [x, y_offset, z]
            contacts.append(pos)
            normals.append([0.0, 0.0, -1.0])
            shank_contacts.append({
                "id": ci,
                "z_mm": z,
                "diameter_um": contact_diameter_um,
            })

        shanks.append({
            "id": si,
            "x_mm": x,
            "y_mm": 0.0,
            "length_mm": length,
            "tip_angle_deg": tip_angle_deg,
            "contacts": shank_contacts,
        })

        # trajectory: entry (proximal) -> tip, aligned with contact coordinates on +Z
        entry = np.array([x, 0.0, length])
        tip = np.array([x, 0.0, 0.0])
        trajectories.append(np.stack([entry, tip]))

    contacts_arr = np.array(contacts, dtype=np.float64) if contacts else np.empty((0, 3), dtype=np.float64)
    normals_arr = np.array(normals, dtype=np.float64) if normals else np.empty((0, 3), dtype=np.float64)

    params: Dict[str, Any] = {
        "n_shanks": n_shanks,
        "contacts_per_shank": contacts_per_shank,
        "contact_pitch_mm": contact_pitch_mm,
        "shank_pitch_mm": shank_pitch_mm,
        "shank_lengths": lengths,
        "default_shank_length_mm": default_shank_length_mm,
        "shank_width_um": shank_width_um,
        "arrangement": arrangement,
        "stagger_mode": stagger_mode,
        "stagger_mm": stagger_mm,
        "tip_angle_deg": tip_angle_deg,
        "contact_diameter_um": contact_diameter_um,
    }

    return ImplantSpec(
        name=name,
        family="multishank",
        params=params,
        contacts_local=contacts_arr,
        contact_normals_local=normals_arr,
        trajectories_local=trajectories,
        shanks=shanks,
        materials_meta={"shank_material": "Si", "contact_material": "Pt"},
    )


def _apply_stagger(n: int, base_length: float, mode: str, amount: float) -> List[float]:
    """Compute per-shank lengths with staggering."""
    if mode == "none" or amount == 0.0:
        return [base_length] * n
    elif mode == "linear":
        # linear ramp: shank 0 = base, shank N-1 = base + (N-1)*amount
        return [base_length + i * amount for i in range(n)]
    elif mode == "alternating":
        return [base_length + (amount if i % 2 == 1 else 0.0) for i in range(n)]
    elif mode == "custom":
        # custom mode expects caller to supply shank_lengths directly
        return [base_length] * n
    else:
        raise ValueError(f"Unknown stagger_mode '{mode}'")
