"""
implants_core.export – Serialisation & file export
===================================================
* JSON/YAML spec files (the canonical format)
* CSV contact lists (for analysis / external tools)
* OBJ mesh export (for figures)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .spec import ImplantSpec, ImplantInstance


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def save_spec_json(spec: ImplantSpec, path: str | Path) -> None:
    """Save an ``ImplantSpec`` to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f, indent=2)


def load_spec_json(path: str | Path) -> ImplantSpec:
    """Load a v2 ``ImplantSpec`` from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return ImplantSpec.from_dict(d)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_contacts_csv(
    instance_or_spec,
    path: str | Path,
    *,
    rf_data: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """
    Export per-contact CSV.

    Parameters
    ----------
    instance_or_spec : ImplantSpec or ImplantInstance
        If ``ImplantInstance``, world-space coordinates are used.
        If ``ImplantSpec``, local coordinates are used.
    path : output file path
    rf_data : optional dict with keys ``"ecc"``, ``"pol"``, ``"area"``
              each an (N,) array matching contact order.
    """
    is_instance = isinstance(instance_or_spec, ImplantInstance)
    if is_instance:
        inst: ImplantInstance = instance_or_spec
        spec = inst.spec
        pts = inst.contacts_world
        nrm = inst.contact_normals_world
    else:
        spec = instance_or_spec
        pts = spec.contacts_local
        nrm = spec.contact_normals_local

    n = pts.shape[0]

    # build shank_id + contact_idx per contact
    shank_ids = []
    contact_idxs = []
    for sd in spec.shanks:
        for c in sd.get("contacts", []):
            shank_ids.append(sd["id"])
            contact_idxs.append(c["id"])
    # pad if shanks list doesn't cover all contacts (custom implants)
    while len(shank_ids) < n:
        shank_ids.append(-1)
        contact_idxs.append(-1)

    header = [
        "contact_id", "x_mm", "y_mm", "z_mm",
        "nx", "ny", "nz",
        "shank_id", "contact_idx",
    ]
    has_rf = rf_data is not None
    if has_rf:
        header.extend(["ecc", "pol", "area_label"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n):
            row = [
                i,
                f"{pts[i, 0]:.6f}",
                f"{pts[i, 1]:.6f}",
                f"{pts[i, 2]:.6f}",
                f"{nrm[i, 0]:.6f}" if i < nrm.shape[0] else "0",
                f"{nrm[i, 1]:.6f}" if i < nrm.shape[0] else "0",
                f"{nrm[i, 2]:.6f}" if i < nrm.shape[0] else "0",
                shank_ids[i],
                contact_idxs[i],
            ]
            if has_rf:
                row.append(f"{rf_data['ecc'][i]:.4f}" if i < len(rf_data.get("ecc", [])) else "")
                row.append(f"{rf_data['pol'][i]:.4f}" if i < len(rf_data.get("pol", [])) else "")
                row.append(rf_data.get("area", [""])[i] if i < len(rf_data.get("area", [])) else "")
            writer.writerow(row)


# ---------------------------------------------------------------------------
# OBJ mesh export (optional, for figures)
# ---------------------------------------------------------------------------

def save_mesh_obj(spec: ImplantSpec, path: str | Path) -> None:
    """
    Export a simple OBJ mesh representing shanks as rectangular prisms
    and contacts as small cubes. Suitable for 3-D figure rendering.
    """
    vertices: List[np.ndarray] = []
    faces: List[List[int]] = []
    v_offset = 0

    for sd in spec.shanks:
        x, y = sd["x_mm"], sd["y_mm"]
        length = sd["length_mm"]
        hw = 0.02  # half-width of shank cross section (mm)

        # shank body: rectangular prism from z=0 (tip) to z=-length (entry)
        corners = np.array([
            [x - hw, y - hw, 0.0],
            [x + hw, y - hw, 0.0],
            [x + hw, y + hw, 0.0],
            [x - hw, y + hw, 0.0],
            [x - hw, y - hw, -length],
            [x + hw, y - hw, -length],
            [x + hw, y + hw, -length],
            [x - hw, y + hw, -length],
        ])
        vertices.extend(corners)
        # 6 faces (quads, 1-indexed in OBJ)
        base = v_offset + 1
        box_faces = [
            [base, base+1, base+2, base+3],       # tip face
            [base+4, base+7, base+6, base+5],     # entry face
            [base, base+4, base+5, base+1],       # front
            [base+1, base+5, base+6, base+2],     # right
            [base+2, base+6, base+7, base+3],     # back
            [base+3, base+7, base+4, base],       # left
        ]
        faces.extend(box_faces)
        v_offset += 8

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# ImplantSpec OBJ export: {spec.name}\n")
        f.write(f"# {spec.total_contacts()} contacts, {len(spec.shanks)} shanks\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write("f " + " ".join(str(vi) for vi in face) + "\n")
