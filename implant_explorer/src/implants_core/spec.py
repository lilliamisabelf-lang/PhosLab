"""
implants_core.spec – Canonical data model
==========================================
``ImplantSpec``  = pure design (no anatomy).
``ImplantInstance`` = placed design on a subject.
``ConstraintResult``  = one validation finding.

"""
from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_hash(obj: Any) -> str:
    """Deterministic SHA-256 of a JSON-serialisable object."""
    raw = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _rot_matrix_4x4(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """4×4 homogeneous rotation Rx @ Ry @ Rz (intrinsic, degrees)."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    R = np.eye(4, dtype=np.float64)
    R[:3, :3] = np.array([
        [cy * cz,              -cy * sz,               sy],
        [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
        [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz,  cx * cy],
    ])
    return R


def _translation_4x4(dx: float, dy: float, dz: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [dx, dy, dz]
    return T


# ---------------------------------------------------------------------------
# ConstraintResult
# ---------------------------------------------------------------------------

@dataclass
class ConstraintResult:
    """One validation finding."""
    rule: str
    status: str            # "ok" | "warn" | "block"
    margin: float = 0.0    # numeric slack (positive = OK)
    message: str = ""
    suggested_fix: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConstraintResult":
        return cls(**d)


# ---------------------------------------------------------------------------
# ImplantSpec (pure design, no anatomy)
# ---------------------------------------------------------------------------

@dataclass
class ImplantSpec:
    """
    Parametric implant design – no placement info.

    ``contacts_local``  (N, 3) float64 mm – electrode tips in implant-local frame.
    ``contact_normals_local`` (N, 3) float64 – unit normals at each contact.
    ``trajectories_local``  list of (M, 3) arrays – per-shank/thread polylines
                             (first point = entry, last = tip).
    ``params``  stores the exact generator arguments so the design is reproducible.
    ``shanks``  low-level per-shank/per-contact list (kept for v1 compat & per-site edits).
    ``materials_meta``  labels only (Pt, IrOx, notes…).
    """

    name: str
    family: str                                   # "utah" | "thread" | "multishank"
    params: Dict[str, Any] = field(default_factory=dict)

    # -- geometry (populated by generators) --
    contacts_local: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    contact_normals_local: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    trajectories_local: List[np.ndarray] = field(default_factory=list)

    # -- low-level per-shank representation (v1 compat) --
    shanks: List[Dict[str, Any]] = field(default_factory=list)

    # -- metadata --
    materials_meta: Dict[str, Any] = field(default_factory=dict)
    transform_meta: Dict[str, Any] = field(default_factory=dict)
    placement_snapshot: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "2.0"
    created: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def total_contacts(self) -> int:
        return int(self.contacts_local.shape[0])

    def footprint_mm(self) -> tuple[float, float]:
        """(width, height) bounding box of contact projection."""
        if self.contacts_local.shape[0] == 0:
            return 0.0, 0.0
        if self.family in {"utah", "thread"}:
            x = self.contacts_local[:, 0]
            y = self.contacts_local[:, 1]
            z = self.contacts_local[:, 2]
            y_span = float(y.max() - y.min())
            z_span = float(z.max() - z.min())
            h_span = z_span if z_span > y_span else y_span
            return float(x.max() - x.min()), h_span
        mn = self.contacts_local[:, :2].min(axis=0)
        mx = self.contacts_local[:, :2].max(axis=0)
        return float(mx[0] - mn[0]), float(mx[1] - mn[1])

    def depth_range_mm(self) -> tuple[float, float]:
        """(min_z, max_z) of contacts in local frame."""
        if self.contacts_local.shape[0] == 0:
            return 0.0, 0.0
        if self.family == "thread":
            axis = 1
        else:
            axis = 2
        return float(self.contacts_local[:, axis].min()), float(self.contacts_local[:, axis].max())

    def spec_hash(self) -> str:
        """Deterministic hash of design parameters + contact positions."""
        obj = {
            "family": self.family,
            "params": self.params,
            "contacts": self.contacts_local.round(6).tolist(),
        }
        return _stable_hash(obj)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "schema_version": self.schema_version,
            "name": self.name,
            "family": self.family,
            "params": self.params,
            "contacts_local": self.contacts_local.tolist(),
            "contact_normals_local": self.contact_normals_local.tolist(),
            "trajectories_local": [t.tolist() for t in self.trajectories_local],
            "shanks": self.shanks,
            "materials_meta": self.materials_meta,
            "created": self.created,
        }
        if self.transform_meta:
            out["transform_meta"] = self.transform_meta
        if self.placement_snapshot:
            out["placement_snapshot"] = self.placement_snapshot
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImplantSpec":
        d = dict(d)  # shallow copy
        ver = str(d.get("schema_version", ""))
        if not ver.startswith("2"):
            raise ValueError(f"Unsupported implant schema_version '{ver or 'missing'}'; expected 2.x")
        contacts = np.array(d.pop("contacts_local", []), dtype=np.float64)
        if contacts.ndim == 1 and contacts.size == 0:
            contacts = np.empty((0, 3), dtype=np.float64)
        normals = np.array(d.pop("contact_normals_local", []), dtype=np.float64)
        if normals.ndim == 1 and normals.size == 0:
            normals = np.empty((0, 3), dtype=np.float64)
        trajs = [np.array(t, dtype=np.float64) for t in d.pop("trajectories_local", [])]
        d.pop("schema_version", None)
        return cls(
            contacts_local=contacts,
            contact_normals_local=normals,
            trajectories_local=trajs,
            schema_version="2.0",
            **{k: v for k, v in d.items() if k in cls.__dataclass_fields__},
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ImplantSpec":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

# ---------------------------------------------------------------------------
# ImplantInstance (placed on anatomy)
# ---------------------------------------------------------------------------

@dataclass
class ImplantInstance:
    """
    A placed implant: ``ImplantSpec`` + rigid transform + world-space contacts.
    """
    spec: ImplantSpec
    transform_world_from_implant: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )

    # world-space derived geometry (call ``apply_transform`` to refresh)
    contacts_world: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    contact_normals_world: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    trajectories_world: List[np.ndarray] = field(default_factory=list)

    constraints_report: List[ConstraintResult] = field(default_factory=list)

    # ------------------------------------------------------------------

    def __post_init__(self):
        if self.contacts_world.shape[0] == 0 and self.spec.contacts_local.shape[0] > 0:
            self.apply_transform(self.transform_world_from_implant)

    def apply_transform(self, T: np.ndarray) -> None:
        """Update world-space geometry from a 4×4 transform."""
        self.transform_world_from_implant = np.array(T, dtype=np.float64)
        R = T[:3, :3]
        t = T[:3, 3]

        cl = self.spec.contacts_local
        self.contacts_world = (cl @ R.T + t).astype(np.float64)

        nl = self.spec.contact_normals_local
        self.contact_normals_world = (nl @ R.T).astype(np.float64)

        self.trajectories_world = [
            (tr @ R.T + t).astype(np.float64) for tr in self.spec.trajectories_local
        ]

    def set_position(self, x: float, y: float, z: float) -> None:
        """Set translation, keeping current rotation."""
        T = self.transform_world_from_implant.copy()
        T[:3, 3] = [x, y, z]
        self.apply_transform(T)

    def set_rotation(self, rx_deg: float, ry_deg: float, rz_deg: float,
                     pivot: Optional[np.ndarray] = None) -> None:
        """Set rotation (keeps current translation unless pivot given)."""
        R = _rot_matrix_4x4(rx_deg, ry_deg, rz_deg)
        if pivot is not None:
            pivot = np.asarray(pivot, dtype=np.float64)
            T_pre = _translation_4x4(-pivot[0], -pivot[1], -pivot[2])
            T_post = _translation_4x4(pivot[0], pivot[1], pivot[2])
            R = T_post @ R @ T_pre
        R[:3, 3] = self.transform_world_from_implant[:3, 3]
        self.apply_transform(R)

    def snap_to_surface(
        self,
        surface_point: np.ndarray,
        surface_normal: np.ndarray,
    ) -> None:
        """
        Orient implant so its local -Z axis aligns with surface normal,
        and translate so implant origin sits at *surface_point*.
        """
        n = np.asarray(surface_normal, dtype=np.float64)
        n = n / (np.linalg.norm(n) + 1e-12)

        # target z-axis of implant in world = -surface_normal (shanks point into cortex)
        z_axis = -n
        # pick an arbitrary up direction that isn't parallel to z_axis
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(up, z_axis)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis) + 1e-12
        y_axis = np.cross(z_axis, x_axis)

        T = np.eye(4, dtype=np.float64)
        T[:3, 0] = x_axis
        T[:3, 1] = y_axis
        T[:3, 2] = z_axis
        T[:3, 3] = np.asarray(surface_point, dtype=np.float64)
        self.apply_transform(T)

    def instance_hash(self) -> str:
        obj = {
            "spec_hash": self.spec.spec_hash(),
            "transform": self.transform_world_from_implant.round(8).tolist(),
        }
        return _stable_hash(obj)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "transform_world_from_implant": self.transform_world_from_implant.tolist(),
            "constraints_report": [c.to_dict() for c in self.constraints_report],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImplantInstance":
        spec = ImplantSpec.from_dict(d["spec"])
        T = np.array(d.get("transform_world_from_implant", np.eye(4).tolist()), dtype=np.float64)
        inst = cls(spec=spec, transform_world_from_implant=T)
        inst.constraints_report = [
            ConstraintResult.from_dict(c) for c in d.get("constraints_report", [])
        ]
        inst.apply_transform(T)
        return inst


