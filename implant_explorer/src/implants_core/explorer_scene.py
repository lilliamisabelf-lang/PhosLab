"""Explorer scene helpers for duplicate implant layouts."""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from implants_core.transforms import apply_transform_to_contacts


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def normalize_instance_placement(placement: Dict[str, Any] | None = None) -> Dict[str, float]:
    raw = dict(placement or {})
    return {
        "tx_mm": float(raw.get("tx_mm", 0.0)),
        "ty_mm": float(raw.get("ty_mm", 0.0)),
        "tz_mm": float(raw.get("tz_mm", 0.0)),
        "rx_deg": float(raw.get("rx_deg", 0.0)),
        "ry_deg": float(raw.get("ry_deg", 0.0)),
        "rz_deg": float(raw.get("rz_deg", 0.0)),
    }


@dataclass
class ExplorerImplantInstance:
    instance_id: str
    label: str
    design_revision_id: str
    visible: bool = True
    selected: bool = False
    placement: Dict[str, float] = field(default_factory=normalize_instance_placement)
    created_utc: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "label": self.label,
            "design_revision_id": self.design_revision_id,
            "visible": bool(self.visible),
            "selected": bool(self.selected),
            "placement": normalize_instance_placement(self.placement),
            "created_utc": str(self.created_utc),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExplorerImplantInstance":
        raw = dict(d or {})
        return cls(
            instance_id=str(raw.get("instance_id", "")),
            label=str(raw.get("label", "Implant")),
            design_revision_id=str(raw.get("design_revision_id", "")),
            visible=bool(raw.get("visible", True)),
            selected=bool(raw.get("selected", False)),
            placement=normalize_instance_placement(raw.get("placement", {})),
            created_utc=str(raw.get("created_utc", _utc_now_iso())),
        )


@dataclass
class ExplorerDesignRevision:
    revision_id: str
    label: str
    design_payload: Dict[str, Any]
    source_path: str = ""
    created_utc: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "label": self.label,
            "design_payload": dict(self.design_payload),
            "source_path": str(self.source_path),
            "created_utc": str(self.created_utc),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExplorerDesignRevision":
        raw = dict(d or {})
        return cls(
            revision_id=str(raw.get("revision_id", "")),
            label=str(raw.get("label", "Design")),
            design_payload=dict(raw.get("design_payload", {}) or {}),
            source_path=str(raw.get("source_path", "")),
            created_utc=str(raw.get("created_utc", _utc_now_iso())),
        )


@dataclass
class ExplorerScene:
    schema_version: str = "explorer_scene_v1"
    dataset_context: Dict[str, Any] = field(default_factory=dict)
    active_design_revision_id: str = ""
    design_revisions: List[ExplorerDesignRevision] = field(default_factory=list)
    instances: List[ExplorerImplantInstance] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_context": dict(self.dataset_context),
            "active_design_revision_id": str(self.active_design_revision_id),
            "design_revisions": [r.to_dict() for r in self.design_revisions],
            "instances": [inst.to_dict() for inst in self.instances],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExplorerScene":
        raw = dict(d or {})
        return cls(
            schema_version=str(raw.get("schema_version", "explorer_scene_v1")),
            dataset_context=dict(raw.get("dataset_context", {}) or {}),
            active_design_revision_id=str(raw.get("active_design_revision_id", "")),
            design_revisions=[
                ExplorerDesignRevision.from_dict(row)
                for row in list(raw.get("design_revisions", []) or [])
            ],
            instances=[
                ExplorerImplantInstance.from_dict(row)
                for row in list(raw.get("instances", []) or [])
            ],
        )

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ExplorerScene":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def transform_instance_contacts_mm(
    template_mm: np.ndarray,
    placement: Dict[str, Any] | None,
    anchor_mm: np.ndarray,
    *,
    entry_index: int = 0,
) -> np.ndarray:
    pts = np.asarray(template_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    place = normalize_instance_placement(placement)
    idx = int(min(max(0, entry_index), pts.shape[0] - 1))
    local_mm = pts - pts[idx]
    placed = apply_transform_to_contacts(
        local_mm,
        tx_mm=place["tx_mm"],
        ty_mm=place["ty_mm"],
        tz_mm=place["tz_mm"],
        rx_deg=place["rx_deg"],
        ry_deg=place["ry_deg"],
        rz_deg=place["rz_deg"],
        scale=1.0,
        mirror_x=False,
        pivot_mm=np.zeros(3, dtype=np.float64),
    )
    return placed + np.asarray(anchor_mm, dtype=np.float64).reshape(1, 3)


def hit_test_instance_contacts(
    world_xyz: Iterable[float] | None,
    contacts_by_instance: Dict[str, np.ndarray],
    threshold_vox: float,
) -> str | None:
    if world_xyz is None:
        return None
    world = np.asarray(list(world_xyz)[:3], dtype=np.float64)
    if world.shape != (3,) or not np.all(np.isfinite(world)):
        return None
    best_id = None
    best_dist = float("inf")
    for instance_id, pts in contacts_by_instance.items():
        arr = np.asarray(pts, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            continue
        d = np.linalg.norm(arr - world.reshape(1, 3), axis=1)
        cur = float(np.min(d))
        if cur < best_dist:
            best_dist = cur
            best_id = instance_id
    if best_dist <= float(threshold_vox):
        return best_id
    return None
