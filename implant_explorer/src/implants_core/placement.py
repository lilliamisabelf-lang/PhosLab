"""PlacementController — multi-implant placement manager
=======================================================

Sits between the designer's ImplantSpec and the explorer's RF pipeline.
Manages one or more ImplantInstance objects with per-instance transforms,
surface-snap, and voxel coordinate conversion.

Provides a single combined contacts-in-voxels array that the explorer can
feed directly into ``collect_rfs`` / ``update_electrodes``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .spec import ImplantSpec, ImplantInstance
from .constraints import validate_spec, validate_instance


# ---------------------------------------------------------------------------
# Voxel conversion helpers
# ---------------------------------------------------------------------------

def mm_to_voxel(pts_mm: np.ndarray, voxel_size_mm: float = 0.5) -> np.ndarray:
    """Convert (N,3) mm coordinates to voxel coordinates (integer-ish)."""
    return pts_mm / voxel_size_mm


def voxel_to_mm(pts_vox: np.ndarray, voxel_size_mm: float = 0.5) -> np.ndarray:
    """Convert (N,3) voxel coordinates to mm."""
    return pts_vox * voxel_size_mm


# ---------------------------------------------------------------------------
# Per-slot metadata
# ---------------------------------------------------------------------------

@dataclass
class ImplantSlot:
    """One implant in the scene, with placement state."""
    instance: ImplantInstance
    visible: bool = True
    color: str = "#61afef"
    label: str = ""
    
    # Per-slot display cache
    _contacts_vox: np.ndarray | None = field(default=None, repr=False)

    def contacts_world(self) -> np.ndarray:
        return self.instance.contacts_world

    def contacts_voxel(self, voxel_size_mm: float = 0.5) -> np.ndarray:
        if self._contacts_vox is None:
            self._contacts_vox = mm_to_voxel(self.instance.contacts_world, voxel_size_mm)
        return self._contacts_vox

    def invalidate_cache(self):
        self._contacts_vox = None


# ---------------------------------------------------------------------------
# PlacementController
# ---------------------------------------------------------------------------

SLOT_COLORS = [
    "#e06c75", "#61afef", "#98c379", "#e5c07b",
    "#c678dd", "#56b6c2", "#be5046", "#d19a66",
]


class PlacementController:
    """Manages multiple implant placements on a brain surface.

    Usage::

        pc = PlacementController(voxel_size_mm=0.5)
        idx = pc.add_implant(spec, position_mm=np.array([10, 20, 5]))
        pc.translate(idx, np.array([1, 0, 0]))
        contacts_vox = pc.combined_contacts_voxel()  # feed to explorer
    """

    def __init__(self, voxel_size_mm: float = 0.5, max_implants: int = 8):
        self.voxel_size_mm = voxel_size_mm
        self.max_implants = max_implants
        self._slots: list[ImplantSlot] = []

    # -- Slot CRUD -------------------------------------------------------

    def add_implant(
        self,
        spec: ImplantSpec,
        position_mm: np.ndarray | None = None,
        rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        label: str = "",
    ) -> int:
        """Add an implant to the scene.  Returns the slot index."""
        if len(self._slots) >= self.max_implants:
            raise ValueError(f"Max {self.max_implants} implants reached")

        instance = ImplantInstance(spec=spec)
        if position_mm is not None:
            pos = np.asarray(position_mm, dtype=np.float64).ravel()
            instance.set_position(float(pos[0]), float(pos[1]), float(pos[2]))
        if any(r != 0 for r in rotation_deg):
            instance.set_rotation(*rotation_deg)

        idx = len(self._slots)
        color = SLOT_COLORS[idx % len(SLOT_COLORS)]
        slot = ImplantSlot(
            instance=instance,
            visible=True,
            color=color,
            label=label or f"{spec.name} #{idx}",
        )
        self._slots.append(slot)
        return idx

    def remove_implant(self, idx: int) -> None:
        if 0 <= idx < len(self._slots):
            self._slots.pop(idx)

    def clear(self) -> None:
        self._slots.clear()

    @property
    def num_implants(self) -> int:
        return len(self._slots)

    def slot(self, idx: int) -> ImplantSlot:
        return self._slots[idx]

    def slots(self) -> list[ImplantSlot]:
        return list(self._slots)

    # -- Transform operations -------------------------------------------

    def translate(self, idx: int, delta_mm: np.ndarray) -> None:
        """Translate implant *idx* by *delta_mm* (3-vector)."""
        s = self._slots[idx]
        current = s.instance.contacts_world.mean(axis=0) if s.instance.contacts_world.shape[0] else np.zeros(3)
        new_pos = current + np.asarray(delta_mm, dtype=np.float64)
        s.instance.set_position(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
        s.invalidate_cache()

    def set_position(self, idx: int, position_mm: np.ndarray) -> None:
        s = self._slots[idx]
        pos = np.asarray(position_mm, dtype=np.float64).ravel()
        s.instance.set_position(float(pos[0]), float(pos[1]), float(pos[2]))
        s.invalidate_cache()

    def set_rotation(self, idx: int, rx: float, ry: float, rz: float) -> None:
        s = self._slots[idx]
        s.instance.set_rotation(rx, ry, rz)
        s.invalidate_cache()

    def set_transform(self, idx: int, transform_4x4: np.ndarray) -> None:
        s = self._slots[idx]
        s.instance.apply_transform(np.asarray(transform_4x4, dtype=np.float64))
        s.invalidate_cache()

    def snap_to_surface(
        self,
        idx: int,
        surface_point_mm: np.ndarray,
        surface_normal: np.ndarray | None = None,
    ) -> None:
        """Snap implant *idx* so its entry face sits at *surface_point_mm*,
        optionally aligning the implant normal to *surface_normal*."""
        s = self._slots[idx]
        s.instance.snap_to_surface(
            np.asarray(surface_point_mm, dtype=np.float64),
            np.asarray(surface_normal, dtype=np.float64) if surface_normal is not None else None,
        )
        s.invalidate_cache()

    # -- Visibility -----------------------------------------------------

    def set_visible(self, idx: int, visible: bool) -> None:
        self._slots[idx].visible = visible

    def toggle_visible(self, idx: int) -> bool:
        self._slots[idx].visible = not self._slots[idx].visible
        return self._slots[idx].visible

    # -- Combined output -------------------------------------------------

    def combined_contacts_world(self, visible_only: bool = True) -> np.ndarray:
        """Return (M,3) contacts from all visible implants, concatenated."""
        arrays = []
        for s in self._slots:
            if visible_only and not s.visible:
                continue
            w = s.contacts_world()
            if w.shape[0] > 0:
                arrays.append(w)
        if not arrays:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack(arrays)

    def combined_contacts_voxel(self, visible_only: bool = True) -> np.ndarray:
        """Return (M,3) contacts in voxel space for all visible implants."""
        arrays = []
        for s in self._slots:
            if visible_only and not s.visible:
                continue
            v = s.contacts_voxel(self.voxel_size_mm)
            if v.shape[0] > 0:
                arrays.append(v)
        if not arrays:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack(arrays)

    def per_implant_contacts_voxel(self, visible_only: bool = True) -> list[np.ndarray]:
        """Return list of per-implant (Ni,3) voxel arrays (matches explorer's per-comb pattern)."""
        out = []
        for s in self._slots:
            if visible_only and not s.visible:
                continue
            out.append(s.contacts_voxel(self.voxel_size_mm))
        return out

    def contact_slot_indices(self, visible_only: bool = True) -> np.ndarray:
        """Return (M,) int array mapping each combined contact to its slot index."""
        parts = []
        for i, s in enumerate(self._slots):
            if visible_only and not s.visible:
                continue
            n = s.contacts_world().shape[0]
            parts.append(np.full(n, i, dtype=np.int32))
        if not parts:
            return np.empty(0, dtype=np.int32)
        return np.concatenate(parts)

    # -- Validation ------------------------------------------------------

    def validate_all(self, **kwargs) -> dict[int, list]:
        """Run constraints on each slot. Returns {slot_idx: [ConstraintResult]}."""
        results = {}
        for i, s in enumerate(self._slots):
            spec_results = validate_spec(s.instance.spec)
            inst_results = validate_instance(s.instance, **kwargs)
            results[i] = spec_results + inst_results
        return results

    # -- Serialization ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "voxel_size_mm": self.voxel_size_mm,
            "slots": [
                {
                    "instance": s.instance.to_dict(),
                    "visible": s.visible,
                    "color": s.color,
                    "label": s.label,
                }
                for s in self._slots
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlacementController":
        pc = cls(voxel_size_mm=d.get("voxel_size_mm", 0.5))
        for sd in d.get("slots", []):
            inst = ImplantInstance.from_dict(sd["instance"])
            slot = ImplantSlot(
                instance=inst,
                visible=sd.get("visible", True),
                color=sd.get("color", "#61afef"),
                label=sd.get("label", ""),
            )
            pc._slots.append(slot)
        return pc
