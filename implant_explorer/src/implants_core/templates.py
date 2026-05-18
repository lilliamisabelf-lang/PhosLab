"""
implants_core.templates - Preset implant library
================================================
Each template is a thin wrapper calling the appropriate generator
with published / commonly used parameters.

Usage::

    from implants_core import list_templates, get_template
    names = list_templates()               # ["Utah-96", ...]
    spec  = get_template("Utah-96")        # -> ImplantSpec
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List

from .spec import ImplantSpec
from .generators import generate_utah, generate_thread_bundle, generate_multishank


def _full_span_pitch(length_mm: float, n_contacts: int) -> float:
    if int(n_contacts) <= 1:
        return max(float(length_mm), 0.001)
    return max(float(length_mm) / float(int(n_contacts) - 1), 0.001)


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

def _utah_96() -> ImplantSpec:
    """Standard 96-channel Utah array (10x10 with 4 corner sites removed)."""
    mask = np.ones((10, 10), dtype=bool)
    mask[0, 0] = False
    mask[0, 9] = False
    mask[9, 0] = False
    mask[9, 9] = False
    return generate_utah(
        rows=10,
        cols=10,
        pitch_mm=0.4,
        shank_length_mm=1.5,
        tip_angle_deg=25.0,
        contact_diameter_um=20.0,
        site_mask=mask,
        name="Utah-96",
    )


def _utah_100() -> ImplantSpec:
    """Full 100-channel Utah array (10x10, no exclusions)."""
    return generate_utah(
        rows=10,
        cols=10,
        pitch_mm=0.4,
        shank_length_mm=1.5,
        tip_angle_deg=25.0,
        contact_diameter_um=20.0,
        name="Utah-100",
    )


def _utah_64() -> ImplantSpec:
    """64-channel Utah array (8x8)."""
    return generate_utah(
        rows=8,
        cols=8,
        pitch_mm=0.4,
        shank_length_mm=1.0,
        tip_angle_deg=25.0,
        contact_diameter_um=20.0,
        name="Utah-64",
    )


def _single_shank_32() -> ImplantSpec:
    """Single-shank 32-contact probe with full-span contact coverage."""
    contacts = 32
    length = 8.0
    return generate_multishank(
        n_shanks=1,
        contacts_per_shank=contacts,
        contact_pitch_mm=_full_span_pitch(length, contacts),
        shank_pitch_mm=0.0,
        default_shank_length_mm=length,
        arrangement="single_row",
        stagger_mode="none",
        tip_angle_deg=15.0,
        contact_diameter_um=15.0,
        name="Single-Shank-32",
    )


def _single_shank_64() -> ImplantSpec:
    """Single-shank 64-contact probe with full-span contact coverage."""
    contacts = 64
    length = 8.0
    return generate_multishank(
        n_shanks=1,
        contacts_per_shank=contacts,
        contact_pitch_mm=_full_span_pitch(length, contacts),
        shank_pitch_mm=0.0,
        default_shank_length_mm=length,
        arrangement="single_row",
        stagger_mode="none",
        tip_angle_deg=15.0,
        contact_diameter_um=15.0,
        name="Single-Shank-64",
    )


def _thread_1024() -> ImplantSpec:
    """Neuralink-style 1024-channel thread bundle (64 threads x 16 contacts)."""
    contacts = 16
    length = 4.0
    return generate_thread_bundle(
        n_threads=64,
        contacts_per_thread=contacts,
        contact_spacing_mm=_full_span_pitch(length, contacts),
        thread_length_mm=length,
        hub_radius_mm=3.5,
        layout="circular",
        contact_diameter_um=12.0,
        name="Thread-1024",
    )


def _thread_3072() -> ImplantSpec:
    """High-density thread bundle (96 threads x 32 contacts)."""
    contacts = 32
    length = 6.0
    return generate_thread_bundle(
        n_threads=96,
        contacts_per_thread=contacts,
        contact_spacing_mm=_full_span_pitch(length, contacts),
        thread_length_mm=length,
        hub_radius_mm=4.0,
        layout="circular",
        contact_diameter_um=10.0,
        name="Thread-3072",
    )


TEMPLATES: Dict[str, Callable[[], ImplantSpec]] = {
    "Utah-96": _utah_96,
    "Utah-100": _utah_100,
    "Utah-64": _utah_64,
    "Single-Shank-32": _single_shank_32,
    "Single-Shank-64": _single_shank_64,
    "Thread-1024": _thread_1024,
    "Thread-3072": _thread_3072,
}

TEMPLATE_FAMILIES: Dict[str, str] = {
    "Utah-96": "utah",
    "Utah-100": "utah",
    "Utah-64": "utah",
    "Single-Shank-32": "multishank",
    "Single-Shank-64": "multishank",
    "Thread-1024": "thread",
    "Thread-3072": "thread",
}


def list_templates(family: str | None = None) -> List[str]:
    """Return template names, optionally filtered by family."""
    if family is None:
        return list(TEMPLATES.keys())
    return [k for k, v in TEMPLATE_FAMILIES.items() if v == family]


def get_template(name: str) -> ImplantSpec:
    """Instantiate a template by name. Raises ``KeyError`` if unknown."""
    if name not in TEMPLATES:
        raise KeyError(f"Unknown template '{name}'. Available: {list(TEMPLATES.keys())}")
    return TEMPLATES[name]()
