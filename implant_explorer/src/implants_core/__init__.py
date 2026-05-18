"""
implants_core – Parametric implant design & placement library
=============================================================
Pure Python (no Qt).  All spatial units millimetres unless noted.

Public API
----------
* Data model:   ``ImplantSpec``, ``ImplantInstance``, ``ConstraintResult``
* Generators:   ``generate_utah``, ``generate_thread_bundle``, ``generate_multishank``
* Templates:    ``list_templates``, ``get_template``
* Validation:   ``validate_spec``, ``validate_instance``
* Export:       ``save_spec_json``, ``load_spec_json``, ``save_contacts_csv``
* Caching:      ``ContactCache``, ``RFCache``
* Placement:    ``PlacementController``, ``ImplantSlot``
"""

from .spec import (
    ImplantSpec,
    ImplantInstance,
    ConstraintResult,
)
from .generators import (
    generate_utah,
    generate_thread_bundle,
    generate_multishank,
)
from .templates import list_templates, get_template
from .constraints import validate_spec, validate_instance
from .export import save_spec_json, load_spec_json, save_contacts_csv
from .cache import ContactCache, RFCache
from .placement import PlacementController, ImplantSlot
from .transforms import compose_transform_matrix, apply_transform_to_contacts, validate_contacts_soft

__all__ = [
    # data model
    "ImplantSpec",
    "ImplantInstance",
    "ConstraintResult",
    # generators
    "generate_utah",
    "generate_thread_bundle",
    "generate_multishank",
    # templates
    "list_templates",
    "get_template",
    # validation
    "validate_spec",
    "validate_instance",
    # export
    "save_spec_json",
    "load_spec_json",
    "save_contacts_csv",
    "compose_transform_matrix",
    "apply_transform_to_contacts",
    "validate_contacts_soft",
    # caching
    "ContactCache",
    "RFCache",
]
