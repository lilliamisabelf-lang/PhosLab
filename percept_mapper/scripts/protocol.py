"""Experiment protocol schema.

The plan is to migrate `main.py` from being the *definition* of the
experiment to being a *runner* that consumes a declarative protocol.
This module defines the schema today (no runtime change yet) so we have
a typed contract to write the existing experiment against and validate
future protocols against.

A `ProtocolSpec` carries:
- top-level identity (name, version, schema_version)
- a list of `PhaseSpec` describing the screens the participant sees
  in order on each trial
- the trial-sequence config (randomization, catch trials, practice)
- timing defaults consumed by phases

Phases declare *what* they do (screen + gate) but not *how* — the
runner maps `screen` strings to concrete pygame screens and `gate`
strings to actual wait conditions. This indirection is the whole
point: changing the protocol becomes editing YAML, not Python.

All records are round-trippable through `from_dict` / `to_dict` like
the records in `schemas.py`, and carry an `extras` escape hatch so the
schema can evolve without dropping fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

PROTOCOL_SCHEMA_VERSION = 1


# Allowed values for the discriminator fields. The runner will dispatch
# on these strings. Extending requires adding both a literal here and a
# handler in the runner — the validator below checks the literal set.
SCREEN_TYPES = frozenset({"anchor", "stimulation", "saccade", "drawing"})

GATE_TYPES = frozenset({
    # Wait for `value` ms, then advance.
    "time_ms",
    # Wait until participant accumulates `value` ms of continuous fixation
    # on the screen's anchor target. Failing the gate retries the phase.
    "continuous_fixation_ms",
    # Wait `value` ms, but abort the trial if fixation is lost any time
    # during the window.
    "time_or_lost_fixation_ms",
    # Wait until the response capture (drawing/saccade) signals finished.
    "response_finished",
})

ON_FIXATION_LOST = frozenset({"retry_phase", "abort_trial", "continue"})


@dataclass(frozen=True)
class PhaseSpec:
    """One phase in a single trial.

    `screen` is the symbolic name of the screen to display
    (`anchor`, `stimulation`, `saccade`, `drawing`).
    `gate` is the wait condition that ends the phase. `value` is the
    numerical parameter consumed by the gate (typically ms).
    `on_fixation_lost` controls retry policy when a fixation-aware gate
    detects loss-of-fixation mid-phase. Defaults to `retry_phase` which
    matches the current implementation.
    """
    name: str
    screen: str
    gate: str
    value: float = 0.0
    on_fixation_lost: str = "retry_phase"
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> PhaseSpec:
        known = {"name", "screen", "gate", "value", "on_fixation_lost"}
        extras = {k: v for k, v in d.items() if k not in known}
        return cls(
            name=str(d.get("name", "")),
            screen=str(d.get("screen", "")),
            gate=str(d.get("gate", "time_ms")),
            value=float(d.get("value", 0.0)),
            on_fixation_lost=str(d.get("on_fixation_lost", "retry_phase")),
            extras=extras,
        )

    def to_dict(self) -> dict:
        out = {
            "name": self.name,
            "screen": self.screen,
            "gate": self.gate,
            "value": float(self.value),
            "on_fixation_lost": self.on_fixation_lost,
        }
        out.update(self.extras)
        return out


@dataclass(frozen=True)
class TrialSequenceSpec:
    """Same fields as `schemas.TrialSequenceConfig` — duplicated here so a
    protocol file is self-describing without referencing the in-flight
    session schema. Validation tools can compare them for drift."""
    randomize: bool = True
    random_seed: int | None = None
    catch_trial_rate: float = 0.0
    no_immediate_repeat: bool = True
    num_practice_trials: int = 0
    isi_jitter_ms: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> TrialSequenceSpec:
        return cls(
            randomize=bool(d.get("randomize", True)),
            random_seed=d.get("random_seed"),
            catch_trial_rate=float(d.get("catch_trial_rate", 0.0)),
            no_immediate_repeat=bool(d.get("no_immediate_repeat", True)),
            num_practice_trials=int(d.get("num_practice_trials", 0)),
            isi_jitter_ms=float(d.get("isi_jitter_ms", 0.0)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProtocolSpec:
    """Top-level protocol record. Loaded from `config/protocol.yaml`
    (or any other path passed to a future `run_protocol.py`)."""
    name: str
    version: str
    phases: list[PhaseSpec]
    trial_sequence: TrialSequenceSpec
    response_mode: str  # "saccade" | "drawing"
    schema_version: int = PROTOCOL_SCHEMA_VERSION
    description: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    _KNOWN = frozenset({
        "name", "version", "phases", "trial_sequence",
        "response_mode", "schema_version", "description",
    })

    @classmethod
    def from_dict(cls, d: dict) -> ProtocolSpec:
        known = {k: d[k] for k in cls._KNOWN if k in d}
        extras = {k: v for k, v in d.items() if k not in cls._KNOWN}
        phases_raw = known.get("phases") or []
        seq_raw = known.get("trial_sequence") or {}
        return cls(
            name=str(known.get("name", "")),
            version=str(known.get("version", "0")),
            phases=[PhaseSpec.from_dict(p) for p in phases_raw],
            trial_sequence=TrialSequenceSpec.from_dict(seq_raw),
            response_mode=str(known.get("response_mode", "saccade")),
            schema_version=int(known.get("schema_version", PROTOCOL_SCHEMA_VERSION)),
            description=str(known.get("description", "")),
            extras=extras,
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "schema_version": int(self.schema_version),
            "description": self.description,
            "response_mode": self.response_mode,
            "trial_sequence": self.trial_sequence.to_dict(),
            "phases": [p.to_dict() for p in self.phases],
        }
        out.update(self.extras)
        return out


# ---- validation ------------------------------------------------------------


class ProtocolValidationError(ValueError):
    """Raised by `validate_protocol` when the protocol violates one of the
    structural constraints. Carries the full list of issues found, not
    just the first — so a YAML author sees everything wrong in one shot."""

    def __init__(self, issues: list[str]):
        self.issues = list(issues)
        super().__init__("Protocol validation failed:\n  - " + "\n  - ".join(self.issues))


def validate_protocol(protocol: ProtocolSpec) -> list[str]:
    """Return a list of structural issues. Empty list means the protocol
    is internally consistent. Does *not* raise — callers decide whether
    to treat issues as fatal."""
    issues: list[str] = []

    if not protocol.name:
        issues.append("protocol.name is empty")
    if not protocol.phases:
        issues.append("protocol.phases is empty (need at least one phase)")
    if protocol.response_mode not in {"saccade", "drawing"}:
        issues.append(
            f"response_mode={protocol.response_mode!r} not in {{'saccade','drawing'}}"
        )

    seen_names: set[str] = set()
    has_response_phase = False
    for i, phase in enumerate(protocol.phases):
        prefix = f"phases[{i}] (name={phase.name!r})"
        if not phase.name:
            issues.append(f"{prefix}: empty name")
        elif phase.name in seen_names:
            issues.append(f"{prefix}: duplicate phase name")
        seen_names.add(phase.name)
        if phase.screen not in SCREEN_TYPES:
            issues.append(
                f"{prefix}: screen={phase.screen!r} not in {sorted(SCREEN_TYPES)}"
            )
        if phase.gate not in GATE_TYPES:
            issues.append(
                f"{prefix}: gate={phase.gate!r} not in {sorted(GATE_TYPES)}"
            )
        if phase.on_fixation_lost not in ON_FIXATION_LOST:
            issues.append(
                f"{prefix}: on_fixation_lost={phase.on_fixation_lost!r} not in "
                f"{sorted(ON_FIXATION_LOST)}"
            )
        if phase.gate in {"time_ms", "continuous_fixation_ms", "time_or_lost_fixation_ms"} and phase.value <= 0:
            issues.append(f"{prefix}: gate={phase.gate} requires value > 0, got {phase.value}")
        if phase.screen in {"saccade", "drawing"}:
            has_response_phase = True

    if not has_response_phase:
        issues.append(
            "protocol.phases contains no saccade or drawing phase — "
            "participant has no way to report a percept"
        )

    seq = protocol.trial_sequence
    if not (0.0 <= seq.catch_trial_rate < 1.0):
        issues.append(
            f"trial_sequence.catch_trial_rate={seq.catch_trial_rate} not in [0, 1)"
        )
    if seq.num_practice_trials < 0:
        issues.append(
            f"trial_sequence.num_practice_trials={seq.num_practice_trials} < 0"
        )
    if seq.isi_jitter_ms < 0:
        issues.append(f"trial_sequence.isi_jitter_ms={seq.isi_jitter_ms} < 0")

    return issues


def assert_valid(protocol: ProtocolSpec) -> None:
    """Like `validate_protocol`, but raises `ProtocolValidationError` if
    any issues exist. Convenience for code paths that don't want to
    propagate an issue list."""
    issues = validate_protocol(protocol)
    if issues:
        raise ProtocolValidationError(issues)


# ---- loader ---------------------------------------------------------------


def load_protocol(path) -> ProtocolSpec:
    """Load + parse a YAML/JSON protocol file. Does *not* validate —
    call `validate_protocol(...)` afterwards for that. Splitting lets a
    caller report parse errors and structural errors separately."""
    from pathlib import Path
    import yaml
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ProtocolSpec.from_dict(raw)
