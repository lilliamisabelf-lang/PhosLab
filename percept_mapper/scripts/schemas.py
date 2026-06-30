"""Typed schemas for the load-bearing records in percept_mapper.

Why this module exists:
- Every payload between phases (per-trial metadata, session metadata,
  analyzer results) is currently a hand-rolled dict. Analyzers
  defensively `.get()` everything because the shape drifts silently
  across releases.
- A versioned dataclass for each record gives one place to look up the
  current shape, a `schema_version` field so old data is identifiable,
  and an `extras: dict` escape hatch so we can roll this out
  incrementally without dropping any field on the floor.

The shape is intentionally JSON-round-trippable — `dataclasses.asdict`
on every record produces a dict that `from_dict` will re-hydrate
identically. Hypothesis tests in `schemas_property_test.py` lock that
invariant in.

Schema versioning: bump `SCHEMA_VERSION` only when an existing field
*changes meaning*. Adding fields is non-breaking — old data loads with
the new fields at their defaults. Removing or renaming a field is
breaking and requires migration logic in `_upgrade()`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


# ---- helpers ---------------------------------------------------------------


def _pop_known(d: dict, names: set[str]) -> tuple[dict, dict]:
    """Split d into (known_fields, extras) by name."""
    known = {}
    extras = {}
    for k, v in d.items():
        (known if k in names else extras)[k] = v
    return known, extras


def _upgrade(d: dict, target_version: int) -> dict:
    """Migrate an older record dict up to `target_version`. No-op when
    the version is already current or absent (data older than the
    versioning scheme is treated as v0 → v1).

    Migration rules live here. They must be small, explicit, and one-way:
    no record should ever be written at < SCHEMA_VERSION.
    """
    v = int(d.get("schema_version", 0))
    out = dict(d)

    # v0 → v1: stamp the version. No field renames yet.
    if v < 1 and target_version >= 1:
        out["schema_version"] = 1
        v = 1

    return out


# ---- TrialSequenceConfig (nested inside SessionMetadata) -------------------


@dataclass(frozen=True)
class TrialSequenceConfig:
    randomize: bool = True
    random_seed: int = 0
    catch_trial_rate: float = 0.0
    no_immediate_repeat: bool = True
    num_practice_trials: int = 0
    isi_jitter_ms: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> TrialSequenceConfig:
        names = {f.name for f in fields(cls)}
        known, _ = _pop_known(d, names)
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)


# ---- SessionMetadata -------------------------------------------------------


@dataclass
class SessionMetadata:
    """Top-level record persisted at `<run_dir>/session_metadata.json`.

    Carries everything needed to identify the run (seed, electrode set,
    trial order) so an analyzer can later reproduce or audit the
    sequence without re-running the experiment.
    """
    session_started: str  # ISO-8601
    valid_electrode_indices: list[int]
    num_repetitions: int
    trial_sequence_config: TrialSequenceConfig
    mapping_method: str = "absolute"
    coords_csv: str = ""  # filename of the implant geometry CSV used (e.g. "prueba4utah.csv")
    summary: dict[str, Any] = field(default_factory=dict)
    trial_order: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    extras: dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS = frozenset({
        "session_started", "valid_electrode_indices", "num_repetitions",
        "trial_sequence_config", "mapping_method", "coords_csv", "summary",
        "trial_order", "schema_version",
    })

    @classmethod
    def from_dict(cls, d: dict) -> SessionMetadata:
        d = _upgrade(d, SCHEMA_VERSION)
        known, extras = _pop_known(d, cls._KNOWN_FIELDS)
        tsc_raw = known.get("trial_sequence_config", {}) or {}
        return cls(
            session_started=str(known.get("session_started", "")),
            valid_electrode_indices=list(known.get("valid_electrode_indices", []) or []),
            num_repetitions=int(known.get("num_repetitions", 0)),
            trial_sequence_config=TrialSequenceConfig.from_dict(tsc_raw),
            mapping_method=str(known.get("mapping_method", "absolute")),
            coords_csv=str(known.get("coords_csv", "")),
            summary=dict(known.get("summary", {}) or {}),
            trial_order=list(known.get("trial_order", []) or []),
            schema_version=int(known.get("schema_version", SCHEMA_VERSION)),
            extras=extras,
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "session_started": self.session_started,
            "valid_electrode_indices": list(self.valid_electrode_indices),
            "num_repetitions": int(self.num_repetitions),
            "trial_sequence_config": self.trial_sequence_config.to_dict(),
            "mapping_method": str(self.mapping_method),
            "coords_csv": str(self.coords_csv),
            "summary": dict(self.summary),
            "trial_order": list(self.trial_order),
            "schema_version": int(self.schema_version),
        }
        # Extras after known fields so they don't shadow them on re-load.
        out.update(self.extras)
        return out


# ---- StimulationParameters (nested inside TrialRecord) ---------------------


@dataclass(frozen=True)
class StimulationParameters:
    current_uA: float = 0.0
    pulse_width_us: float = 0.0
    frequency_hz: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> StimulationParameters:
        names = {f.name for f in fields(cls)}
        known, _ = _pop_known(d, names)
        return cls(**{k: float(v) for k, v in known.items()})

    def to_dict(self) -> dict:
        return asdict(self)


# ---- TrialRecord -----------------------------------------------------------


@dataclass
class TrialRecord:
    """Per-trial record. Used both in-flight (the trial loop builds one
    per trial) and on-disk (serialized into per-electrode metadata.json).

    Many fields default to None/empty — a catch or practice trial fills
    fewer than a real trial does. Consumers should check `is_catch` /
    `is_practice` before reading response-specific fields.
    """
    repetition_number: int
    electrode_index: int | None
    trial_idx: int | None = None
    is_catch: bool = False
    is_practice: bool = False
    position: list[float] = field(default_factory=list)
    stimulation_parameters: StimulationParameters = field(default_factory=StimulationParameters)
    start_time: str = ""
    end_time: str = ""
    events: dict[str, Any] = field(default_factory=dict)
    fixation_losses: int = 0
    trial_attempts: int = 1
    gaze_tracking: dict[str, list] = field(default_factory=lambda: {
        "prestim": [], "stim": [], "poststim": [], "drawing": [],
    })
    # Response metadata — present once the response phase completes.
    response_mode: str | None = None
    response_status: str | None = None
    response_xy: list[float] | None = None
    response_file: str | None = None
    response_file_type: str | None = None
    response_attempts: int | None = None
    response_extraction: str | None = None
    raw_file: str | None = None
    # Legacy aliases written by the old `apply_response_metadata`:
    drawing_file: str | None = None
    saccade_samples_file: str | None = None
    schema_version: int = SCHEMA_VERSION
    extras: dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS = frozenset({
        "repetition_number", "electrode_index", "trial_idx",
        "is_catch", "is_practice", "position", "stimulation_parameters",
        "start_time", "end_time", "events", "fixation_losses",
        "trial_attempts", "gaze_tracking",
        "response_mode", "response_status", "response_xy", "response_file",
        "response_file_type", "response_attempts", "response_extraction",
        "raw_file", "drawing_file", "saccade_samples_file",
        "schema_version",
    })

    @classmethod
    def from_dict(cls, d: dict) -> TrialRecord:
        d = _upgrade(d, SCHEMA_VERSION)
        known, extras = _pop_known(d, cls._KNOWN_FIELDS)
        stim_raw = known.get("stimulation_parameters", {}) or {}
        return cls(
            repetition_number=int(known.get("repetition_number", 0)),
            electrode_index=known.get("electrode_index"),
            trial_idx=known.get("trial_idx"),
            is_catch=bool(known.get("is_catch", False)),
            is_practice=bool(known.get("is_practice", False)),
            position=list(known.get("position", []) or []),
            stimulation_parameters=StimulationParameters.from_dict(stim_raw),
            start_time=str(known.get("start_time", "") or ""),
            end_time=str(known.get("end_time", "") or ""),
            events=dict(known.get("events", {}) or {}),
            fixation_losses=int(known.get("fixation_losses", 0)),
            trial_attempts=int(known.get("trial_attempts", 1)),
            gaze_tracking=dict(known.get("gaze_tracking") or {
                "prestim": [], "stim": [], "poststim": [], "drawing": [],
            }),
            response_mode=known.get("response_mode"),
            response_status=known.get("response_status"),
            response_xy=(list(known["response_xy"]) if known.get("response_xy") is not None else None),
            response_file=known.get("response_file"),
            response_file_type=known.get("response_file_type"),
            response_attempts=known.get("response_attempts"),
            response_extraction=known.get("response_extraction"),
            raw_file=known.get("raw_file"),
            drawing_file=known.get("drawing_file"),
            saccade_samples_file=known.get("saccade_samples_file"),
            schema_version=int(known.get("schema_version", SCHEMA_VERSION)),
            extras=extras,
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "repetition_number": int(self.repetition_number),
            "electrode_index": self.electrode_index,
            "trial_idx": self.trial_idx,
            "is_catch": bool(self.is_catch),
            "is_practice": bool(self.is_practice),
            "position": list(self.position),
            "stimulation_parameters": self.stimulation_parameters.to_dict(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "events": dict(self.events),
            "fixation_losses": int(self.fixation_losses),
            "trial_attempts": int(self.trial_attempts),
            "gaze_tracking": dict(self.gaze_tracking),
            "response_mode": self.response_mode,
            "response_status": self.response_status,
            "response_xy": list(self.response_xy) if self.response_xy is not None else None,
            "response_file": self.response_file,
            "response_file_type": self.response_file_type,
            "response_attempts": self.response_attempts,
            "response_extraction": self.response_extraction,
            "raw_file": self.raw_file,
            "drawing_file": self.drawing_file,
            "saccade_samples_file": self.saccade_samples_file,
            "schema_version": int(self.schema_version),
        }
        out.update(self.extras)
        return out


# ---- ElectrodeAnalysisResult -----------------------------------------------


@dataclass
class ElectrodeAnalysisResult:
    """Output of `mapping_analyzer.analyze_electrode_repetitions`.

    Captures only the load-bearing top-level fields — there's enough
    derived structure inside (per_repetition, boxplot_stats, ellipse
    params, reliability, catch stats) that a flat schema would be
    unwieldy. The extras escape hatch carries the rest unchanged.
    """
    electrode_index: int
    num_total_repetitions: int
    num_valid_repetitions: int
    num_invalid_repetitions: int
    centroids: list[list[float]] = field(default_factory=list)
    centroids_deg: list[list[float]] = field(default_factory=list)
    mean_position_deg: dict[str, float] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    extras: dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS = frozenset({
        "electrode_index", "num_total_repetitions",
        "num_valid_repetitions", "num_invalid_repetitions",
        "centroids", "centroids_deg", "mean_position_deg",
        "schema_version",
    })

    @classmethod
    def from_dict(cls, d: dict) -> ElectrodeAnalysisResult:
        d = _upgrade(d, SCHEMA_VERSION)
        known, extras = _pop_known(d, cls._KNOWN_FIELDS)
        return cls(
            electrode_index=int(known.get("electrode_index", -1)),
            num_total_repetitions=int(known.get("num_total_repetitions", 0)),
            num_valid_repetitions=int(known.get("num_valid_repetitions", 0)),
            num_invalid_repetitions=int(known.get("num_invalid_repetitions", 0)),
            centroids=[list(c) for c in (known.get("centroids") or [])],
            centroids_deg=[list(c) for c in (known.get("centroids_deg") or [])],
            mean_position_deg=dict(known.get("mean_position_deg") or {}),
            schema_version=int(known.get("schema_version", SCHEMA_VERSION)),
            extras=extras,
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "electrode_index": int(self.electrode_index),
            "num_total_repetitions": int(self.num_total_repetitions),
            "num_valid_repetitions": int(self.num_valid_repetitions),
            "num_invalid_repetitions": int(self.num_invalid_repetitions),
            "centroids": [list(c) for c in self.centroids],
            "centroids_deg": [list(c) for c in self.centroids_deg],
            "mean_position_deg": dict(self.mean_position_deg),
            "schema_version": int(self.schema_version),
        }
        out.update(self.extras)
        return out


# ---- loader helpers --------------------------------------------------------


def load_session_metadata(path) -> SessionMetadata:
    """Load a `session_metadata.json` file as a `SessionMetadata`. Records
    written by older code (no `schema_version`) are upgraded on read."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return SessionMetadata.from_dict(raw)


def load_electrode_trials(metadata_json_path) -> list[TrialRecord]:
    """Load every trial from a per-electrode `metadata.json` as a list of
    `TrialRecord`s. Skips items that are not valid dicts so corrupted
    legacy files don't crash an analyzer halfway through a sweep."""
    p = Path(metadata_json_path)
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    reps = raw.get("repetitions") or []
    out: list[TrialRecord] = []
    for r in reps:
        if isinstance(r, dict):
            out.append(TrialRecord.from_dict(r))
    return out
