"""Property-based tests for the typed schemas.

Round-trip is the load-bearing invariant: from_dict(to_dict(x)) == x. If
that ever breaks, an older saved session JSON would silently lose
fields when re-loaded under new code.

Run:
    uv run --project percept_mapper python percept_mapper/scripts/schemas_property_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, strategies as st  # noqa: E402

import tempfile  # noqa: E402

from scripts.schemas import (  # noqa: E402
    SCHEMA_VERSION,
    ElectrodeAnalysisResult,
    SessionMetadata,
    StimulationParameters,
    TrialRecord,
    TrialSequenceConfig,
    load_electrode_trials,
    load_session_metadata,
)


# --- strategies --------------------------------------------------------------

safe_float = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
safe_int = st.integers(min_value=-10_000, max_value=10_000)
short_string = st.text(min_size=0, max_size=24, alphabet=st.characters(blacklist_categories=("Cs",)))


tsc_strategy = st.builds(
    TrialSequenceConfig,
    randomize=st.booleans(),
    random_seed=st.integers(min_value=0, max_value=2**31 - 1),
    catch_trial_rate=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    no_immediate_repeat=st.booleans(),
    num_practice_trials=st.integers(min_value=0, max_value=10),
    isi_jitter_ms=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)


stim_strategy = st.builds(
    StimulationParameters,
    current_uA=safe_float,
    pulse_width_us=safe_float,
    frequency_hz=safe_float,
)


session_strategy = st.builds(
    SessionMetadata,
    session_started=short_string,
    valid_electrode_indices=st.lists(safe_int, min_size=0, max_size=8),
    num_repetitions=st.integers(min_value=0, max_value=20),
    trial_sequence_config=tsc_strategy,
    summary=st.dictionaries(short_string, safe_int, max_size=4),
    trial_order=st.lists(
        st.dictionaries(short_string, safe_int, max_size=3), max_size=20
    ),
    schema_version=st.just(SCHEMA_VERSION),
)


trial_strategy = st.builds(
    TrialRecord,
    repetition_number=st.integers(min_value=0, max_value=100),
    electrode_index=st.one_of(st.none(), safe_int),
    trial_idx=st.one_of(st.none(), st.integers(min_value=0, max_value=1000)),
    is_catch=st.booleans(),
    is_practice=st.booleans(),
    position=st.lists(safe_float, min_size=2, max_size=2),
    stimulation_parameters=stim_strategy,
    start_time=short_string,
    end_time=short_string,
    events=st.dictionaries(short_string, short_string, max_size=4),
    fixation_losses=st.integers(min_value=0, max_value=50),
    trial_attempts=st.integers(min_value=1, max_value=10),
    response_mode=st.one_of(st.none(), st.sampled_from(["drawing", "saccade"])),
    response_status=st.one_of(st.none(), short_string),
    response_xy=st.one_of(st.none(), st.lists(safe_float, min_size=2, max_size=2)),
    response_file=st.one_of(st.none(), short_string),
    response_file_type=st.one_of(st.none(), short_string),
    response_attempts=st.one_of(st.none(), st.integers(min_value=1, max_value=10)),
    response_extraction=st.one_of(st.none(), short_string),
    schema_version=st.just(SCHEMA_VERSION),
)


electrode_result_strategy = st.builds(
    ElectrodeAnalysisResult,
    electrode_index=safe_int,
    num_total_repetitions=st.integers(min_value=0, max_value=20),
    num_valid_repetitions=st.integers(min_value=0, max_value=20),
    num_invalid_repetitions=st.integers(min_value=0, max_value=20),
    centroids=st.lists(st.lists(safe_float, min_size=2, max_size=2), max_size=10),
    centroids_deg=st.lists(st.lists(safe_float, min_size=2, max_size=2), max_size=10),
    mean_position_deg=st.dictionaries(short_string, safe_float, max_size=2),
    schema_version=st.just(SCHEMA_VERSION),
)


# --- properties --------------------------------------------------------------


@given(tsc_strategy)
@settings(max_examples=100, deadline=None)
def prop_tsc_round_trip(x):
    assert TrialSequenceConfig.from_dict(x.to_dict()) == x


@given(stim_strategy)
@settings(max_examples=100, deadline=None)
def prop_stim_round_trip(x):
    assert StimulationParameters.from_dict(x.to_dict()) == x


@given(session_strategy)
@settings(max_examples=100, deadline=None)
def prop_session_round_trip(x):
    assert SessionMetadata.from_dict(x.to_dict()) == x


@given(trial_strategy)
@settings(max_examples=100, deadline=None)
def prop_trial_round_trip(x):
    assert TrialRecord.from_dict(x.to_dict()) == x


@given(electrode_result_strategy)
@settings(max_examples=100, deadline=None)
def prop_electrode_result_round_trip(x):
    assert ElectrodeAnalysisResult.from_dict(x.to_dict()) == x


@given(session_strategy)
@settings(max_examples=50, deadline=None)
def prop_session_json_round_trip(x):
    """Stronger version: go through json.dumps/loads, not just dict."""
    blob = json.dumps(x.to_dict())
    restored = SessionMetadata.from_dict(json.loads(blob))
    assert restored == x


@given(trial_strategy)
@settings(max_examples=50, deadline=None)
def prop_trial_extras_pass_through(x):
    """Unknown fields must survive a round trip in `extras`."""
    raw = x.to_dict()
    raw["__future_field__"] = {"foo": 7}
    restored = TrialRecord.from_dict(raw)
    assert restored.extras.get("__future_field__") == {"foo": 7}
    assert restored.to_dict()["__future_field__"] == {"foo": 7}


def prop_legacy_session_loads_with_schema_version_1():
    """An old session JSON without schema_version should still load and
    get stamped as v1 on the way in."""
    legacy = {
        "session_started": "2025-01-01T00:00:00",
        "valid_electrode_indices": [10, 20],
        "num_repetitions": 5,
        "trial_sequence_config": {
            "randomize": True, "random_seed": 42, "catch_trial_rate": 0.0,
            "no_immediate_repeat": True, "num_practice_trials": 0,
            "isi_jitter_ms": 0.0,
        },
        "summary": {},
        "trial_order": [],
        # schema_version intentionally missing
    }
    s = SessionMetadata.from_dict(legacy)
    assert s.schema_version == 1
    assert s.valid_electrode_indices == [10, 20]


def prop_legacy_trial_loads_with_schema_version_1():
    legacy = {
        "repetition_number": 3,
        "electrode_index": 80,
        "trial_idx": 17,
        "is_catch": False,
        "is_practice": False,
        "position": [960.0, 540.0],
        "stimulation_parameters": {
            "current_uA": 90.0, "pulse_width_us": 200.0, "frequency_hz": 50.0,
        },
        "drawing_file": "repetition_003.png",
    }
    t = TrialRecord.from_dict(legacy)
    assert t.schema_version == 1
    assert t.drawing_file == "repetition_003.png"
    assert t.stimulation_parameters.current_uA == 90.0


def prop_load_session_metadata_round_trips_through_disk():
    """Write a SessionMetadata to JSON on disk, load it back via the
    helper, confirm equality."""
    s = SessionMetadata(
        session_started="2026-01-01T00:00:00",
        valid_electrode_indices=[10, 20, 30],
        num_repetitions=5,
        trial_sequence_config=TrialSequenceConfig(
            randomize=True, random_seed=42, catch_trial_rate=0.15,
            no_immediate_repeat=True, num_practice_trials=2, isi_jitter_ms=200.0,
        ),
        summary={"n": 17, "real": 15, "catch": 2},
        trial_order=[{"trial_idx": 0, "electrode_index": 10, "rep_num": 1, "is_catch": False, "is_practice": False}],
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "session_metadata.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(s.to_dict(), f)
        loaded = load_session_metadata(path)
    assert loaded == s


def prop_load_electrode_trials_filters_garbage():
    """A per-electrode metadata.json with one valid + one corrupted rep
    should yield exactly one TrialRecord — the corrupted entry is skipped,
    not crashed on."""
    payload = {
        "experiment_name": "test",
        "repetitions": [
            {  # valid
                "repetition_number": 1, "electrode_index": 5, "trial_idx": 0,
                "is_catch": False, "is_practice": False,
                "position": [100.0, 200.0],
                "stimulation_parameters": {"current_uA": 30.0, "pulse_width_us": 200.0, "frequency_hz": 50.0},
            },
            "not a dict",  # corrupted
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metadata.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        trials = load_electrode_trials(path)
    assert len(trials) == 1
    assert trials[0].electrode_index == 5
    assert trials[0].schema_version == SCHEMA_VERSION


def main():
    print("[schemas_property_test] running properties...")
    prop_tsc_round_trip()
    print("  ✓ TrialSequenceConfig round-trip")
    prop_stim_round_trip()
    print("  ✓ StimulationParameters round-trip")
    prop_session_round_trip()
    print("  ✓ SessionMetadata round-trip")
    prop_trial_round_trip()
    print("  ✓ TrialRecord round-trip")
    prop_electrode_result_round_trip()
    print("  ✓ ElectrodeAnalysisResult round-trip")
    prop_session_json_round_trip()
    print("  ✓ SessionMetadata json.dumps/loads round-trip")
    prop_trial_extras_pass_through()
    print("  ✓ TrialRecord extras pass-through")
    prop_legacy_session_loads_with_schema_version_1()
    print("  ✓ legacy SessionMetadata (no schema_version) loads as v1")
    prop_legacy_trial_loads_with_schema_version_1()
    print("  ✓ legacy TrialRecord (no schema_version) loads as v1")
    prop_load_session_metadata_round_trips_through_disk()
    print("  ✓ load_session_metadata round-trips through disk")
    prop_load_electrode_trials_filters_garbage()
    print("  ✓ load_electrode_trials filters non-dict entries")
    print("All schemas property tests passed.")


if __name__ == "__main__":
    main()
