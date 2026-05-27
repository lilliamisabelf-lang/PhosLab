"""Property-based tests for scripts.protocol.

Two contracts being pinned:
1. Round-trip: from_dict(to_dict(x)) == x for every record type.
2. Validator: catches the specific structural mistakes it claims to.

Run:
    uv run --project percept_mapper python percept_mapper/scripts/protocol_property_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, strategies as st  # noqa: E402

from scripts.protocol import (  # noqa: E402
    PROTOCOL_SCHEMA_VERSION,
    GATE_TYPES,
    ON_FIXATION_LOST,
    PhaseSpec,
    ProtocolSpec,
    ProtocolValidationError,
    SCREEN_TYPES,
    TrialSequenceSpec,
    assert_valid,
    load_protocol,
    validate_protocol,
)


# --- strategies -------------------------------------------------------------

short_name = st.text(min_size=1, max_size=16, alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"))
safe_float = st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False)

phase_strategy = st.builds(
    PhaseSpec,
    name=short_name,
    screen=st.sampled_from(sorted(SCREEN_TYPES)),
    gate=st.sampled_from(sorted(GATE_TYPES)),
    value=safe_float,
    on_fixation_lost=st.sampled_from(sorted(ON_FIXATION_LOST)),
)

trial_seq_strategy = st.builds(
    TrialSequenceSpec,
    randomize=st.booleans(),
    random_seed=st.one_of(st.none(), st.integers(min_value=0, max_value=2**31 - 1)),
    catch_trial_rate=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    no_immediate_repeat=st.booleans(),
    num_practice_trials=st.integers(min_value=0, max_value=10),
    isi_jitter_ms=safe_float,
)


# --- round-trip -------------------------------------------------------------


@given(phase_strategy)
@settings(max_examples=100, deadline=None)
def prop_phase_round_trip(p):
    assert PhaseSpec.from_dict(p.to_dict()) == p


@given(trial_seq_strategy)
@settings(max_examples=100, deadline=None)
def prop_trial_seq_round_trip(s):
    assert TrialSequenceSpec.from_dict(s.to_dict()) == s


@given(
    short_name,
    short_name,
    st.lists(phase_strategy, min_size=0, max_size=6),
    trial_seq_strategy,
    st.sampled_from(["saccade", "drawing"]),
)
@settings(max_examples=50, deadline=None)
def prop_protocol_round_trip(name, version, phases, seq, response_mode):
    proto = ProtocolSpec(
        name=name,
        version=version,
        phases=phases,
        trial_sequence=seq,
        response_mode=response_mode,
    )
    restored = ProtocolSpec.from_dict(proto.to_dict())
    assert restored == proto


@given(
    short_name, short_name, st.lists(phase_strategy, min_size=1, max_size=4),
    trial_seq_strategy, st.sampled_from(["saccade", "drawing"]),
)
@settings(max_examples=30, deadline=None)
def prop_protocol_extras_pass_through(name, version, phases, seq, response_mode):
    proto = ProtocolSpec(
        name=name, version=version, phases=phases,
        trial_sequence=seq, response_mode=response_mode,
    )
    raw = proto.to_dict()
    raw["__future_field__"] = {"foo": [1, 2]}
    restored = ProtocolSpec.from_dict(raw)
    assert restored.extras.get("__future_field__") == {"foo": [1, 2]}


# --- validator catches what it claims to -----------------------------------


def _good_protocol() -> ProtocolSpec:
    return ProtocolSpec(
        name="ok",
        version="1.0",
        phases=[
            PhaseSpec(name="p1", screen="anchor", gate="continuous_fixation_ms", value=200),
            PhaseSpec(name="p2", screen="saccade", gate="response_finished", value=1500),
        ],
        trial_sequence=TrialSequenceSpec(),
        response_mode="saccade",
    )


def test_validator_accepts_good_protocol():
    print("\n=== test: validator accepts a well-formed protocol ===")
    assert validate_protocol(_good_protocol()) == []
    # Also: the on-disk default protocol must validate.
    default = load_protocol(Path(__file__).resolve().parent.parent / "config" / "protocol.yaml")
    assert validate_protocol(default) == [], f"default protocol invalid: {validate_protocol(default)}"
    print("  ✓ default config/protocol.yaml is valid")


def test_validator_catches_empty_phases():
    print("\n=== test: validator rejects empty phases ===")
    p = _good_protocol()
    p.phases.clear()
    issues = validate_protocol(p)
    assert any("phases is empty" in i for i in issues), issues
    print(f"  ✓ caught: {[i for i in issues if 'phases' in i][0]}")


def test_validator_catches_unknown_screen():
    print("\n=== test: validator rejects unknown screen type ===")
    p = _good_protocol()
    p.phases[0] = PhaseSpec(name="p1", screen="bogus", gate="time_ms", value=100)
    issues = validate_protocol(p)
    assert any("screen=" in i for i in issues), issues
    print(f"  ✓ caught: {[i for i in issues if 'screen=' in i][0]}")


def test_validator_catches_unknown_gate():
    print("\n=== test: validator rejects unknown gate ===")
    p = _good_protocol()
    p.phases[0] = PhaseSpec(name="p1", screen="anchor", gate="bogus_gate", value=100)
    issues = validate_protocol(p)
    assert any("gate=" in i for i in issues), issues
    print(f"  ✓ caught: {[i for i in issues if 'gate=' in i][0]}")


def test_validator_catches_duplicate_phase_name():
    print("\n=== test: validator rejects duplicate phase names ===")
    p = _good_protocol()
    p.phases.append(PhaseSpec(name=p.phases[0].name, screen="anchor", gate="time_ms", value=10))
    issues = validate_protocol(p)
    assert any("duplicate phase name" in i for i in issues), issues
    print(f"  ✓ caught: {[i for i in issues if 'duplicate' in i][0]}")


def test_validator_catches_no_response_phase():
    print("\n=== test: validator rejects protocol with no saccade/drawing phase ===")
    p = ProtocolSpec(
        name="bad",
        version="1.0",
        phases=[
            PhaseSpec(name="only", screen="anchor", gate="time_ms", value=100),
        ],
        trial_sequence=TrialSequenceSpec(),
        response_mode="saccade",
    )
    issues = validate_protocol(p)
    assert any("no saccade or drawing phase" in i for i in issues), issues
    print(f"  ✓ caught: {[i for i in issues if 'no saccade' in i][0]}")


def test_validator_catches_invalid_response_mode():
    print("\n=== test: validator rejects bad response_mode ===")
    p = _good_protocol()
    p_dict = p.to_dict()
    p_dict["response_mode"] = "thoughts"
    p_bad = ProtocolSpec.from_dict(p_dict)
    issues = validate_protocol(p_bad)
    assert any("response_mode=" in i for i in issues), issues
    print(f"  ✓ caught: {[i for i in issues if 'response_mode=' in i][0]}")


def test_assert_valid_raises_with_issue_list():
    print("\n=== test: assert_valid raises ProtocolValidationError with all issues ===")
    p = ProtocolSpec(
        name="",
        version="1.0",
        phases=[],
        trial_sequence=TrialSequenceSpec(),
        response_mode="saccade",
    )
    try:
        assert_valid(p)
        raise AssertionError("expected ProtocolValidationError")
    except ProtocolValidationError as e:
        # Must report ALL issues, not just the first.
        assert len(e.issues) >= 2, f"expected multiple issues, got {e.issues}"
        print(f"  ✓ raised with {len(e.issues)} issues")


def main():
    print("[protocol_property_test] running...")
    prop_phase_round_trip()
    prop_trial_seq_round_trip()
    prop_protocol_round_trip()
    prop_protocol_extras_pass_through()
    print("  ✓ all round-trip properties pass")
    test_validator_accepts_good_protocol()
    test_validator_catches_empty_phases()
    test_validator_catches_unknown_screen()
    test_validator_catches_unknown_gate()
    test_validator_catches_duplicate_phase_name()
    test_validator_catches_no_response_phase()
    test_validator_catches_invalid_response_mode()
    test_assert_valid_raises_with_issue_list()
    print("\nAll protocol property tests passed.")


if __name__ == "__main__":
    main()
