"""Drift-detector test: protocol.yaml vs params.yaml.

`config/protocol.yaml` is the declarative description of the experiment.
`config/params.yaml` is the live runtime config. They overlap on timing,
trial sequence, and response mode — and they will silently drift apart
unless something compares them.

This test loads both and asserts the overlapping fields agree. When it
fails, the failure message tells the author *which* field drifted and
*which file* to update.

Run:
    uv run --project percept_mapper python percept_mapper/scripts/protocol_drift_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from scripts.protocol import load_protocol  # noqa: E402


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_PARAMS_PATH = _CONFIG_DIR / "params.yaml"
_PROTOCOL_PATH = _CONFIG_DIR / "protocol.yaml"


def _load_params() -> dict:
    with open(_PARAMS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_response_mode_matches():
    print("\n=== test: response_mode agrees between protocol.yaml and params.yaml ===")
    params = _load_params()
    proto = load_protocol(_PROTOCOL_PATH)
    params_mode = (params.get("response_mode") or "drawing").lower()
    proto_mode = proto.response_mode.lower()
    assert params_mode == proto_mode, (
        f"response_mode drift: params.yaml={params_mode!r}, "
        f"protocol.yaml={proto_mode!r}"
    )
    print(f"  ✓ both files agree on response_mode={proto_mode!r}")


def test_phase_timings_match_params_timing():
    print("\n=== test: phase 'value' fields match params.timing.* ===")
    params = _load_params()
    proto = load_protocol(_PROTOCOL_PATH)

    timing = params["timing"]
    # Map phase name → expected timing key in params.yaml. response phase
    # has no equivalent in params.timing (its value is a soft saccade
    # capture timeout that lives in params.saccade.capture_duration_ms).
    expected = {
        "prestim": timing["prestimulation"],
        "stim": timing["stimulation"],
        "poststim": timing["poststimulation"],
    }

    by_name = {p.name: p for p in proto.phases}
    for name, want in expected.items():
        assert name in by_name, f"protocol.yaml missing required phase {name!r}"
        got = by_name[name].value
        assert int(got) == int(want), (
            f"phase {name!r} drift: protocol.yaml value={got}, "
            f"params.yaml timing.{name}~={want}"
        )
        print(f"  ✓ phase {name!r}: {int(got)} ms (both files agree)")

    # Response phase value should equal saccade.capture_duration_ms when
    # response_mode is saccade. Different field name in params.yaml.
    if proto.response_mode == "saccade" and "response" in by_name:
        saccade = params.get("saccade") or {}
        expected_response = saccade.get("capture_duration_ms")
        if expected_response is not None:
            got = by_name["response"].value
            assert int(got) == int(expected_response), (
                f"response phase value={got} ≠ "
                f"params.yaml saccade.capture_duration_ms={expected_response}"
            )
            print(f"  ✓ phase 'response': {int(got)} ms (matches saccade.capture_duration_ms)")


def test_trial_sequence_block_matches():
    print("\n=== test: trial_sequence agrees with params.phosphene_mapping ===")
    params = _load_params()
    proto = load_protocol(_PROTOCOL_PATH)

    mapping = params.get("phosphene_mapping") or {}
    p_seq = proto.trial_sequence

    pairs = [
        ("randomize", p_seq.randomize, mapping.get("randomize", True)),
        ("catch_trial_rate", p_seq.catch_trial_rate, mapping.get("catch_trial_rate", 0.0)),
        ("no_immediate_repeat", p_seq.no_immediate_repeat, mapping.get("no_immediate_repeat", True)),
        ("num_practice_trials", p_seq.num_practice_trials, mapping.get("num_practice_trials", 0)),
        ("isi_jitter_ms", p_seq.isi_jitter_ms, mapping.get("isi_jitter_ms", 0.0)),
    ]
    for name, proto_val, params_val in pairs:
        # Allow type-coerced equality (float vs int, bool vs bool).
        if isinstance(proto_val, float) or isinstance(params_val, float):
            assert float(proto_val) == float(params_val), (
                f"trial_sequence.{name}: protocol={proto_val} vs params={params_val}"
            )
        else:
            assert proto_val == params_val, (
                f"trial_sequence.{name}: protocol={proto_val} vs params={params_val}"
            )
        print(f"  ✓ {name}: {proto_val} (both agree)")


def test_default_protocol_validates():
    """Belt + suspenders: this assertion is in protocol_property_test too,
    but it's load-bearing enough that the drift suite should fail loud
    if the on-disk protocol stops being internally consistent."""
    print("\n=== test: default protocol.yaml is internally valid ===")
    from scripts.protocol import validate_protocol
    proto = load_protocol(_PROTOCOL_PATH)
    issues = validate_protocol(proto)
    assert issues == [], f"protocol.yaml has issues: {issues}"
    print("  ✓ default protocol passes validate_protocol()")


def main():
    print("[protocol_drift_test] checking protocol.yaml ↔ params.yaml...")
    test_response_mode_matches()
    test_phase_timings_match_params_timing()
    test_trial_sequence_block_matches()
    test_default_protocol_validates()
    print("\nAll protocol drift tests passed — protocol.yaml is in sync with params.yaml.")


if __name__ == "__main__":
    main()
