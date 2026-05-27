"""Tests for the skeletal phase runner.

Pin the runner's contract against mock handlers so the production
wiring in main.py (Layer 3c, future session) can adopt this without
guessing at semantics.

Run:
    uv run --project percept_mapper python percept_mapper/scripts/phase_runner_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.protocol import PhaseSpec, ProtocolSpec, TrialSequenceSpec  # noqa: E402
from scripts.phase_runner import (  # noqa: E402
    HandlerRegistry,
    PhaseStatus,
    TrialStatus,
    UnknownHandlerError,
    run_protocol,
    run_trial,
)


def _simple_protocol(phases: list[PhaseSpec]) -> ProtocolSpec:
    return ProtocolSpec(
        name="test",
        version="1.0",
        phases=phases,
        trial_sequence=TrialSequenceSpec(),
        response_mode="saccade",
    )


def _always(status: PhaseStatus):
    """Build a handler that returns a fixed status, ignoring inputs."""
    def _h(_phase, _ctx):
        return status
    return _h


def _sequenced(*statuses):
    """Build a handler that returns the given statuses in order on
    successive calls. After the list is exhausted, returns OK."""
    state = {"i": 0}
    seq = list(statuses)

    def _h(_phase, _ctx):
        if state["i"] < len(seq):
            s = seq[state["i"]]
            state["i"] += 1
            return s
        return PhaseStatus.OK

    return _h


def test_all_phases_ok_completes():
    print("\n=== test: every handler returns OK → COMPLETED ===")
    proto = _simple_protocol([
        PhaseSpec(name="p1", screen="anchor", gate="time_ms", value=10),
        PhaseSpec(name="p2", screen="anchor", gate="time_ms", value=10),
    ])
    reg = HandlerRegistry()
    reg.register("anchor", "time_ms", _always(PhaseStatus.OK))
    result = run_trial(proto, reg)
    assert result.status is TrialStatus.COMPLETED
    assert [s for _, s in result.phases_run] == [PhaseStatus.OK, PhaseStatus.OK]
    assert result.retry_count == 0
    print(f"  ✓ {result.phases_run}")


def test_cancellation_stops_immediately():
    print("\n=== test: CANCELLED halts the trial at the current phase ===")
    proto = _simple_protocol([
        PhaseSpec(name="p1", screen="anchor", gate="time_ms", value=10),
        PhaseSpec(name="p2", screen="anchor", gate="time_ms", value=10),
    ])
    reg = HandlerRegistry()
    reg.register("anchor", "time_ms", _sequenced(PhaseStatus.CANCELLED))
    result = run_trial(proto, reg)
    assert result.status is TrialStatus.CANCELLED
    assert result.aborted_at_phase == "p1"
    # The second phase should NOT have run.
    assert len(result.phases_run) == 1
    print(f"  ✓ stopped at {result.aborted_at_phase!r}")


def test_fixation_lost_retry_phase_loops_and_eventually_succeeds():
    print("\n=== test: retry_phase policy retries until the handler returns OK ===")
    proto = _simple_protocol([
        PhaseSpec(
            name="prestim", screen="anchor", gate="continuous_fixation_ms",
            value=200, on_fixation_lost="retry_phase",
        ),
    ])
    reg = HandlerRegistry()
    # Handler: lose fixation 3 times, then succeed.
    reg.register(
        "anchor", "continuous_fixation_ms",
        _sequenced(
            PhaseStatus.FIXATION_LOST,
            PhaseStatus.FIXATION_LOST,
            PhaseStatus.FIXATION_LOST,
            PhaseStatus.OK,
        ),
    )
    result = run_trial(proto, reg)
    assert result.status is TrialStatus.COMPLETED
    assert result.retry_count == 3
    print(f"  ✓ completed after {result.retry_count} retries")


def test_fixation_lost_abort_trial_stops_immediately():
    print("\n=== test: abort_trial policy aborts on first loss ===")
    proto = _simple_protocol([
        PhaseSpec(
            name="stim", screen="stimulation", gate="time_or_lost_fixation_ms",
            value=400, on_fixation_lost="abort_trial",
        ),
        PhaseSpec(name="never_runs", screen="anchor", gate="time_ms", value=10),
    ])
    reg = HandlerRegistry()
    reg.register("stimulation", "time_or_lost_fixation_ms", _always(PhaseStatus.FIXATION_LOST))
    reg.register("anchor", "time_ms", _always(PhaseStatus.OK))
    result = run_trial(proto, reg)
    assert result.status is TrialStatus.ABORTED
    assert result.aborted_at_phase == "stim"
    assert len(result.phases_run) == 1
    print(f"  ✓ aborted at {result.aborted_at_phase!r}, never_runs was skipped")


def test_fixation_lost_continue_advances_anyway():
    print("\n=== test: continue policy advances despite fixation loss ===")
    proto = _simple_protocol([
        PhaseSpec(
            name="response", screen="saccade", gate="response_finished",
            value=1500, on_fixation_lost="continue",
        ),
        PhaseSpec(name="after", screen="anchor", gate="time_ms", value=10),
    ])
    reg = HandlerRegistry()
    reg.register("saccade", "response_finished", _always(PhaseStatus.FIXATION_LOST))
    reg.register("anchor", "time_ms", _always(PhaseStatus.OK))
    result = run_trial(proto, reg)
    assert result.status is TrialStatus.COMPLETED
    assert len(result.phases_run) == 2
    print("  ✓ both phases ran despite fixation_lost on the first")


def test_max_retries_aborts():
    print("\n=== test: retry_phase respects max_retries_per_phase ===")
    proto = _simple_protocol([
        PhaseSpec(
            name="prestim", screen="anchor", gate="continuous_fixation_ms",
            value=200, on_fixation_lost="retry_phase",
        ),
    ])
    reg = HandlerRegistry()
    # Handler that NEVER recovers.
    reg.register("anchor", "continuous_fixation_ms", _always(PhaseStatus.FIXATION_LOST))
    result = run_trial(proto, reg, max_retries_per_phase=3)
    assert result.status is TrialStatus.ABORTED
    assert result.retry_count == 4  # 3 retries + 1 final attempt that pushed it over
    print(f"  ✓ aborted after retry_count={result.retry_count} (cap=3+1)")


def test_unknown_handler_raises():
    print("\n=== test: unregistered (screen, gate) raises UnknownHandlerError ===")
    proto = _simple_protocol([
        PhaseSpec(name="p1", screen="mysteries", gate="time_ms", value=10),
    ])
    reg = HandlerRegistry()  # empty
    try:
        run_trial(proto, reg)
    except UnknownHandlerError as e:
        print(f"  ✓ {e}")
        return
    raise AssertionError("expected UnknownHandlerError")


def test_run_protocol_iterates_trial_contexts():
    print("\n=== test: run_protocol iterates over trial contexts ===")
    proto = _simple_protocol([
        PhaseSpec(name="p1", screen="anchor", gate="time_ms", value=10),
    ])
    reg = HandlerRegistry()
    seen_ids = []

    def _h(_phase, ctx):
        seen_ids.append(ctx.get("trial_idx"))
        return PhaseStatus.OK

    reg.register("anchor", "time_ms", _h)
    trial_ctxs = [{"trial_idx": i} for i in range(4)]
    results = run_protocol(proto, reg, trial_contexts=trial_ctxs)
    assert len(results) == 4
    assert all(r.status is TrialStatus.COMPLETED for r in results)
    assert seen_ids == [0, 1, 2, 3]
    print("  ✓ handler saw trial_idx in expected order")


def test_handler_receives_phase_and_context():
    """The handler must get the actual PhaseSpec instance, so it can
    read e.g. phase.value to know how long to wait."""
    print("\n=== test: handler gets the right PhaseSpec + context ===")
    proto = _simple_protocol([
        PhaseSpec(name="p1", screen="anchor", gate="time_ms", value=42.0),
    ])
    reg = HandlerRegistry()
    seen = {}

    def _h(phase, ctx):
        seen["name"] = phase.name
        seen["value"] = phase.value
        seen["ctx"] = ctx
        return PhaseStatus.OK

    reg.register("anchor", "time_ms", _h)
    run_trial(proto, reg, context={"hello": "world"})
    assert seen["name"] == "p1"
    assert seen["value"] == 42.0
    assert seen["ctx"]["hello"] == "world"
    print(f"  ✓ handler observed: {seen}")


def main():
    print("[phase_runner_test] running...")
    test_all_phases_ok_completes()
    test_cancellation_stops_immediately()
    test_fixation_lost_retry_phase_loops_and_eventually_succeeds()
    test_fixation_lost_abort_trial_stops_immediately()
    test_fixation_lost_continue_advances_anyway()
    test_max_retries_aborts()
    test_unknown_handler_raises()
    test_run_protocol_iterates_trial_contexts()
    test_handler_receives_phase_and_context()
    print("\nAll phase runner tests passed.")


if __name__ == "__main__":
    main()
