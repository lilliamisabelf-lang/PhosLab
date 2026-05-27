"""Tests for the headless protocol runner.

Drives the real `config/protocol.yaml` through `phase_runner.run_trial`
with the headless handlers and asserts:

  - a clean trial completes with all phases in order,
  - a forced-loss trial triggers retry until budget exhausted,
  - cancellation halts immediately,
  - a trial-list built by `build_trial_list` can be fed through
    `run_protocol` and every trial completes.

Run:
    uv run --project percept_mapper python percept_mapper/scripts/headless_runner_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.headless_runner import HeadlessContext, build_headless_registry  # noqa: E402
from scripts.phase_runner import PhaseStatus, TrialStatus, run_protocol, run_trial  # noqa: E402
from scripts.protocol import load_protocol  # noqa: E402
from scripts.trial_sequence import build_trial_list  # noqa: E402

_PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "config" / "protocol.yaml"


def test_default_protocol_runs_to_completion():
    print("\n=== test: real default protocol runs end-to-end headless ===")
    proto = load_protocol(_PROTOCOL_PATH)
    reg = build_headless_registry()
    hctx = HeadlessContext()
    result = run_trial(proto, reg, context={"headless": hctx})
    assert result.status is TrialStatus.COMPLETED, (
        f"got {result.status}; phases={result.phases_run}"
    )
    phase_names = [p[0] for p in result.phases_run]
    assert phase_names == ["prestim", "stim", "poststim", "response"], phase_names
    statuses = [p[1] for p in result.phases_run]
    assert all(s is PhaseStatus.OK for s in statuses), statuses
    assert result.retry_count == 0
    print(f"  ✓ phases ran in order: {phase_names}")


def test_forced_fixation_loss_exhausts_retry_budget():
    """`fixation_lost_probability=1.0` makes every fixation-aware phase
    fail. The protocol's first such phase is `prestim` with
    `on_fixation_lost=retry_phase`, so we expect the trial to retry
    until max_retries_per_phase is hit and then ABORT."""
    print("\n=== test: forced fixation loss exhausts retry and aborts ===")
    proto = load_protocol(_PROTOCOL_PATH)
    reg = build_headless_registry()
    hctx = HeadlessContext(fixation_lost_probability=1.0)
    result = run_trial(proto, reg, context={"headless": hctx}, max_retries_per_phase=4)
    assert result.status is TrialStatus.ABORTED, result.status
    assert result.aborted_at_phase == "prestim"
    # Every recorded attempt at prestim must have failed.
    prestim_attempts = [h for h in hctx.history if h["phase"] == "prestim"]
    assert len(prestim_attempts) >= 5, f"expected ≥5 prestim attempts, got {len(prestim_attempts)}"
    assert all(h["status"] == "fixation_lost" for h in prestim_attempts)
    print(f"  ✓ aborted after {result.retry_count} retries at {result.aborted_at_phase!r}")


def test_partial_loss_eventually_completes():
    """A 30% loss rate should usually let the trial complete within a
    handful of retries (still deterministic via the rng seed)."""
    print("\n=== test: 30% loss rate completes with bounded retries ===")
    proto = load_protocol(_PROTOCOL_PATH)
    reg = build_headless_registry()
    hctx = HeadlessContext(fixation_lost_probability=0.3, rng_seed=42)
    result = run_trial(proto, reg, context={"headless": hctx})
    assert result.status is TrialStatus.COMPLETED
    print(
        f"  ✓ completed after retry_count={result.retry_count} "
        f"(prestim attempts={sum(1 for h in hctx.history if h['phase'] == 'prestim')})"
    )


def test_cancellation_halts():
    print("\n=== test: cancellation halts the trial at the current phase ===")
    proto = load_protocol(_PROTOCOL_PATH)
    reg = build_headless_registry()
    hctx = HeadlessContext(cancelled=True)
    result = run_trial(proto, reg, context={"headless": hctx})
    assert result.status is TrialStatus.CANCELLED
    assert result.aborted_at_phase == "prestim"
    assert len(result.phases_run) == 1
    print(f"  ✓ stopped at {result.aborted_at_phase!r}")


def test_trial_list_runs_through_protocol():
    """End-to-end: build a trial list, feed each trial as a context
    into run_protocol, every trial completes cleanly."""
    print("\n=== test: trial_list × run_protocol completes every trial ===")
    proto = load_protocol(_PROTOCOL_PATH)
    reg = build_headless_registry()
    trials = build_trial_list(
        [10, 20, 30], num_repetitions=4, seed=7,
        catch_trial_rate=0.0,  # keep this test about the runner, not catches
    )
    # Each trial gets its own headless context, but the same RNG seed
    # gives reproducible results across runs.
    contexts = [
        {"headless": HeadlessContext(rng_seed=t.trial_idx), "trial": t.to_dict()}
        for t in trials
    ]
    results = run_protocol(proto, reg, trial_contexts=contexts)
    assert len(results) == len(trials), (len(results), len(trials))
    completed = [r for r in results if r.status is TrialStatus.COMPLETED]
    assert len(completed) == len(trials), (
        f"only {len(completed)}/{len(trials)} trials completed; "
        f"first failure status={results[len(completed)].status if len(completed) < len(trials) else None}"
    )
    print(f"  ✓ all {len(trials)} trials completed")


def test_history_records_every_phase_invocation():
    """The `history` list in HeadlessContext should grow by exactly one
    entry per handler call, including retries — useful for forensic
    debugging when a participant complains."""
    print("\n=== test: history records every phase invocation, including retries ===")
    proto = load_protocol(_PROTOCOL_PATH)
    reg = build_headless_registry()
    hctx = HeadlessContext(fixation_lost_probability=0.5, rng_seed=11)
    result = run_trial(proto, reg, context={"headless": hctx})
    expected_count = len(result.phases_run)
    assert len(hctx.history) == expected_count, (len(hctx.history), expected_count)
    # All history entries should have the same shape.
    for h in hctx.history:
        assert set(h.keys()) == {"phase", "screen", "gate", "value", "status", "note"}
    print(f"  ✓ history len={len(hctx.history)} matches phases_run len={expected_count}")


def main():
    print("[headless_runner_test] running...")
    test_default_protocol_runs_to_completion()
    test_forced_fixation_loss_exhausts_retry_budget()
    test_partial_loss_eventually_completes()
    test_cancellation_halts()
    test_trial_list_runs_through_protocol()
    test_history_records_every_phase_invocation()
    print("\nAll headless runner tests passed.")


if __name__ == "__main__":
    main()
