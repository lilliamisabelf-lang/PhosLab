"""Smoke tests for trial sequence construction.

Run:
    uv run --project percept_mapper python percept_mapper/scripts/trial_sequence_smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.trial_sequence import build_trial_list, summary  # noqa: E402


def test_count_real_trials():
    print("\n=== test: count of real trials ===")
    trials = build_trial_list([10, 20, 30], num_repetitions=5, seed=42)
    real = [t for t in trials if not t.is_catch and not t.is_practice]
    assert len(real) == 15, f"expected 15 real trials, got {len(real)}"
    print(f"  3 electrodes * 5 reps -> {len(real)} real trials  ✓ PASS")


def test_reproducibility_same_seed():
    print("\n=== test: same seed -> same order ===")
    a = build_trial_list([10, 20, 30, 40], num_repetitions=4, seed=123)
    b = build_trial_list([10, 20, 30, 40], num_repetitions=4, seed=123)
    a_ids = [(t.electrode_index, t.rep_num) for t in a]
    b_ids = [(t.electrode_index, t.rep_num) for t in b]
    assert a_ids == b_ids, "same seed must produce identical sequence"
    print("  ✓ PASS")


def test_randomization_changes_order():
    print("\n=== test: random order != deterministic order ===")
    deterministic = build_trial_list([10, 20, 30], num_repetitions=5, randomize=False)
    randomized = build_trial_list([10, 20, 30], num_repetitions=5, seed=7)
    d_ids = [(t.electrode_index, t.rep_num) for t in deterministic]
    r_ids = [(t.electrode_index, t.rep_num) for t in randomized]
    assert d_ids != r_ids, "shuffle should change the order vs deterministic"
    assert set(d_ids) == set(r_ids), "shuffle should preserve trial set"
    print(f"  deterministic[:5]={d_ids[:5]}  randomized[:5]={r_ids[:5]}  ✓ PASS")


def test_no_immediate_repeat_best_effort():
    print("\n=== test: no_immediate_repeat reduces adjacent repeats ===")
    with_constraint = build_trial_list(
        [10, 20, 30, 40, 50], num_repetitions=4, seed=42, no_immediate_repeat=True
    )
    without_constraint = build_trial_list(
        [10, 20, 30, 40, 50], num_repetitions=4, seed=42, no_immediate_repeat=False
    )
    runs_with = summary(with_constraint)["adjacent_repeats"]
    runs_without = summary(without_constraint)["adjacent_repeats"]
    print(f"  adjacent repeats: with constraint={runs_with}  without={runs_without}")
    assert runs_with <= runs_without, "constraint must not make it worse"
    print("  ✓ PASS")


def test_catch_trials_inserted_at_requested_rate():
    print("\n=== test: catch trials at ~15% ===")
    trials = build_trial_list(
        [10, 20, 30, 40], num_repetitions=10, seed=99, catch_trial_rate=0.15
    )
    s = summary(trials)
    expected_catch = round(0.15 * 40)
    assert s["catch"] == expected_catch, (
        f"expected {expected_catch} catch trials, got {s['catch']}"
    )
    assert s["real"] == 40
    assert s["n"] == 40 + expected_catch
    print(f"  {s}  ✓ PASS")


def test_catch_trials_marked_correctly():
    print("\n=== test: catch trials have electrode_index=None and is_catch=True ===")
    trials = build_trial_list(
        [10, 20], num_repetitions=10, seed=5, catch_trial_rate=0.2
    )
    for t in trials:
        if t.is_catch:
            assert t.electrode_index is None
            assert t.rep_num == 0
        else:
            assert isinstance(t.electrode_index, int)
            assert t.rep_num >= 1
    print("  ✓ PASS")


def test_practice_trials_prepended():
    print("\n=== test: practice trials come first ===")
    trials = build_trial_list(
        [10, 20, 30], num_repetitions=3, seed=1, num_practice_trials=4
    )
    assert all(t.is_practice for t in trials[:4])
    assert not any(t.is_practice for t in trials[4:])
    practice_electrodes = [t.electrode_index for t in trials[:4]]
    assert practice_electrodes == [10, 20, 30, 10], (
        f"practice should round-robin electrodes, got {practice_electrodes}"
    )
    print(f"  first 4 practice electrodes = {practice_electrodes}  ✓ PASS")


def test_trial_idx_dense_and_in_order():
    print("\n=== test: trial_idx is 0..N-1 in order ===")
    trials = build_trial_list(
        [10, 20, 30], num_repetitions=4, seed=1, catch_trial_rate=0.2,
        num_practice_trials=2,
    )
    ids = [t.trial_idx for t in trials]
    assert ids == list(range(len(trials))), f"trial_idx must be dense and sorted: {ids}"
    print(f"  trial_idx[:6]={ids[:6]} ... last={ids[-1]}  ✓ PASS")


def test_single_electrode_falls_through_gracefully():
    print("\n=== test: 1 electrode still works ===")
    trials = build_trial_list([42], num_repetitions=5, seed=1)
    assert all(t.electrode_index == 42 for t in trials)
    assert sorted(t.rep_num for t in trials) == [1, 2, 3, 4, 5], (
        "all 5 reps should be present (shuffle order irrelevant for a single electrode)"
    )
    print("  ✓ PASS")


def main():
    test_count_real_trials()
    test_reproducibility_same_seed()
    test_randomization_changes_order()
    test_no_immediate_repeat_best_effort()
    test_catch_trials_inserted_at_requested_rate()
    test_catch_trials_marked_correctly()
    test_practice_trials_prepended()
    test_trial_idx_dense_and_in_order()
    test_single_electrode_falls_through_gracefully()
    print("\nAll trial-sequence smoke tests passed.")


if __name__ == "__main__":
    main()
