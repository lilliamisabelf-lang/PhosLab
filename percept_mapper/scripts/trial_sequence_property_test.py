"""Property-based tests for trial_sequence.build_trial_list.

The smoke tests pin concrete examples. These pin *invariants* over a wide
parameter space — Hypothesis searches for inputs that break them.

Run:
    uv run --project percept_mapper python percept_mapper/scripts/trial_sequence_property_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, strategies as st  # noqa: E402

from scripts.trial_sequence import build_trial_list, summary  # noqa: E402


# Constrain the inputs to a realistic-but-fast envelope. Wider envelopes
# pass but slow down the suite; we want this to run in <2 s on CI.
electrode_lists = st.lists(
    st.integers(min_value=0, max_value=2047),
    min_size=1,
    max_size=8,
    unique=True,
)
reps = st.integers(min_value=1, max_value=6)
seeds = st.integers(min_value=0, max_value=2**31 - 1)
catch_rates = st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)
practice_counts = st.integers(min_value=0, max_value=4)


@given(electrode_lists, reps, seeds, catch_rates, practice_counts)
@settings(max_examples=200, deadline=None)
def prop_total_count_matches_formula(electrodes, n_reps, seed, catch_rate, practice):
    """N = practice + (electrodes × reps) + round(catch_rate × electrodes × reps)."""
    trials = build_trial_list(
        electrodes, n_reps, seed=seed,
        catch_trial_rate=catch_rate,
        num_practice_trials=practice,
    )
    real_count = len(electrodes) * n_reps
    expected_catch = round(catch_rate * real_count)
    assert len(trials) == practice + real_count + expected_catch, (
        f"len={len(trials)} expected={practice + real_count + expected_catch} "
        f"electrodes={electrodes} reps={n_reps} catch_rate={catch_rate} practice={practice}"
    )


@given(electrode_lists, reps, seeds)
@settings(max_examples=200, deadline=None)
def prop_every_real_trial_appears_exactly_once(electrodes, n_reps, seed):
    """Across the shuffled trial list, every (electrode, rep) pair from the
    cartesian product appears exactly once."""
    trials = build_trial_list(electrodes, n_reps, seed=seed, catch_trial_rate=0.0)
    real_pairs = [(t.electrode_index, t.rep_num) for t in trials if not t.is_catch and not t.is_practice]
    expected = sorted((e, r) for e in electrodes for r in range(1, n_reps + 1))
    assert sorted(real_pairs) == expected


@given(electrode_lists, reps, seeds, catch_rates, practice_counts)
@settings(max_examples=200, deadline=None)
def prop_trial_idx_is_dense_0_to_n_minus_1(electrodes, n_reps, seed, catch_rate, practice):
    """trial_idx should be exactly range(0, N) — no gaps, no duplicates."""
    trials = build_trial_list(
        electrodes, n_reps, seed=seed,
        catch_trial_rate=catch_rate,
        num_practice_trials=practice,
    )
    ids = [t.trial_idx for t in trials]
    assert ids == list(range(len(trials)))


@given(electrode_lists, reps, seeds, catch_rates, practice_counts)
@settings(max_examples=100, deadline=None)
def prop_same_seed_same_output(electrodes, n_reps, seed, catch_rate, practice):
    """The shuffle is fully seeded — same seed → identical sequence."""
    a = build_trial_list(
        electrodes, n_reps, seed=seed,
        catch_trial_rate=catch_rate,
        num_practice_trials=practice,
    )
    b = build_trial_list(
        electrodes, n_reps, seed=seed,
        catch_trial_rate=catch_rate,
        num_practice_trials=practice,
    )
    a_ids = [(t.electrode_index, t.rep_num, t.is_catch, t.is_practice) for t in a]
    b_ids = [(t.electrode_index, t.rep_num, t.is_catch, t.is_practice) for t in b]
    assert a_ids == b_ids


@given(electrode_lists, reps, seeds, catch_rates, practice_counts)
@settings(max_examples=100, deadline=None)
def prop_catch_trials_well_formed(electrodes, n_reps, seed, catch_rate, practice):
    """Catch trials must have electrode_index=None, rep_num=0, is_catch=True;
    non-catch trials must NOT have electrode_index=None."""
    trials = build_trial_list(
        electrodes, n_reps, seed=seed,
        catch_trial_rate=catch_rate,
        num_practice_trials=practice,
    )
    for t in trials:
        if t.is_catch:
            assert t.electrode_index is None
            assert t.rep_num == 0
            assert not t.is_practice
        else:
            assert isinstance(t.electrode_index, int)
            assert t.rep_num >= 0


@given(electrode_lists, reps, seeds)
@settings(max_examples=100, deadline=None)
def prop_no_immediate_repeat_constraint_holds_when_feasible(electrodes, n_reps, seed):
    """When the constraint is feasible (no electrode dominates more than
    half the list), the realized sequence must have zero adjacent repeats."""
    total = len(electrodes) * n_reps
    # Feasible iff no single electrode contributes more than half of the
    # non-catch portion. Since all electrodes contribute n_reps each, this
    # holds iff len(electrodes) >= 2 OR n_reps == 1.
    if len(electrodes) < 2 and n_reps > 1:
        return  # infeasible — skip
    trials = build_trial_list(
        electrodes, n_reps, seed=seed,
        catch_trial_rate=0.0,
        no_immediate_repeat=True,
    )
    runs = summary(trials)["adjacent_repeats"]
    assert runs == 0, (
        f"adjacent repeats={runs} for electrodes={electrodes} reps={n_reps} "
        f"seed={seed}; first 10 ids="
        f"{[(t.electrode_index, t.rep_num) for t in trials[:10]]}"
    )


def main():
    print("[trial_sequence_property_test] running properties...")
    prop_total_count_matches_formula()
    print("  ✓ total count matches formula")
    prop_every_real_trial_appears_exactly_once()
    print("  ✓ every (electrode, rep) appears exactly once")
    prop_trial_idx_is_dense_0_to_n_minus_1()
    print("  ✓ trial_idx is dense and sorted")
    prop_same_seed_same_output()
    print("  ✓ same seed → same output")
    prop_catch_trials_well_formed()
    print("  ✓ catch trials are well-formed")
    prop_no_immediate_repeat_constraint_holds_when_feasible()
    print("  ✓ no-immediate-repeat holds when feasible")
    print("All trial_sequence property tests passed.")


if __name__ == "__main__":
    main()
