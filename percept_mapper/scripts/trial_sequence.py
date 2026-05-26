"""Trial-sequence construction for phosphene mapping.

Builds the full list of trials for a session — interleaving electrodes,
shuffling with a seeded RNG, inserting catch trials, and (optionally)
guaranteeing that the same electrode does not fire twice in a row.

A `Trial` is the unit of work the experiment loop consumes. For real
trials, `electrode_index` is the electrode being stimulated. For catch
trials, `electrode_index is None` and `is_catch is True` — the experiment
runs the same screen sequence but skips the stimulation draw.

The realized order (including the seed used) is recorded in the experiment
metadata so any session is reproducible from the saved JSON.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class Trial:
    trial_idx: int            # 0-based position in the realized sequence
    electrode_index: int | None
    rep_num: int              # 1-based repetition for that electrode (catch: 0)
    is_catch: bool = False
    is_practice: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def build_trial_list(
    electrode_indices: Iterable[int],
    num_repetitions: int,
    *,
    seed: int | None = None,
    catch_trial_rate: float = 0.0,
    no_immediate_repeat: bool = True,
    randomize: bool = True,
    num_practice_trials: int = 0,
) -> list[Trial]:
    """Return the full ordered trial list for one session.

    Args:
        electrode_indices: electrodes to map this session.
        num_repetitions: real repetitions per electrode.
        seed: RNG seed for reproducibility. None -> non-deterministic.
        catch_trial_rate: fraction of total trials that are catch (no stim).
            Catch trials are inserted *after* the real-trial shuffle so they
            scatter uniformly. 0.0 disables catch trials.
        no_immediate_repeat: if True, swap adjacent items so no electrode
            fires twice in a row (best-effort; impossible if one electrode
            dominates the list).
        randomize: if False, return the deterministic [electrode][rep] order
            with no shuffling and no catch trials. Useful for debugging.
        num_practice_trials: K practice trials at the start of the session.
            Practice trials pick electrodes round-robin from the input list,
            are tagged is_practice=True, and do not contribute to the
            num_repetitions budget.

    Returns:
        list of Trial. Length = num_practice_trials + (electrodes * reps) +
        round(catch_trial_rate * electrodes * reps).
    """
    electrode_list = list(electrode_indices)
    if not electrode_list:
        return []
    if num_repetitions < 1:
        raise ValueError(f"num_repetitions must be >= 1, got {num_repetitions}")
    if not (0.0 <= catch_trial_rate < 1.0):
        raise ValueError(f"catch_trial_rate must be in [0, 1), got {catch_trial_rate}")

    rng = random.Random(seed)

    real_trials = [
        (electrode, rep) for electrode in electrode_list for rep in range(1, num_repetitions + 1)
    ]
    if randomize:
        rng.shuffle(real_trials)
        if no_immediate_repeat:
            real_trials = _avoid_immediate_repeats(real_trials, rng)

    catch_count = round(catch_trial_rate * len(real_trials)) if randomize else 0
    output: list[Trial] = []

    if num_practice_trials > 0:
        for i in range(num_practice_trials):
            output.append(
                Trial(
                    trial_idx=i,
                    electrode_index=electrode_list[i % len(electrode_list)],
                    rep_num=0,
                    is_catch=False,
                    is_practice=True,
                )
            )

    for electrode, rep in real_trials:
        output.append(
            Trial(
                trial_idx=len(output),
                electrode_index=electrode,
                rep_num=rep,
                is_catch=False,
            )
        )

    if catch_count > 0:
        catch_positions = rng.sample(
            range(num_practice_trials, len(output) + catch_count), catch_count
        )
        catch_positions.sort()
        for slot in catch_positions:
            output.insert(
                slot,
                Trial(
                    trial_idx=slot,
                    electrode_index=None,
                    rep_num=0,
                    is_catch=True,
                ),
            )
        for i, t in enumerate(output):
            if t.trial_idx != i:
                output[i] = Trial(
                    trial_idx=i,
                    electrode_index=t.electrode_index,
                    rep_num=t.rep_num,
                    is_catch=t.is_catch,
                    is_practice=t.is_practice,
                )

    return output


def _avoid_immediate_repeats(
    trials: list[tuple[int, int]], rng: random.Random, max_passes: int = 8
) -> list[tuple[int, int]]:
    """Swap adjacent (electrode, rep) pairs to break runs of the same
    electrode. Best-effort: if one electrode dominates more than half the
    list, some adjacent repeats are unavoidable and will remain."""
    trials = list(trials)
    n = len(trials)
    for _ in range(max_passes):
        runs = [
            i for i in range(1, n) if trials[i][0] == trials[i - 1][0]
        ]
        if not runs:
            return trials
        for i in runs:
            candidates = [
                j for j in range(n)
                if j not in (i - 1, i, i + 1 if i + 1 < n else -1)
                and trials[j][0] != trials[i][0]
                and (j == 0 or trials[j - 1][0] != trials[i][0])
                and (j == n - 1 or trials[j + 1][0] != trials[i][0])
            ]
            if candidates:
                j = rng.choice(candidates)
                trials[i], trials[j] = trials[j], trials[i]
    return trials


def summary(trials: list[Trial]) -> dict:
    """Tiny human-readable summary of a built trial list."""
    if not trials:
        return {"n": 0}
    real = [t for t in trials if not t.is_catch and not t.is_practice]
    catches = [t for t in trials if t.is_catch]
    practices = [t for t in trials if t.is_practice]
    runs = 0
    for i in range(1, len(trials)):
        a, b = trials[i - 1].electrode_index, trials[i].electrode_index
        if a is not None and a == b:
            runs += 1
    return {
        "n": len(trials),
        "real": len(real),
        "catch": len(catches),
        "practice": len(practices),
        "adjacent_repeats": runs,
    }
