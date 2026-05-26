# Phosphene Mapping — Experimental Rigor Plan

What is left between the current PhosLab pipeline and a phosphene-mapping
evaluation that would survive scientific scrutiny. Grouped by category, with
concrete files to touch and a suggested order at the bottom.

The current pipeline is fine for pilot data with one cooperative participant.
The items below are what would have to change to claim "this measures
phosphene mapping accurately" in a paper or a clinical report.

---

## 1. Experimental rigor — missing entirely

### 1.1 No trial randomization or shuffling
- **Where:** [percept_mapper/scripts/phosphene_mapping.py](percept_mapper/scripts/phosphene_mapping.py),
  [percept_mapper/main.py](percept_mapper/main.py)
- **Symptom:** electrodes fire in the order listed in
  [config/params.yaml](percept_mapper/config/params.yaml)
  (`phosphene_mapping.electrodes_by_implant[].electrode_index`), and
  repetitions run as a contiguous block per electrode. Every session
  reproduces the same order.
- **Impact:** fatigue, learning, anticipation, and adaptation effects are all
  confounded with electrode identity. Any positional bias that depends on
  trial number gets baked into the per-electrode centroid.
- **Minimal fix:** at trial-list build time, build the full `(electrode, rep)`
  sequence, shuffle it once with a seeded RNG (so the run is reproducible
  from metadata), and record the realized order in
  `experiment_metadata["trial_order"]`. Counterbalance interleaving so an
  electrode never fires twice in a row.

### 1.2 No catch / sham trials
- **Where:** trial-list construction in
  [percept_mapper/scripts/phosphene_mapping.py](percept_mapper/scripts/phosphene_mapping.py).
- **Symptom:** every trial actually stimulates. There is no way to separate
  "real percept reported" from "participant always reports something."
- **Impact:** false-positive rate is unmeasurable. Per-electrode centroids
  silently absorb guessing/confabulation.
- **Minimal fix:** add ~10–20% catch trials with zero current (or stimulator
  disabled). Same screen sequence; just no stim. Tag each trial as
  `is_catch: bool` in metadata and report catch-trial response rate per
  participant.

### 1.3 `max_fixation_wait` is a timeout, not a guarantee
- **Where:** [percept_mapper/config/params.yaml](percept_mapper/config/params.yaml#L50)
  (`timing.max_fixation_wait: 100000`) and the prestim phase.
- **Symptom:** the parameter exists but I have not verified it gates
  stimulation on actual fixation rather than just timing out and proceeding.
- **Minimal fix:** read through `run_prestimulation` in
  [percept_mapper/main.py](percept_mapper/main.py) and confirm the exit
  conditions. Either gate stim strictly on fixation acquired or rename the
  parameter and document that it is just a budget.

### 1.4 No practice block, no jittered ISI
- Practice trials let the participant warm up to the task before real data
  starts. Currently the first electrode is real data.
- `timing.interstimulation: 500` is constant, which lets the participant
  anticipate the next stim. A small jitter (e.g. 500–900 ms) breaks the
  cadence.

---

## 2. Statistical evaluation — missing

### 2.1 No cross-validation
- **Where:** [percept_mapper/run_learning.py](percept_mapper/run_learning.py),
  [percept_mapper/scripts/learning/](percept_mapper/scripts/learning/).
- **Symptom:** `learning.neural.train_split: 0.8` is a single holdout. No
  k-fold, no leave-one-electrode-out.
- **Impact:** the "corrected map is better" claim cannot be supported.
- **Minimal fix:** add k-fold CV (k=5 by default) for the bayesian and
  neural models. Report mean ± std test error across folds, not just the
  single-split number.

### 2.2 No significance test on the correction
- After CV, you still need: "is the corrected map's error distribution
  significantly lower than the uncorrected map's, on held-out electrodes?"
- **Minimal fix:** paired bootstrap or Wilcoxon signed-rank on per-trial
  errors, reported in `evaluation_metrics.json`.

### 2.3 No per-electrode uncertainty
- Only centroids are reported. With `num_repetitions: 7` per electrode
  there are enough samples for a 2D confidence ellipse (or just std on x
  and y).
- **Minimal fix:** when the analyzer aggregates a phosphene's reps,
  compute and persist `(x_std, y_std, n_valid_reps)` and
  `(ellipse_major, ellipse_minor, ellipse_angle)`. Plot ellipses on
  `visual_field_comparison.png` instead of bare dots.

### 2.4 No test-retest reliability
- `num_repetitions: 7` already gives within-electrode replicates, but
  nothing computes the within-vs-between-electrode variance ratio.
- **Minimal fix:** in `mapping_analyzer.py`, alongside the centroid,
  emit `within_electrode_std` and (if the protocol runs the same
  electrode set across two sessions) `test_retest_r`. Without two
  sessions, at least report within-electrode variance as the noise
  floor.

---

## 3. Scope honesty

### 3.1 README does not flag simulation
- **Where:** [README.md](README.md)
- **Symptom:** says "Pipeline de Protesis Cortical Visual" without
  flagging that stimulation is `dynaphos` simulation, not real ICMS.
- **Impact:** anyone evaluating the tool assumes hardware-in-the-loop
  until told otherwise. Reviewer trust evaporates the moment they
  discover the gap on their own.
- **Minimal fix:** add a "Scope and limitations" section to the README
  stating exactly what is simulated, what is real (eye tracker, response
  capture, screen presentation), and what would have to change for a
  hardware-in-the-loop run.

---

## 4. Robustness

### 4.1 No resume / checkpoint
- **Where:** the trial loops in
  [percept_mapper/scripts/phosphene_mapping.py](percept_mapper/scripts/phosphene_mapping.py)
  and [percept_mapper/main.py](percept_mapper/main.py).
- **Symptom:** if a 7-rep × N-electrode session crashes at trial 40,
  everything since the last per-electrode JSON write is lost. No way to
  resume.
- **Minimal fix:** write the partial `experiment_metadata` JSON after
  every trial (atomic rename pattern), and add a `--resume <run_dir>`
  flag that loads it, finds the highest completed trial index, and
  picks up the trial list from there.

### 4.2 No experimenter dashboard
- During the session there is no live view of: gaze tracking quality,
  fraction of trials accepted vs failed-saccade, time elapsed, time
  remaining. If gaze quality degrades mid-session, you only notice
  during analysis the next day.
- **Minimal fix:** a sidecar terminal log (or a tiny second pygame
  surface on the experimenter screen) showing a 1 Hz status: current
  trial, last status, running accept rate, last gaze sample age.

---

## 5. Validation of the apparatus itself

### 5.1 No independent eye-tracker accuracy check
- **Symptom:** PhosLab trusts whatever Pupil Capture's own calibration
  produces. There is no separate validation grid (e.g. 9-point) that
  scores tracker accuracy against ground truth before each session.
- **Minimal fix:** a `validate_eye_tracker.py` script that puts 9 dots
  in a 3×3 grid, asks the participant to fixate each for ~1 s, records
  gaze, and reports per-point error in degrees. Refuse to start the
  experiment if mean error > some threshold.

### 5.2 No display/timing calibration
- **Symptom:** stimulus onset latency is whatever `pygame.display.flip`
  + the screen's input lag produce. Currently un-measured.
- **Minimal fix:** if a photodiode is ever available, instrument the
  stim-onset path to flash a black/white square at a known screen
  corner and log the timestamp. Until then, document the assumption.

---

## 6. Participant UI & sound design

The participant currently sees the anchor circle in one color, a static
instruction title, and no audio confirmation that the system has seen their
fixation. Cleaner state communication makes the task easier and reduces
guess-rate.

### 6.1 Anchor traffic-light state machine
- **Where:** [percept_mapper/scripts/anchor_screen.py](percept_mapper/scripts/anchor_screen.py),
  [percept_mapper/scripts/saccade_screen.py](percept_mapper/scripts/saccade_screen.py)
- **Symptom:** the anchor is one color (red outline) regardless of whether
  the participant is fixating, the system is about to stim, or stim is
  active. The participant has no signal that the system has accepted their
  fixation.
- **Minimal fix:** drive four states from the existing
  `is_looking_at_point()` signal:
  - **white outline** — idle, fixation not yet acquired
  - **green outline** — fixation acquired, stim about to fire ("you're ready")
  - **red filled** — stimulation active ("hold still")
  - **dim white outline** — response phase, still a saccade target
- **Risk:** any color change *during* prestim/stim risks being interpreted
  as part of the phosphene. Confine transitions to phase boundaries, or use
  thickness/saturation rather than hue if hue confounds suspected.

### 6.2 Fixation-acquired audio tick
- **Where:** [percept_mapper/scripts/audio_cue.py](percept_mapper/scripts/audio_cue.py)
- **Symptom:** the participant has no audio confirmation that the system
  saw their fixation. The 880 Hz saccade-onset cue exists, but there is no
  earlier "go" signal.
- **Minimal fix:** soft low tick (~200 Hz, 30 ms, 0.3 volume) at the moment
  `is_looking_at_point()` first returns true after a phase start. Mirrors
  the green anchor transition.
- **Risk:** add nothing else. Failure tones would defeat the silent-retry
  policy. A confirmation tone after the response would bias the next trial.

### 6.3 Hide timing UI
- **Where:** [percept_mapper/scripts/phosphene_mapping.py:378-388](percept_mapper/scripts/phosphene_mapping.py#L378-L388)
- **Symptom:** the interstimulation screen renders a live "X.X s" countdown.
  A visible countdown trains participants to time their next saccade to the
  clock instead of the stimulus — the exact confound the audio cue exists to
  avoid.
- **Minimal fix:** drop the numeric countdown. Keep the phase text ("descanso
  antes de la repetición N") but no remaining-time number. Audio cue plays
  when the next trial is ready to start.

### 6.4 Fade the instruction title
- **Where:** [percept_mapper/scripts/saccade_screen.py:218-224](percept_mapper/scripts/saccade_screen.py#L218-L224)
- **Symptom:** the saccade instruction ("Mira el fosfeno y vuelve al centro")
  is rendered for the full 1500 ms capture window. That is reading material
  competing with the percept memory.
- **Minimal fix:** show full-opacity title for first ~200 ms after the cue,
  then linearly fade to alpha 0 by ~500 ms. After the first three trials of
  a session, suppress it entirely (participant has learned the task).

---

## Priority order

If picking the highest-leverage items to add first:

1. **Trial randomization (§1.1)** — single highest-value change. Without
   it, none of the per-electrode numbers can be trusted.
2. **Catch trials (§1.2)** — lets you put a number on guessing /
   confabulation.
3. **CV + per-electrode uncertainty (§2.1, §2.3)** — required to make any
   claim about the learning models being useful.
4. **UI traffic-light + hide timing UI (§6.1, §6.3)** — cheap, immediate
   participant-experience win, and removes a known confound.
5. **README scope honesty (§3)** — five-minute write, protects reviewer
   trust.

Robustness (§4) and apparatus validation (§5) are deliberately last:
they protect the data but do not change the conclusions you can draw
from a clean session.

---

## Out of scope for this plan

- Group-level analysis (multiple participants). Single-subject design is
  fine for now.
- Hardware ICMS integration. That is a separate, larger effort.
- IRB / regulatory / data-management process. Pipeline-side only.
