# Saccade Mode — Setup & Usage Guide

Alternative response mode for percept_mapper trials. Instead of drawing the perceived phosphene location, the participant fixates the center anchor, sees the stimulus, then makes an eye saccade toward where they perceived the phosphene and returns to center. The system records the gaze trajectory during a fixed capture window and extracts a single `(x, y)` response point.

## Architecture

```
Anchor fixation ──▶ Stimulation ──▶ Poststim ──▶ SaccadeScreen.update() (capture window) ──▶ Extractor ──▶ response_xy
                                                  - audio "go" cue                            (one of 3 methods)
                                                  - polls eye_tracker.last_smooth_gaze
                                                  - draws live gaze trace
                                                  - silent retry if extraction fails
```

`SaccadeScreen` and `DrawingTablet` share the same interface (`reset` / `update(screen, events) → (finished, payload)` / `close`), so swapping between them in [main.py](main.py) is a single config flag.

## Quick start

In [`config/params.yaml`](config/params.yaml):

```yaml
response_mode: saccade   # drawing | saccade
```

Then run as usual:

```
cd percept_mapper
uv run python main.py
```

The drawing tablet step is replaced by the saccade capture window. Everything else (anchor fixation, stimulation, mapping mode, learning pipeline) stays the same.

## Full configuration

```yaml
response_mode: saccade

saccade:
  capture_duration_ms: 1500          # capture window length after stimulus
  extraction: idt_first_fixation     # idt_first_fixation | peak_distance | velocity_endpoint
  idt:
    dispersion_px: 60                # ≈1° visual angle @ 60 cm on 2560×1440
    min_duration_ms: 100             # minimum fixation duration
    skip_anchor_radius_px: 40        # ignore initial fixation at anchor
  velocity:
    onset_threshold_px_s: 1500       # px/s, saccade onset
    settle_threshold_px_s: 300       # px/s, saccade landing
    smoothing_window: 5              # boxcar velocity smoothing
  min_response_distance_px: 30       # gates trivial "didn't saccade" responses
  on_failure: rerun_max_3            # skip | rerun | rerun_max_<N>
  show_gaze_trace: true              # live trace overlay during capture
  audio_cue:
    enabled: true
    frequency_hz: 880
    duration_ms: 80
    volume: 0.4
```

## Extraction methods

| Method | Description | Best for |
|--------|-------------|----------|
| `idt_first_fixation` (default) | Identification-by-Dispersion-Threshold (Salvucci & Goldberg 2000). Scans for the first contiguous run where x/y dispersion < `dispersion_px` for at least `min_duration_ms` ms, returns its centroid. Skips initial anchor fixation via `skip_anchor_radius_px`. | Most experiments. Robust to overshoot, tracker spikes, and corrective microsaccades. Gold standard in saccadic-localization literature. |
| `peak_distance` | Sample with maximum Euclidean distance from anchor. | Quick baseline. Vulnerable to overshoot and tracker spikes. |
| `velocity_endpoint` | Smooths per-sample velocity, finds first saccade onset (velocity > threshold) then first landing (velocity < threshold). Returns landing sample's `(x, y)`. | Clean physiological model. Needs more tuning; less tolerant of dropouts than IDT. |

### Why IDT beats peak_distance

Human saccades systematically undershoot their target by ~5-10%, then a corrective microsaccade brings gaze the rest of the way. The `peak` sample lands somewhere along the overshoot trajectory. The first *stable fixation* after the saccade lands is closer to the perceived target. IDT also rejects tracker spikes — a single noisy sample can't form a stable fixation cluster.

## Eye tracker compatibility

Works with any `input_mode`:

- **`pupil`** — calibrated gaze from Pupil Capture surface tracker. Best quality.
- **`gaze`** — webcam MediaPipe iris tracking. Lower quality, useful for testing without a headset.
- **`mouse`** — mouse cursor position. **Debug only**; lets you test the trial flow without an eye tracker by "saccading" with the mouse.
- **Wacom tablet** — same as mouse mode (Wacom appears as a pointer device).

The class polls (in order) `eye_tracker.last_smooth_gaze` → `eye_tracker.last_raw_gaze` → `pygame.mouse.get_pos()`. Mouse fallback always works.

## Output

### Trial files

For each phosphene/repetition with `response_mode: saccade`, a JSON sidecar is written next to the metadata:

- Standard mode: `experiment_dir/saccade_samples_<N>.json`
- Mapping mode: `electrode_dir/saccade_samples_<NNN>.json`

```json
{
  "response_xy": [1500.0, 600.0],
  "status": "ok",
  "extraction": "idt_first_fixation",
  "attempts": 1,
  "max_attempts": 3,
  "capture_duration_ms": 1500,
  "anchor_xy": [1280.0, 720.0],
  "samples": [
    {"t": 0.0, "x": 1281.2, "y": 719.8},
    {"t": 0.0167, "x": 1280.5, "y": 720.3},
    ...
  ]
}
```

### Metadata additions

Each `phosphene` (standard) or `repetition` (mapping) entry in `metadata.json` gets:

```json
{
  "response_mode": "saccade",
  "saccade_samples_file": "saccade_samples_1.json",
  "response_xy": [1500.0, 600.0],
  "response_status": "ok",
  "response_extraction": "idt_first_fixation",
  "response_attempts": 1
}
```

## Status codes

| Status | Meaning |
|--------|---------|
| `ok` | Extraction succeeded, `response_xy` is valid. |
| `failed_no_fixation` | IDT found no stable fixation in the capture window. Likely no clean saccade made, or tracker dropouts. |
| `failed_no_endpoint` | Velocity-based extractor found no saccade landing. |
| `failed_no_motion` | No gaze samples captured at all (tracker offline). |
| `failed_too_close_to_anchor` | Response was inside `min_response_distance_px` of the anchor. Treated as no saccade. |
| `aborted_by_user` | ESC was pressed during the capture window. |

## Retry policy

On non-`ok` status, `on_failure` controls behavior:

- `skip` — accept the failed payload and continue. Trial is recorded with `status != "ok"`.
- `rerun` — retry indefinitely until success.
- `rerun_max_<N>` — retry up to N times total (default `rerun_max_3`). After the budget is exhausted, the last failed payload is accepted.

Retries are **silent** — no UI feedback to the participant. The capture window simply restarts. Press `R` during a capture to manually trigger a retry (counts toward the budget).

## Learning pipeline integration

Both [standard_analyzer.py](scripts/standard_analyzer.py) and [mapping_analyzer.py](scripts/mapping_analyzer.py) auto-detect `response_mode: saccade` and use `response_xy` directly as the centroid. The `centroids_deg` output is unchanged, so [`scripts/learning/data_loader.py`](scripts/learning/data_loader.py) needs no modifications. Bayesian and neural models train on saccade data exactly like on drawing data.

## Diagnostic / testing

```
uv run --project percept_mapper python percept_mapper/scripts/saccade_smoke_test.py
```

Runs the 6 unit tests:
- IDT extractor on clean synthetic saccade
- peak_distance on the same
- velocity_endpoint on the same
- no-saccade trial → expect failure
- retry budget exhausts at exactly N
- audio cue builds correctly

Also:
```
uv run --project percept_mapper python percept_mapper/scripts/saccade_extractors.py
```
Runs extractor self-test against a synthetic saccade with overshoot + correction.

## Known limitations / TODO

- **Fixed-duration capture window.** A natural extension is "end on return saccade to anchor" — once the first stable fixation lands and gaze returns within `tolerance_radius` of the anchor, terminate. This is more ecological and would yield variable but shorter trials. Marked as TODO in [`saccade_screen.py`](scripts/saccade_screen.py).
- **No source discrimination for retries.** A retry restarts the entire capture window. If the failure was due to a single bad blink, a more sophisticated policy could splice in a partial window.
- **No real-time saccade detection.** All extraction happens at end-of-window. For closed-loop paradigms (e.g., stimulus contingent on saccade onset), real-time detection would be needed — out of scope for now.

## File reference

| Path | Role |
|------|------|
| [`scripts/saccade_screen.py`](scripts/saccade_screen.py) | `SaccadeScreen` class — same interface as `DrawingTablet`. |
| [`scripts/saccade_extractors.py`](scripts/saccade_extractors.py) | Three pure extractor functions. |
| [`scripts/audio_cue.py`](scripts/audio_cue.py) | Synthesized sine-tone Sound (`pygame.sndarray`). |
| [`scripts/saccade_smoke_test.py`](scripts/saccade_smoke_test.py) | Headless integration tests. |
| [`config/params.yaml`](config/params.yaml) | `response_mode` + `saccade:` config block. |
| [`main.py`](main.py) | Picks `DrawingTablet` vs `SaccadeScreen` based on `response_mode`. |
| [`scripts/phosphene_mapping.py`](scripts/phosphene_mapping.py) | Mapping mode — same branching as `main.py`. |
| [`scripts/standard_analyzer.py`](scripts/standard_analyzer.py) | Detects saccade trials, uses `response_xy` as centroid. |
| [`scripts/mapping_analyzer.py`](scripts/mapping_analyzer.py) | Same as above for mapping mode. |
