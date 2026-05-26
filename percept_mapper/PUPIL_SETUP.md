# Pupil Core — Setup & Usage Guide

End-to-end guide for installing Pupil Capture, calibrating, defining the screen surface, and running PhosLab in `input_mode: pupil`.

## Architecture

```
Pupil Core headset ── USB ──▶ Pupil Capture (app) ── ZMQ:tcp://127.0.0.1:50020 ──▶ PhosLab PupilTracker ──▶ Experiment
                              (owns cameras, runs           (publishes surfaces.<name>            (consumer only;
                               pupil detection + calibration,        with gaze_on_surfaces)        no camera I/O)
                               + Surface Tracker)
```

PhosLab is a *consumer* over a network socket. It does not import any code from `C:\Users\admin\pupil\` (the source folder is reference-only).

---

## Part 1 — One-time software install

### 1.1 Pupil Capture (the app)

1. Download the latest `.msi` from <https://github.com/pupil-labs/pupil/releases/latest> (already present at `C:\Users\admin\pupil\pupil_v3.5-1-g1cdbe38_windows_x64.msi`).
2. Install. "Pupil Capture" should appear in Start Menu.

### 1.2 PhosLab Python deps

Already covered when you run `uv sync --project percept_mapper`. The relevant packages are `pyzmq` and `msgpack`.

---

## Part 2 — One-time hardware + surface setup

### 2.1 Plug in the headset

Connect the Pupil Core via USB. Launch **Pupil Capture**. You should get three windows: world view, Eye 0, Eye 1.

### 2.2 Pupil detection check (the #1 calibration killer)

For each eye window:

- Press `a` to enable **Algorithm Mode**.
- The **green circle** must track the pupil at every gaze angle (top-left, top-right, bottom-left, bottom-right of the screen). If it disappears or jumps, calibration cannot work.
- Tune in the **Pupil Detector 2D** plugin sidebar:
  - `Pupil Min/Max` — drag the red circles so they bracket the green one even at extreme angles.
  - `Intensity Range` — blue-tinted pixels must cover the pupil without leaking into eyelashes/eyebrows.
- Glasses cause severe IR reflection. Contacts are recommended.

### 2.3 Calibration (must redo each session or after slippage)

1. Sidebar → **Calibration** → Choreography: **Screen Marker**.
2. Set **Monitor** to the stimulus screen; enable **Use fullscreen**.
3. Bump **Sample duration** to ~2 s and **Marker size** to a comfortable level.
4. Press **C**. Follow the markers with your eyes only — *do not move your head*.
5. Enable the **Accuracy Visualizer** plugin. Press **T** to run an accuracy test. Target: **<1.5° angular error**. If >2°, return to step 2.2.

### 2.4 Surface definition

PhosLab subscribes to surface-relative gaze. The "surface" is a 2D plane in world-camera coordinates that maps to the stimulus screen.

1. From the repo root, in one terminal:
   ```
   uv run --project percept_mapper python percept_mapper/scripts/show_apriltags_fullscreen.py
   ```
   Displays 4 AprilTags at the screen corners (borderless windowed, survives focus loss). Don't close it.

2. In Pupil Capture: **Plugin Manager** → enable **Surface Tracker**.

3. In the world view you should see the 4 tags outlined with IDs (0, 1, 2, 3).

4. Press **A** (or click **Add surface**). A new surface appears, default name `Surface 1` / `Surface 2` / ...

5. Click **edit surface**, freeze the scene, drag the 4 corner handles so the green rectangle matches the screen edges exactly.

6. **Note the surface name** — must match `pupil.surface_name` in `params.yaml`.

The definition is saved to `pupil_capture_settings/surface_definitions` and persists across Pupil Capture restarts.

---

## Part 3 — Configure PhosLab

[`percept_mapper/config/params.yaml`](config/params.yaml):

```yaml
input_mode: pupil           # mouse | gaze (webcam mediapipe) | pupil
pupil:
  address: 127.0.0.1
  port: 50020               # Pupil Remote default
  surface_name: Surface 2   # MUST match Pupil Capture's surface name
  min_confidence: 0.7       # drop samples below this confidence
  max_sample_age_s: 0.25    # stale-gaze guard: treat last_smooth_gaze as None
                            # if newest sample is older than this (seconds)
  one_euro:                 # smoothing filter parameters
    fps: 60
    min_cutoff: 0.3         # lower = more smoothing on slow gaze
    beta: 0.001             # lower = less velocity-driven adaptation
    d_cutoff: 1.0
apriltag_overlay:           # render 4 tags every frame inside experiment
  enabled: true
  png_dir: percept_mapper/assets/apriltags
  tag_files: [tag_0.png, tag_1.png, tag_2.png, tag_3.png]
  tag_size_px: 300
  margin_px: 30
```

---

## Part 4 — Run an experiment

Session-by-session sequence:

1. Plug in headset.
2. Launch **Pupil Capture**. Wait for cameras to load. Look at the screen — verify the surface outline turns **green** (= tags detected and surface locked).
3. **Calibrate** (Screen Marker, **C**). Validate with **T**.
4. Run PhosLab from `percept_mapper/`:
   ```
   cd percept_mapper
   uv run python main.py
   ```

Expected startup output:
```
[INIT] Modo de entrada: pupil
[PupilTracker] SUB_PORT=...
[PupilTracker] Suscrito a tópico: surfaces.Surface 2
[PupilTracker] One-Euro: min_cutoff=0.3 beta=0.001 d_cutoff=1.0
[PupilTracker] ✓ Inicializado correctamente
[AprilTagOverlay] flip hook instalado: tags visibles en cada frame.
```

If any of those is missing, jump to Troubleshooting.

---

## Diagnostic tools

| Tool | When to use |
|------|-------------|
| `scripts/pupil_smoke_test.py` | Tier-by-tier check: are `pupil.*`, `gaze.*`, `surfaces.*` flowing? Use after every Pupil-side change. |
| `scripts/verify_overlay.py` | Headless render of the AprilTag overlay → `assets/apriltag_overlay_preview.png`. Confirms overlay module works without launching the experiment. |
| `scripts/show_apriltags_fullscreen.py` | Displays the overlay borderless-fullscreen. Use when defining the surface or after recalibrating. |
| In-tracker diagnostic log | Auto-printed every 2 s during `main.py`. See fields below. |

The in-tracker line:
```
[PupilTracker][diag] msgs=120 con_gaze=118 sin_screen=0 norm=(0.105, 0.627) conf=0.977 smooth=(269, 538) screen=(2560, 1440)
```

- `msgs` — surface packets received per window (≈60/s, so ~120 per 2 s)
- `con_gaze` — samples passing `min_confidence`
- `sin_screen` — dropped because screen size not yet known; should drop to 0 once the experiment hits its first frame
- `norm/conf` — last raw `norm_pos` and confidence from Pupil
- `smooth` — pixel-space output the experiment reads (`last_smooth_gaze`)

---

## Troubleshooting

### `msgs = 0` for the entire run
Pupil Capture isn't publishing surface data. Causes:
- **AprilTags not actually on screen.** Look at the monitor during the experiment — the 4 corner tags must be visible. If not, check `apriltag_overlay.enabled: true` and that PNGs exist at `percept_mapper/assets/apriltags/`.
- **Surface name mismatch.** Run `pupil_smoke_test.py` — it reports the actual surface name. Update `pupil.surface_name` in `params.yaml`.
- **Surface not defined yet.** Repeat Part 2.4.

### `con_gaze = 0` but `msgs > 0`
Surface is detected but no gaze passes the confidence threshold.
- **Not calibrated** — calibrate (Part 2.3).
- **Confidence too high** — temporarily lower `min_confidence` to `0.5`. If samples flow, your calibration is poor; revisit Part 2.2.

### Gaze is super unstable / jittery
Tune `pupil.one_euro`:
- Lower `min_cutoff` (0.3–0.5) for heavier smoothing of slow gaze.
- Lower `beta` (0.001) for less velocity-driven adaptation.
- Raise `min_confidence` (0.85) to drop bad samples — but don't combine very high confidence + adaptive dt (current implementation uses fixed dt, so this is safe here).

### Calibration is bad (accuracy >2°)
- Don't recalibrate first — fix pupil detection (Part 2.2).
- Tighten the headset, reseat.
- Try 3D Gaze Mapper (sidebar → Calibration → Gaze Mapper) if you suspect slippage.

### Surface drops mid-experiment
- Tags too small in world camera view. Bump `apriltag_overlay.tag_size_px` to 300–400.
- World camera not focused on the screen.

### Experiment freezes at "Esperando fijación..."
`PupilTracker.last_smooth_gaze` is None or stale → `is_looking_at_point` always returns False.
- Check the diag line — is `smooth` `None`? Then nothing is reaching the tracker.
- Check that AprilTag overlay installed (`[AprilTagOverlay] flip hook instalado` printed at startup).

---

## Tuning cheatsheet

| Symptom | Knob | Direction |
|---------|------|-----------|
| Gaze too jittery | `pupil.one_euro.min_cutoff` | Lower (→ 0.3) |
| Cursor lags real gaze | `pupil.one_euro.beta` | Higher (→ 0.005) |
| Frequent dropouts | `pupil.min_confidence` | Lower (→ 0.6) |
| Bad samples leak through | `pupil.min_confidence` | Higher (→ 0.85) |
| Tags lose detection | `apriltag_overlay.tag_size_px` | Higher (→ 300+) |
| Anchor never accepts fixation | anchor `tolerance_radius` in `params.yaml > screen.anchor_circle` | Higher (→ 80–100) |

---

## File reference

| Path | Role |
|------|------|
| [`core/pupil_tracker.py`](core/pupil_tracker.py) | ZMQ subscriber + One-Euro filter. Implements the same interface as `EyeTracker` / `MouseTracker`. |
| [`scripts/apriltag_overlay.py`](scripts/apriltag_overlay.py) | Renders 4 AprilTag PNGs at screen corners via a `pygame.display.flip` monkey-patch. |
| [`scripts/extract_apriltags.py`](scripts/extract_apriltags.py) | One-shot script that cropped the 4 individual tag PNGs from Pupil Labs' sheet. |
| [`scripts/show_apriltags_fullscreen.py`](scripts/show_apriltags_fullscreen.py) | Borderless fullscreen tag display for surface definition. |
| [`scripts/pupil_smoke_test.py`](scripts/pupil_smoke_test.py) | Tier-by-tier pipeline check. |
| [`scripts/verify_overlay.py`](scripts/verify_overlay.py) | Headless render check of the overlay. |
| [`assets/apriltags/tag_*.png`](assets/apriltags/) | The 4 extracted AprilTag PNGs (tag36h11 IDs 0–3, with 20% white quiet zone baked in). |
| [`config/params.yaml`](config/params.yaml) | All Pupil-related configuration. |
