# Screen Geometry & Visual-Field Calibration — Setup & Usage

How PhosLab maps **degrees of visual angle ↔ pixels**, why that mapping depends
on the physical screen and viewing distance, how to keep it coherent
automatically, and the safety gate that stops an experiment from placing
invisible phosphenes off-screen.

## Why this matters

Every phosphene position in the coordinate CSV is in **degrees of visual field**.
To draw it, PhosLab converts degrees → pixels. That conversion is only
*physically meaningful* if it accounts for the real screen size and how far the
participant sits. Get it wrong and a phosphene labelled "10°" is shown at some
other angle — quietly invalidating the mapping.

---

## The deg ↔ px mapping

PhosLab uses a single **isotropic** pixels-per-degree (panel pixels are square,
so one degree must span the same number of pixels in x and y):

```
ppd = min(width_px, height_px) / (2 · vf_scope_deg)

x_px = center_x + x_deg · ppd
y_px = center_y − y_deg · ppd        (y inverted: + is up)
```

- **Isotropic** ⇒ iso-eccentricity contours are true **circles**, and polar
  angles are preserved. (A previous bug scaled x and y independently, turning
  circles into ellipses and misplacing phosphenes — see git history.)
- **Anchored to the shorter side** (`min(W,H)` = height on a landscape monitor)
  so the full ±`vf_scope_deg` fits with **no clipping**; the longer side just
  keeps unused margin.

`vf_scope_deg` is therefore the **half-FOV (max eccentricity) mapped to the
screen's shorter side** — i.e. the largest *full* iso-eccentricity ring that
fits on screen.

> Note: the conversion is linear (`px = deg · ppd`), not a flat-screen `tan`
> projection. Anchored at the shorter-side edge it is exact at the edge and
> within ~0.25° elsewhere for the ranges used here. Good enough for behavioural
> mapping; switch to `tan` placement if you need sub-0.1° at large eccentricity.

---

## `vf_scope_deg` — the one value that must match your screen

For the degrees to be **physically true**, `vf_scope_deg` must equal the
screen's shorter-side physical half-FOV at the viewing distance:

```
vf_scope_deg = atan( (shorter_physical_side_cm / 2) / dist_to_screen_cm )
```

You do **not** have to compute this by hand. Set it to `auto`:

```yaml
screen:
  width: 2560
  height: 1440
  screen_diagonal_inches: 27.15   # physical size (use the screen detector)
  dist_to_screen_cm: 50           # measured viewing distance
  vf_scope_deg: auto              # derived from the geometry above — can't drift
```

| `vf_scope_deg` value | Behaviour at startup |
|----------------------|----------------------|
| `auto` (or `physical` / `screen` / `max`) | Derived from `resolution + diagonal + distance`. Recommended — updates itself if the distance or monitor changes. |
| a number (e.g. `18.68`) | Used as-is, but a **warning** prints if it is >5% off the physical value. Use this only to force a deliberately nominal field. |
| missing / unparseable | Falls back to `15.0` with a warning. |

### Physical geometry, from first principles

From the diagonal + resolution (square pixels):

```
diag_cm   = diagonal_inches · 2.54
width_cm  = diag_cm · W / hypot(W, H)
height_cm = diag_cm · H / hypot(W, H)
half_fov(side_cm) = atan( (side_cm / 2) / dist_cm )
```

Worked example — the 27.15″ 2560×1440 dev monitor (60.1 × 33.8 cm):

| Distance | Horizontal ± | Vertical ± | Corner | **Full-ring max = `vf_scope` (auto)** |
|----------|-------------|-----------|--------|----------------------------------------|
| 50 cm | 31.0° | 18.7° | 34.6° | **18.68°** |
| 60 cm | 26.6° | 15.7° | 29.9° | **15.73°** |
| 29 cm | 46.0° | 30.2° | 49.9° | **30.24°** |

The full-ring max is **height-limited** on a landscape screen. To reach a target
full-ring eccentricity `E`, sit at `dist = (shorter_half_cm) / tan(E)` — e.g. a
complete 30° ring on this monitor needs ~29 cm.

---

## Keeping geometry honest — the screen detector

`screen_diagonal_inches` is *per machine* and drifts when a config is copied
between PCs. [`scripts/screen_detect.py`](scripts/screen_detect.py) reads the
real geometry from the OS (resolution always; physical size via EDID/WMI on
Windows) and reports it.

**CLI:**
```
uv run --project percept_mapper python percept_mapper/scripts/screen_detect.py          # report
uv run --project percept_mapper python percept_mapper/scripts/screen_detect.py --write  # persist resolution + diagonal into params.yaml
```

Sample report (note the **max-eccentricity** feedback and the `vf_scope`
coherence line):
```
  resolution: config 2560x1440  →  detected 2560x1440
  diagonal:   config 27.15"     →  detected 27.15"
  vf_scope:   config auto        →  physical (short side) 18.68°   [±31.0° H, ±18.7° V at 50cm]
  max ecc:    full ring 18.68° (short side)  |  horizontal ±31.0°  |  corner 34.6°   (at 50cm)
```

The same is available in the launcher's **Pantalla** page (the "Detectar
pantalla" button): detected vs configured resolution/diagonal, a one-click
**Sobrescribir** to write them into `params.yaml`, and the max-eccentricity +
`vf_scope` coherence feedback.

---

## Off-screen feasibility gate

Before each run, PhosLab checks that **every selected electrode's phosphene
lands on-screen** for the current `vf_scope_deg`. A phosphene fits vertically
only if `ecc · |sin θ| ≤ vf_scope_deg`, so a `vf_scope` that is too small for
your electrode set pushes outer/near-vertical phosphenes off the screen — where
they'd be invisible yet recorded as if shown.

**Default: it blocks the launch** and lists the offenders:

```
[INIT] ✗ FOSFENOS FUERA DE PANTALLA: con vf_scope_deg=10°, 3 de los electrodos
        seleccionados caen fuera de 2560x1440 y NO serían visibles:
    electrodo 8:  px=(1280,1638), ecc=12.8°
    electrodo 11: px=(1892,1780), ecc=17.0°
    electrodo 12: px=(1892,-340), ecc=17.0°
  El ecc máximo visible (en cualquier ángulo) con esta geometría es ~10°.
  Soluciones: sube vf_scope_deg (o ponlo a 'auto'), acerca la pantalla, o
  reduce la excentricidad de los electrodos.
  Para ejecutar igualmente: screen.allow_offscreen: true
```

To run anyway (e.g. you intentionally want partial-arc coverage at high
eccentricity):

```yaml
screen:
  allow_offscreen: true   # gate warns instead of blocking
```

---

## Behaviour summary

| Situation | What happens |
|-----------|--------------|
| `vf_scope_deg: auto` | Derived from physical geometry + distance. Always coherent. |
| Numeric `vf_scope_deg` incoherent with the screen (degrees ≠ real angle) but phosphenes on-screen | **Warning**, runs as-is (respects an intentional nominal field). |
| `vf_scope_deg` so small that selected phosphenes fall off-screen | **Blocks** by default; lists offenders + max usable ecc. Override with `allow_offscreen: true`. |
| `screen_diagonal_inches` off >10% from the detected monitor | **Warning** at startup; fix with `screen_detect --write`. |

PhosLab never *silently* changes an explicit value — it warns, blocks, or (only
for `auto`) derives.

---

## File reference

| Path | Role |
|------|------|
| [`scripts/screen_detect.py`](scripts/screen_detect.py) | OS geometry detection (EDID/WMI); `physical_fov_deg()` / `coherent_vf_scope_deg()` first-principles FOV; params writer; CLI. |
| [`scripts/dynaphos_adapter.py`](scripts/dynaphos_adapter.py) | Isotropic deg→px placement (`pixels_per_degree`, `vf_scope` geometry). |
| [`main.py`](main.py) | Resolves `vf_scope_deg: auto`, coherence warning, and the off-screen feasibility gate (`_assert_phosphenes_onscreen`). |
| [`launcher/launcher.py`](../launcher/launcher.py) | "Pantalla" page: detect / compare / override / max-ecc feedback. |
| [`config/params.yaml`](config/params.yaml) | `screen:` block — `width`, `height`, `screen_diagonal_inches`, `dist_to_screen_cm`, `vf_scope_deg`, `allow_offscreen`. |

---

## Troubleshooting

### Launch aborts with "FOSFENOS FUERA DE PANTALLA"
A selected electrode's phosphene maps off-screen for the current `vf_scope_deg`.
Set `vf_scope_deg: auto`, move the screen closer, reduce the electrodes'
eccentricity, or — if intentional — set `screen.allow_offscreen: true`.

### "vf_scope_deg=X° no coincide con el half-FOV físico"
Your hand-typed `vf_scope_deg` doesn't match the screen at this distance. Set
`vf_scope_deg: auto` to derive it, or update the number.

### "screen_diagonal_inches no coincide con el monitor detectado"
The config's physical size is stale (copied from another PC). Run
`screen_detect --write`, or use the launcher's **Pantalla → Sobrescribir**.

### Physical size shows `n/a` / `vf_scope: auto` falls back to 15°
EDID wasn't readable (non-Windows, a KVM/adapter, or a virtual display). Set
`screen_diagonal_inches` manually with a tape-measured value; resolution +
distance + diagonal are all `auto` needs.

### Rings look like ellipses instead of circles
That was an old anisotropic-mapping bug (now fixed). If you still see it, you're
on an old build — pull `main`.
