# phosLab — Implant Explorer

Interactive 3D explorer for cortical implant placement and receptive-field
coverage analysis. Lets you load parametric implant designs (Utah arrays,
multi-shank combs, thread bundles), drag and rotate them on macaque or human
cortex, and inspect the visual-field coverage of every contact in real time.

Supports two datasets out of the box:

- **NHP mode** — macaque D99 atlas + measured pRF maps from Monkey D / Monkey E
- **Human demo** — fsaverage cortex + Benson14 inferred retinotopy

Both datasets ship with the repo (~33 MB total). No external downloads are
required to launch.

## Quick Start

Requires Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/) (install:
`pip install uv` or `winget install astral-sh.uv`).

```bash
git clone https://github.com/antonio-lozano/phosLab.git
cd phosLab
uv sync
```

Run the explorer:

```bash
# NHP mode (default)
uv run python src/implant_explorer.py

# Human demo (fsaverage)
uv run python src/implant_explorer.py --dataset human_demo --human-subject fsaverage
```

### 30-second smoke test

After the window opens, in the **Setup** tab:

1. Find the **Implant Geometry** panel (right column).
2. Click *Load Implant…* and pick `implant_designs/Utah Array.json`.
3. You should see a 10×10 grid of electrodes rendered on the macaque brain.
4. Switch to the **Analysis** tab → the polar plot shows the RF coverage
   of the loaded array.

If you got that far, everything works.

## Using the Explorer

The window has three tabs on the right: **Setup** (data + implant), **Analysis**
(visual areas, RF picking, RF export), and **View** (display options). The 3D
viewer fills the left side.

### 1. Load an implant design

In **Setup → Implant Geometry**, click *Load Implant…* and pick one of the
shipped designs in `implant_designs/`:

| File | What it is |
|---|---|
| `Utah Array.json` | 10×10 grid (100 contacts) — recommended first try |
| `Shank Array.json` | 5-shank probe (160 contacts) |
| `Comb 10x10 5mm.json` | 10-shank linear comb, 4.5×5 mm (100 contacts) |
| `Comb 32x32 30mm.json` | 32-shank linear comb, 10×30 mm (1024 contacts) |
| `Thread-32.json` | 32-thread bundle (512 contacts) |
| `Thread-1024_30mm_fulls.json` | Neuralink-style circular bundle (1024 contacts) |
| `Thread-3072.json` | 96-thread bundle (3072 contacts) |

### 2. Move the implant

In **Setup → Placement** and **Manipulation**:

- Type X/Y/Z values and press *Apply* to translate, rotate, scale, mirror
- Or check *Enable drag* in **Implant Drag**, then **Ctrl + left-drag** the
  implant in the 3D view (plain left-drag still rotates the camera)
- *Reset Position* (under **Actions**) snaps it back to the default anchor
- Undo/Redo with **Ctrl+Z** / **Ctrl+Y**

### 3. See receptive-field coverage

Switch to the **Analysis** tab. The polar plot on the right shows where the
visible electrodes' receptive fields land in visual space.

- **Visual Areas**: pick which V1/V2/V3/V4 contacts to include
- **R² threshold**: filter out poor pRF fits (disabled in human-demo mode —
  the inferred maps don't have a meaningful R²)
- **pRF Source**: switch between Monkey D / Monkey E (NHP) or the human map

### 4. Hover the cortex to inspect RFs

In **Analysis → RF Picking**, keep *Enable RF picking* on. Then in the 3D view:

- **Hover** over cortex → live preview of the local RF neighborhood
- **Click** → lock the selection
- **Click empty space** (or *Clear Selection*) → unlock

### 5. Place multiple implants

In **Setup → Placed Implants**:

- *Duplicate Current Design* → adds another copy of the current implant. Each
  copy can be moved independently.
- *Hide Selected* / *Show All* / *Remove Selected* → manage scene visibility
- Tip: select one or several from the list, then drag them as a group.

### 6. Save and export

| Where | What it saves |
|---|---|
| **Implant Geometry → Save Design Template…** | Just the implant geometry (reusable design) |
| **Implant Geometry → Save Placement Snapshot…** | Geometry + the current placement transform |
| **Placed Implants → Save Layout…** | Full multi-implant scene (all placements) |
| **Analysis → RF Export → Export RFs CSV / JSON** | Receptive fields currently visible in the polar plot, with `x_deg`, `y_deg`, `polar_deg`, `ecc_deg`, etc. |

## Repo Layout

```
phosLab/
├── src/
│   ├── implant_explorer.py    # main Qt application
│   ├── dataset_adapters.py    # NHP / human dataset loaders
│   ├── rf_selection_utils.py  # RF picking helpers
│   └── implants_core/         # parametric implant library (designs, validation, export)
├── implant_designs/           # ready-to-load JSON designs
├── data/
│   ├── nhp/                   # macaque atlas + Monkey D/E pRF maps
│   └── human/demo_subject/.../fsaverage/T1w/.../mri/  # fsaverage Benson14 retinotopy
├── pyproject.toml
└── uv.lock
```

## License

MIT — see [LICENSE](LICENSE).

## Credits

Antonio Lozano — Eduardo Fernández's lab (NBIO, UMH) · a.lozano@umh.es
