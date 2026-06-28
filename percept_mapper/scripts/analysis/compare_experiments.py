"""Compara hasta 4 modalidades de entrada: mouse, gaze, pupil, wacom.

Produce en --out-dir:
  1. error_vs_ecc_comparison.png   -- boxplots agrupados por excentricidad
  2. map_comparison.png            -- true vs measured overlay de todas las condiciones

Solo se grafican las condiciones cuyos --exp-dir existen.

Uso:
    uv run python percept_mapper/compare_experiments.py \
        --mouse   mapping_experiments/mapping_..._mouse \
        --gaze    mapping_experiments/mapping_..._gaze \
        --pupil   mapping_experiments/mapping_..._pupil \
        --wacom   mapping_experiments/mapping_..._wacom \
        --out-dir comparison_results/input_mode
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Paleta fija por modalidad
MODALITY_STYLE: dict[str, dict] = {
    "mouse":  {"color": "#4C72B0", "marker": "o",  "label": "Mouse"},
    "gaze":   {"color": "#DD8452", "marker": "^",  "label": "Gaze (saccade)"},
    "pupil":  {"color": "#55A868", "marker": "s",  "label": "Pupil"},
    "wacom":  {"color": "#C44E52", "marker": "D",  "label": "WACOM"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(exp_dir: Path) -> dict:
    f = exp_dir / "consolidated_analysis" / "consolidated_results.json"
    if not f.exists():
        raise SystemExit(f"No se encontró {f}")
    return json.loads(f.read_text(encoding="utf-8"))


def assign_rings(results: dict, tol: float = 1.0, snap: float = 0.05) -> dict[int, float]:
    raw = {
        int(k): math.hypot(*v["stimulation_position_deg"])
        for k, v in results["electrodes"].items()
    }
    order = sorted(raw, key=lambda e: raw[e])
    rings: dict[int, float] = {}
    cluster: list[int] = []

    def flush(cl):
        m = sum(raw[e] for e in cl) / len(cl)
        rv = round(round(m / snap) * snap, 2)
        for e in cl:
            rings[e] = rv

    for e in order:
        if cluster and raw[e] - raw[cluster[-1]] > tol:
            flush(cluster)
            cluster = []
        cluster.append(e)
    if cluster:
        flush(cluster)
    return rings


def collect_errors(results: dict, rings: dict[int, float]) -> dict[float, list[float]]:
    groups: dict[float, list[float]] = {}
    for k, rec in results["electrodes"].items():
        ecc = rings[int(k)]
        for rep in rec.get("per_repetition_metrics", []) or []:
            groups.setdefault(ecc, []).append(float(rep["distance_to_stim_deg"]))
    return groups


# ---------------------------------------------------------------------------
# Plot 1: grouped boxplot (N condiciones × M excentricidades)
# ---------------------------------------------------------------------------

def plot_error_comparison(
    conditions: list[tuple[str, dict[float, list[float]]]],
    eccs: list[float],
    out: Path,
) -> None:
    n = len(conditions)
    width = min(0.3, 0.9 / n)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * (width + 0.04)

    # Usar posiciones categóricas equidistantes para que el salto 8°→12°
    # no aparezca más ancho que los demás en el eje X
    # Posiciones categóricas equidistantes (independientemente del salto en grados)
    display_eccs = eccs
    cat_pos = list(range(len(eccs)))
    ecc_to_cat = {e: i for i, e in enumerate(eccs)}

    fig, ax = plt.subplots(figsize=(max(9, 2 * len(eccs) * n), 6))
    rng = np.random.default_rng(42)

    for i, (modality, groups) in enumerate(conditions):
        style = MODALITY_STYLE[modality]
        color = style["color"]
        label = style["label"]
        positions = [ecc_to_cat[e] + offsets[i] for e in eccs]
        data = [groups.get(e, []) for e in eccs]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            medianprops=dict(color="black", lw=1.5),
            meanprops=dict(color=color, ls="--", lw=1.4),
            boxprops=dict(facecolor=color, alpha=0.30, edgecolor="black", lw=1.0),
            whiskerprops=dict(color="black", lw=1.0),
            capprops=dict(color="black", lw=1.0),
            flierprops=dict(marker="", alpha=0),
        )

        for j, (pos, e) in enumerate(zip(positions, eccs)):
            pts = groups.get(e, [])
            if not pts:
                continue
            jitter = (rng.random(len(pts)) - 0.5) * width * 0.6
            ax.scatter(
                np.array([pos] * len(pts)) + jitter,
                pts,
                color=color,
                s=24,
                edgecolors="black",
                linewidths=0.4,
                alpha=0.75,
                zorder=5,
            )
            # Mediana encima del bigote superior
            top_cap_y = bp["caps"][2 * j + 1].get_ydata()[0]
            med = float(np.median(pts))
            ax.text(pos, top_cap_y + 0.04, f"Md={med:.2f}°",
                    ha="center", va="bottom", fontsize=7, color="black")

    # x labels: ecc + n per condition
    def _n_label(e):
        parts = "/".join(
            str(len(g.get(e, [])))
            for _, g in conditions
        )
        return f"{round(e):d}°\n(n={parts})"

    ax.set_xticks(cat_pos)
    ax.set_xticklabels([_n_label(e) for e in eccs])
    ax.set_xlabel("Eccentricity (deg)", fontsize=12)
    ax.set_ylabel("Radial error  |response − stimulus|  (deg)", fontsize=12)
    ax.set_ylim(0, None)
    ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        Line2D([], [], color=MODALITY_STYLE[m]["color"], lw=8, alpha=0.4,
               label=MODALITY_STYLE[m]["label"])
        for m, _ in conditions
    ]
    legend_handles += [
        Line2D([], [], color="black", lw=1.5, label="mediana"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# Plot 2: true vs measured overlay
# ---------------------------------------------------------------------------

def _implant_labels(elec_ids: list[int], results: dict) -> tuple[dict[int, str], dict[str, str]]:
    """Per-electrode plot labels + a short-code legend for multi-implant CSVs.

    With a single implant (or YAML mode, where implant_id is None) this is a
    no-op: labels are just the global index. With 2+ implants, labels become
    "<code>:<local_index>" (e.g. "B:67") so the implant origin is visible
    directly on the plot instead of requiring the offset table.
    """
    implant_of = {e: results["electrodes"][str(e)].get("implant_id") for e in elec_ids}
    unique_implants = list(dict.fromkeys(v for v in implant_of.values() if v is not None))
    if len(unique_implants) < 2:
        return {e: str(e) for e in elec_ids}, {}

    import string
    codes = {imp: string.ascii_uppercase[i] for i, imp in enumerate(unique_implants)}
    label_of = {}
    for e in elec_ids:
        imp = implant_of[e]
        local = results["electrodes"][str(e)].get("implant_local_index")
        label_of[e] = f"{codes[imp]}:{local}" if imp is not None else str(e)
    legend = {code: imp for imp, code in codes.items()}
    return label_of, legend


def plot_map_comparison(
    conditions: list[tuple[str, dict]],   # (modality, results_dict)
    rings_ref: dict[int, float],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    max_r = 1.0
    eccs_seen: set[float] = set(rings_ref.values())

    # true positions (one set — all conditions share the same stimuli)
    ref_results = conditions[0][1]
    elec_ids = sorted(int(k) for k in ref_results["electrodes"])
    label_of, implant_legend = _implant_labels(elec_ids, ref_results)
    first_true = True
    for e in elec_ids:
        tx, ty = ref_results["electrodes"][str(e)]["stimulation_position_deg"]
        max_r = max(max_r, abs(tx), abs(ty))
        ax.scatter([tx], [ty], marker="s", s=60, facecolors="none",
                   edgecolors="black", linewidths=1.3, zorder=4,
                   label="true (stimulus)" if first_true else "")
        ax.annotate(label_of[e], (tx, ty), textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color="#444444")
        first_true = False

    for modality, results in conditions:
        style = MODALITY_STYLE[modality]
        color = style["color"]
        marker = style["marker"]
        label = style["label"]
        first = True
        for k, rec in results["electrodes"].items():
            mx = rec["mean_position_deg"]["x"]
            my = rec["mean_position_deg"]["y"]
            cents = np.array(rec["centroids_deg"], dtype=float)
            sx = float(np.std(cents[:, 0], ddof=1)) if len(cents) > 1 else 0.0
            sy = float(np.std(cents[:, 1], ddof=1)) if len(cents) > 1 else 0.0
            max_r = max(max_r, abs(mx), abs(my))
            ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt=marker, ms=8,
                        color=color, ecolor=color, elinewidth=1.2,
                        capsize=3, mec="black", mew=0.5, zorder=5,
                        label=label if first else "")
            first = False

    for r in sorted(eccs_seen):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls=":", lw=1.0,
                                 ec="#aaaaaa", zorder=1))
        ax.text(r * math.cos(math.radians(42)), r * math.sin(math.radians(42)),
                f"{r:g}°", fontsize=7, color="#888888")

    lim = max_r * 1.15
    ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
    ax.axvline(0, color="#dddddd", lw=0.8, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Visual field X (deg)   [+ = right]", fontsize=12)
    ax.set_ylabel("Visual field Y (deg)   [+ = up]", fontsize=12)
    ax.grid(alpha=0.2)
    if implant_legend:
        legend_txt = "\n".join(f"{code} = {imp}" for code, imp in sorted(implant_legend.items()))
        ax.text(0.02, 0.02, legend_txt, transform=ax.transAxes, fontsize=7,
                color="#555555", va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.85))
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse",   default=None, help="Directorio experimento mouse")
    ap.add_argument("--gaze",    default=None, help="Directorio experimento gaze/saccade")
    ap.add_argument("--pupil",   default=None, help="Directorio experimento pupil")
    ap.add_argument("--wacom",   default=None, help="Directorio experimento WACOM")
    ap.add_argument("--out-dir", default=None, help="Carpeta de salida (default: mapping_experiments/comparison_input_mode)")
    args = ap.parse_args()

    # Recoger las condiciones que tienen datos (wacom primero para que aparezca
    # a la izquierda en los boxplots agrupados)
    raw = [
        ("wacom",  args.wacom),
        ("mouse",  args.mouse),
        ("gaze",   args.gaze),
        ("pupil",  args.pupil),
    ]
    active = [(m, Path(p)) for m, p in raw if p is not None]

    if len(active) < 2:
        raise SystemExit("Necesitas al menos dos condiciones (--mouse, --gaze, --pupil o --wacom).")

    out_dir = Path(args.out_dir) if args.out_dir else Path("mapping_experiments/comparison_input_mode")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare] Condiciones: {[m for m, _ in active]}")
    print(f"[compare] Salida: {out_dir}")

    loaded = [(m, load_results(d)) for m, d in active]
    rings_ref = assign_rings(loaded[0][1])

    error_conditions = [(m, collect_errors(r, assign_rings(r))) for m, r in loaded]
    eccs = sorted({e for _, g in error_conditions for e in g})

    plot_error_comparison(error_conditions, eccs,
                          out_dir / "error_vs_ecc_comparison.png")
    plot_map_comparison(loaded, rings_ref,
                        out_dir / "map_comparison.png")

    # Resumen numérico
    print(f"\n{'ecc':>7}", end="")
    for m, _ in error_conditions:
        print(f"  {MODALITY_STYLE[m]['label']:>16}", end="")
    print()
    for e in eccs:
        print(f"{e:>6.2f}°", end="")
        for _, g in error_conditions:
            pts = g.get(e, [])
            s = f"{np.mean(pts):.3f} ± {np.std(pts, ddof=1):.3f}°" if pts else "—"
            print(f"  {s:>16}", end="")
        print()


if __name__ == "__main__":
    main()
