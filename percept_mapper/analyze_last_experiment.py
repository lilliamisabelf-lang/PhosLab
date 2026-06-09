"""Extra analysis plots for a finished mapping experiment.

Reads consolidated_analysis/consolidated_results.json and produces, in degrees
of visual angle (the session is distance-calibrated):

  1. error_vs_eccentricity.png  -- per-trial radial error grouped by
     eccentricity ring, as boxes + the individual single-trial points.
  2. true_vs_measured_map.png   -- visual-field map of the TRUE (stimulus)
     position vs the MEASURED mean response per electrode, with +/-1 SD error
     bars in x and y, and iso-eccentricity reference rings.

Usage:
    uv run --project percept_mapper python percept_mapper/analyze_last_experiment.py
    uv run --project percept_mapper python percept_mapper/analyze_last_experiment.py --exp-dir <path>
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

MAPPING_DIR = Path(__file__).resolve().parent / "mapping_experiments"


def find_latest_experiment() -> Path:
    cands = [
        d for d in MAPPING_DIR.iterdir()
        if d.is_dir() and (d / "consolidated_analysis" / "consolidated_results.json").exists()
    ]
    if not cands:
        raise SystemExit(f"No analyzed experiments found under {MAPPING_DIR}")
    return max(cands, key=lambda d: d.stat().st_mtime)


def load_results(exp_dir: Path) -> dict:
    f = exp_dir / "consolidated_analysis" / "consolidated_results.json"
    if not f.exists():
        raise SystemExit(f"Missing {f}")
    return json.load(open(f, encoding="utf-8"))


def _ecc_of(stim_deg) -> float:
    return float(math.hypot(stim_deg[0], stim_deg[1]))


def assign_rings(results: dict, tol: float = 1.0, snap: float = 0.05) -> dict[int, float]:
    """Cluster electrodes into eccentricity rings.

    The stimulus is stored at integer pixels, so the back-computed ecc varies by
    a few hundredths of a degree between electrodes that share a design ring.
    Group electrodes whose ecc are within `tol` deg into one ring and label the
    ring by its mean ecc (snapped to `snap` for a clean value).
    """
    raw = {int(k): _ecc_of(v["stimulation_position_deg"]) for k, v in results["electrodes"].items()}
    order = sorted(raw, key=lambda e: raw[e])
    rings: dict[int, float] = {}
    cluster: list[int] = []

    def flush(cl: list[int]) -> None:
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


def plot_error_vs_ecc(results: dict, out: Path, rings: dict[int, float]) -> None:
    # Group per-trial radial errors (deg) by eccentricity ring.
    groups: dict[float, list[float]] = {}
    pts_x, pts_y, pts_c = [], [], []
    elec_ids = sorted((int(k) for k in results["electrodes"]), key=int)
    cmap = plt.cm.tab20(np.linspace(0, 1, len(elec_ids)))
    color_of = {e: cmap[i] for i, e in enumerate(elec_ids)}

    for e in elec_ids:
        rec = results["electrodes"][str(e)]
        ecc = rings[e]
        for rep in rec.get("per_repetition_metrics", []) or []:
            err = float(rep["distance_to_stim_deg"])
            groups.setdefault(ecc, []).append(err)
            pts_x.append(ecc)
            pts_y.append(err)
            pts_c.append(color_of[e])

    eccs = sorted(groups)
    data = [groups[k] for k in eccs]

    fig, ax = plt.subplots(figsize=(9, 6))
    width = 0.6
    bp = ax.boxplot(
        data, positions=eccs, widths=width, patch_artist=True,
        showmeans=True, meanline=True,
        medianprops=dict(color="black", lw=1.6),
        meanprops=dict(color="crimson", ls="--", lw=1.4),
        boxprops=dict(facecolor="#dfe7f5", edgecolor="#5b6b8c", alpha=0.9),
        whiskerprops=dict(color="#5b6b8c"), capprops=dict(color="#5b6b8c"),
        flierprops=dict(marker="", alpha=0),  # hide default fliers; we plot all points
    )
    # single-trial points with horizontal jitter
    rng = np.random.default_rng(0)
    jitter = (rng.random(len(pts_x)) - 0.5) * width * 0.7
    ax.scatter(
        np.array(pts_x) + jitter, pts_y, c=pts_c, s=42, edgecolors="black",
        linewidths=0.5, zorder=5, alpha=0.9,
    )
    # per-ring group mean error annotation
    for k in eccs:
        m = float(np.mean(groups[k]))
        ax.text(k, ax.get_ylim()[1], f"  mean {m:.2f}°", ha="center", va="bottom",
                fontsize=8, color="crimson")

    ax.set_xlabel("Eccentricity (deg)", fontsize=12)
    ax.set_ylabel("Radial error  |response - stimulus|  (deg)", fontsize=12)
    ax.set_title(f"Localization error vs eccentricity\n{results['experiment_name']}",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(eccs)
    ax.set_xticklabels([f"{k:g}°\n(n={len(groups[k])})" for k in eccs])
    ax.set_ylim(0, max(pts_y) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.margins(x=0.08)
    # legend for box / mean
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color="black", lw=1.6, label="median"),
        Line2D([], [], color="crimson", ls="--", lw=1.4, label="mean"),
        Line2D([], [], marker="o", ls="", mfc="gray", mec="black", label="single trial"),
    ], loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")
    # also return a small summary
    return {k: (float(np.mean(groups[k])), float(np.std(groups[k], ddof=1))) for k in eccs}


def plot_true_vs_measured(results: dict, out: Path, rings: dict[int, float]) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    elec_ids = sorted((int(k) for k in results["electrodes"]), key=int)
    cmap = plt.cm.tab20(np.linspace(0, 1, len(elec_ids)))

    eccs_seen = set(rings.values())
    max_r = 1.0
    for i, e in enumerate(elec_ids):
        rec = results["electrodes"][str(e)]
        tx, ty = rec["stimulation_position_deg"]
        mx, my = rec["mean_position_deg"]["x"], rec["mean_position_deg"]["y"]
        cents = np.array(rec["centroids_deg"], dtype=float)
        sx = float(np.std(cents[:, 0], ddof=1)) if len(cents) > 1 else 0.0
        sy = float(np.std(cents[:, 1], ddof=1)) if len(cents) > 1 else 0.0
        max_r = max(max_r, abs(tx), abs(ty), abs(mx), abs(my))

        c = cmap[i]
        # offset line true -> measured
        ax.plot([tx, mx], [ty, my], color=c, lw=1.0, alpha=0.6, zorder=2)
        # true position (open black square)
        ax.scatter([tx], [ty], marker="s", s=70, facecolors="none",
                   edgecolors="black", linewidths=1.4, zorder=4)
        # measured mean with +/-1 SD error bars in x and y
        ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt="o", ms=9, color=c,
                    ecolor=c, elinewidth=1.4, capsize=3, mec="black", mew=0.6,
                    zorder=5)
        ax.annotate(str(e), (mx, my), textcoords="offset points", xytext=(7, 5),
                    fontsize=8, color="black")

    # iso-eccentricity reference rings + axes
    for r in sorted(eccs_seen):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls=":", lw=1.0,
                                 ec="#999999", zorder=1))
        ax.text(r * math.cos(math.radians(45)), r * math.sin(math.radians(45)),
                f"{r:g}°", fontsize=7, color="#777777", ha="left", va="bottom")
    lim = max_r * 1.15
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Visual field X (deg)   [+ = right]", fontsize=12)
    ax.set_ylabel("Visual field Y (deg)   [+ = up]", fontsize=12)
    ax.set_title(f"True (stimulus) vs measured (mean +/-1 SD) positions\n{results['experiment_name']}",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", mfc="none", mec="black", label="true (stimulus)"),
        Line2D([], [], marker="o", ls="", mfc="gray", mec="black", label="measured mean +/-1 SD"),
    ], loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default=None, help="Experiment dir (default: latest analyzed)")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir) if args.exp_dir else find_latest_experiment()
    print(f"[analyze] experiment: {exp_dir.name}")
    results = load_results(exp_dir)
    outdir = exp_dir / "consolidated_analysis"
    rings = assign_rings(results)

    summary = plot_error_vs_ecc(results, outdir / "error_vs_eccentricity.png", rings)
    plot_true_vs_measured(results, outdir / "true_vs_measured_map.png", rings)

    print("\nError-by-eccentricity (mean +/-1 SD, deg):")
    for ecc, (m, s) in summary.items():
        print(f"  ecc {ecc:>6.2f}°:  {m:.3f} +/- {s:.3f}")


if __name__ == "__main__":
    main()
