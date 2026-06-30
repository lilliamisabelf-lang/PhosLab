"""Visualización ilustrativa del método de mapeo relativo por pares.

Genera figuras para *evaluar visualmente* el método (no solo medianas):

  1. truth_map.png            — el mapa verdadero del CSV (electrodos en el campo
                                 visual, coloreados por excentricidad/implante).
  2. pair_graph.png           — qué pares se miden (aristas del grafo) para cada
                                 estrategia, superpuestas al mapa.
  3. recovery_overlay.png     — verdad vs recuperado (LSQ y MDS), con vectores de
                                 error por electrodo y elipse de error agregado.
  4. recovery_vs_npairs.png   — pequeños múltiplos: el mapa recuperado a 1×, 1.5×,
                                 2×, 4× pares mínimos → se VE cómo se ordena.
  5. convergence.png          — traza de stress de SMACOF (MDS) y de error LSQ vs
                                 nº de pares, lado a lado.
  6. error_vs_pairs_detail.png— error mediano + nube de repeticiones Monte-Carlo,
                                 LSQ vs MDS, con suelo de ruido.

No necesita hardware. Reutiliza el cargador, el modelo de ruido y los
estimadores de simulate_pair_count.py / relative_map.py.

Uso (PowerShell):
    cd percept_mapper
    uv run python scripts/analysis/visualize_pair_mapping.py `
        --csv config/synthetic_4ecc_3elec_17deg.csv `
        --out-dir comparison_results/pair_mapping_viz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.relative_map import (
    build_pair_graph,
    embed_displacement_lsq,
    embed_mds,
    align_procrustes,
    map_error,
)
from scripts.analysis.simulate_pair_count import (
    load_truth_csv,
    simulate_observations,
    pick_anchors,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 200,
})


# ---------------------------------------------------------------------------
# small drawing helpers
# ---------------------------------------------------------------------------

def _setup_vf_axis(ax, truth, title):
    """Eje de campo visual: cuadrado, centrado en fijación, con cruz y anillos."""
    lim = np.max(np.abs(truth)) * 1.15 + 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
    ax.axvline(0, color="gray", lw=0.8, alpha=0.6)
    # anillos de iso-excentricidad
    for r in _nice_rings(lim):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="gray",
                                 lw=0.6, alpha=0.35, ls=":"))
        ax.text(0, r, f"{r:g}°", color="gray", fontsize=7,
                ha="center", va="bottom", alpha=0.7)
    ax.plot(0, 0, "+", color="black", ms=10, mew=1.5)
    ax.set_xlabel("x (°)")
    ax.set_ylabel("y (°)")
    ax.set_title(title)


def _nice_rings(lim):
    step = 2 if lim <= 12 else (5 if lim <= 30 else 10)
    return np.arange(step, lim, step)


def _ecc_colors(truth):
    ecc = np.hypot(truth[:, 0], truth[:, 1])
    return ecc


# ---------------------------------------------------------------------------
# figure 1 — truth map
# ---------------------------------------------------------------------------

def fig_truth_map(truth, labels, path):
    fig, ax = plt.subplots(figsize=(7, 7))
    _setup_vf_axis(ax, truth, "Mapa verdadero (CSV)")
    ecc = _ecc_colors(truth)
    sc = ax.scatter(truth[:, 0], truth[:, 1], c=ecc, cmap="viridis",
                    s=140, edgecolors="black", linewidths=1.2, zorder=5)
    for i, (x, y) in enumerate(truth):
        ax.annotate(str(i), (x, y), fontsize=7, color="white",
                    ha="center", va="center", zorder=6)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("Excentricidad (°)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure 2 — pair graph per strategy
# ---------------------------------------------------------------------------

def fig_pair_graph(truth, path):
    strategies = [
        ("spanning", dict(strategy="spanning")),
        ("knn", dict(strategy="knn", k=3)),
        ("knn+struts", dict(strategy="knn+struts", k=3, n_long=6)),
        ("complete", dict(strategy="complete")),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    for ax, (name, kw) in zip(axes, strategies):
        edges = build_pair_graph(truth, **kw)
        _setup_vf_axis(ax, truth, f"{name}\n{len(edges)} pares")
        for (i, j) in edges:
            ax.plot([truth[i, 0], truth[j, 0]], [truth[i, 1], truth[j, 1]],
                    color="C0", lw=0.8, alpha=0.45, zorder=2)
        ax.scatter(truth[:, 0], truth[:, 1], c="crimson", s=70,
                   edgecolors="black", linewidths=1.0, zorder=5)
    fig.suptitle("Topología del grafo de pares por estrategia", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure 3 — recovery overlay (truth vs recovered) with error vectors
# ---------------------------------------------------------------------------

def _recover(truth, edges, vecs, dists, anchors):
    lsq, _ = embed_displacement_lsq(edges, vecs, truth.shape[0], anchors=anchors)
    mds_raw, _ = embed_mds(edges, dists, truth.shape[0], method="smacof",
                           n_init=4, seed=0)
    try:
        mds, _ = align_procrustes(mds_raw, anchors, allow_scale=True,
                                  allow_reflection=True)
    except ValueError:
        mds = mds_raw
    return lsq, mds


def _draw_recovery(ax, truth, est, anchors, title):
    _setup_vf_axis(ax, truth, title)
    # vectores de error verdad→estimado
    for i in range(truth.shape[0]):
        if np.any(np.isnan(est[i])):
            continue
        ax.annotate("", xy=(est[i, 0], est[i, 1]), xytext=(truth[i, 0], truth[i, 1]),
                    arrowprops=dict(arrowstyle="->", color="crimson",
                                    lw=1.2, alpha=0.8), zorder=4)
    ax.scatter(truth[:, 0], truth[:, 1], c="black", s=70, marker="o",
               label="verdad", zorder=5)
    ax.scatter(est[:, 0], est[:, 1], c="crimson", s=55, marker="x",
               label="recuperado", zorder=6)
    # anclajes destacados
    aidx = list(anchors.keys())
    ax.scatter(truth[aidx, 0], truth[aidx, 1], facecolors="none",
               edgecolors="dodgerblue", s=240, linewidths=2.2,
               label="anclaje", zorder=7)
    err = map_error(est, truth)
    ax.text(0.02, 0.98,
            f"mediana {err['median']:.2f}°\np95 {err['p95']:.2f}°\nmáx {err['max']:.2f}°",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def fig_recovery_overlay(truth, cfg, rng, path):
    edges0 = build_pair_graph(truth, strategy="knn+struts", k=3, n_long=6)
    anchors = pick_anchors(truth, cfg["n_anchors"], rng)
    e2, vecs, dists = simulate_observations(
        truth, edges0, sigma_draw=cfg["sigma_draw"], serial_bias=cfg["serial_bias"],
        warp_gain=cfg["warp_gain"], rng=rng, both_directions=cfg["both_directions"])
    lsq, mds = _recover(truth, e2, vecs, dists, anchors)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    _draw_recovery(axes[0], truth, lsq, anchors,
                   f"Displacement LSQ  ({len(edges0)} pares, {cfg['n_anchors']} anclajes)")
    _draw_recovery(axes[1], truth, mds, anchors,
                   f"MDS (distances)  ({len(edges0)} pares, {cfg['n_anchors']} anclajes)")
    fig.suptitle("Verdad vs. mapa recuperado (flechas = error por electrodo)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure 4 — recovered map as #pairs grows (small multiples)
# ---------------------------------------------------------------------------

def fig_recovery_vs_npairs(truth, cfg, rng, path):
    full = build_pair_graph(truth, strategy="complete")
    n = truth.shape[0]
    base = max(2, n - 1)
    targets = sorted(set(int(min(len(full), round(base * m)))
                         for m in (1.0, 1.5, 2.0, 4.0)))
    anchors = pick_anchors(truth, cfg["n_anchors"], rng)

    fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 5.6))
    if len(targets) == 1:
        axes = [axes]
    for ax, npairs in zip(axes, targets):
        e = list(full)
        rng.shuffle(e)
        e = e[:npairs]
        e2, vecs, dists = simulate_observations(
            truth, e, sigma_draw=cfg["sigma_draw"], serial_bias=cfg["serial_bias"],
            warp_gain=cfg["warp_gain"], rng=rng, both_directions=cfg["both_directions"])
        lsq, _info = embed_displacement_lsq(e2, vecs, n, anchors=anchors)
        err = map_error(lsq, truth)
        conn = "" if _info["n_components"] == 1 else f" · {_info['n_components']} comp!"
        _setup_vf_axis(ax, truth, f"{npairs} pares  (med {err['median']:.2f}°{conn})")
        for i in range(n):
            if np.any(np.isnan(lsq[i])):
                continue
            ax.annotate("", xy=(lsq[i, 0], lsq[i, 1]),
                        xytext=(truth[i, 0], truth[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="crimson",
                                        lw=1.0, alpha=0.75), zorder=4)
        ax.scatter(truth[:, 0], truth[:, 1], c="black", s=45, zorder=5)
        ax.scatter(lsq[:, 0], lsq[:, 1], c="crimson", s=40, marker="x", zorder=6)
    fig.suptitle("Mapa recuperado (LSQ) según crece el nº de pares", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure 5 — convergence (SMACOF stress trace + LSQ error vs pairs)
# ---------------------------------------------------------------------------

def fig_convergence(truth, cfg, rng, path):
    full = build_pair_graph(truth, strategy="complete")
    anchors = pick_anchors(truth, cfg["n_anchors"], rng)
    e2, vecs, dists = simulate_observations(
        truth, full, sigma_draw=cfg["sigma_draw"], serial_bias=cfg["serial_bias"],
        warp_gain=cfg["warp_gain"], rng=rng, both_directions=cfg["both_directions"])
    _coords, info = embed_mds(full, dists, truth.shape[0], method="smacof",
                              n_init=4, seed=0, return_trace=True)
    trace = info.get("stress_trace", [])

    n = truth.shape[0]
    grid = np.unique(np.clip(
        np.round(np.geomspace(max(2, n - 1), len(full), 10)).astype(int),
        2, len(full)))
    lsq_curve = []
    for npairs in grid:
        e = list(full)
        rng.shuffle(e)
        e = e[:npairs]
        ee, vv, _dd = simulate_observations(
            truth, e, sigma_draw=cfg["sigma_draw"], serial_bias=cfg["serial_bias"],
            warp_gain=cfg["warp_gain"], rng=rng, both_directions=cfg["both_directions"])
        est, _ = embed_displacement_lsq(ee, vv, n, anchors=anchors)
        lsq_curve.append(map_error(est, truth)["median"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(range(1, len(trace) + 1), trace, "o-", color="C1", ms=3)
    axes[0].set_xlabel("Iteración SMACOF")
    axes[0].set_ylabel("Stress")
    axes[0].set_title("Convergencia del MDS (stress por iteración)")
    axes[0].set_yscale("log")

    axes[1].plot(grid, lsq_curve, "o-", color="C0")
    axes[1].axhline(cfg["sigma_draw"], ls=":", color="gray",
                    label=f"σ_draw = {cfg['sigma_draw']}°")
    axes[1].set_xlabel("Nº de pares")
    axes[1].set_ylabel("Error mediano LSQ (°)")
    axes[1].set_title("Convergencia del LSQ (error vs nº de pares)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure 6 — error vs pairs with Monte-Carlo cloud
# ---------------------------------------------------------------------------

def fig_error_vs_pairs_detail(truth, cfg, rng, path):
    full = build_pair_graph(truth, strategy="complete")
    n = truth.shape[0]
    grid = np.unique(np.clip(
        np.round(np.geomspace(max(2, n - 1), len(full), 10)).astype(int),
        2, len(full)))
    reps = cfg["reps"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for key, color, marker, run in (
        ("LSQ", "C0", "o", "lsq"),
        ("MDS", "C1", "s", "mds"),
    ):
        meds = []
        for npairs in grid:
            errs = []
            for _ in range(reps):
                e = list(full)
                rng.shuffle(e)
                e = e[:npairs]
                ee, vv, dd = simulate_observations(
                    truth, e, sigma_draw=cfg["sigma_draw"],
                    serial_bias=cfg["serial_bias"], warp_gain=cfg["warp_gain"],
                    rng=rng, both_directions=cfg["both_directions"])
                anchors = pick_anchors(truth, cfg["n_anchors"], rng)
                if run == "lsq":
                    est, _ = embed_displacement_lsq(ee, vv, n, anchors=anchors)
                else:
                    raw, _ = embed_mds(ee, dd, n, method="smacof", n_init=3, seed=0)
                    try:
                        est, _ = align_procrustes(raw, anchors, allow_scale=True)
                    except ValueError:
                        est = raw
                errs.append(map_error(est, truth)["median"])
            meds.append(np.nanmedian(errs))
            jitter = (np.random.default_rng(int(npairs)).uniform(-0.4, 0.4, len(errs)))
            ax.scatter(np.full(len(errs), npairs) + jitter, errs, s=8,
                       color=color, alpha=0.18, zorder=2)
        ax.plot(grid, meds, marker + "-", color=color, lw=2, ms=7,
                label=f"{key} (mediana)", zorder=5)

    ax.axhline(cfg["sigma_draw"], ls=":", color="gray",
               label=f"σ_draw = {cfg['sigma_draw']}° (suelo de ruido)")
    ax.set_xlabel("Número de pares medidos")
    ax.set_ylabel("Error mediano de localización (°)")
    ax.set_title("Error vs nº de pares — nube Monte-Carlo + mediana")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path,
                   default=Path("comparison_results/pair_mapping_viz"))
    p.add_argument("--implant-id", default="all")
    p.add_argument("--reps", type=int, default=40)
    p.add_argument("--sigma-draw", type=float, default=0.7)
    p.add_argument("--serial-bias", type=float, default=0.3)
    p.add_argument("--warp-gain", type=float, default=0.0)
    p.add_argument("--n-anchors", type=int, default=3)
    p.add_argument("--both-directions", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    truth, labels = load_truth_csv(args.csv, args.implant_id)
    valid = ~np.any(np.isnan(truth), axis=1)
    truth, labels = truth[valid], [l for l, v in zip(labels, valid) if v]
    n = truth.shape[0]
    if n < 3:
        print(f"✗ Se necesitan ≥3 electrodos válidos; el CSV tiene {n}.")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(reps=args.reps, sigma_draw=args.sigma_draw,
               serial_bias=args.serial_bias, warp_gain=args.warp_gain,
               n_anchors=args.n_anchors, both_directions=args.both_directions)

    print("=" * 70)
    print("VISUALIZACIÓN — Mapeo relativo por pares")
    print("=" * 70)
    print(f"CSV: {args.csv}   electrodos: {n}   pares posibles: {n*(n-1)//2}")
    print(f"σ_draw={args.sigma_draw}°  serial_bias={args.serial_bias}°  "
          f"anclajes={args.n_anchors}  both_dir={args.both_directions}")
    print("-" * 70)

    od = args.out_dir
    fig_truth_map(truth, labels, od / "truth_map.png")
    print("✓ truth_map.png")
    fig_pair_graph(truth, od / "pair_graph.png")
    print("✓ pair_graph.png")
    fig_recovery_overlay(truth, cfg, np.random.default_rng(args.seed),
                         od / "recovery_overlay.png")
    print("✓ recovery_overlay.png")
    fig_recovery_vs_npairs(truth, cfg, np.random.default_rng(args.seed + 1),
                           od / "recovery_vs_npairs.png")
    print("✓ recovery_vs_npairs.png")
    fig_convergence(truth, cfg, np.random.default_rng(args.seed + 2),
                    od / "convergence.png")
    print("✓ convergence.png")
    fig_error_vs_pairs_detail(truth, cfg, np.random.default_rng(args.seed + 3),
                              od / "error_vs_pairs_detail.png")
    print("✓ error_vs_pairs_detail.png")

    print(f"\n✓ 6 figuras en: {od}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
