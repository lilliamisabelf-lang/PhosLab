"""Compara N sesiones con etiquetas libres (métodos de mapeo, implantes, etc.).

Produce en --out-dir:
  1. error_comparison.png  — boxplots agrupados por excentricidad, una caja por sesión
  2. map_comparison.png    — overlay de posiciones medias (verdaderas vs medidas)

Uso (PowerShell):
    cd percept_mapper; uv run python scripts/analysis/compare_sessions.py --sessions mapping_experiments/A mapping_experiments/B mapping_experiments/C --labels "Absolute" "Relative" "Forced adjustment" --out-dir comparison_results/exp4_mapping_method
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

# ---------------------------------------------------------------------------
# Estilo global (coherente con el resto de scripts de análisis)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# Paleta para hasta 6 sesiones (colorblind-safe)
SESSION_COLORS = [
    "#4C72B0",  # azul
    "#DD8452",  # naranja
    "#55A868",  # verde
    "#C44E52",  # rojo
    "#8172B2",  # morado
    "#937860",  # marrón
]
SESSION_MARKERS = ["o", "s", "D", "^", "v", "P"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(session_dir: Path) -> dict:
    f = session_dir / "consolidated_analysis" / "consolidated_results.json"
    if not f.exists():
        raise SystemExit(f"No se encontro {f}\nEjecuta primero el analisis consolidado.")
    return json.loads(f.read_text(encoding="utf-8"))


def assign_rings(results: dict, tol: float = 1.0) -> dict[int, float]:
    """Agrupa electrodos por anillo de excentricidad."""
    raw = {
        int(k): math.hypot(*v["stimulation_position_deg"])
        for k, v in results["electrodes"].items()
    }
    order = sorted(raw, key=lambda e: raw[e])
    rings: dict[int, float] = {}
    cluster: list[int] = []

    def flush(cl):
        m = round(sum(raw[e] for e in cl) / len(cl), 2)
        for e in cl:
            rings[e] = m

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
# Figura 1 — Boxplots agrupados por excentricidad
# ---------------------------------------------------------------------------

def plot_error_comparison(
    sessions: list[tuple[str, dict[float, list[float]]]],
    eccs: list[float],
    out: Path,
) -> None:
    n = len(sessions)
    width = min(0.28, 0.85 / n)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * (width + 0.04)
    cat_pos = list(range(len(eccs)))
    ecc_to_cat = {e: i for i, e in enumerate(eccs)}

    fig, ax = plt.subplots(figsize=(max(9, 2 * len(eccs) * n), 6))
    rng = np.random.default_rng(42)

    for i, (label, groups) in enumerate(sessions):
        color = SESSION_COLORS[i % len(SESSION_COLORS)]
        positions = [ecc_to_cat[e] + offsets[i] for e in eccs]
        data = [groups.get(e, []) for e in eccs]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            medianprops=dict(color="black", lw=1.5),
            boxprops=dict(facecolor=color, alpha=0.35, edgecolor="black", lw=1.0),
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
                np.array([pos] * len(pts)) + jitter, pts,
                color=color, s=22, edgecolors="black",
                linewidths=0.4, alpha=0.70, zorder=5,
            )
            top_cap_y = bp["caps"][2 * j + 1].get_ydata()[0]
            med = float(np.median(pts))
            ax.text(pos, top_cap_y + 0.04, f"Md={med:.2f}°",
                    ha="center", va="bottom", fontsize=7, color="black")

    def _n_label(e):
        parts = "/".join(str(len(g.get(e, []))) for _, g in sessions)
        return f"{round(e):d}°\n(n={parts})"

    ax.set_xticks(cat_pos)
    ax.set_xticklabels([_n_label(e) for e in eccs])
    ax.set_xlabel("Excentricidad del electrodo (°)", labelpad=8)
    ax.set_ylabel("Error de localizacion (°)")
    y_cap = 4.0
    ax.set_ylim(0, y_cap)

    # Indicar cuántos valores quedan fuera de escala
    all_vals = [v for _, g in sessions for pts in g.values() for v in pts]
    n_out_total = sum(1 for v in all_vals if v > y_cap)
    if n_out_total:
        ax.text(0.99, 0.99, f"{n_out_total} valor(es) fuera de escala (>{y_cap}°)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#888888",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.8))

    legend_handles = [
        Line2D([], [], color=SESSION_COLORS[i % len(SESSION_COLORS)],
               lw=8, alpha=0.45, label=label)
        for i, (label, _) in enumerate(sessions)
    ]
    legend_handles += [
        Line2D([], [], color="black", lw=1.5, label="mediana"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10)

    fig.subplots_adjust(bottom=0.15)
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] Guardado: {out}")


# ---------------------------------------------------------------------------
# Figura 2 — Overlay de posiciones (verdaderas vs medidas)
# ---------------------------------------------------------------------------

def plot_map_comparison(
    sessions: list[tuple[str, dict]],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    max_r = 1.0
    eccs_seen: set[float] = set()

    ref_results = sessions[0][1]
    first_true = True
    for k, rec in ref_results["electrodes"].items():
        tx, ty = rec["stimulation_position_deg"]
        ecc = math.hypot(tx, ty)
        eccs_seen.add(round(ecc, 1))
        max_r = max(max_r, abs(tx), abs(ty))
        ax.scatter([tx], [ty], marker="s", s=55, facecolors="none",
                   edgecolors="black", linewidths=1.3, zorder=4,
                   label="Estimulo (verdad)" if first_true else "")
        ax.annotate(str(k), (tx, ty), textcoords="offset points",
                    xytext=(5, 4), fontsize=7, color="#444444")
        first_true = False

    for i, (label, results) in enumerate(sessions):
        color = SESSION_COLORS[i % len(SESSION_COLORS)]
        marker = SESSION_MARKERS[i % len(SESSION_MARKERS)]
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
    ax.set_xlabel("Campo visual X (°)   [+ = derecha]")
    ax.set_ylabel("Campo visual Y (°)   [+ = arriba]")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] Guardado: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compara N sesiones con etiquetas libres."
    )
    ap.add_argument("--sessions", nargs="+", required=True,
                    help="Rutas a las carpetas de sesion")
    ap.add_argument("--labels", nargs="+", required=True,
                    help="Etiquetas para cada sesion (mismo orden)")
    ap.add_argument("--out-dir", default=None,
                    help="Carpeta de salida")
    args = ap.parse_args()

    if len(args.sessions) != len(args.labels):
        raise SystemExit("--sessions y --labels deben tener el mismo numero de elementos.")

    out_dir = Path(args.out_dir) if args.out_dir else Path("comparison_results/compare_sessions")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare_sessions] Sesiones: {args.labels}")
    print(f"[compare_sessions] Salida:   {out_dir}\n")

    loaded_results = [(label, load_results(Path(s)))
                      for label, s in zip(args.labels, args.sessions)]

    rings_ref = assign_rings(loaded_results[0][1])
    error_sessions = [(label, collect_errors(r, assign_rings(r)))
                      for label, r in loaded_results]
    eccs = sorted({e for _, g in error_sessions for e in g})

    # Resumen numérico
    print(f"{'ecc':>8}", end="")
    for label, _ in error_sessions:
        print(f"  {label:>22}", end="")
    print()
    for e in eccs:
        print(f"{e:>7.2f}°", end="")
        for _, groups in error_sessions:
            pts = groups.get(e, [])
            if pts:
                print(f"  {np.mean(pts):6.3f} ± {np.std(pts):.3f}°", end="")
            else:
                print(f"  {'':>22}", end="")
        print()
    print()

    plot_error_comparison(error_sessions, eccs,
                          out_dir / "error_comparison.png")
    plot_map_comparison(loaded_results,
                        out_dir / "map_comparison.png")

    print("\n[OK] Exp 4 completado.")


if __name__ == "__main__":
    main()
