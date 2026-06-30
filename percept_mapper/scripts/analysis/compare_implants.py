"""Exp 3 — Comparación de configuraciones de implante.

Compara hasta N sesiones con distintos tipos de implante en términos de:
  - Cobertura del campo visual (posiciones de los electrodos)
  - Error de predicción del atlas de Benson (distancia predicción → respuesta)

Produce en --out-dir:
  1. coverage_map.png   — distribución espacial de electrodos + posiciones medidas
  2. error_comparison.png — boxplots de error por tipo de implante

Uso (PowerShell):
    cd percept_mapper
    uv run python scripts/analysis/compare_implants.py `
        --sessions mapping_experiments/utah mapping_experiments/comb mapping_experiments/thread `
        --labels   "4x Utah" "Comb 10x10" "Thread-1024" `
        --out-dir  comparison_results/exp3_implants
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
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Estilo global
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.30,
    "grid.linestyle":    "--",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

SESSION_COLORS  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
SESSION_MARKERS = ["o", "s", "D", "^", "v", "P"]


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_results(session_dir: Path) -> dict:
    f = session_dir / "consolidated_analysis" / "consolidated_results.json"
    if not f.exists():
        raise SystemExit(f"No se encontró {f}\nEjecuta primero el análisis consolidado.")
    return json.loads(f.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Convex hull para área de cobertura (en grados²)
# ---------------------------------------------------------------------------

def _convex_hull_area(points: list[tuple[float, float]]) -> float:
    """Área del convex hull de una lista de puntos (deg²). 0 si < 3 puntos."""
    if len(points) < 3:
        return 0.0
    pts = np.array(points)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        return float(hull.volume)  # en 2D, .volume es el área
    except Exception:
        return 0.0


def _hull_vertices(points: list[tuple[float, float]]) -> np.ndarray | None:
    if len(points) < 3:
        return None
    pts = np.array(points)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        verts = pts[np.append(hull.vertices, hull.vertices[0])]
        return verts
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Figura 1 — Mapa de cobertura del campo visual
# ---------------------------------------------------------------------------

def plot_coverage_map(
    sessions: list[tuple[str, dict]],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))
    max_r = 1.0
    eccs_seen: set[float] = set()

    for i, (label, results) in enumerate(sessions):
        color  = SESSION_COLORS[i % len(SESSION_COLORS)]
        marker = SESSION_MARKERS[i % len(SESSION_MARKERS)]

        true_pts: list[tuple[float, float]] = []
        meas_pts: list[tuple[float, float]] = []

        for k, rec in results["electrodes"].items():
            tx, ty = rec["stimulation_position_deg"]
            mx = rec["mean_position_deg"]["x"]
            my = rec["mean_position_deg"]["y"]

            ecc = round(math.hypot(tx, ty), 1)
            eccs_seen.add(ecc)
            max_r = max(max_r, abs(tx), abs(ty), abs(mx), abs(my))

            true_pts.append((tx, ty))
            meas_pts.append((mx, my))

            # línea estímulo → media
            ax.plot([tx, mx], [ty, my], color=color, lw=0.8, alpha=0.45, zorder=3)

        # convex hull de posiciones verdaderas (cobertura)
        hull_v = _hull_vertices(true_pts)
        if hull_v is not None:
            ax.fill(hull_v[:, 0], hull_v[:, 1],
                    color=color, alpha=0.08, zorder=1)
            ax.plot(hull_v[:, 0], hull_v[:, 1],
                    color=color, lw=1.2, ls="--", alpha=0.55, zorder=2)

        # posiciones verdaderas (cuadrados huecos negros, anotados con implante)
        txs, tys = zip(*true_pts)
        ax.scatter(txs, tys,
                   marker="s", s=50, facecolors="none",
                   edgecolors="black", linewidths=1.1, zorder=5)

        # posiciones medidas
        mxs, mys = zip(*meas_pts)
        cents = np.array(list(results["electrodes"].values()), dtype=object)
        errors_deg = [rec["distance_mean_to_stimulus_deg"]
                      for rec in results["electrodes"].values()]
        ax.scatter(mxs, mys,
                   marker=marker, s=60, color=color,
                   edgecolors="black", linewidths=0.5,
                   zorder=6, label=label)

    # anillos de excentricidad
    for r in sorted(eccs_seen):
        circle = plt.Circle((0, 0), r, fill=False, ls=":", lw=0.9,
                             ec="#aaaaaa", zorder=0)
        ax.add_patch(circle)
        ax.text(r * math.cos(math.radians(38)),
                r * math.sin(math.radians(38)),
                f"{r:g}°", fontsize=7, color="#888888")

    lim = max_r * 1.18
    ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
    ax.axvline(0, color="#dddddd", lw=0.8, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Campo visual X (°)   [+ = derecha]")
    ax.set_ylabel("Campo visual Y (°)   [+ = arriba]")
    ax.set_title("Cobertura del campo visual por tipo de implante")

    # leyenda manual
    legend_handles = [
        Line2D([], [], color=SESSION_COLORS[i % len(SESSION_COLORS)],
               marker=SESSION_MARKERS[i % len(SESSION_MARKERS)],
               ms=7, lw=0, mec="black", mew=0.5, label=label)
        for i, (label, _) in enumerate(sessions)
    ]
    legend_handles += [
        Line2D([], [], color="black", marker="s", ms=7,
               lw=0, mec="black", mew=1.1, mfc="none", label="Estímulo (verdad)"),
        Line2D([], [], color="#888888", lw=0.8, alpha=0.5,
               label="Error individual"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] Guardado: {out}")


# ---------------------------------------------------------------------------
# Figura 2 — Boxplots de error por tipo de implante
# ---------------------------------------------------------------------------

def collect_all_errors(results: dict) -> list[float]:
    """Todos los errores per-repetición de una sesión."""
    errors = []
    for rec in results["electrodes"].values():
        for rep in rec.get("per_repetition_metrics", []) or []:
            errors.append(float(rep["distance_to_stim_deg"]))
    return errors


def plot_error_comparison(
    sessions: list[tuple[str, dict]],
    out: Path,
) -> None:
    n = len(sessions)
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * n), 6))
    rng = np.random.default_rng(42)

    all_data  = [(label, collect_all_errors(r)) for label, r in sessions]
    positions = list(range(n))
    data_only = [d for _, d in all_data]
    labels    = [lbl for lbl, _ in all_data]
    colors    = [SESSION_COLORS[i % len(SESSION_COLORS)] for i in range(n)]

    bp = ax.boxplot(
        data_only,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color="black", lw=1.8),
        boxprops=dict(facecolor="white", edgecolor="black", lw=1.0),
        whiskerprops=dict(color="black", lw=1.0),
        capprops=dict(color="black", lw=1.0),
        flierprops=dict(marker="", alpha=0),
    )

    # colorear las cajas
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    for i, (pos, color, pts) in enumerate(zip(positions, colors, data_only)):
        if not pts:
            continue
        jitter = (rng.random(len(pts)) - 0.5) * 0.3
        ax.scatter(
            np.array([pos] * len(pts)) + jitter, pts,
            color=color, s=25, edgecolors="black",
            linewidths=0.4, alpha=0.70, zorder=5,
        )
        # mediana y stats encima del whisker superior
        top_cap_y = bp["caps"][2 * i + 1].get_ydata()[0]
        med = float(np.median(pts))
        mean = float(np.mean(pts))
        n_rep = len(pts)
        ax.text(pos, top_cap_y + 0.015,
                f"Md={med:.2f}°",
                ha="center", va="bottom", fontsize=8, color="black")

    y_cap = max((max(pts) if pts else 0) for pts in data_only) * 1.35 + 0.05
    y_cap = max(y_cap, 0.5)
    ax.set_ylim(0, y_cap)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)

    for i, (_, r) in enumerate(sessions):
        true_pts = [tuple(rec["stimulation_position_deg"])
                    for rec in r["electrodes"].values()]
        area = _convex_hull_area(true_pts)
        area_str = f"{area:.2f}°²" if area > 0 else "N/A"
        ax.text(i, -0.09, f"área = {area_str}",
                ha="center", va="top", fontsize=8, color="#888888",
                transform=ax.get_xaxis_transform())
    ax.set_ylabel("Error de localización (°)")
    ax.set_title("Error de predicción por tipo de implante")

    fig.subplots_adjust(bottom=0.20)
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] Guardado: {out}")


# ---------------------------------------------------------------------------
# Resumen numérico en consola
# ---------------------------------------------------------------------------

def print_summary(sessions: list[tuple[str, dict]]) -> None:
    print(f"\n{'Implante':>20}  {'N electr':>8}  {'Área cob (°²)':>14}  "
          f"{'Error medio (°)':>16}  {'Std (°)':>8}")
    print("-" * 75)
    for label, results in sessions:
        true_pts = [tuple(rec["stimulation_position_deg"])
                    for rec in results["electrodes"].values()]
        errors = collect_all_errors(results)
        area = _convex_hull_area(true_pts)
        n_el = len(results["electrodes"])
        mean_err = float(np.mean(errors)) if errors else float("nan")
        std_err  = float(np.std(errors))  if errors else float("nan")
        print(f"{label:>20}  {n_el:>8}  {area:>14.3f}  {mean_err:>16.4f}  {std_err:>8.4f}°")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Exp 3 — Comparación de tipos de implante."
    )
    ap.add_argument("--sessions", nargs="+", required=True,
                    help="Rutas a las carpetas de sesión")
    ap.add_argument("--labels", nargs="+", required=True,
                    help="Etiquetas para cada sesión (mismo orden)")
    ap.add_argument("--out-dir", default=None,
                    help="Carpeta de salida")
    args = ap.parse_args()

    if len(args.sessions) != len(args.labels):
        raise SystemExit("--sessions y --labels deben tener el mismo número de elementos.")

    out_dir = Path(args.out_dir) if args.out_dir else Path("comparison_results/exp3_implants")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare_implants] Implantes: {args.labels}")
    print(f"[compare_implants] Salida:    {out_dir}\n")

    loaded = [(label, load_results(Path(s)))
              for label, s in zip(args.labels, args.sessions)]

    print_summary(loaded)
    plot_coverage_map(loaded, out_dir / "coverage_map.png")
    plot_error_comparison(loaded, out_dir / "error_comparison.png")
    print("\n[OK] Exp 3 completado.")


if __name__ == "__main__":
    main()
