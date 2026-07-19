"""Experimento 1 — Error de localización vs. Excentricidad.

Produce dos figuras en --out-dir:
  1. error_vs_ecc_boxplot.png  — boxplot por anillo de excentricidad con puntos superpuestos
  2. error_vs_ecc_scatter.png  — scatter por ensayo con línea de regresión lineal

Además imprime en consola el r de Pearson entre excentricidad y error, con su
IC95% y p-valor calculados por bootstrap por electrodo (ver stats_utils.py) —
no por ensayo individual, ya que las 10 repeticiones de un mismo electrodo no
son observaciones independientes entre sí.

Uso (PowerShell):
    cd percept_mapper; uv run python scripts/analysis/plot_error_vs_ecc.py --session mapping_experiments/mapping_mapeo_multiples_electrodo_20260626_165130 --out-dir comparison_results/exp1_error_vs_ecc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from stats_utils import report_r, format_r_report

# ---------------------------------------------------------------------------
# Estilo global para TFG
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

# Paleta accesible (colorblind-safe) — un color por anillo
RING_COLORS = {
    2:  "#4C72B0",
    4:  "#DD8452",
    6:  "#55A868",
    8:  "#C44E52",
    12: "#8172B2",
}

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_data(session_dir: Path) -> tuple[list[float], list[float], list[float]]:
    """Devuelve (eccs, errors_per_trial, electrode_eccs_mean).

    eccs             — excentricidad del estímulo por ensayo (°)
    errors_per_trial — error euclidiano por ensayo (°)
    """
    results_file = session_dir / "consolidated_analysis" / "consolidated_results.json"
    if not results_file.exists():
        raise SystemExit(f"No se encontró {results_file}\n"
                         "Asegúrate de haber ejecutado el análisis consolidado primero.")

    data = json.loads(results_file.read_text(encoding="utf-8"))

    eccs: list[float] = []
    errors: list[float] = []

    for e in data["electrodes"].values():
        stim_deg = e.get("stimulation_position_deg")
        if not stim_deg:
            continue
        ecc = float(np.hypot(stim_deg[0], stim_deg[1]))

        for rep in e.get("per_repetition_metrics", []):
            err = rep.get("distance_to_stim_deg")
            if err is not None:
                eccs.append(ecc)
                errors.append(float(err))

    return eccs, errors


def round_ring(ecc: float) -> int:
    """Redondea la excentricidad al anillo más cercano."""
    rings = sorted(RING_COLORS.keys())
    return min(rings, key=lambda r: abs(r - ecc))


# ---------------------------------------------------------------------------
# Figura 1 — Boxplot por anillo
# ---------------------------------------------------------------------------

def plot_boxplot(eccs: list[float], errors: list[float], out_path: Path) -> None:
    rings = sorted(RING_COLORS.keys())
    ring_errors: dict[int, list[float]] = {r: [] for r in rings}
    for ecc, err in zip(eccs, errors):
        ring_errors[round_ring(ecc)].append(err)

    fig, ax = plt.subplots(figsize=(7, 5))

    positions = list(range(len(rings)))
    data_by_ring = [ring_errors[r] for r in rings]

    bp = ax.boxplot(
        data_by_ring,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="x", markersize=5, alpha=0.4),
        showfliers=False,
    )

    for patch, ring in zip(bp["boxes"], rings):
        patch.set_facecolor(RING_COLORS[ring])
        patch.set_alpha(0.65)

    # Jitter overlay
    rng = np.random.default_rng(42)
    for pos, ring in zip(positions, rings):
        vals = ring_errors[ring]
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(
            [pos + j for j in jitter], vals,
            color=RING_COLORS[ring], s=22, alpha=0.55,
            edgecolors="white", linewidths=0.4, zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{r:d}°\n(n={len(ring_errors[r])})" for r in rings],
        linespacing=1.4,
    )
    ax.set_xlabel("Excentricidad del electrodo (°)", labelpad=10)
    ax.set_ylabel("Error de localización (°)")
    # Mediana anotada encima del bigote superior (cap superior de cada caja)
    # bp["caps"] tiene 2 elementos por caja: [cap_inf, cap_sup, cap_inf, cap_sup, ...]
    for i, ring in enumerate(rings):
        top_cap_y = bp["caps"][2 * i + 1].get_ydata()[0]
        med = float(np.median(ring_errors[ring]))
        ax.text(
            positions[i], top_cap_y + 0.04,
            f"Md={med:.2f}°",
            ha="center", va="bottom", fontsize=8.5, color="#222222",
        )

    fig.subplots_adjust(bottom=0.15, top=0.88)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Guardado: {out_path}")


# ---------------------------------------------------------------------------
# Figura 2 — Scatter por ensayo con regresión
# ---------------------------------------------------------------------------

def plot_scatter(eccs: list[float], errors: list[float], out_path: Path) -> None:
    eccs_arr = np.array(eccs)
    errors_arr = np.array(errors)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Puntos coloreados por anillo
    colors = [RING_COLORS[round_ring(e)] for e in eccs]
    ax.scatter(eccs_arr, errors_arr,
               c=colors, s=28, alpha=0.55,
               edgecolors="white", linewidths=0.4, zorder=3)

    # Regresión lineal
    coeffs = np.polyfit(eccs_arr, errors_arr, 1)
    x_line = np.linspace(eccs_arr.min() - 0.5, eccs_arr.max() + 0.5, 200)
    y_line = np.polyval(coeffs, x_line)
    r = float(np.corrcoef(eccs_arr, errors_arr)[0, 1])
    ax.plot(x_line, y_line, color="#333333", linewidth=1.8, zorder=4,
            label=f"Regresión lineal (r = {r:.2f})")

    ax.set_xlabel("Excentricidad del electrodo (°)")
    ax.set_ylabel("Error de localización (°)")
    # Leyenda de anillos
    legend_patches = [
        mpatches.Patch(facecolor=RING_COLORS[r], alpha=0.75, label=f"{r}°")
        for r in sorted(RING_COLORS)
    ]
    legend_patches.append(
        plt.Line2D([0], [0], color="#333333", linewidth=1.8, label=f"Regresión (r={r:.2f})")
    )
    ax.legend(handles=legend_patches, title="Excentricidad",
              loc="upper left", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Guardado: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Exp 1 — Error de localización vs. excentricidad."
    )
    ap.add_argument(
        "--session", required=True,
        help="Ruta a la carpeta de sesión (p.ej. mapping_experiments/mapping_...)",
    )
    ap.add_argument(
        "--out-dir", default=None,
        help="Carpeta de salida (default: comparison_results/exp1_error_vs_ecc)",
    )
    args = ap.parse_args()

    session_dir = Path(args.session)
    out_dir = Path(args.out_dir) if args.out_dir else Path("comparison_results/exp1_error_vs_ecc")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Exp 1] Sesión:  {session_dir}")
    print(f"[Exp 1] Salida:  {out_dir}\n")

    eccs, errors = load_data(session_dir)
    print(f"[Exp 1] Ensayos cargados: {len(errors)}")
    print(f"[Exp 1] Error medio global: {np.mean(errors):.3f}°  |  mediana: {np.median(errors):.3f}°\n")

    results_file = session_dir / "consolidated_analysis" / "consolidated_results.json"
    data = json.loads(results_file.read_text(encoding="utf-8"))
    rep = report_r(data)
    print(f"[Exp 1] {format_r_report('Correlación excentricidad-error', rep)}")
    print("[Exp 1] IC y p calculados remuestreando electrodos completos (no ensayos), "
          "porque las repeticiones de un mismo electrodo no son independientes entre sí.\n")

    plot_boxplot(eccs, errors, out_dir / "error_vs_ecc_boxplot.png")
    plot_scatter(eccs, errors, out_dir / "error_vs_ecc_scatter.png")

    print("\n[OK] Exp 1 completado.")


if __name__ == "__main__":
    main()
