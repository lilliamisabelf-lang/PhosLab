"""Experimento 5 — Curva de convergencia del modelo Bayesiano.

Reproduce la actualizacion bayesiana trial a trial usando los errores
almacenados en learning_results/bayesian_model.json y muestra como el
sesgo estimado converge hacia el sesgo inyectado artificialmente.

Produce en --out-dir:
  1. learning_curve.png — sesgo estimado (X e Y) por numero de trials
                         vs lineas de referencia del sesgo verdadero

Uso (PowerShell):
    cd percept_mapper; uv run python scripts/analysis/plot_learning_curve.py --session mapping_experiments/mapping_mapeo_multiples_electrodo_20260626_183704 --bias-true 2.0 1.0 --out-dir comparison_results/exp5_learning
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def bayesian_update_series(
    errors: list[float],
    prior_mean: float,
    prior_std: float,
    noise_std: float,
) -> list[float]:
    """Devuelve (medias, stds) del posterior tras cada observacion."""
    post_mean = prior_mean
    post_var  = prior_std ** 2
    noise_var = noise_std ** 2
    means, stds = [], []
    for obs in errors:
        new_var   = 1.0 / (1.0 / post_var + 1.0 / noise_var)
        post_mean = new_var * (post_mean / post_var + obs / noise_var)
        post_var  = new_var
        means.append(post_mean)
        stds.append(np.sqrt(post_var))
    return np.array(means), np.array(stds)


def plot_learning_curve(
    errors_x: list[float],
    errors_y: list[float],
    bias_true: tuple[float, float],
    prior_mean: float,
    prior_std: float,
    noise_std: float,
    out_path: Path,
) -> None:
    n = len(errors_x)
    trials = np.arange(1, n + 1)

    means_x, stds_x = bayesian_update_series(errors_x, prior_mean, prior_std, noise_std)
    means_y, stds_y = bayesian_update_series(errors_y, prior_mean, prior_std, noise_std)

    COLOR_X = "#2166AC"
    COLOR_Y = "#B2182B"

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.subplots_adjust(hspace=0.45)

    for ax, means, stds, bias, color, axis_label in [
        (axes[0], means_x, stds_x, bias_true[0], COLOR_X, "X (horizontal)"),
        (axes[1], means_y, stds_y, bias_true[1], COLOR_Y, "Y (vertical)"),
    ]:
        # Banda de incertidumbre ±1σ (68% CI posterior)
        ax.fill_between(trials, means - stds, means + stds,
                        color=color, alpha=0.18, label="±1σ posterior")

        # Media posterior
        ax.plot(trials, means, color=color, lw=2.0,
                label="Media posterior")

        # Sesgo verdadero
        ax.axhline(bias, color="black", lw=1.3, ls="--",
                   label=f"Sesgo inyectado = {bias}°")

        ax.set_ylabel(f"Sesgo estimado {axis_label} (°)")
        ax.set_xlim(1, n)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    axes[1].set_xlabel("Numero de ensayos acumulados")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Guardado: {out_path}")


def plot_error_reduction(metrics: dict, out_path: Path) -> None:
    """Barras de error medio (original / Bayesiano / Red neuronal) con std."""
    split = "test" if "test" in metrics else "train"
    data = metrics[split]

    labels  = ["Sin corrección", "Bayesiano", "Red neuronal"]
    means   = [data["original"]["mean_error_deg"],
               data["bayesian"]["mean_error_deg"],
               data["neural"]["mean_error_deg"]]
    stds    = [data["original"]["std_error_deg"],
               data["bayesian"]["std_error_deg"],
               data["neural"]["std_error_deg"]]
    colors  = ["#888888", "#2166AC", "#55A868"]
    improv  = [None,
               data["bayesian"]["improvement_pct"],
               data["neural"]["improvement_pct"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))

    bars = ax.bar(x, means, width=0.5,
                  color=colors, alpha=0.75,
                  edgecolor="black", linewidth=0.9)

    # Anotacion: valor medio encima de cada barra
    for i, (m, s, imp) in enumerate(zip(means, stds, improv)):
        y_top = m
        ax.text(i, y_top + 0.04, f"{m:.3f}°",
                ha="center", va="bottom", fontsize=9, color="black")
        if imp is not None:
            ax.text(i, m / 2, f"−{imp:.1f}%",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Error de localizacion medio (°)")
    ax.set_ylim(0, max(means) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Guardado: {out_path}")


def plot_visual_field_map(
    session_dir: Path,
    model: dict,
    metrics: dict | None,
    out_path: Path,
) -> None:
    """Mapa de campo visual: estimulacion / Bayesiano / Neuronal."""
    import csv, math

    # Posiciones atlas (sin sesgo) desde CSV
    atlas_csv_candidates = [
        Path("C:/PhosLab/implant_explorer/data/exported_RFs/synthetic_4ecc_4el.csv"),
        Path("config/synthetic_4ecc_4el.csv"),
    ]
    atlas_csv = next((p for p in atlas_csv_candidates if p.exists()), None)
    atlas: dict[int, tuple[float, float]] = {}
    if atlas_csv:
        for row in csv.DictReader(atlas_csv.read_text(encoding="utf-8").splitlines()):
            atlas[int(row["electrode_index"])] = (float(row["x_deg"]), float(row["y_deg"]))

    cr = json.loads(
        (session_dir / "consolidated_analysis" / "consolidated_results.json")
        .read_text(encoding="utf-8")
    )

    # Correcciones
    bias_bx = float(model["posterior_mean_x"])
    bias_by = float(model["posterior_mean_y"])

    # Corrección neuronal: estimada a partir de las métricas de evaluación
    if metrics:
        split = "test" if "test" in metrics else "train"
        orig_x  = metrics[split]["original"]["mean_error_x_deg"]
        orig_y  = metrics[split]["original"]["mean_error_y_deg"]
        neur_x  = metrics[split]["neural"]["mean_error_x_deg"]
        neur_y  = metrics[split]["neural"]["mean_error_y_deg"]
        bias_nx = orig_x - neur_x
        bias_ny = orig_y - neur_y
    else:
        bias_nx = bias_bx
        bias_ny = bias_by

    fig, ax = plt.subplots(figsize=(9, 9))
    max_r = 1.0
    eccs_seen: set[float] = set()

    first_stim = first_bay = first_neur = True

    for idx_str, rec in cr["electrodes"].items():
        idx = int(idx_str)
        if idx not in atlas:
            continue
        ax_deg, ay_deg = atlas[idx]
        ecc = math.hypot(ax_deg, ay_deg)
        eccs_seen.add(round(ecc))

        # 1 — Estimulacion: posicion medida por el participante (respuesta)
        cents = np.array(rec["centroids_deg"], dtype=float)
        mx = rec["mean_position_deg"]["x"]
        my = rec["mean_position_deg"]["y"]
        sx = float(np.std(cents[:, 0], ddof=1)) if len(cents) > 1 else 0.0
        sy = float(np.std(cents[:, 1], ddof=1)) if len(cents) > 1 else 0.0
        max_r = max(max_r, abs(mx), abs(my))
        ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt="o", ms=7,
                    color="#555555", ecolor="#555555", elinewidth=0.9,
                    capsize=2, mec="black", mew=0.5, zorder=3,
                    label="Estimulacion (medida)" if first_stim else "")
        ax.annotate(str(idx), (mx, my),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=6, color="#666666")
        first_stim = False

        # 2 — Prediccion corregida: Bayesiano
        bx, by = ax_deg + bias_bx, ay_deg + bias_by
        max_r = max(max_r, abs(bx), abs(by))
        ax.scatter([bx], [by], marker="D", s=55,
                   color="#2166AC", edgecolors="black", linewidths=0.5,
                   zorder=5, label="Bayesiano" if first_bay else "")
        first_bay = False

        # 3 — Prediccion corregida: Red neuronal
        nx, ny = ax_deg + bias_nx, ay_deg + bias_ny
        max_r = max(max_r, abs(nx), abs(ny))
        ax.scatter([nx], [ny], marker="^", s=55,
                   color="#55A868", edgecolors="black", linewidths=0.5,
                   zorder=4, label="Red neuronal" if first_neur else "")
        first_neur = False

    # Circulos de excentricidad (referencia)
    for r in sorted(eccs_seen):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls=":", lw=1.0,
                                ec="#cccccc", zorder=1))
        angle = math.radians(40)
        ax.text(r * math.cos(angle), r * math.sin(angle),
                f"{r:g}°", fontsize=7, color="#aaaaaa")

    lim = max_r * 1.1
    ax.axhline(0, color="#eeeeee", lw=0.8, zorder=0)
    ax.axvline(0, color="#eeeeee", lw=0.8, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Campo visual X (°)   [+ = derecha]")
    ax.set_ylabel("Campo visual Y (°)   [+ = arriba]")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Guardado: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Exp 5 — Curva de convergencia Bayesiana."
    )
    ap.add_argument("--session", required=True,
                    help="Ruta a la carpeta de sesion")
    ap.add_argument("--bias-true", nargs=2, type=float, default=[2.0, 1.0],
                    metavar=("BX", "BY"),
                    help="Sesgo verdadero inyectado [X Y] en grados (default: 2.0 1.0)")
    ap.add_argument("--out-dir", default=None,
                    help="Carpeta de salida (default: comparison_results/exp5_learning)")
    args = ap.parse_args()

    session_dir = Path(args.session)
    out_dir = Path(args.out_dir) if args.out_dir else Path("comparison_results/exp5_learning")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Buscar bayesian_model.json: primero en learning_results/ global,
    # luego dentro de la propia sesion
    candidates = [
        Path("learning_results/bayesian_model.json"),
        session_dir / "bayesian_model.json",
        session_dir / "learning" / "bayesian_model.json",
    ]
    model_file = next((p for p in candidates if p.exists()), None)
    if model_file is None:
        raise SystemExit(
            "No se encontro bayesian_model.json.\n"
            "Ejecuta primero: uv run python scripts/learning/run_learning.py"
        )

    print(f"[Exp 5] Sesion:  {session_dir}")
    print(f"[Exp 5] Modelo:  {model_file}")
    print(f"[Exp 5] Sesgo verdadero: X={args.bias_true[0]}°  Y={args.bias_true[1]}°")
    print(f"[Exp 5] Salida:  {out_dir}\n")

    model = json.loads(model_file.read_text(encoding="utf-8"))
    errors_x = model["observed_errors_x"]
    errors_y = model["observed_errors_y"]
    prior_mean = float(model.get("prior_mean", 0.0))
    prior_std  = float(model.get("prior_std",  5.0))
    noise_std  = float(model.get("noise_std",  0.5))

    print(f"[Exp 5] Observaciones cargadas: {len(errors_x)}")
    print(f"[Exp 5] Posterior final X: {model['posterior_mean_x']:.4f}° "
          f"(verdad={args.bias_true[0]}°)")
    print(f"[Exp 5] Posterior final Y: {model['posterior_mean_y']:.4f}° "
          f"(verdad={args.bias_true[1]}°)")
    print()

    plot_learning_curve(
        errors_x, errors_y,
        bias_true=tuple(args.bias_true),
        prior_mean=prior_mean,
        prior_std=prior_std,
        noise_std=noise_std,
        out_path=out_dir / "learning_curve.png",
    )

    # Figura 2: reduccion de error (original / Bayesiano / Red neuronal)
    metrics_candidates = [
        Path("learning_results/evaluation_metrics.json"),
        session_dir / "evaluation_metrics.json",
    ]
    metrics_file = next((p for p in metrics_candidates if p.exists()), None)
    if metrics_file:
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        plot_error_reduction(metrics, out_dir / "error_reduction.png")
    else:
        print("[AVISO] No se encontro evaluation_metrics.json — omitiendo error_reduction.png")

    # Figura 3: mapa de campo visual
    metrics_for_map = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file else None
    plot_visual_field_map(session_dir, model, metrics_for_map, out_dir / "visual_field_map.png")

    print("\n[OK] Exp 5 completado.")


if __name__ == "__main__":
    main()
