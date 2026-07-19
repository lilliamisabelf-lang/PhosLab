"""Exp 3 — Comparación de configuraciones de implante.

Compara hasta N sesiones con distintos tipos de implante en términos de:
  - Cobertura del campo visual (posiciones de los electrodos)
  - Error de predicción del atlas de Benson (distancia predicción → respuesta)
  - Correlación excentricidad-error por sesión (r de Pearson + IC95 bootstrap
    por electrodo, ya que las repeticiones de un mismo electrodo no son
    observaciones independientes entre sí)

Produce en --out-dir:
  1. coverage_map.png            — distribución espacial de electrodos + posiciones medidas
  2. error_comparison.png        — boxplots de error por tipo de implante
  3. map_<implante>_split.png    — un archivo por sesión: estímulo / percepción
                                    media ± std / superposición, en 3 paneles
  4. error_vs_ecc_exp3.png       — boxplots por anillo de excentricidad, con una
                                    recta de regresión por sesión (r en la leyenda)

Todas las figuras usan la misma paleta de color por sesión (RING_PLOT_COLORS,
con SESSION_COLORS como respaldo para labels no reconocidos).

El resumen de consola (print_summary) incluye, además del error medio/SD por
implante, el r y su IC95 bootstrap por electrodo. Con --compare-ring se añade
además un test de Mann-Whitney y un IC95 bootstrap de la diferencia de
medianas para cada par de sesiones que comparta datos en esa excentricidad.

Uso (PowerShell):
    cd percept_mapper
    uv run python scripts/analysis/compare_implants.py `
        --sessions mapping_experiments/utah mapping_experiments/comb mapping_experiments/thread `
        --labels   "4x Utah" "Comb 10x10" "Thread-1024" `
        --out-dir  comparison_results/exp3_implants `
        --compare-ring 4
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

from stats_utils import (
    RINGS_DEFAULT, N_BOOT_DEFAULT, SEED_DEFAULT,
    round_ring, collect_electrode_data, pearson_r_by_trial,
    cluster_bootstrap_r, bootstrap_ci, bootstrap_p_two_sided,
    report_r, format_r_report, mannwhitney_compare,
)
from map_plot_utils import plot_split_maps

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

# Paleta fija para plot_error_vs_ecc_by_ring, por nombre de sesión (independiente
# de SESSION_COLORS, que usan plot_coverage_map/plot_error_comparison y cuyo
# orden ya está descrito en el caption de coverage_map.png). Cualquier label no
# listado aquí recurre a SESSION_COLORS por posición.
RING_PLOT_COLORS = {
    "Comb 10x10": "#C44E52",
    "Thread-1024": "#55A868",
    "Utah Array": "#8172B2",
}


def _color_for_label(label: str, index: int) -> str:
    return RING_PLOT_COLORS.get(label, SESSION_COLORS[index % len(SESSION_COLORS)])


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
        color  = _color_for_label(label, i)
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
        Line2D([], [], color=_color_for_label(label, i),
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


def _scatter_with_clip(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    y_cap: float,
    s: float = 25,
    alpha: float = 0.7,
    zorder: int = 5,
) -> int:
    """Dibuja el scatter recortando en y_cap: los puntos por encima se marcan
    con un triángulo pegado al techo en vez de estirar el eje Y. Devuelve
    cuántos puntos quedaron recortados."""
    x = np.asarray(x)
    y = np.asarray(y)
    visible = y <= y_cap
    ax.scatter(x[visible], y[visible], color=color, s=s, edgecolors="black",
               linewidths=0.4, alpha=alpha, zorder=zorder)
    n_clipped = int((~visible).sum())
    if n_clipped:
        ax.scatter(x[~visible], np.full(n_clipped, y_cap), color=color,
                   s=s * 1.5, marker="^", edgecolors="black", linewidths=0.7,
                   alpha=min(alpha + 0.2, 1.0), zorder=zorder + 1)
    return n_clipped


def _robust_y_cap(pooled: np.ndarray, percentile: float = 99.0, floor: float = 0.5) -> float:
    """Límite superior del eje Y basado en un percentil de los datos, para que
    unos pocos outliers extremos no aplasten la visualización del resto."""
    return max(float(np.percentile(pooled, percentile)), floor)


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
    colors    = [_color_for_label(lbl, i) for i, lbl in enumerate(labels)]

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

    pooled = np.concatenate([np.array(pts) for pts in data_only if pts])
    y_cap = _robust_y_cap(pooled) * 1.15

    for i, (pos, color, pts) in enumerate(zip(positions, colors, data_only)):
        if not pts:
            continue
        jitter = (rng.random(len(pts)) - 0.5) * 0.3
        n_clipped = _scatter_with_clip(
            ax, np.array([pos] * len(pts)) + jitter, np.array(pts), color, y_cap,
        )
        if n_clipped:
            print(f"  [aviso] {n_clipped} punto(s) fuera de escala en '{labels[i]}' "
                  f"(> {y_cap:.2f}°), marcados con un triángulo")
        # mediana y stats encima del whisker superior
        top_cap_y = bp["caps"][2 * i + 1].get_ydata()[0]
        med = float(np.median(pts))
        mean = float(np.mean(pts))
        n_rep = len(pts)
        ax.text(pos, top_cap_y + 0.015,
                f"Md={med:.2f}°",
                ha="center", va="bottom", fontsize=8, color="black")

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
# Figura 3 — Error vs. excentricidad por anillo, con rectas de regresión
# ---------------------------------------------------------------------------

def plot_error_vs_ecc_by_ring(
    sessions: list[tuple[str, dict]],
    out: Path,
    rings: list[int] = RINGS_DEFAULT,
    seed: int = SEED_DEFAULT,
    extrapolate_lines: bool = False,
    clip_outliers: bool = True,
) -> None:
    """Boxplot por anillo de excentricidad y sesión, con una recta de regresión
    por sesión (r de Pearson en la leyenda).

    Por defecto cada recta se dibuja únicamente dentro del rango de
    excentricidad real de esa sesión (sin extrapolar) y el eje Y se recorta a
    un percentil robusto para que los outliers extremos no aplasten el resto
    (quedan marcados con un triángulo). Con extrapolate_lines=True y/o
    clip_outliers=False se recupera el estilo "clásico" usado en figuras de
    póster ya publicadas, donde las rectas cubren todo el eje y el eje Y se
    autoescala al máximo de los datos."""
    fig, ax = plt.subplots(figsize=(22, 5.5))
    rng = np.random.default_rng(seed)
    dodge_width = 1.3

    legend_handles = []
    all_ring_data: dict[int, list[tuple[str, str, np.ndarray]]] = {}

    for i, (label, results) in enumerate(sessions):
        color = _color_for_label(label, i)
        electrode_ecc, electrode_errs = collect_electrode_data(results)
        trial_ecc, trial_err = [], []
        for ecc, reps in zip(electrode_ecc, electrode_errs):
            for e in reps:
                trial_ecc.append(ecc)
                trial_err.append(e)
        trial_ecc = np.array(trial_ecc)
        trial_err = np.array(trial_err)

        r = float(np.corrcoef(trial_ecc, trial_err)[0, 1])
        legend_handles.append(
            Line2D([0], [0], color=color, ls="--", lw=2.2,
                   label=f"{label} (r={r:.2f})".replace(".", ","))
        )

        coeffs = np.polyfit(trial_ecc, trial_err, 1)
        if extrapolate_lines:
            x_lo, x_hi = min(rings) - 0.3, max(rings) + 0.6
        else:
            x_lo, x_hi = trial_ecc.min(), trial_ecc.max()
        x_line = np.linspace(x_lo, x_hi, 200)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line, y_line, color=color, ls="--", lw=2.2, zorder=4, alpha=0.9)

        for ecc, reps in zip(electrode_ecc, electrode_errs):
            ring = round_ring(ecc, rings)
            all_ring_data.setdefault(ring, []).append((label, color, np.array(reps)))

    # agrupa por anillo los electrodos ya asignados arriba, pero antes hay que
    # fusionar los distintos electrodos de una misma sesión que caen en el
    # mismo anillo, para dibujar una sola caja por (sesión, anillo)
    merged: dict[int, dict[str, tuple[str, list[float]]]] = {}
    for ring, entries in all_ring_data.items():
        bucket: dict[str, tuple[str, list[float]]] = {}
        for label, color, vals in entries:
            key = label
            if key not in bucket:
                bucket[key] = (color, [])
            bucket[key][1].extend(vals.tolist())
        merged[ring] = bucket

    pooled = np.concatenate([np.array(vals) for bucket in merged.values() for _, vals in bucket.values()])
    y_cap = _robust_y_cap(pooled) * 1.2 if clip_outliers else float(pooled.max()) * 1.15

    for ring, bucket in merged.items():
        items = list(bucket.items())
        n = len(items)
        offsets = [0.0] if n == 1 else np.linspace(-dodge_width / 2, dodge_width / 2, n)
        for (label, (color, vals)), off in zip(items, offsets):
            vals = np.array(vals)
            pos = ring + off
            bp = ax.boxplot(
                [vals], positions=[pos], widths=0.5,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(linewidth=1.2, color="black"),
                capprops=dict(linewidth=1.2, color="black"),
                boxprops=dict(facecolor=color, edgecolor="black", alpha=0.45),
                zorder=5,
            )
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            if clip_outliers:
                n_clipped = _scatter_with_clip(
                    ax, pos + jitter, vals, color, y_cap, s=28, alpha=0.6, zorder=6,
                )
                if n_clipped:
                    print(f"  [aviso] {n_clipped} punto(s) fuera de escala en '{label}' "
                          f"anillo {ring}° (> {y_cap:.2f}°), marcados con un triángulo")
            else:
                ax.scatter(pos + jitter, vals, color=color, s=28, alpha=0.6,
                           edgecolors="white", linewidths=0.5, zorder=6)
            top_cap_y = bp["caps"][1].get_ydata()[0]
            med = float(np.median(vals))
            ax.text(pos, top_cap_y + 0.03, f"Md={med:.2f}°".replace(".", ","),
                    ha="center", va="bottom", fontsize=11, color="#222222")

    ax.set_ylim(0, y_cap)
    ax.set_xticks(rings)
    ax.set_xticklabels([f"{r}°" for r in rings])
    ax.set_xlabel("Excentricidad del electrodo (°)", labelpad=10)
    ax.set_ylabel("Error de localización (°)")
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9, fontsize=13)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] Guardado: {out}")


# ---------------------------------------------------------------------------
# Figura 4 — Mapa estímulo / percepción / superposición, por implante
# ---------------------------------------------------------------------------
# (implementación compartida en map_plot_utils.py, reutilizada también por
# compare_mapmethod.py — ver plot_split_maps allí)


# ---------------------------------------------------------------------------
# Comparación formal entre dos sesiones en un anillo de excentricidad
# ---------------------------------------------------------------------------

def compare_at_ring(
    sessions: list[tuple[str, dict]],
    ring: int,
    rings: list[int] = RINGS_DEFAULT,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> None:
    """Para cada par de sesiones con datos en `ring`, imprime un test de
    Mann-Whitney (por ensayo) y un IC95 bootstrap por electrodo de la
    diferencia de medianas."""
    per_session: dict[str, list[list[float]]] = {}
    for label, results in sessions:
        electrode_ecc, electrode_errs = collect_electrode_data(results)
        in_ring = [reps for ecc, reps in zip(electrode_ecc, electrode_errs) if round_ring(ecc, rings) == ring]
        if in_ring:
            per_session[label] = in_ring

    labels = list(per_session.keys())
    print(f"\nComparación en el anillo de {ring}°:")
    if len(labels) < 2:
        print(f"  Menos de dos sesiones con datos en {ring}°; nada que comparar.")
        return

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label_a, label_b = labels[i], labels[j]
            cmp = mannwhitney_compare(label_a, per_session[label_a], label_b, per_session[label_b], n_boot, seed)
            print(f"  {label_a} (n_el={cmp['n_el_a']}, Md={cmp['md_a']:.3f}°) vs "
                  f"{label_b} (n_el={cmp['n_el_b']}, Md={cmp['md_b']:.3f}°): "
                  f"diff={cmp['diff']:.3f}°, Mann-Whitney U={cmp['u']:.1f} p={cmp['p_mannwhitney']:.4f}, "
                  f"IC95 bootstrap por electrodo=[{cmp['ci_lo']:.3f}, {cmp['ci_hi']:.3f}]")


# ---------------------------------------------------------------------------
# Resumen numérico en consola
# ---------------------------------------------------------------------------

def print_summary(
    sessions: list[tuple[str, dict]],
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> None:
    print(f"\n{'Implante':>20}  {'N electr':>8}  {'Área cob (°²)':>14}  "
          f"{'Error medio (°)':>16}  {'Std (°)':>8}")
    print("-" * 78)
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

    print("Correlación excentricidad-error (bootstrap por electrodo):")
    for label, results in sessions:
        rep = report_r(results, n_boot, seed)
        print(f"  {format_r_report(label, rep)}")
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
    ap.add_argument("--compare-ring", type=int, default=None,
                    help="Excentricidad (°) en la que comparar formalmente cada par de sesiones "
                         "(Mann-Whitney + IC95 bootstrap por electrodo de la diferencia de medianas)")
    ap.add_argument("--extrapolate-lines", action="store_true",
                    help="Dibuja las rectas de regresión en todo el eje X en vez de recortarlas "
                         "al rango real de excentricidad de cada sesión (estilo póster clásico).")
    ap.add_argument("--no-clip-outliers", action="store_true",
                    help="No recorta el eje Y por percentil ni marca outliers con triángulo; "
                         "autoescala al máximo de los datos (estilo póster clásico).")
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
    plot_split_maps(loaded, out_dir, color_fn=_color_for_label, title_fn=lambda label: f"Implante: {label}")
    plot_error_vs_ecc_by_ring(
        loaded, out_dir / "error_vs_ecc_exp3.png",
        extrapolate_lines=args.extrapolate_lines,
        clip_outliers=not args.no_clip_outliers,
    )
    if args.compare_ring is not None:
        compare_at_ring(loaded, args.compare_ring)
    print("\n[OK] Exp 3 completado.")


if __name__ == "__main__":
    main()
