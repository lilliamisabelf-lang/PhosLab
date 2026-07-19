"""Utilidades de trazado compartidas para los mapas estímulo/percepción de
Exp2, Exp3 y Exp4 — el patrón de "3 paneles" (estímulo / percepción media ±
std / superposición) que antes vivía duplicado en cada script.

Cualquier script que necesite este tipo de figura debe importar de aquí en
vez de reimplementarla — así un arreglo o cambio de estilo se aplica a todos
los experimentos a la vez.

Este módulo es una librería, no un script — no se ejecuta directamente
(`python map_plot_utils.py` no hace nada útil). Se usa con `from
map_plot_utils import plot_split_maps` desde el script de análisis
correspondiente (compare_implants.py, compare_mapmethod.py).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def electrode_map_points(results: dict) -> list[dict]:
    """Para cada electrodo: posición estímulo, media percibida y su std (en
    grados), calculada a partir de los centroides por repetición.

    Usa `centroids_deg` (lista de [x, y] por repetición), presente en todos
    los formatos de sesión — a diferencia de `per_repetition[].centroid_deg`,
    que no existe en las sesiones de método relativo/pareado (reconstruidas
    por MDS/Procrustes en vez de centroide directo por ensayo)."""
    points = []
    for key, rec in results["electrodes"].items():
        tx, ty = rec["stimulation_position_deg"]
        mean = rec.get("mean_position_deg")
        centroids = rec.get("centroids_deg") or []
        if not mean or not centroids:
            continue
        cents = np.asarray(centroids, dtype=float)
        points.append({
            "idx": rec.get("electrode_index", key),
            "tx": tx, "ty": ty,
            "mx": mean["x"], "my": mean["y"],
            "sx": float(np.std(cents[:, 0])), "sy": float(np.std(cents[:, 1])),
        })
    return points


def draw_ecc_rings(ax, max_r: float, step: float = 2.0) -> None:
    n_rings = max(1, math.ceil(max_r / step))
    for i in range(1, n_rings + 1):
        r = step * i
        circle = plt.Circle((0, 0), r, fill=False, ls=":", lw=0.9, ec="#aaaaaa", zorder=0)
        ax.add_patch(circle)
        ax.text(r * math.cos(math.radians(38)), r * math.sin(math.radians(38)),
                f"{r:g}°", fontsize=7, color="#888888")
    ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
    ax.axvline(0, color="#dddddd", lw=0.8, zorder=0)


def slug(label: str) -> str:
    return label.lower().replace(" ", "_")


def plot_split_maps(
    sessions: list[tuple[str, dict]],
    out_dir: Path,
    color_fn: Callable[[str, int], str],
    filename_fn: Callable[[str], str] | None = None,
    title_fn: Callable[[str], str] | None = None,
) -> None:
    """Por cada sesión: 3 paneles (estímulo / percepción media ± std /
    superposición) con los electrodos etiquetados por su índice.

    color_fn(label, index) -> color hex, para que cada script decida su
    propia paleta (por posición o por nombre de sesión).
    filename_fn(label) -> nombre de archivo (sin extensión); por defecto
    "map_<slug(label)>_split.png".
    title_fn(label) -> título de la figura; por defecto el label tal cual.
    """
    if filename_fn is None:
        filename_fn = lambda label: f"map_{slug(label)}_split"
    if title_fn is None:
        title_fn = lambda label: label

    for i, (label, results) in enumerate(sessions):
        color = color_fn(label, i)
        points = electrode_map_points(results)
        if not points:
            continue

        max_r = max(
            max(abs(p["tx"]), abs(p["ty"]), abs(p["mx"]), abs(p["my"]))
            + max(p["sx"], p["sy"])
            for p in points
        )
        lim = max_r * 1.2

        fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharex=True, sharey=True)

        titles = ["Estímulo", "Percepción media ± std", "Superposición"]
        for ax, title in zip(axes, titles):
            draw_ecc_rings(ax, max_r)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xlabel("Campo visual X (°)")
            ax.set_title(title)
        axes[0].set_ylabel("Campo visual Y (°)")

        tx = [p["tx"] for p in points]
        ty = [p["ty"] for p in points]
        mx = [p["mx"] for p in points]
        my = [p["my"] for p in points]
        sx = [p["sx"] for p in points]
        sy = [p["sy"] for p in points]

        axes[0].scatter(tx, ty, marker="D", s=60, facecolors="none",
                         edgecolors="black", linewidths=1.1, label="Estímulo", zorder=5)

        axes[1].errorbar(mx, my, xerr=sx, yerr=sy, fmt="D", ms=7,
                          color=color, ecolor=color, elinewidth=1.0, capsize=2,
                          mec="black", mew=0.6, label="Percepción media ± std", zorder=5)

        axes[2].scatter(tx, ty, marker="D", s=60, facecolors="none",
                         edgecolors="black", linewidths=1.1, label="Estímulo", zorder=5)
        axes[2].errorbar(mx, my, xerr=sx, yerr=sy, fmt="D", ms=7,
                          color=color, ecolor=color, elinewidth=1.0, capsize=2,
                          mec="black", mew=0.6, label="Percepción media ± std", zorder=6)

        for p in points:
            axes[0].annotate(str(p["idx"]), (p["tx"], p["ty"]), fontsize=6,
                              color="#555555", xytext=(3, 3), textcoords="offset points")
            axes[1].annotate(str(p["idx"]), (p["mx"], p["my"]), fontsize=6,
                              color="#555555", xytext=(3, 3), textcoords="offset points")
            axes[2].annotate(str(p["idx"]), (p["mx"], p["my"]), fontsize=6,
                              color="#555555", xytext=(3, 3), textcoords="offset points")

        for ax in axes:
            ax.legend(loc="upper left", fontsize=9)

        fig.suptitle(title_fn(label), y=1.04, fontsize=14)
        fig.tight_layout()
        out = out_dir / f"{filename_fn(label)}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"[OK] Guardado: {out}")
