"""Per-electrode mapping scatter plot. Pure visualization layer.

`plot_electrode_map(results, ...)` consumes the dict returned by
`PhospheneMappingAnalyzer.analyze_electrode_repetitions` and emits a
matplotlib Figure. No state on the caller, no IO beyond the optional
`output_path` save.

Matplotlib is imported inside the function so this module's top-level
import is cheap — anything that needs to know the contract (function
signature) without painting pixels can `from scripts.plots import
plot_electrode_map` and pay nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.stats import ellipse_from_cov


def plot_electrode_map(
    results: dict,
    *,
    screen_size: tuple[int, int],
    pixels_per_degree: tuple[float, float],
    output_path: Path | None = None,
    title_suffix: str = "",
):
    """Render the per-electrode mapping scatter and return the Figure.

    Args:
        results: analyzer output dict. Must contain `centroids`,
            `mean_position`, `stimulation_position`, `std_position`,
            `electrode_index`, `num_valid_repetitions`,
            `num_invalid_repetitions`, `mean_distance_from_average*`,
            `max_distance_from_average*`.
        screen_size: (width, height) in pixels, used to clip the zoom box
            so it never extends beyond the actual screen.
        pixels_per_degree: (x, y) px/° conversion, used only for the
            stats text panel showing the offset in degrees.
        output_path: if set, the figure is saved here at 150 dpi. The
            figure is returned either way so the caller can compose
            further or close it.
        title_suffix: optional extra string appended to the title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    if results is None:
        return None

    screen_width, screen_height = screen_size
    ppdx, ppdy = pixels_per_degree

    centroids = np.array(results["centroids"])
    mean_pos = np.array([results["mean_position"]["x"], results["mean_position"]["y"]])
    stim_pos = np.array(results["stimulation_position"])
    std_x = results["std_position"]["x"]
    std_y = results["std_position"]["y"]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Auto-zoom: ~3.5σ around the mean, clipped to the screen.
    margin = max(std_x, std_y) * 3.5
    x_min = max(0, mean_pos[0] - margin)
    x_max = min(screen_width, mean_pos[0] + margin)
    y_min = max(0, mean_pos[1] - margin)
    y_max = min(screen_height, mean_pos[1] + margin)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Y grows downward in image coords

    # Individual repetitions
    ax.scatter(
        centroids[:, 0], centroids[:, 1],
        c="yellow", s=100, alpha=0.6, edgecolors="orange", linewidths=2,
        label="Dibujos individuales",
    )
    # Mean
    ax.scatter(
        mean_pos[0], mean_pos[1],
        c="red", s=300, marker="X", edgecolors="darkred", linewidths=2,
        zorder=10, label="Posición promedio",
    )
    # Stim position
    ax.scatter(
        stim_pos[0], stim_pos[1],
        c="cyan", s=300, marker="*", edgecolors="blue", linewidths=2,
        zorder=9, label="Posición del estímulo",
    )
    # Spokes from each rep to the mean
    for centroid in centroids:
        ax.plot(
            [centroid[0], mean_pos[0]], [centroid[1], mean_pos[1]],
            "gray", alpha=0.3, linewidth=1,
        )

    # Std-radius dashed circle
    std_radius = float(np.hypot(std_x, std_y))
    ax.add_patch(plt.Circle(
        mean_pos, std_radius, color="red", fill=False, linestyle="--", linewidth=2,
    ))

    # 95% ellipses (dispersion + IC of the mean) when we have ≥ 2 reps
    if centroids.shape[0] >= 2:
        cov = np.cov(centroids.T, ddof=1)
        params = ellipse_from_cov(cov, confidence=0.95)
        if params is not None:
            w, h, angle = params
            ax.add_patch(Ellipse(
                xy=mean_pos, width=w, height=h, angle=angle,
                fill=False, edgecolor="red", linestyle="-", linewidth=2, alpha=0.8,
                label="Elipse 95% (dispersión)",
            ))
            cov_mean = cov / float(centroids.shape[0])
            params_mean = ellipse_from_cov(cov_mean, confidence=0.95)
            if params_mean is not None:
                wm, hm, am = params_mean
                ax.add_patch(Ellipse(
                    xy=mean_pos, width=wm, height=hm, angle=am,
                    fill=False, edgecolor="darkred", linestyle=":", linewidth=2, alpha=0.9,
                    label="Elipse 95% (IC media)",
                ))

    ax.set_xlabel("X (píxeles)", fontsize=12)
    ax.set_ylabel("Y (píxeles)", fontsize=12)
    title = (
        f"Análisis de Mapeo - Electrodo {results['electrode_index']}\n"
        f"{results['num_valid_repetitions']} rep. válidas | "
        f"{results['num_invalid_repetitions']} inválidas"
    )
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Stats text box (offset in deg uses anisotropic px/deg)
    offset = mean_pos - stim_pos
    offset_deg = (offset[0] / max(ppdx, 1e-9), offset[1] / max(ppdy, 1e-9))
    stats_text = (
        f"Posición media: ({mean_pos[0]:.1f}, {mean_pos[1]:.1f}) px\n"
        f"Posición estímulo: ({stim_pos[0]:.1f}, {stim_pos[1]:.1f}) px\n"
        f"Desv. Est.: ({std_x:.1f}, {std_y:.1f}) px\n"
        f"Dist. media: {results['mean_distance_from_average']:.1f} px "
        f"({results['mean_distance_from_average_deg']:.2f}°)\n"
        f"Dist. máx.: {results['max_distance_from_average']:.1f} px "
        f"({results['max_distance_from_average_deg']:.2f}°)\n"
        f"Offset desde estímulo: ({offset[0]:.1f}, {offset[1]:.1f}) px\n"
        f"                       ({offset_deg[0]:.2f}°, {offset_deg[1]:.2f}°)"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    if output_path is not None:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ Visualización guardada en: {Path(output_path).name}")

    return fig
