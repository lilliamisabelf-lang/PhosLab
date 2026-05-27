"""Descriptive statistics. Pure numpy. No IO, no plotting.

These were inlined inside `mapping_analyzer.py` as module-private helpers
(`_robust_sigma_mad`, etc.). Extracted so they can be imported, tested,
and reused from notebooks without loading the analyzer's matplotlib /
PIL stack.
"""

from __future__ import annotations

import numpy as np


def robust_sigma_mad(values) -> float:
    """Normal-consistent robust σ estimator via the MAD.

    Returns NaN when the input is empty. Robust to up to ~50% outliers,
    which is overkill for our use but keeps the same number as
    `np.std(values, ddof=1)` for a clean Gaussian sample.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(1.4826 * mad)


def trimmed_mean(values, proportion_to_cut: float = 0.1) -> float:
    """Trimmed mean: drop the top and bottom `proportion_to_cut` of the
    sample before averaging. Returns NaN when the input is empty.
    `proportion_to_cut` is clamped to [0, 0.49]; when the trim leaves zero
    samples, falls back to the plain mean."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    proportion_to_cut = float(np.clip(proportion_to_cut, 0.0, 0.49))
    sorted_vals = np.sort(values)
    k = int(np.floor(sorted_vals.size * proportion_to_cut))
    if sorted_vals.size - 2 * k <= 0:
        return float(np.mean(sorted_vals))
    return float(np.mean(sorted_vals[k : sorted_vals.size - k]))


def tukey_boxplot_stats(values) -> dict:
    """Five-number summary + outliers (Tukey 1.5·IQR rule).

    Returns a JSON-serializable dict with keys:
      n, q1, median, q3, iqr, whisker_low, whisker_high, outliers.
    Empty / all-NaN inputs return an `n=0` skeleton with all numeric
    fields set to `None`.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "q1": None,
            "median": None,
            "q3": None,
            "iqr": None,
            "whisker_low": None,
            "whisker_high": None,
            "outliers": [],
        }

    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr

    inliers = values[(values >= low_fence) & (values <= high_fence)]
    if inliers.size == 0:
        whisker_low = float(np.min(values))
        whisker_high = float(np.max(values))
    else:
        whisker_low = float(np.min(inliers))
        whisker_high = float(np.max(inliers))

    outliers = values[(values < whisker_low) | (values > whisker_high)]

    return {
        "n": int(values.size),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "iqr": float(iqr),
        "whisker_low": float(whisker_low),
        "whisker_high": float(whisker_high),
        "outliers": [float(v) for v in outliers.tolist()],
    }


def distance_deg(
    dx_px, dy_px, px_per_deg_x: float, px_per_deg_y: float
):
    """Radial distance in degrees of visual angle, respecting anisotropic
    px/deg ratios on the two screen axes. Element-wise; accepts scalar
    or ndarray inputs."""
    dx_deg = np.asarray(dx_px, dtype=float) / float(px_per_deg_x)
    dy_deg = np.asarray(dy_px, dtype=float) / float(px_per_deg_y)
    return np.hypot(dx_deg, dy_deg)
