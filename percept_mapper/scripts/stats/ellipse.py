"""95% confidence ellipse from a 2D covariance. Pure numpy + optional scipy.

Extracted from `mapping_analyzer.py`. Returns axes in plot coordinates
(width, height, angle_deg), suitable for feeding directly into
`matplotlib.patches.Ellipse`. None when scipy is unavailable or the
covariance matrix is degenerate.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.stats import chi2
except Exception:  # pragma: no cover
    chi2 = None


def ellipse_from_cov(
    cov, confidence: float
) -> tuple[float, float, float] | None:
    """Compute the (width, height, angle_deg) of a `confidence`-level
    error ellipse from a 2x2 covariance matrix.

    Eigendecomposes the covariance. Axis lengths are 2·s·sqrt(λᵢ) where
    s = sqrt(chi2.ppf(confidence, df=2)). Returns None when scipy is
    unavailable (no χ² quantile) or the covariance is non-finite.

    A tiny diagonal regularizer (1e-9) is added to avoid singular-matrix
    failure modes when two repetitions land at the exact same pixel.
    """
    if chi2 is None:
        return None
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None

    cov = cov + np.eye(2) * 1e-9

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    scale = float(np.sqrt(chi2.ppf(confidence, df=2)))
    width = 2.0 * scale * float(np.sqrt(max(eigvals[0], 0.0)))
    height = 2.0 * scale * float(np.sqrt(max(eigvals[1], 0.0)))
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    return width, height, angle
