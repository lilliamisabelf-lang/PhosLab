"""Pure statistical helpers for percept_mapper.

Everything in this package is:
- a pure function of its arguments (no IO, no global state),
- numpy-only at runtime (scipy optional for the ellipse helper),
- safe to import from a Jupyter notebook without dragging in pygame /
  matplotlib / the experiment loop.

Use `from scripts.stats import ...` to keep the import surface flat:

    from scripts.stats import (
        robust_sigma_mad,
        trimmed_mean,
        tukey_boxplot_stats,
        distance_deg,
        ellipse_from_cov,
    )
"""

from scripts.stats.descriptive import (
    distance_deg,
    robust_sigma_mad,
    trimmed_mean,
    tukey_boxplot_stats,
)
from scripts.stats.ellipse import ellipse_from_cov

__all__ = [
    "distance_deg",
    "robust_sigma_mad",
    "trimmed_mean",
    "tukey_boxplot_stats",
    "ellipse_from_cov",
]
