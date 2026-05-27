"""Property-based tests for scripts.stats — descriptive helpers + ellipse.

Run:
    uv run --project percept_mapper python percept_mapper/scripts/stats/stats_property_test.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from hypothesis import given, settings, strategies as st  # noqa: E402

from scripts.stats import (  # noqa: E402
    distance_deg,
    ellipse_from_cov,
    robust_sigma_mad,
    trimmed_mean,
    tukey_boxplot_stats,
)


safe_float = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
positive_float = st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False)
sample_list = st.lists(safe_float, min_size=0, max_size=200)


# --- robust_sigma_mad -------------------------------------------------------


def prop_robust_sigma_mad_empty_is_nan():
    assert math.isnan(robust_sigma_mad([]))


@given(sample_list)
@settings(max_examples=100, deadline=None)
def prop_robust_sigma_mad_non_negative(values):
    """MAD is always ≥ 0 (or NaN for empty)."""
    s = robust_sigma_mad(values)
    if not values:
        assert math.isnan(s)
    else:
        assert s >= 0.0 or math.isnan(s)


@given(sample_list, safe_float)
@settings(max_examples=100, deadline=None)
def prop_robust_sigma_mad_translation_invariant(values, shift):
    """σ̂ is unchanged when every sample is shifted by the same constant."""
    if not values:
        return
    a = robust_sigma_mad(values)
    b = robust_sigma_mad([v + shift for v in values])
    assert abs(a - b) < 1e-6, f"shift broke MAD: a={a} b={b} shift={shift}"


# --- trimmed_mean -----------------------------------------------------------


def prop_trimmed_mean_empty_is_nan():
    assert math.isnan(trimmed_mean([]))


@given(sample_list)
@settings(max_examples=100, deadline=None)
def prop_trimmed_mean_within_minmax(values):
    """Trimmed mean lies between min and max of the input (or NaN)."""
    if not values:
        return
    m = trimmed_mean(values, 0.1)
    assert min(values) - 1e-9 <= m <= max(values) + 1e-9


@given(sample_list)
@settings(max_examples=100, deadline=None)
def prop_trimmed_mean_equals_mean_at_zero_trim(values):
    """0% trim is just the regular mean."""
    if not values:
        return
    a = trimmed_mean(values, 0.0)
    b = float(np.mean(values))
    assert abs(a - b) < 1e-6


# --- tukey_boxplot_stats ----------------------------------------------------


def prop_tukey_empty_returns_n_zero():
    s = tukey_boxplot_stats([])
    assert s["n"] == 0
    assert s["q1"] is None
    assert s["outliers"] == []


@given(sample_list)
@settings(max_examples=100, deadline=None)
def prop_tukey_ordering(values):
    """When at least one finite value is present: q1 ≤ median ≤ q3."""
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return
    s = tukey_boxplot_stats(values)
    assert s["q1"] <= s["median"] <= s["q3"] + 1e-9


# --- distance_deg ----------------------------------------------------------


@given(safe_float, safe_float, positive_float, positive_float)
@settings(max_examples=100, deadline=None)
def prop_distance_deg_non_negative_and_isotropic_check(dx, dy, ppdx, ppdy):
    """distance_deg is always >= 0 and equals hypot(dx/ppdx, dy/ppdy)."""
    d = float(distance_deg(dx, dy, ppdx, ppdy))
    assert d >= 0
    expected = math.hypot(dx / ppdx, dy / ppdy)
    assert abs(d - expected) < 1e-6


# --- ellipse_from_cov -------------------------------------------------------


def prop_ellipse_from_identity_is_unit_scale():
    """For cov=I, the major and minor axes should be equal and roughly
    2·sqrt(chi2.ppf(0.95, 2)) ≈ 4.89."""
    result = ellipse_from_cov(np.eye(2), 0.95)
    if result is None:
        return  # scipy not installed
    w, h, _ = result
    assert abs(w - h) < 1e-6, f"identity cov should give equal axes; got w={w} h={h}"
    expected = 2.0 * math.sqrt(5.99146)  # chi2.ppf(0.95, 2)
    assert abs(w - expected) < 0.05


def prop_ellipse_from_degenerate_returns_finite_or_none():
    """Degenerate (rank-1) covariance shouldn't crash — either returns a
    finite tuple thanks to the 1e-9 regularizer, or None when scipy is
    unavailable."""
    cov = np.array([[1.0, 0.0], [0.0, 0.0]])
    result = ellipse_from_cov(cov, 0.95)
    if result is None:
        return
    w, h, _ = result
    assert math.isfinite(w) and math.isfinite(h)
    assert h < w  # second axis is the small one


def main():
    print("[stats_property_test] running properties...")
    prop_robust_sigma_mad_empty_is_nan()
    prop_robust_sigma_mad_non_negative()
    prop_robust_sigma_mad_translation_invariant()
    print("  ✓ robust_sigma_mad: empty=NaN, non-negative, translation-invariant")
    prop_trimmed_mean_empty_is_nan()
    prop_trimmed_mean_within_minmax()
    prop_trimmed_mean_equals_mean_at_zero_trim()
    print("  ✓ trimmed_mean: empty=NaN, within [min,max], 0% == mean")
    prop_tukey_empty_returns_n_zero()
    prop_tukey_ordering()
    print("  ✓ tukey: empty skeleton, q1 ≤ median ≤ q3")
    prop_distance_deg_non_negative_and_isotropic_check()
    print("  ✓ distance_deg: matches hypot(dx/px, dy/py)")
    prop_ellipse_from_identity_is_unit_scale()
    prop_ellipse_from_degenerate_returns_finite_or_none()
    print("  ✓ ellipse_from_cov: identity → unit, degenerate → finite")
    print("All stats property tests passed.")


if __name__ == "__main__":
    main()
