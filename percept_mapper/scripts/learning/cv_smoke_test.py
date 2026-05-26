"""Smoke tests for cross_validation.

Run:
    uv run --project percept_mapper python percept_mapper/scripts/learning/cv_smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from scripts.learning.cross_validation import (  # noqa: E402
    k_fold_cv,
    paired_bootstrap_pvalue,
    bayesian_corrector,
)


def _synthetic_dataset(n=200, bias=(2.0, -1.5), noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    pred = rng.uniform(-5, 5, size=(n, 2))
    obs = pred + np.array(bias) + rng.normal(0, noise, size=(n, 2))
    return pred, obs


def test_cv_finds_improvement_when_bias_present():
    print("\n=== test: CV detects real bias correction ===")
    pred, obs = _synthetic_dataset(n=200, bias=(2.0, -1.5), noise=0.3, seed=42)
    result = k_fold_cv(pred, obs, bayesian_corrector(prior_std=10), k=5, seed=1, bootstrap_iter=500)
    print(
        f"  MSE uncorrected={result.mean_mse_uncorrected:.3f}  "
        f"corrected={result.mean_mse_corrected:.3f}  "
        f"improvement={result.improvement_pct:.1f}%  p={result.bootstrap_p_value:.4f}"
    )
    assert result.improvement_abs > 0, "correction should reduce error on biased data"
    assert result.bootstrap_p_value is not None and result.bootstrap_p_value < 0.05, (
        f"improvement should be significant (p<0.05), got p={result.bootstrap_p_value}"
    )
    print("  ✓ PASS")


def test_cv_returns_no_improvement_when_no_bias():
    print("\n=== test: CV returns near-zero improvement on unbiased data ===")
    pred, obs = _synthetic_dataset(n=200, bias=(0.0, 0.0), noise=1.0, seed=7)
    result = k_fold_cv(pred, obs, bayesian_corrector(prior_std=10), k=5, seed=1, bootstrap_iter=500)
    print(
        f"  MSE uncorrected={result.mean_mse_uncorrected:.3f}  "
        f"corrected={result.mean_mse_corrected:.3f}  "
        f"improvement={result.improvement_pct:.1f}%"
    )
    # With no bias, the corrector adds noise (fitting noise mean ≠ 0).
    # We expect at most modest improvement and a non-tiny p-value.
    assert abs(result.improvement_pct) < 20.0, "no large gain when there's no bias"
    print("  ✓ PASS")


def test_paired_bootstrap_centered_at_zero_when_equal():
    print("\n=== test: paired bootstrap CI brackets 0 for identical errors ===")
    err_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    err_b = err_a.copy()
    p, lo, hi = paired_bootstrap_pvalue(err_a, err_b, n_iter=500, seed=0)
    print(f"  p={p:.3f}  CI=({lo:.3f}, {hi:.3f})")
    assert abs(lo) < 1e-9 and abs(hi) < 1e-9, "CI should be exactly zero when diffs are all zero"
    print("  ✓ PASS")


def test_cv_shapes_and_per_trial_consistency():
    print("\n=== test: per-trial arrays match fold MSEs ===")
    pred, obs = _synthetic_dataset(n=50, seed=3)
    result = k_fold_cv(pred, obs, bayesian_corrector(), k=5, seed=2, bootstrap_iter=200)
    assert len(result.per_trial_sq_error_uncorrected) == 50
    assert len(result.per_trial_sq_error_corrected) == 50
    # Mean of per-trial == grand mean across folds (up to fold-size weighting).
    grand_unc = float(np.mean(result.per_trial_sq_error_uncorrected))
    grand_cor = float(np.mean(result.per_trial_sq_error_corrected))
    print(f"  grand uncorrected={grand_unc:.3f}  fold-mean={result.mean_mse_uncorrected:.3f}")
    assert abs(grand_unc - result.mean_mse_uncorrected) < 0.5  # weighted-vs-unweighted slack
    assert abs(grand_cor - result.mean_mse_corrected) < 0.5
    print("  ✓ PASS")


def main():
    test_cv_finds_improvement_when_bias_present()
    test_cv_returns_no_improvement_when_no_bias()
    test_paired_bootstrap_centered_at_zero_when_equal()
    test_cv_shapes_and_per_trial_consistency()
    print("\nAll cross-validation smoke tests passed.")


if __name__ == "__main__":
    main()
