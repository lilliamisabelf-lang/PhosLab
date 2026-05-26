"""K-fold cross-validation and paired-bootstrap significance test for the
phosphene-correction models.

Why this module exists: `train_split: 0.8` in run_learning.py is a single
holdout. A single fold can't tell you whether the corrected map is
*meaningfully* better than the uncorrected map — only that the model fit a
particular train/test split. K-fold with paired bootstrap on the per-trial
squared errors gives a CI and a p-value, which is what you'd need to claim
"corrected < uncorrected" in any report.

Inputs are model-agnostic: a `fit_correct_fn(train_pred, train_obs,
test_pred) -> test_corrected` so the same harness works for Bayesian /
neural / future correctors. Bootstrap statistic is mean squared error
per trial; the null is "no improvement vs uncorrected predictions".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


CorrectorFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CVResult:
    k: int
    n_total: int
    mse_uncorrected_folds: list[float]
    mse_corrected_folds: list[float]
    mean_mse_uncorrected: float
    mean_mse_corrected: float
    std_mse_uncorrected: float
    std_mse_corrected: float
    improvement_abs: float
    improvement_pct: float
    bootstrap_p_value: float | None
    bootstrap_ci_low: float | None
    bootstrap_ci_high: float | None
    per_trial_sq_error_uncorrected: list[float]
    per_trial_sq_error_corrected: list[float]

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "n_total": self.n_total,
            "mse_uncorrected_folds": self.mse_uncorrected_folds,
            "mse_corrected_folds": self.mse_corrected_folds,
            "mean_mse_uncorrected": self.mean_mse_uncorrected,
            "mean_mse_corrected": self.mean_mse_corrected,
            "std_mse_uncorrected": self.std_mse_uncorrected,
            "std_mse_corrected": self.std_mse_corrected,
            "improvement_abs": self.improvement_abs,
            "improvement_pct": self.improvement_pct,
            "bootstrap_p_value": self.bootstrap_p_value,
            "bootstrap_ci_95": (
                [self.bootstrap_ci_low, self.bootstrap_ci_high]
                if self.bootstrap_ci_low is not None
                else None
            ),
        }


def _sq_error_per_trial(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Squared euclidean error per trial. Shapes: pred,obs (N, 2)."""
    diff = pred - obs
    return np.sum(diff ** 2, axis=1)


def k_fold_cv(
    pred: np.ndarray,
    obs: np.ndarray,
    fit_correct_fn: CorrectorFn,
    *,
    k: int = 5,
    seed: int = 0,
    bootstrap_iter: int = 2000,
) -> CVResult:
    """Run k-fold CV, returning per-fold MSEs plus a paired-bootstrap p-value
    on the per-trial squared errors.

    Args:
        pred: (N, 2) raw model predictions before correction.
        obs:  (N, 2) participant-reported positions.
        fit_correct_fn: callable taking (train_pred, train_obs, test_pred)
            and returning corrected test_pred (N_test, 2). Should fit on
            train and apply to test only (no leakage).
        k: number of folds.
        seed: RNG seed (shuffle the indices once, then fold).
        bootstrap_iter: number of bootstrap resamples for the p-value.

    Returns:
        CVResult with both fold-aggregate and per-trial diagnostics.
    """
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    n = pred.shape[0]
    if obs.shape[0] != n:
        raise ValueError("pred and obs must have the same number of rows")
    if n < k:
        raise ValueError(f"need at least k={k} samples, got {n}")

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    mse_uncorrected_folds: list[float] = []
    mse_corrected_folds: list[float] = []
    sq_err_uncorr_all = np.empty(n, dtype=float)
    sq_err_corr_all = np.empty(n, dtype=float)

    for f, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(k) if j != f])
        train_pred = pred[train_idx]
        train_obs = obs[train_idx]
        test_pred = pred[test_idx]
        test_obs = obs[test_idx]

        corrected = np.asarray(
            fit_correct_fn(train_pred, train_obs, test_pred), dtype=float
        )
        if corrected.shape != test_pred.shape:
            raise ValueError(
                f"corrector returned shape {corrected.shape}, expected {test_pred.shape}"
            )

        sq_err_uncorr = _sq_error_per_trial(test_pred, test_obs)
        sq_err_corr = _sq_error_per_trial(corrected, test_obs)
        sq_err_uncorr_all[test_idx] = sq_err_uncorr
        sq_err_corr_all[test_idx] = sq_err_corr

        mse_uncorrected_folds.append(float(np.mean(sq_err_uncorr)))
        mse_corrected_folds.append(float(np.mean(sq_err_corr)))

    mean_unc = float(np.mean(mse_uncorrected_folds))
    mean_cor = float(np.mean(mse_corrected_folds))
    improvement_abs = mean_unc - mean_cor
    improvement_pct = (improvement_abs / mean_unc * 100.0) if mean_unc > 0 else 0.0

    p, lo, hi = paired_bootstrap_pvalue(
        sq_err_uncorr_all, sq_err_corr_all,
        n_iter=bootstrap_iter, seed=seed + 1,
    )

    return CVResult(
        k=k,
        n_total=n,
        mse_uncorrected_folds=mse_uncorrected_folds,
        mse_corrected_folds=mse_corrected_folds,
        mean_mse_uncorrected=mean_unc,
        mean_mse_corrected=mean_cor,
        std_mse_uncorrected=float(np.std(mse_uncorrected_folds, ddof=1)) if k > 1 else 0.0,
        std_mse_corrected=float(np.std(mse_corrected_folds, ddof=1)) if k > 1 else 0.0,
        improvement_abs=improvement_abs,
        improvement_pct=improvement_pct,
        bootstrap_p_value=p,
        bootstrap_ci_low=lo,
        bootstrap_ci_high=hi,
        per_trial_sq_error_uncorrected=sq_err_uncorr_all.tolist(),
        per_trial_sq_error_corrected=sq_err_corr_all.tolist(),
    )


def paired_bootstrap_pvalue(
    err_a: np.ndarray,
    err_b: np.ndarray,
    *,
    n_iter: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap on per-trial squared-error differences (a - b).

    Returns: (two-sided p-value, 95% CI low, 95% CI high) of the mean
    difference. Pairing matters because both errors come from the same trial.
    """
    err_a = np.asarray(err_a, dtype=float)
    err_b = np.asarray(err_b, dtype=float)
    if err_a.shape != err_b.shape:
        raise ValueError("err_a and err_b must have the same shape")
    n = err_a.shape[0]
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    diff = err_a - err_b  # positive => b is better
    observed = float(np.mean(diff))

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        sample_idx = rng.integers(0, n, size=n)
        boot_means[i] = float(np.mean(diff[sample_idx]))

    # Two-sided p-value: fraction of bootstrap means as extreme as observed
    # under a centered null distribution (subtract observed).
    centered = boot_means - observed
    p = float(np.mean(np.abs(centered) >= abs(observed)))
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return p, lo, hi


def bayesian_corrector(
    prior_mean: float = 0.0,
    prior_std: float = 5.0,
    noise_std: float = 0.5,
) -> CorrectorFn:
    """Return a fit_correct_fn for the simple Bayesian bias corrector used
    in the project: a Gaussian prior on per-axis bias, updated by sample
    mean of (obs - pred). No scipy dependency."""
    def _fn(train_pred: np.ndarray, train_obs: np.ndarray, test_pred: np.ndarray) -> np.ndarray:
        train_err = train_obs - train_pred
        n = train_err.shape[0]
        prior_var = prior_std ** 2
        noise_var = noise_std ** 2
        post_var = 1.0 / (1.0 / prior_var + n / noise_var) if n > 0 else prior_var
        mean_err = np.mean(train_err, axis=0) if n > 0 else np.zeros(train_err.shape[1])
        post_mean = post_var * (prior_mean / prior_var + n * mean_err / noise_var)
        return test_pred + post_mean
    return _fn
