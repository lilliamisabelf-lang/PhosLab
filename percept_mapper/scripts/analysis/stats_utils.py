"""Estadística compartida para los scripts de análisis de Exp1-4.

Centraliza el cálculo de la correlación excentricidad-error (r de Pearson)
y de comparaciones entre sesiones, con bootstrap por electrodo completo en
vez de por ensayo individual: las repeticiones de un mismo electrodo no son
observaciones independientes entre sí (comparten el sesgo propio de ese
electrodo), así que remuestrear ensayos sueltos infla artificialmente la
precisión aparente de cualquier IC o p-valor. El tamaño muestral efectivo de
una sesión es su número de electrodos, no su número de ensayos.

Cualquier script de análisis (plot_error_vs_ecc.py, compare_implants.py,
compare_mapmethod.py) debe importar de aquí en vez de reimplementar su
propia versión de estas funciones — así un cambio de metodología (p.ej. el
número de remuestreos bootstrap) se propaga a todos los experimentos a la
vez, y no hay que reconstruir el análisis desde cero la próxima vez que se
necesite (como pasó con las figuras de Exp3).

Este módulo es una librería, no un script — no se ejecuta directamente
(`python stats_utils.py` no hace nada útil). Se usa con `from stats_utils
import ...` desde el script de análisis correspondiente.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

RINGS_DEFAULT = [0, 2, 4, 6, 8, 10, 12, 14]
N_BOOT_DEFAULT = 10000
SEED_DEFAULT = 42


def round_ring(ecc: float, rings: list[int] = RINGS_DEFAULT) -> int:
    """Redondea una excentricidad al anillo más cercano de una lista dada."""
    return min(rings, key=lambda r: abs(r - ecc))


def collect_electrode_data(results: dict) -> tuple[list[float], list[list[float]]]:
    """A partir de un consolidated_results.json ya cargado, devuelve
    (excentricidad por electrodo, [errores por repetición] por electrodo)."""
    electrode_ecc: list[float] = []
    electrode_errs: list[list[float]] = []
    for rec in results["electrodes"].values():
        sp = rec.get("stimulation_position_deg")
        reps = [rep["distance_to_stim_deg"] for rep in rec.get("per_repetition_metrics", []) or []]
        if not sp or not reps:
            continue
        electrode_ecc.append(float(math.hypot(sp[0], sp[1])))
        electrode_errs.append(reps)
    return electrode_ecc, electrode_errs


def pearson_r_by_trial(electrode_ecc: list[float], electrode_errs: list[list[float]]) -> tuple[float, int, int]:
    """r de Pearson sobre todos los ensayos individuales (no por electrodo).
    Devuelve (r, n_electrodos, n_ensayos)."""
    eccs, errs = [], []
    for ecc, reps in zip(electrode_ecc, electrode_errs):
        for e in reps:
            eccs.append(ecc)
            errs.append(e)
    r = float(np.corrcoef(eccs, errs)[0, 1])
    return r, len(electrode_ecc), len(eccs)


def cluster_bootstrap_r(
    electrode_ecc: list[float],
    electrode_errs: list[list[float]],
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> np.ndarray:
    """Distribución bootstrap de r, remuestreando electrodos completos (con
    sus repeticiones en bloque) en vez de ensayos individuales."""
    rng = np.random.default_rng(seed)
    n_el = len(electrode_ecc)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_el, n_el)
        ecc_b, err_b = [], []
        for j in idx:
            ecc_b.extend([electrode_ecc[j]] * len(electrode_errs[j]))
            err_b.extend(electrode_errs[j])
        rs[i] = np.corrcoef(ecc_b, err_b)[0, 1]
    return rs


def cluster_bootstrap_median_diff(
    electrode_errs_a: list[list[float]],
    electrode_errs_b: list[list[float]],
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> np.ndarray:
    """Distribución bootstrap de la diferencia de medianas (b - a),
    remuestreando electrodos completos en cada grupo por separado."""
    rng = np.random.default_rng(seed)
    n_a, n_b = len(electrode_errs_a), len(electrode_errs_b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx_a = rng.integers(0, n_a, n_a)
        idx_b = rng.integers(0, n_b, n_b)
        vals_a = [v for j in idx_a for v in electrode_errs_a[j]]
        vals_b = [v for j in idx_b for v in electrode_errs_b[j]]
        diffs[i] = np.median(vals_b) - np.median(vals_a)
    return diffs


def bootstrap_ci(boot_values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """IC de (1-alpha) a partir de una distribución bootstrap (percentiles)."""
    lo, hi = np.percentile(boot_values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_p_two_sided(boot_values: np.ndarray) -> float:
    """p-valor bilateral a partir de una distribución bootstrap: el doble de
    la proporción de remuestreos que caen del lado contrario al signo del
    valor observado (equivalente a comprobar si el IC cruza el cero).
    Metodológicamente consistente con el IC por electrodo, a diferencia de un
    p-valor paramétrico (p.ej. scipy.stats.pearsonr), que trataría los
    ensayos como independientes y subestimaría la incertidumbre real."""
    p_below = float(np.mean(boot_values <= 0))
    p_above = float(np.mean(boot_values >= 0))
    return min(1.0, 2 * min(p_below, p_above))


def report_r(
    results: dict,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> dict:
    """Calcula r, IC95% y p-valor (todos por electrodo) para una sesión ya
    cargada. Devuelve un dict listo para imprimir o insertar en una figura."""
    electrode_ecc, electrode_errs = collect_electrode_data(results)
    r, n_el, n_obs = pearson_r_by_trial(electrode_ecc, electrode_errs)
    boot = cluster_bootstrap_r(electrode_ecc, electrode_errs, n_boot, seed)
    ci_lo, ci_hi = bootstrap_ci(boot)
    p = bootstrap_p_two_sided(boot)
    return {
        "r": r, "r2": r ** 2, "n_el": n_el, "n_obs": n_obs,
        "ci_lo": ci_lo, "ci_hi": ci_hi, "p": p,
    }


def format_r_report(label: str, rep: dict) -> str:
    p_str = "<0,001" if rep["p"] < 0.001 else f"={rep['p']:.3f}".replace(".", ",")
    return (
        f"{label}: r={rep['r']:.3f} (r2~{rep['r2']:.3f}) "
        f"IC95%=[{rep['ci_lo']:.3f}; {rep['ci_hi']:.3f}] p{p_str} "
        f"(n_el={rep['n_el']}, n_obs={rep['n_obs']})"
    )


def mannwhitney_compare(
    label_a: str, electrode_errs_a: list[list[float]],
    label_b: str, electrode_errs_b: list[list[float]],
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> dict:
    """Compara dos grupos de electrodos (p.ej. mismo anillo, dos sesiones):
    Mann-Whitney por ensayo + IC95% bootstrap por electrodo de la diferencia
    de medianas."""
    trials_a = np.array([v for e in electrode_errs_a for v in e])
    trials_b = np.array([v for e in electrode_errs_b for v in e])
    md_a, md_b = float(np.median(trials_a)), float(np.median(trials_b))
    u, p = stats.mannwhitneyu(trials_b, trials_a, alternative="two-sided")
    diffs = cluster_bootstrap_median_diff(electrode_errs_a, electrode_errs_b, n_boot, seed)
    ci_lo, ci_hi = bootstrap_ci(diffs)
    return {
        "label_a": label_a, "label_b": label_b,
        "n_el_a": len(electrode_errs_a), "n_el_b": len(electrode_errs_b),
        "md_a": md_a, "md_b": md_b, "diff": md_b - md_a,
        "u": float(u), "p_mannwhitney": float(p),
        "ci_lo": ci_lo, "ci_hi": ci_hi,
    }
