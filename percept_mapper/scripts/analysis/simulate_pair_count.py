"""Experimento 6 (simulación) — Mapeo relativo por pares: error vs. nº de pares.

Responde la pregunta de diseño del método de mapeo *paired*: ¿cuántos pares de
electrodos (y con qué topología de grafo y cuántos anclajes) hacen falta para
recuperar el mapa con un error por debajo del ruido del participante? Y, de paso:
¿recupera mejor el mapa el estimador de **desplazamiento (LSQ)** o el de
**distancias (MDS)**?

No necesita hardware ni pygame. Usa un CSV de implant_explorer como VERDAD
ABSOLUTA (columnas x_deg,y_deg o ecc_deg,polar_deg), simula la respuesta ruidosa
del participante a cada par, y barre:

  1. error vs nº de pares           → error_vs_npairs.png
  2. error vs estrategia de grafo   → error_vs_strategy.png   (a nº de pares igualado)
  3. error vs nº de anclajes        → error_vs_nanchors.png
  4. LSQ vs MDS                     → superpuesto en (1)

Modelo de ruido de respuesta (todo en grados, +x derecha, +y arriba):
  Δ_obs(i→j) = (p_j − p_i)
               + N(0, σ_draw)            ruido de dibujo por endpoint (×√2 por par)
               + serial_bias · û(i→j)    sesgo direccional de orden (memoria A→B)
  con un warp opcional de compresión/expansión radial (magnificación cortical).

Uso (PowerShell):
    cd percept_mapper
    uv run python scripts/analysis/simulate_pair_count.py `
        --csv config/synthetic_4ecc_3elec_17deg.csv `
        --out-dir comparison_results/pair_mapping_sim
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.platform == "win32":
    # La consola de Windows (cp1252) no codifica σ/° → forzar UTF-8 como en
    # los smoke tests del repo.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.relative_map import (
    build_pair_graph,
    embed_displacement_lsq,
    embed_mds,
    align_procrustes,
    map_error,
)

# Estilo coherente con el resto de scripts de analysis/ (TFG).
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 200,
})


# ---------------------------------------------------------------------------
# Ground-truth loader (independiente de dynaphos/torch a propósito)
# ---------------------------------------------------------------------------

def load_truth_csv(csv_path: Path, implant_id_filter: str = "all"):
    """Carga (x_deg, y_deg) verdaderos de un CSV de PhosLab/implant_explorer.

    Acepta (x_deg,y_deg) o (ecc_deg,polar_deg). Devuelve (coords (N,2), labels)
    donde labels[i] = "implant:electrode" para trazabilidad.
    """
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV vacío: {csv_path}")

    fmap = {str(k).strip().lower(): k for k in rows[0].keys()}
    imp_key = fmap.get("implant_id")
    id_key = fmap.get("electrode_index") or fmap.get("contact_id")
    x_key, y_key = fmap.get("x_deg"), fmap.get("y_deg")
    ecc_key = fmap.get("ecc_deg") or fmap.get("ecc")
    pol_key = fmap.get("polar_deg") or fmap.get("polar") or fmap.get("pol")

    if (x_key is None or y_key is None) and (ecc_key is None or pol_key is None):
        raise ValueError(
            "CSV inválido: se requiere (x_deg,y_deg) o (ecc_deg,polar_deg). "
            f"Columnas: {list(rows[0].keys())}"
        )

    if implant_id_filter != "all" and imp_key is not None:
        rows = [r for r in rows
                if str(r.get(imp_key, "")).strip() == str(implant_id_filter)]
        if not rows:
            raise ValueError(f"sin filas con implant_id={implant_id_filter!r}")

    coords, labels = [], []
    for ri, r in enumerate(rows):
        if x_key is not None and y_key is not None:
            x, y = float(r[x_key]), float(r[y_key])
        else:
            ecc, pol = float(r[ecc_key]), math.radians(float(r[pol_key]))
            x, y = ecc * math.cos(pol), ecc * math.sin(pol)
        imp = str(r.get(imp_key, "default")).strip() if imp_key else "default"
        idx = r.get(id_key, ri) if id_key else ri
        coords.append((x, y))
        labels.append(f"{imp}:{idx}")
    return np.array(coords, dtype=np.float64), labels


# ---------------------------------------------------------------------------
# Response-noise model
# ---------------------------------------------------------------------------

def simulate_observations(truth, edges, *, sigma_draw, serial_bias,
                          warp_gain, rng, both_directions=False):
    """Genera Δ_obs y |Δ|_obs ruidosos para cada par.

    Returns (edges_used, vectors_obs (E,2), distances_obs (E,)).
    Si both_directions, añade el par invertido (j→i) con su propio ruido y sesgo
    direccional, lo que permite que el sesgo de orden se promedie en el LSQ.
    """
    use_edges = list(edges)
    if both_directions:
        use_edges = use_edges + [(j, i) for (i, j) in edges]

    vecs, dists = [], []
    for (i, j) in use_edges:
        d_true = truth[j] - truth[i]
        # ruido de dibujo: un endpoint por extremo → ~√2·σ en el vector
        noise = rng.normal(0.0, sigma_draw, size=2) * math.sqrt(2.0)
        # sesgo de orden: a lo largo de la dirección de viaje i→j
        norm = np.hypot(*d_true)
        unit = d_true / norm if norm > 1e-9 else np.zeros(2)
        bias = serial_bias * unit
        # warp radial: comprime/expande según excentricidad media del par
        if warp_gain != 0.0:
            ecc_mid = 0.5 * (np.hypot(*truth[i]) + np.hypot(*truth[j]))
            scale = 1.0 + warp_gain * (ecc_mid)  # lineal en ecc; gain en 1/deg
            d_true = d_true * scale
        v = d_true + noise + bias
        vecs.append(v)
        dists.append(np.hypot(*v))
    return use_edges, np.array(vecs), np.array(dists)


def pick_anchors(truth, n_anchors, rng):
    """Elige n anclajes bien separados (greedy farthest-point) → dict idx:(x,y)."""
    n = truth.shape[0]
    n_anchors = max(1, min(int(n_anchors), n))
    first = int(rng.integers(n))
    chosen = [first]
    while len(chosen) < n_anchors:
        d = np.min(
            [np.hypot(*(truth - truth[c]).T) for c in chosen], axis=0
        )
        d[chosen] = -1.0
        chosen.append(int(np.argmax(d)))
    return {c: tuple(truth[c]) for c in chosen}


# ---------------------------------------------------------------------------
# Estimator runners (return median error in degrees, post-alignment)
# ---------------------------------------------------------------------------

def run_lsq(truth, edges, vecs, anchors, align=False):
    est, info = embed_displacement_lsq(edges, vecs, truth.shape[0], anchors=anchors)
    if align:
        # opcional: re-alinear por Procrustes (corrige escala/rotación si la
        # calibración px→deg no es de fiar). Por defecto el LSQ ya está en grados.
        try:
            est, _ = align_procrustes(est, anchors, allow_scale=True)
        except ValueError:
            pass
    return map_error(est, truth), info


def run_mds(truth, edges, dists, anchors, method="smacof"):
    est, info = embed_mds(edges, dists, truth.shape[0], method=method,
                          n_init=3, seed=0)
    try:
        aligned, _ = align_procrustes(est, anchors, allow_scale=True,
                                      allow_reflection=True)
    except ValueError:
        aligned = est
    return map_error(aligned, truth), info


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def _edge_subset(edges, n_pairs, rng):
    """Submuestra n_pairs aristas manteniendo conectividad lo mejor posible:
    baraja y recorta. (La conectividad se reporta vía n_components del estimador.)"""
    e = list(edges)
    rng.shuffle(e)
    return e[:max(1, min(int(n_pairs), len(e)))]


def sweep_npairs(truth, cfg, rng):
    """Error vs nº de pares, para LSQ y MDS, sobre el grafo 'complete' submuestreado."""
    full = build_pair_graph(truth, strategy="complete")
    max_pairs = len(full)
    n = truth.shape[0]
    # de N-1 (árbol) hasta el total, ~12 puntos log-espaciados
    grid = np.unique(np.clip(
        np.round(np.geomspace(max(2, n - 1), max_pairs, 12)).astype(int),
        2, max_pairs))

    out = {"npairs": [], "lsq_med": [], "lsq_lo": [], "lsq_hi": [],
           "mds_med": [], "mds_lo": [], "mds_hi": [], "frac_connected": []}
    for npairs in grid:
        lsq_errs, mds_errs, conn = [], [], 0
        for _ in range(cfg["reps"]):
            edges = _edge_subset(full, npairs, rng)
            anchors = pick_anchors(truth, cfg["n_anchors"], rng)
            e2, vecs, dists = simulate_observations(
                truth, edges, sigma_draw=cfg["sigma_draw"],
                serial_bias=cfg["serial_bias"], warp_gain=cfg["warp_gain"],
                rng=rng, both_directions=cfg["both_directions"])
            le, linfo = run_lsq(truth, e2, vecs, anchors)
            me, _ = run_mds(truth, e2, dists, anchors)
            if linfo["n_components"] == 1:
                conn += 1
            lsq_errs.append(le["median"])
            mds_errs.append(me["median"])
        out["npairs"].append(int(npairs))
        for key, arr in (("lsq", lsq_errs), ("mds", mds_errs)):
            a = np.array(arr, dtype=float)
            out[f"{key}_med"].append(float(np.nanmedian(a)))
            out[f"{key}_lo"].append(float(np.nanpercentile(a, 25)))
            out[f"{key}_hi"].append(float(np.nanpercentile(a, 75)))
        out["frac_connected"].append(conn / cfg["reps"])
    return out


def sweep_strategy(truth, cfg, rng):
    """Error (LSQ) por estrategia de grafo, a nº de pares comparable."""
    strategies = [
        ("spanning", dict(strategy="spanning")),
        ("knn", dict(strategy="knn", k=3)),
        ("knn+struts", dict(strategy="knn+struts", k=3, n_long=6)),
        ("spanning+redundancy", dict(strategy="spanning+redundancy", k=3)),
        ("complete", dict(strategy="complete")),
    ]
    out = {"labels": [], "npairs": [], "lsq_med": [], "lsq_iqr": [],
           "mds_med": []}
    for name, kw in strategies:
        edges0 = build_pair_graph(truth, **kw)
        lsq_errs, mds_errs = [], []
        for _ in range(cfg["reps"]):
            anchors = pick_anchors(truth, cfg["n_anchors"], rng)
            e2, vecs, dists = simulate_observations(
                truth, edges0, sigma_draw=cfg["sigma_draw"],
                serial_bias=cfg["serial_bias"], warp_gain=cfg["warp_gain"],
                rng=rng, both_directions=cfg["both_directions"])
            le, _ = run_lsq(truth, e2, vecs, anchors)
            me, _ = run_mds(truth, e2, dists, anchors)
            lsq_errs.append(le["median"])
            mds_errs.append(me["median"])
        out["labels"].append(name)
        out["npairs"].append(len(edges0))
        out["lsq_med"].append(float(np.nanmedian(lsq_errs)))
        out["lsq_iqr"].append(float(np.nanpercentile(lsq_errs, 75)
                                    - np.nanpercentile(lsq_errs, 25)))
        out["mds_med"].append(float(np.nanmedian(mds_errs)))
    return out


def sweep_nanchors(truth, cfg, rng):
    """Error vs nº de anclajes (LSQ con 1 basta; MDS necesita ≥3)."""
    n = truth.shape[0]
    grid = [a for a in (1, 2, 3, 4, 6, 8) if a <= n]
    edges0 = build_pair_graph(truth, strategy="knn+struts", k=3, n_long=6)
    out = {"nanchors": [], "lsq_med": [], "mds_med": []}
    for na in grid:
        lsq_errs, mds_errs = [], []
        for _ in range(cfg["reps"]):
            anchors = pick_anchors(truth, na, rng)
            e2, vecs, dists = simulate_observations(
                truth, edges0, sigma_draw=cfg["sigma_draw"],
                serial_bias=cfg["serial_bias"], warp_gain=cfg["warp_gain"],
                rng=rng, both_directions=cfg["both_directions"])
            le, _ = run_lsq(truth, e2, vecs, anchors)
            me, _ = run_mds(truth, e2, dists, anchors)
            lsq_errs.append(le["median"])
            mds_errs.append(me["median"])
        out["nanchors"].append(na)
        out["lsq_med"].append(float(np.nanmedian(lsq_errs)))
        out["mds_med"].append(float(np.nanmedian(mds_errs)))
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_npairs(res, cfg, path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = res["npairs"]
    ax.fill_between(x, res["lsq_lo"], res["lsq_hi"], alpha=0.15, color="C0")
    ax.plot(x, res["lsq_med"], "o-", color="C0", label="Displacement LSQ")
    ax.fill_between(x, res["mds_lo"], res["mds_hi"], alpha=0.15, color="C1")
    ax.plot(x, res["mds_med"], "s-", color="C1", label="MDS (distances)")
    ax.axhline(cfg["sigma_draw"], ls=":", color="gray",
               label=f"σ_draw = {cfg['sigma_draw']}° (noise floor)")
    ax.set_xlabel("Número de pares medidos")
    ax.set_ylabel("Error mediano de localización (°)")
    ax.set_title("Recuperación del mapa vs. número de pares")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_strategy(res, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    xs = np.arange(len(res["labels"]))
    w = 0.38
    ax.bar(xs - w / 2, res["lsq_med"], w, label="LSQ", color="C0")
    ax.bar(xs + w / 2, res["mds_med"], w, label="MDS", color="C1")
    for i, (np_, lm) in enumerate(zip(res["npairs"], res["lsq_med"])):
        ax.text(i, max(res["lsq_med"][i], res["mds_med"][i]),
                f"{np_}p", ha="center", va="bottom", fontsize=9, color="dimgray")
    ax.set_xticks(xs)
    ax.set_xticklabels(res["labels"], rotation=20, ha="right")
    ax.set_ylabel("Error mediano (°)")
    ax.set_title("Error por estrategia de grafo (etiqueta = nº de pares)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_nanchors(res, path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(res["nanchors"], res["lsq_med"], "o-", color="C0", label="LSQ")
    ax.plot(res["nanchors"], res["mds_med"], "s-", color="C1", label="MDS")
    ax.set_xlabel("Número de anclajes absolutos")
    ax.set_ylabel("Error mediano (°)")
    ax.set_title("Sensibilidad al número de anclajes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True,
                   help="CSV de verdad absoluta (implant_explorer / PhosLab)")
    p.add_argument("--out-dir", type=Path,
                   default=Path("comparison_results/pair_mapping_sim"))
    p.add_argument("--implant-id", default="all")
    p.add_argument("--reps", type=int, default=80,
                   help="repeticiones Monte-Carlo por punto del barrido")
    p.add_argument("--sigma-draw", type=float, default=0.7,
                   help="σ del ruido de dibujo por endpoint (grados)")
    p.add_argument("--serial-bias", type=float, default=0.3,
                   help="sesgo direccional de orden A→B (grados)")
    p.add_argument("--warp-gain", type=float, default=0.0,
                   help="ganancia de warp radial por grado de ecc (0 = sin warp)")
    p.add_argument("--n-anchors", type=int, default=3,
                   help="anclajes usados en los barridos de pares/estrategia")
    p.add_argument("--both-directions", action="store_true",
                   help="medir cada par en ambos sentidos (promedia el sesgo de orden)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    truth, labels = load_truth_csv(args.csv, args.implant_id)
    valid = ~np.any(np.isnan(truth), axis=1)
    truth = truth[valid]
    n = truth.shape[0]
    if n < 3:
        print(f"✗ Se necesitan ≥3 electrodos válidos; el CSV tiene {n}.")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cfg = dict(reps=args.reps, sigma_draw=args.sigma_draw,
               serial_bias=args.serial_bias, warp_gain=args.warp_gain,
               n_anchors=args.n_anchors, both_directions=args.both_directions)

    print("=" * 70)
    print("SIMULACIÓN — Mapeo relativo por pares (error vs nº de pares)")
    print("=" * 70)
    print(f"CSV:            {args.csv}")
    print(f"Electrodos:     {n}  (pares posibles = {n*(n-1)//2})")
    print(f"σ_draw:         {args.sigma_draw}°   serial_bias: {args.serial_bias}°"
          f"   warp_gain: {args.warp_gain}")
    print(f"Anclajes:       {args.n_anchors}   both_directions: {args.both_directions}"
          f"   reps: {args.reps}")
    print("-" * 70)

    res_np = sweep_npairs(truth, cfg, rng)
    res_st = sweep_strategy(truth, cfg, rng)
    res_an = sweep_nanchors(truth, cfg, rng)

    plot_npairs(res_np, cfg, args.out_dir / "error_vs_npairs.png")
    plot_strategy(res_st, args.out_dir / "error_vs_strategy.png")
    plot_nanchors(res_an, args.out_dir / "error_vs_nanchors.png")

    # CSV resumen para el TFG
    summary = args.out_dir / "sim_summary.csv"
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sweep", "x_label", "x_value", "lsq_median_deg",
                    "mds_median_deg", "extra"])
        for i, npairs in enumerate(res_np["npairs"]):
            w.writerow(["npairs", "n_pairs", npairs, res_np["lsq_med"][i],
                        res_np["mds_med"][i], f"frac_conn={res_np['frac_connected'][i]:.2f}"])
        for i, lab in enumerate(res_st["labels"]):
            w.writerow(["strategy", lab, res_st["npairs"][i], res_st["lsq_med"][i],
                        res_st["mds_med"][i], ""])
        for i, na in enumerate(res_an["nanchors"]):
            w.writerow(["nanchors", "n_anchors", na, res_an["lsq_med"][i],
                        res_an["mds_med"][i], ""])

    # Recomendación automática: primer nº de pares cuyo LSQ cae bajo el ruido.
    floor = args.sigma_draw
    rec_pairs = None
    for npairs, med in zip(res_np["npairs"], res_np["lsq_med"]):
        if med <= floor:
            rec_pairs = npairs
            break
    best_strat_i = int(np.argmin(res_st["lsq_med"]))

    print("\nRESULTADOS")
    print("-" * 70)
    print(f"LSQ @ pocos pares: {res_np['lsq_med'][0]:.2f}°  →  "
          f"@ máx pares: {res_np['lsq_med'][-1]:.2f}°")
    print(f"MDS @ pocos pares: {res_np['mds_med'][0]:.2f}°  →  "
          f"@ máx pares: {res_np['mds_med'][-1]:.2f}°")
    if rec_pairs is not None:
        print(f"➜ LSQ alcanza el suelo de ruido (≤{floor:.2f}°) con ~{rec_pairs} pares.")
    else:
        print(f"➜ LSQ no baja de {floor:.2f}° en el rango probado "
              f"(mín {min(res_np['lsq_med']):.2f}°).")
    print(f"➜ Mejor estrategia (LSQ): {res_st['labels'][best_strat_i]} "
          f"({res_st['npairs'][best_strat_i]} pares, "
          f"{res_st['lsq_med'][best_strat_i]:.2f}°).")
    print(f"\n✓ Figuras y sim_summary.csv en: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
