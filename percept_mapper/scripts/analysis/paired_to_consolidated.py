"""Convierte una sesión de mapeo PAREADO en consolidated_results.json.

A partir de la carpeta de sesión con `pairs/metadata.json`:
  1. Extrae observaciones directas de endpoints (endpoint_a/b_px → °) por electrodo
  2. Reconstruye el mapa con Displacement-LSQ (+ MDS de respaldo)
  3. Escribe `consolidated_analysis/` con el mismo formato que multi_electrode_analyzer

Los 'centroids_deg' son los endpoints crudos por ensayo (observaciones directas del
usuario antes de consolidar con MDS/LSQ). La posición final 'mean_position_deg' usa
el estimador LSQ (o MDS si LSQ queda fragmentado en varias componentes).

Uso:
    cd percept_mapper
    uv run python scripts/analysis/paired_to_consolidated.py \\
        --session mapping_experiments/mapping_mapeo_multiples_electrodo_20260630_194015
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.build_relative_map import (
    PairedSession,
    _resolve_pairs_dir,
    _px_to_deg,
    reconstruct,
)
from scripts.relative_map import map_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_endpoint_obs(
    sess: PairedSession,
) -> dict[int, list[tuple[float, float]]]:
    """Para cada electrodo, lista de posiciones directas (°) de sus endpoints."""
    obs: dict[int, list[tuple[float, float]]] = {}
    for t in sess.meta.get("trials", []):
        if t.get("is_practice"):
            continue
        status = (t.get("response_status") or "").lower()
        ea, eb = t.get("electrode_a"), t.get("electrode_b")
        a_px, b_px = t.get("endpoint_a_px"), t.get("endpoint_b_px")
        if status == "empty" or ea is None or eb is None:
            continue
        if a_px is None or b_px is None:
            continue
        a_deg = _px_to_deg(a_px, sess.screen_center, sess.ppd_x, sess.ppd_y)
        b_deg = _px_to_deg(b_px, sess.screen_center, sess.ppd_x, sess.ppd_y)
        obs.setdefault(int(ea), []).append((float(a_deg[0]), float(a_deg[1])))
        obs.setdefault(int(eb), []).append((float(b_deg[0]), float(b_deg[1])))
    return obs


def _best_estimate(rec: dict, sess: PairedSession) -> np.ndarray:
    """Devuelve la estimación LSQ si el grafo es conexo, MDS en caso contrario."""
    n_comp = rec["lsq_info"].get("n_components", 1)
    if n_comp == 1:
        return rec["lsq"]
    if rec["mds"] is not None:
        return rec["mds"]
    return rec["lsq"]


# ---------------------------------------------------------------------------
# Construcción del JSON consolidado
# ---------------------------------------------------------------------------

def build_consolidated(
    sess: PairedSession,
    rec: dict,
    obs_by_elec: dict[int, list[tuple[float, float]]],
) -> dict:
    """Devuelve el dict consolidated_results compatible con compare_mapmethod.py."""
    est = _best_estimate(rec, sess)
    truth = sess.coords_true_deg  # (N,2) array o None

    meta = sess.meta
    session_meta = sess.session

    electrodes_out: dict[str, dict] = {}

    for node, elec_idx in enumerate(sess.node_to_electrode):
        # Posición recuperada (estimador LSQ/MDS)
        rx, ry = float(est[node, 0]), float(est[node, 1])

        # Posición verdadera del estímulo
        if truth is not None:
            tx, ty = float(truth[node, 0]), float(truth[node, 1])
        else:
            tx = ty = None

        # Observaciones directas de endpoints (centroids crudos)
        raw_pts: list[tuple[float, float]] = obs_by_elec.get(elec_idx, [])
        centroids_deg = [[x, y] for x, y in raw_pts]

        # Per-repetition metrics (distancia de cada endpoint crudo al estímulo)
        per_rep_metrics: list[dict] = []
        for rep_num, (cx, cy) in enumerate(raw_pts, start=1):
            if tx is not None:
                dx = cx - tx
                dy = cy - ty
                dist = math.hypot(dx, dy)
            else:
                dx = dy = dist = None
            per_rep_metrics.append({
                "repetition_number": rep_num,
                "dx_to_stim_deg": dx,
                "dy_to_stim_deg": dy,
                "distance_to_stim_deg": dist,
            })

        # Std de los centroids crudos
        if len(raw_pts) > 1:
            arr = np.array(raw_pts)
            std_x = float(np.std(arr[:, 0], ddof=1))
            std_y = float(np.std(arr[:, 1], ddof=1))
        else:
            std_x = std_y = 0.0

        # Error de la estimación final vs verdad
        if tx is not None:
            dist_est = math.hypot(rx - tx, ry - ty)
            offset_x = rx - tx
            offset_y = ry - ty
        else:
            dist_est = offset_x = offset_y = None

        electrodes_out[str(elec_idx)] = {
            "electrode_index": elec_idx,
            "num_total_repetitions": len(raw_pts),
            "num_valid_repetitions": len(raw_pts),
            "num_invalid_repetitions": 0,
            "centroids_deg": centroids_deg,
            "mean_position_deg": {"x": rx, "y": ry},
            "std_position": {"x": std_x, "y": std_y},
            "stimulation_position_deg": [tx, ty] if tx is not None else None,
            "per_repetition_metrics": per_rep_metrics,
            "distance_mean_to_stimulus_deg": dist_est,
            "offset_mean_to_stimulus_deg": (
                {"x": offset_x, "y": offset_y} if offset_x is not None else None
            ),
            "status": "ok" if len(raw_pts) > 0 else "no_data",
        }

    # Posiciones medias globales
    mean_positions = {
        str(elec_idx): [
            electrodes_out[str(elec_idx)]["mean_position_deg"]["x"],
            electrodes_out[str(elec_idx)]["mean_position_deg"]["y"],
        ]
        for elec_idx in sess.node_to_electrode
    }

    started = session_meta.get("session_started", "")
    ended = meta.get("end_time", "")

    return {
        "experiment_name": meta.get("experiment_name", meta.get("experiment_id", "")),
        "num_electrodes": sess.n_nodes,
        "mapping_method": "paired",
        "estimator": "lsq" if rec["lsq_info"].get("n_components", 1) == 1 else "mds",
        "n_pairs": len(sess.edges),
        "n_ok": sess.n_ok,
        "n_partial": sess.n_partial,
        "n_empty": sess.n_empty,
        "session_started": started,
        "session_ended": ended,
        "electrodes": electrodes_out,
        "mean_positions": mean_positions,
        "no_response_electrodes": [
            elec_idx for elec_idx in sess.node_to_electrode
            if electrodes_out[str(elec_idx)]["status"] == "no_data"
        ],
    }


# ---------------------------------------------------------------------------
# Reporte de texto
# ---------------------------------------------------------------------------

def write_report(consolidated: dict, sess: PairedSession, rec: dict, out: Path) -> None:
    truth = sess.coords_true_deg
    est = _best_estimate(rec, sess)
    estimator = consolidated["estimator"].upper()

    lines = [
        "=" * 64,
        "MAPA PAREADO — ANÁLISIS CONSOLIDADO",
        "=" * 64,
        f"Sesión:      {sess.pairs_dir.parent}",
        f"Electrodos:  {sess.n_nodes}   pares (aristas): {len(sess.edges)}",
        f"Respuestas:  ok={sess.n_ok}  parcial={sess.n_partial}  "
        f"vacío={sess.n_empty}  omitido={sess.n_skipped}",
        f"Estimador:   {estimator}  "
        f"(componentes LSQ: {rec['lsq_info'].get('n_components', 1)})",
        "-" * 64,
    ]

    if truth is not None:
        err = map_error(est, truth)
        lines += [
            f"Error de recuperación vs verdad:",
            f"  Mediana : {err['median']:.3f}°",
            f"  P95     : {err['p95']:.3f}°",
            f"  Máximo  : {err['max']:.3f}°",
            f"  n       : {err['n_recovered']}/{err['n_total']}",
        ]
        if rec["mds"] is not None and consolidated["estimator"] == "lsq":
            err_m = map_error(rec["mds"], truth)
            lines.append(
                f"  [MDS ref]: med={err_m['median']:.3f}°  p95={err_m['p95']:.3f}°"
            )
    else:
        lines.append("(sin posiciones verdaderas — solo mapa recuperado)")

    lines += ["", "Por electrodo:"]
    header = f"  {'Elec':>5}  {'True X':>7}  {'True Y':>7}  "
    header += f"{'Est X':>7}  {'Est Y':>7}  {'Error':>7}  {'n_obs':>5}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for elec_idx in sess.node_to_electrode:
        rec_e = consolidated["electrodes"][str(elec_idx)]
        tx_ty = rec_e["stimulation_position_deg"]
        tx = f"{tx_ty[0]:>7.3f}" if tx_ty and tx_ty[0] is not None else f"{'---':>7}"
        ty = f"{tx_ty[1]:>7.3f}" if tx_ty and tx_ty[1] is not None else f"{'---':>7}"
        ex = f"{rec_e['mean_position_deg']['x']:>7.3f}"
        ey = f"{rec_e['mean_position_deg']['y']:>7.3f}"
        err_val = rec_e["distance_mean_to_stimulus_deg"]
        er = f"{err_val:>7.3f}°" if err_val is not None else f"{'---':>8}"
        n_obs = rec_e["num_valid_repetitions"]
        lines.append(f"  {elec_idx:>5}  {tx}  {ty}  {ex}  {ey}  {er}  {n_obs:>5}")

    lines += [
        "-" * 64,
        f"Generado: {datetime.now(timezone.utc).isoformat()}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Reporte: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", type=Path, required=True,
                   help="carpeta de sesión pareada (o su subcarpeta pairs/)")
    p.add_argument("--anchors", default="",
                   help="electrode_index de anclas, separados por coma (ej. 1,4,8). "
                        "Por defecto: farthest-point sobre la geometría verdadera.")
    p.add_argument("--n-anchors", type=int, default=3,
                   help="nº de anclas si no se pasan --anchors")
    p.add_argument("--estimator", choices=["auto", "lsq", "mds"], default="auto",
                   help="estimador a usar como mean_position_deg (default: auto)")
    args = p.parse_args(argv)

    pairs_dir = _resolve_pairs_dir(args.session)
    sess = PairedSession(pairs_dir)

    if sess.n_nodes < 2:
        print("✗ La sesión tiene <2 electrodos.")
        return 1
    if len(sess.edges) < 1:
        print("✗ No hay pares válidos (todos partial/empty). Nada que reconstruir.")
        return 1

    anchor_idx = [int(s) for s in args.anchors.split(",") if s.strip()] or None
    rng = np.random.default_rng(0)
    anchors = sess.anchors_from_true(
        indices=anchor_idx, n_anchors=args.n_anchors, rng=rng)

    print(f"[paired] Sesión:     {pairs_dir.parent}")
    print(f"[paired] Electrodos: {sess.n_nodes}  pares: {len(sess.edges)}")
    print(f"[paired] Respuestas: ok={sess.n_ok}  partial={sess.n_partial}  "
          f"empty={sess.n_empty}")
    print(f"[paired] Anclas:     nodos {sorted(anchors.keys())}")

    rec = reconstruct(sess, anchors)

    if args.estimator == "mds":
        if rec["mds"] is None:
            print("✗ MDS no disponible (requiere ≥1 par).")
            return 1
        # fuerza MDS sobreescribiendo n_components ficticiamente
        rec["lsq_info"]["n_components"] = 99
    elif args.estimator == "lsq":
        # fuerza LSQ aunque esté fragmentado
        rec["lsq_info"]["n_components"] = 1

    obs_by_elec = _collect_endpoint_obs(sess)
    consolidated = build_consolidated(sess, rec, obs_by_elec)

    out_dir = pairs_dir.parent / "consolidated_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_out = out_dir / "consolidated_results.json"
    json_out.write_text(
        json.dumps(consolidated, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] JSON:   {json_out}")

    write_report(consolidated, sess, rec, out_dir / "consolidated_report.txt")

    # Estadísticas globales
    errors = [
        e["distance_mean_to_stimulus_deg"]
        for e in consolidated["electrodes"].values()
        if e["distance_mean_to_stimulus_deg"] is not None
    ]
    if errors:
        print(f"\n[paired] Error global — "
              f"mediana={float(np.median(errors)):.3f}°  "
              f"media={float(np.mean(errors)):.3f}°  "
              f"n={len(errors)}")
    else:
        print("\n[paired] No se pudo calcular error (sin posiciones verdaderas).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
