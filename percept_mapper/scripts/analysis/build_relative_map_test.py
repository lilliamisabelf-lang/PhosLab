"""Tests para scripts/analysis/build_relative_map.py (reconstrucción pareada).

Sin hardware: sintetiza una sesión `pairs/` en disco (metadata.json +
session_metadata.json) con geometría conocida, escribe los extremos en píxeles
usando la MISMA convención px↔deg que el analizador, y verifica que:

  - PairedSession recupera Δ_obs en grados a partir de los endpoints en px
  - LSQ con endpoints limpios reconstruye el mapa casi exactamente
  - respuestas partial/empty se descartan (no contaminan)
  - direcciones inversas (A→B y B→A) se promedian (mismo par, una arista)
  - el CLI completo corre y escribe recovered_map.csv + overlay
  - sin coords verdaderas, reconstruye igual (solo que sin error)

Ejecución:
    uv run python scripts/analysis/build_relative_map_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.build_relative_map import (
    PairedSession,
    reconstruct,
    main as cli_main,
)
from scripts.relative_map import build_pair_graph, map_error

# Geometría px↔deg fija para los tests (debe coincidir con la del analizador).
PPD = 50.0
CENTER = (960.0, 540.0)


def _deg_to_px(xy_deg):
    """Inversa de _px_to_deg del analizador: +y arriba → y de pantalla invertida."""
    x = CENTER[0] + xy_deg[0] * PPD
    y = CENTER[1] - xy_deg[1] * PPD
    return [int(round(x)), int(round(y))]


def _ring(n=8, radius=6.0):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ecc = radius * (0.5 + 0.5 * ((np.arange(n) % 3) / 2.0))
    return np.column_stack([ecc * np.cos(ang), ecc * np.sin(ang)])


def _write_session(tmp, truth, edges, *, both_directions=False,
                   include_truth=True, partial_every=0, empty_every=0,
                   noise=0.0, seed=0):
    """Escribe una sesión pareada sintética. Devuelve la carpeta pairs/."""
    rng = np.random.default_rng(seed)
    pairs_dir = Path(tmp) / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    node_to_electrode = [100 + i for i in range(truth.shape[0])]  # ids no triviales

    directed = [(i, j) for (i, j) in edges]
    if both_directions:
        directed = directed + [(j, i) for (i, j) in edges]

    trials = []
    for k, (i, j) in enumerate(directed):
        pa = truth[i] + (rng.normal(0, noise, 2) if noise else 0.0)
        pb = truth[j] + (rng.normal(0, noise, 2) if noise else 0.0)
        a_px, b_px = _deg_to_px(pa), _deg_to_px(pb)
        status = "ok"
        if partial_every and (k % partial_every == 0):
            status, b_px = "partial", None
        elif empty_every and (k % empty_every == 0):
            status, a_px, b_px = "empty", None, None
        trials.append({
            "pair_index": k + 1,
            "electrode_a": node_to_electrode[i],
            "electrode_b": node_to_electrode[j],
            "response_status": status,
            "endpoint_a_px": a_px,
            "endpoint_b_px": b_px,
            "displacement_px": (None if (a_px is None or b_px is None)
                                else [b_px[0] - a_px[0], b_px[1] - a_px[1]]),
            "is_practice": False,
        })

    meta = {
        "mapping_method": "paired",
        "display": {
            "screen_center_px": [int(CENTER[0]), int(CENTER[1])],
            "screen_resolution_px": [1920, 1080],
            "pixels_per_degree_x": PPD,
            "pixels_per_degree_y": PPD,
        },
        "trials": trials,
    }
    (pairs_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    session = {
        "mapping_method": "paired",
        "node_to_electrode": node_to_electrode,
    }
    if include_truth:
        session["coords_deg_true"] = truth.tolist()
    (pairs_dir / "session_metadata.json").write_text(
        json.dumps(session, indent=2), encoding="utf-8")
    return pairs_dir


# ---------------------------------------------------------------------------

def test_clean_lsq_recovers_map():
    truth = _ring(8)
    edges = build_pair_graph(truth, strategy="knn+struts", k=4, n_long=4)
    with tempfile.TemporaryDirectory() as tmp:
        pairs_dir = _write_session(tmp, truth, edges)
        sess = PairedSession(pairs_dir)
        assert sess.n_nodes == 8
        assert len(sess.edges) == len(edges), "todas las aristas deben recuperarse"
        anchors = sess.anchors_from_true(n_anchors=1)
        rec = reconstruct(sess, anchors)
        # endpoints limpios → recuperación casi exacta (solo redondeo a px)
        err = map_error(rec["lsq"], truth)
        assert err["median"] < 0.05, f"LSQ limpio debe ser exacto, med={err['median']}"
        assert rec["lsq_info"]["n_components"] == 1
    print(f"[test_clean_lsq_recovers_map] ok (med={err['median']:.3f}°)")


def test_partial_and_empty_discarded():
    truth = _ring(8)
    edges = build_pair_graph(truth, strategy="complete")
    with tempfile.TemporaryDirectory() as tmp:
        pairs_dir = _write_session(tmp, truth, edges,
                                   partial_every=5, empty_every=7)
        sess = PairedSession(pairs_dir)
        assert sess.n_partial > 0 and sess.n_empty > 0, "deben existir descartes"
        # ninguna arista debe provenir de un trial partial/empty
        n_kept = sess.n_ok
        assert len(sess.edges) <= n_kept
        # con los pares restantes aún se reconstruye sin error grande
        anchors = sess.anchors_from_true(n_anchors=1)
        rec = reconstruct(sess, anchors)
        err = map_error(rec["lsq"], truth)
        assert err["median"] < 0.1, f"med={err['median']}"
    print(f"[test_partial_and_empty_discarded] ok "
          f"(partial={sess.n_partial}, empty={sess.n_empty}, edges={len(sess.edges)})")


def test_both_directions_averaged_to_single_edge():
    truth = _ring(6)
    edges = build_pair_graph(truth, strategy="knn", k=3)
    with tempfile.TemporaryDirectory() as tmp:
        pairs_dir = _write_session(tmp, truth, edges, both_directions=True)
        sess = PairedSession(pairs_dir)
        # 2× trials dirigidos, pero deben colapsar a |edges| aristas no dirigidas
        assert sess.n_ok == 2 * len(edges)
        assert len(sess.edges) == len(edges), \
            "A→B y B→A deben promediarse en una sola arista"
        anchors = sess.anchors_from_true(n_anchors=1)
        rec = reconstruct(sess, anchors)
        assert map_error(rec["lsq"], truth)["median"] < 0.05
    print("[test_both_directions_averaged_to_single_edge] ok")


def test_no_truth_still_reconstructs():
    truth = _ring(7)
    edges = build_pair_graph(truth, strategy="knn+struts", k=3, n_long=3)
    with tempfile.TemporaryDirectory() as tmp:
        pairs_dir = _write_session(tmp, truth, edges, include_truth=False)
        sess = PairedSession(pairs_dir)
        assert sess.coords_true_deg is None
        # sin verdad no se pueden elegir anclas farthest-point → pasar explícitas
        # (usando posiciones absolutas conocidas como referencia de orientación)
        try:
            sess.anchors_from_true(n_anchors=1)
            raise AssertionError("debe exigir anclas explícitas sin verdad")
        except ValueError:
            pass
        # con 1 ancla explícita (nodo 0 en su posición real) sí reconstruye
        anchors = {0: tuple(truth[0])}
        rec = reconstruct(sess, anchors)
        assert rec["lsq"].shape == (7, 2)
    print("[test_no_truth_still_reconstructs] ok")


def test_cli_writes_outputs():
    truth = _ring(8)
    edges = build_pair_graph(truth, strategy="knn+struts", k=4, n_long=4)
    with tempfile.TemporaryDirectory() as tmp:
        pairs_dir = _write_session(tmp, truth, edges)
        out_dir = Path(tmp) / "out"
        rc = cli_main([
            "--session", str(pairs_dir),
            "--out-dir", str(out_dir),
            "--n-anchors", "3",
            "--reps", "3",
        ])
        assert rc == 0
        assert (out_dir / "recovered_map.csv").exists()
        assert (out_dir / "recovery_overlay.png").exists()
        assert (out_dir / "recovery_report.txt").exists()
        # el CSV debe tener una fila por electrodo con columnas verdad+lsq
        rows = (out_dir / "recovered_map.csv").read_text(encoding="utf-8").splitlines()
        assert len(rows) == 1 + 8, "header + 8 electrodos"
        assert "lsq_x_deg" in rows[0] and "true_x_deg" in rows[0]
    print("[test_cli_writes_outputs] ok")


# ---------------------------------------------------------------------------

def main() -> int:
    test_clean_lsq_recovers_map()
    test_partial_and_empty_discarded()
    test_both_directions_averaged_to_single_edge()
    test_no_truth_still_reconstructs()
    test_cli_writes_outputs()
    print("\n[build_relative_map_test] todos los tests pasaron ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
