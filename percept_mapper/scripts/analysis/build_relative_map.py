"""Experimento 6 (análisis) — Reconstruye el mapa desde una sesión PAREADA.

Toma una sesión capturada con `mapping_method: paired` (carpeta `pairs/` con
`metadata.json` + `session_metadata.json`), extrae los extremos ordenados de
cada línea trazada (endpoint_a_px, endpoint_b_px), los convierte a grados
relativos a la fijación, y reconstruye el mapa con AMBOS estimadores:

  • Displacement-LSQ : embed por mínimos cuadrados del grafo de desplazamientos
                       (Δ = pos_B − pos_A). Necesita 1 ancla; sin ambigüedad de
                       rotación/reflexión. Es el estimador primario.
  • MDS (distancias) : Torgerson/SMACOF sobre |Δ|. Baseline; necesita ≥3 anclas
                       y se orienta por Procrustes.

Las anclas absolutas (posiciones conocidas de unos pocos electrodos) orientan
el mapa. En modo evaluación usamos posiciones VERDADERAS del session_metadata
como anclas para poder medir el error de recuperación sobre el resto.

Salidas (en --out-dir, por defecto <session>/relative_map):
  recovered_map.csv          posición recuperada por electrodo (LSQ y MDS, en °)
  recovery_overlay.png       verdad vs recuperado con flechas de error
  error_vs_npairs.png        error al submuestrear los pares capturados
  recovery_report.txt        resumen numérico

Uso (PowerShell):
    cd percept_mapper
    uv run python scripts/analysis/build_relative_map.py `
        --session mapping_experiments/mapping_<ts>/pairs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.platform == "win32":
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
# Carga de la sesión capturada
# ---------------------------------------------------------------------------

def _resolve_pairs_dir(session_path: Path) -> Path:
    """Acepta tanto .../pairs como la carpeta de experimento que la contiene."""
    session_path = Path(session_path)
    if (session_path / "metadata.json").exists():
        return session_path
    cand = session_path / "pairs"
    if (cand / "metadata.json").exists():
        return cand
    raise FileNotFoundError(
        f"No se encontró metadata.json en {session_path} ni en {cand}. "
        "Pasa la carpeta 'pairs/' de una sesión pareada."
    )


def _px_to_deg(xy_px, screen_center, ppd_x, ppd_y):
    """Convención del pipeline (mapping_analyzer._px_to_deg): origen en fijación,
    +x derecha, +y ARRIBA (y de pantalla invertida)."""
    x = (float(xy_px[0]) - float(screen_center[0])) / float(ppd_x)
    y = -(float(xy_px[1]) - float(screen_center[1])) / float(ppd_y)
    return np.array([x, y], dtype=np.float64)


class PairedSession:
    """Vista cargada de una sesión pareada lista para reconstruir el mapa.

    Atributos clave:
      node_to_electrode : list[int]   nodo del grafo (0..N-1) → electrode_index
      coords_true_deg   : (N,2) | None  geometría verdadera (si está en session_meta)
      edges             : list[(i,j)]   pares dirigidos efectivamente medidos (status ok)
      vectors_obs       : (E,2)         Δ_obs en grados, promediado si hay reverso
      distances_obs     : (E,)         |Δ|_obs en grados
      n_partial/n_empty : conteos de respuestas descartadas
    """

    def __init__(self, pairs_dir: Path):
        self.pairs_dir = Path(pairs_dir)
        meta = json.loads((self.pairs_dir / "metadata.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.display = meta.get("display") or {}

        sess_path = self.pairs_dir / "session_metadata.json"
        self.session = (
            json.loads(sess_path.read_text(encoding="utf-8"))
            if sess_path.exists() else {}
        )

        # px → deg
        sc = self.display.get("screen_center_px") or [
            (self.display.get("screen_resolution_px") or [0, 0])[0] / 2,
            (self.display.get("screen_resolution_px") or [0, 0])[1] / 2,
        ]
        self.screen_center = (float(sc[0]), float(sc[1]))
        self.ppd_x = float(self.display.get("pixels_per_degree_x") or 60.0)
        self.ppd_y = float(self.display.get("pixels_per_degree_y") or self.ppd_x)

        # node ↔ electrode
        n2e = self.session.get("node_to_electrode") or \
            self.session.get("valid_electrode_indices")
        if not n2e:
            # Reconstruir desde los electrodos vistos en los trials (orden estable)
            seen = []
            for t in meta.get("trials", []):
                for k in ("electrode_a", "electrode_b"):
                    e = t.get(k)
                    if e is not None and e not in seen:
                        seen.append(int(e))
            n2e = sorted(seen)
        self.node_to_electrode = [int(e) for e in n2e]
        self.electrode_to_node = {e: i for i, e in enumerate(self.node_to_electrode)}
        self.n_nodes = len(self.node_to_electrode)

        ct = self.session.get("coords_deg_true")
        self.coords_true_deg = (
            np.asarray(ct, dtype=np.float64) if ct else None
        )

        self._extract_observations()

    # ------------------------------------------------------------------ #

    def _extract_observations(self):
        """Construye Δ_obs por par a partir de los extremos dibujados. Promedia
        las dos direcciones (i→j y j→i) si ambas se midieron, lo que cancela el
        sesgo de orden serial. Descarta pares partial/empty."""
        acc: dict[tuple[int, int], list[np.ndarray]] = {}
        self.n_ok = self.n_partial = self.n_empty = self.n_skipped = 0

        for t in self.meta.get("trials", []):
            if t.get("is_practice"):
                continue
            status = (t.get("response_status") or "").lower()
            ea, eb = t.get("electrode_a"), t.get("electrode_b")
            a_px, b_px = t.get("endpoint_a_px"), t.get("endpoint_b_px")
            if status == "empty" or ea is None or eb is None:
                self.n_empty += 1
                continue
            if status == "partial" or a_px is None or b_px is None:
                self.n_partial += 1
                continue
            if ea not in self.electrode_to_node or eb not in self.electrode_to_node:
                self.n_skipped += 1
                continue

            i = self.electrode_to_node[ea]
            j = self.electrode_to_node[eb]
            pa = _px_to_deg(a_px, self.screen_center, self.ppd_x, self.ppd_y)
            pb = _px_to_deg(b_px, self.screen_center, self.ppd_x, self.ppd_y)
            delta = pb - pa  # Δ(A→B) = pos_B − pos_A en grados

            # Clave no dirigida (i<j); guardamos el vector orientado i→j
            if i < j:
                acc.setdefault((i, j), []).append(delta)
            else:
                acc.setdefault((j, i), []).append(-delta)
            self.n_ok += 1

        self.edges = sorted(acc.keys())
        self.vectors_obs = np.array(
            [np.mean(acc[e], axis=0) for e in self.edges], dtype=np.float64
        ) if self.edges else np.zeros((0, 2))
        self.distances_obs = (
            np.hypot(self.vectors_obs[:, 0], self.vectors_obs[:, 1])
            if len(self.edges) else np.zeros((0,))
        )

    # ------------------------------------------------------------------ #

    def anchors_from_true(self, indices=None, n_anchors=3, rng=None):
        """dict node→(x,y) de anclas a partir de la geometría VERDADERA.

        Si `indices` (electrode_index) se da, se usan esos; si no, se eligen
        n_anchors por farthest-point. Requiere coords_true_deg (modo evaluación).
        """
        if self.coords_true_deg is None:
            raise ValueError(
                "No hay coords verdaderas en session_metadata; pasa --anchors "
                "con posiciones absolutas conocidas para orientar el mapa."
            )
        truth = self.coords_true_deg
        if indices:
            nodes = [self.electrode_to_node[int(e)] for e in indices
                     if int(e) in self.electrode_to_node]
        else:
            rng = rng or np.random.default_rng(0)
            n_anchors = max(1, min(int(n_anchors), self.n_nodes))
            first = int(rng.integers(self.n_nodes))
            nodes = [first]
            while len(nodes) < n_anchors:
                d = np.min([np.hypot(*(truth - truth[c]).T) for c in nodes], axis=0)
                d[nodes] = -1.0
                nodes.append(int(np.argmax(d)))
        return {c: tuple(truth[c]) for c in nodes}


# ---------------------------------------------------------------------------
# Reconstrucción
# ---------------------------------------------------------------------------

def reconstruct(sess: PairedSession, anchors):
    """Corre LSQ y MDS sobre los pares observados; devuelve dict con estimaciones
    (en grados) e info de componentes/stress. MDS se alinea por Procrustes."""
    n = sess.n_nodes
    lsq_est, lsq_info = embed_displacement_lsq(
        sess.edges, sess.vectors_obs, n, anchors=anchors
    )
    mds_aligned = None
    mds_info = {}
    if len(sess.edges) >= 1 and len(anchors) >= 1:
        mds_est, mds_info = embed_mds(
            sess.edges, sess.distances_obs, n, method="smacof", n_init=4, seed=0
        )
        try:
            mds_aligned, _ = align_procrustes(
                mds_est, anchors, allow_scale=True, allow_reflection=True
            )
        except ValueError:
            mds_aligned = mds_est
    return {
        "lsq": lsq_est, "lsq_info": lsq_info,
        "mds": mds_aligned, "mds_info": mds_info,
    }


def _subset_error_vs_npairs(sess, anchors, reps=40, seed=0):
    """Submuestrea los pares capturados y mide el error mediano (LSQ y MDS) vs
    nº de pares — la curva de convergencia con DATOS REALES, no simulados.
    Solo disponible en modo evaluación (hay verdad)."""
    if sess.coords_true_deg is None:
        return None
    rng = np.random.default_rng(seed)
    E = len(sess.edges)
    n = sess.n_nodes
    truth = sess.coords_true_deg
    grid = np.unique(np.clip(
        np.round(np.geomspace(max(2, n - 1), E, min(10, E))).astype(int), 2, E))
    out = {"npairs": [], "lsq_med": [], "mds_med": [], "frac_connected": []}
    idx_all = np.arange(E)
    for npairs in grid:
        lsq_e, mds_e, conn = [], [], 0
        for _ in range(reps):
            sel = rng.choice(idx_all, size=int(npairs), replace=False)
            edges = [sess.edges[s] for s in sel]
            vecs = sess.vectors_obs[sel]
            dists = sess.distances_obs[sel]
            le, linfo = embed_displacement_lsq(edges, vecs, n, anchors=anchors)
            err_l = map_error(le, truth)
            lsq_e.append(err_l["median"])
            if linfo["n_components"] == 1:
                conn += 1
            me, _ = embed_mds(edges, dists, n, method="smacof", n_init=2, seed=0)
            try:
                ma, _ = align_procrustes(me, anchors, allow_scale=True,
                                         allow_reflection=True)
            except ValueError:
                ma = me
            mds_e.append(map_error(ma, truth)["median"])
        out["npairs"].append(int(npairs))
        out["lsq_med"].append(float(np.nanmedian(lsq_e)))
        out["mds_med"].append(float(np.nanmedian(mds_e)))
        out["frac_connected"].append(conn / reps)
    return out


# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------

def write_csv(sess, rec, out_csv):
    truth = sess.coords_true_deg
    lsq, mds = rec["lsq"], rec["mds"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["electrode_index", "node",
                    "true_x_deg", "true_y_deg",
                    "lsq_x_deg", "lsq_y_deg",
                    "mds_x_deg", "mds_y_deg"])
        for node, e in enumerate(sess.node_to_electrode):
            tx = ty = ""
            if truth is not None:
                tx, ty = float(truth[node, 0]), float(truth[node, 1])
            lx, ly = float(lsq[node, 0]), float(lsq[node, 1])
            if mds is not None:
                mx, my = float(mds[node, 0]), float(mds[node, 1])
            else:
                mx = my = ""
            w.writerow([e, node, tx, ty, lx, ly, mx, my])


def plot_overlay(sess, rec, anchors, out_png):
    truth = sess.coords_true_deg
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, key, title in ((axes[0], "lsq", "Displacement-LSQ"),
                           (axes[1], "mds", "MDS (distances)")):
        est = rec[key]
        ax.axhline(0, color="0.8", lw=0.8)
        ax.axvline(0, color="0.8", lw=0.8)
        ax.plot(0, 0, "+", color="0.4", ms=12, mew=2)  # fijación
        if est is None:
            ax.set_title(f"{title} (no disponible)")
            continue
        if truth is not None:
            for node in range(sess.n_nodes):
                ax.annotate("", xy=est[node], xytext=truth[node],
                            arrowprops=dict(arrowstyle="->", color="crimson",
                                            lw=1.0, alpha=0.8))
            ax.scatter(truth[:, 0], truth[:, 1], s=45, c="black",
                       label="Verdad", zorder=3)
        ax.scatter(est[:, 0], est[:, 1], s=30, marker="x", c="crimson",
                   label="Recuperado", zorder=4)
        for node in anchors:
            ax.scatter(*est[node], s=130, facecolors="none",
                       edgecolors="C0", lw=2, zorder=5)
        sub = ""
        if truth is not None:
            err = map_error(est, truth)
            sub = (f"  med={err['median']:.2f}°  p95={err['p95']:.2f}°"
                   f"  max={err['max']:.2f}°")
        info = rec[f"{key}_info"]
        ncomp = info.get("n_components", 1)
        comp = f"  ({ncomp} comp.)" if ncomp > 1 else ""
        ax.set_title(f"{title}{sub}{comp}")
        ax.set_xlabel("x (°)")
        ax.set_ylabel("y (°)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", fontsize=9)
    n_anchor = len(anchors)
    fig.suptitle(
        f"Mapa pareado recuperado — {sess.n_ok} pares OK, "
        f"{len(sess.edges)} aristas, {n_anchor} anclas "
        f"(○ azul = ancla)", fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out_png)
    plt.close(fig)


def plot_npairs(res, out_png):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = res["npairs"]
    ax.plot(x, res["lsq_med"], "o-", color="C0", label="Displacement LSQ")
    ax.plot(x, res["mds_med"], "s-", color="C1", label="MDS (distances)")
    ax.set_xlabel("Número de pares (submuestreo de los capturados)")
    ax.set_ylabel("Error mediano de localización (°)")
    ax.set_title("Convergencia con los pares realmente medidos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", type=Path, required=True,
                   help="carpeta 'pairs/' (o el experimento que la contiene)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="por defecto <session>/relative_map")
    p.add_argument("--anchors", default="",
                   help="electrode_index de anclas, separados por coma "
                        "(ej. 1,4,8). Por defecto: farthest-point sobre la verdad.")
    p.add_argument("--n-anchors", type=int, default=3,
                   help="nº de anclas si no se pasan --anchors")
    p.add_argument("--reps", type=int, default=40,
                   help="repeticiones del submuestreo error-vs-pares")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    pairs_dir = _resolve_pairs_dir(args.session)
    out_dir = args.out_dir or (pairs_dir.parent / "relative_map")
    out_dir.mkdir(parents=True, exist_ok=True)

    sess = PairedSession(pairs_dir)
    if sess.n_nodes < 2:
        print("✗ La sesión tiene <2 electrodos; no se puede reconstruir.")
        return 1
    if len(sess.edges) < 1:
        print("✗ No hay pares válidos (todos partial/empty). Nada que reconstruir.")
        return 1

    anchor_idx = [int(s) for s in args.anchors.split(",") if s.strip()] or None
    rng = np.random.default_rng(args.seed)
    anchors = sess.anchors_from_true(
        indices=anchor_idx, n_anchors=args.n_anchors, rng=rng)

    print("=" * 70)
    print("RECONSTRUCCIÓN DEL MAPA PAREADO")
    print("=" * 70)
    print(f"Sesión:     {pairs_dir}")
    print(f"Electrodos: {sess.n_nodes}   pares (aristas): {len(sess.edges)}")
    print(f"Respuestas: ok={sess.n_ok}  partial={sess.n_partial}  "
          f"empty={sess.n_empty}  skipped={sess.n_skipped}")
    print(f"px→deg:     centro={sess.screen_center}  "
          f"ppd=({sess.ppd_x:.1f},{sess.ppd_y:.1f})")
    print(f"Anclas:     nodos {sorted(anchors.keys())} "
          f"(electrodos {[sess.node_to_electrode[n] for n in sorted(anchors)]})")
    print("-" * 70)

    rec = reconstruct(sess, anchors)

    write_csv(sess, rec, out_dir / "recovered_map.csv")
    plot_overlay(sess, rec, anchors, out_dir / "recovery_overlay.png")

    res_np = _subset_error_vs_npairs(sess, anchors, reps=args.reps, seed=args.seed)
    if res_np is not None and len(res_np["npairs"]) >= 2:
        plot_npairs(res_np, out_dir / "error_vs_npairs.png")

    # Reporte
    lines = []
    lines.append("RECONSTRUCCIÓN DEL MAPA PAREADO")
    lines.append(f"Sesión: {pairs_dir}")
    lines.append(f"Electrodos: {sess.n_nodes}   aristas: {len(sess.edges)}")
    lines.append(f"Respuestas ok/partial/empty/skipped: "
                 f"{sess.n_ok}/{sess.n_partial}/{sess.n_empty}/{sess.n_skipped}")
    lines.append(f"LSQ componentes: {rec['lsq_info'].get('n_components')}")
    if sess.coords_true_deg is not None:
        err_l = map_error(rec["lsq"], sess.coords_true_deg)
        lines.append(f"LSQ error: median={err_l['median']:.3f}°  "
                     f"p95={err_l['p95']:.3f}°  max={err_l['max']:.3f}°  "
                     f"(n={err_l['n_recovered']}/{err_l['n_total']})")
        if rec["mds"] is not None:
            err_m = map_error(rec["mds"], sess.coords_true_deg)
            lines.append(f"MDS error: median={err_m['median']:.3f}°  "
                         f"p95={err_m['p95']:.3f}°  max={err_m['max']:.3f}°")
        winner = "LSQ"
        if rec["mds"] is not None and \
           map_error(rec["mds"], sess.coords_true_deg)["median"] < err_l["median"]:
            winner = "MDS"
        lines.append(f"➜ Mejor estimador en esta sesión: {winner}")
    else:
        lines.append("(sin verdad → solo mapa recuperado, sin error)")
    report = "\n".join(lines)
    (out_dir / "recovery_report.txt").write_text(report + "\n", encoding="utf-8")

    print("\n" + report)
    print(f"\n✓ Resultados en: {out_dir}")
    print("   recovered_map.csv · recovery_overlay.png · recovery_report.txt"
          + (" · error_vs_npairs.png" if res_np is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
