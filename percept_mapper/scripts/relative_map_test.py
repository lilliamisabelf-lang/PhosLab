"""Tests for scripts/relative_map.py (paired-stimulation map recovery).

Verifies the estimator core without any display or hardware:
  - noiseless displacement-LSQ recovers the map exactly (up to translation)
  - noiseless MDS recovers the map exactly (up to a similarity transform)
  - LSQ needs only 1 anchor; MDS needs >=3 non-collinear anchors to orient
  - a disconnected pair-graph is detected and reported, not silently wrong
  - displacement asymmetry Δ(A→B) != −Δ(B→A) is averaged, not assumed away
  - pair-graph builders produce connected graphs (knn+struts, spanning+redundancy)
  - Procrustes alignment recovers a known rotation/reflection/scale

Run:
    uv run python scripts/relative_map_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.relative_map import (
    build_pair_graph,
    embed_displacement_lsq,
    embed_mds,
    align_procrustes,
    map_error,
    _component_labels,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ring_map(n=12, radius=6.0):
    """N points on a ring + center-ish jitter → a non-degenerate 2D layout."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ecc = radius * (0.4 + 0.6 * ((np.arange(n) % 4) / 3.0))  # 4 eccentricities
    return np.column_stack([ecc * np.cos(ang), ecc * np.sin(ang)])


def _vectors_from_truth(edges, truth, order="forward"):
    """Build clean displacement vectors Δ(i→j) for each edge."""
    vecs = []
    for (i, j) in edges:
        if order == "reverse":
            vecs.append(truth[i] - truth[j])
        else:
            vecs.append(truth[j] - truth[i])
    return np.array(vecs, dtype=np.float64)


def _distances_from_truth(edges, truth):
    return np.array([np.hypot(*(truth[j] - truth[i])) for (i, j) in edges])


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_lsq_noiseless_exact():
    truth = _ring_map(12)
    edges = build_pair_graph(truth, strategy="knn+struts", k=4, n_long=6)
    vecs = _vectors_from_truth(edges, truth)
    anchor = {0: tuple(truth[0])}
    est, info = embed_displacement_lsq(edges, vecs, len(truth), anchors=anchor)
    err = map_error(est, truth)
    assert info["n_components"] == 1, "graph should be connected"
    assert err["max"] < 1e-6, f"noiseless LSQ should be exact, got max={err['max']}"
    print("[test_lsq_noiseless_exact] ok")


def test_lsq_one_anchor_enough():
    """A single anchor fully pins the displacement-LSQ map (translation only)."""
    truth = _ring_map(10)
    edges = build_pair_graph(truth, strategy="knn", k=3)
    vecs = _vectors_from_truth(edges, truth)
    est, _ = embed_displacement_lsq(edges, vecs, len(truth), anchors={3: tuple(truth[3])})
    err = map_error(est, truth)
    assert err["max"] < 1e-6, f"1 anchor should suffice for LSQ, max={err['max']}"
    print("[test_lsq_one_anchor_enough] ok")


def test_lsq_no_anchor_is_translation_only():
    """Without anchors the LSQ map is centroid-zeroed: correct shape, offset by a
    constant translation equal to the truth centroid."""
    truth = _ring_map(8)
    edges = build_pair_graph(truth, strategy="knn+struts", k=4, n_long=4)
    vecs = _vectors_from_truth(edges, truth)
    est, _ = embed_displacement_lsq(edges, vecs, len(truth), anchors=None)
    shift = np.nanmean(truth - est, axis=0)
    residual = (est + shift) - truth
    assert np.nanmax(np.abs(residual)) < 1e-6, "shape must match up to one translation"
    print("[test_lsq_no_anchor_is_translation_only] ok")


def test_mds_noiseless_recovers_shape():
    """Classical MDS on clean distances recovers the map up to a similarity
    transform; after Procrustes onto 3 anchors the error vanishes."""
    truth = _ring_map(12)
    edges = build_pair_graph(truth, strategy="complete")
    dists = _distances_from_truth(edges, truth)
    est, info = embed_mds(edges, dists, len(truth), method="classical")
    anchors = {0: tuple(truth[0]), 4: tuple(truth[4]), 8: tuple(truth[8])}
    aligned, tf = align_procrustes(est, anchors, allow_scale=True, allow_reflection=True)
    err = map_error(aligned, truth)
    assert info["n_components"] == 1
    assert err["median"] < 1e-3, f"noiseless MDS should recover shape, median={err['median']}"
    print("[test_mds_noiseless_recovers_shape] ok")


def test_mds_smacof_matches_classical_noiseless():
    truth = _ring_map(10)
    edges = build_pair_graph(truth, strategy="complete")
    dists = _distances_from_truth(edges, truth)
    est, info = embed_mds(edges, dists, len(truth), method="smacof", n_init=3, seed=1)
    anchors = {0: tuple(truth[0]), 3: tuple(truth[3]), 7: tuple(truth[7])}
    aligned, _ = align_procrustes(est, anchors)
    err = map_error(aligned, truth)
    assert err["median"] < 0.05, f"SMACOF noiseless median too high: {err['median']}"
    print("[test_mds_smacof_matches_classical_noiseless] ok")


def test_disconnected_graph_is_flagged():
    """Two clusters with no edge between them → 2 components reported, and the
    cross-cluster error is undefined (not silently zero)."""
    truth = _ring_map(8)
    # edges only within {0,1,2,3} and within {4,5,6,7}; no bridge.
    edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]
    vecs = _vectors_from_truth(edges, truth)
    est, info = embed_displacement_lsq(edges, vecs, len(truth), anchors={0: tuple(truth[0])})
    assert info["n_components"] == 2, f"expected 2 components, got {info['n_components']}"
    # component without an anchor is centroid-zeroed → not aligned to truth.
    err_second = map_error(est[4:8], truth[4:8])
    assert err_second["median"] > 1e-3, "unanchored component must not match truth by luck"
    print("[test_disconnected_graph_is_flagged] ok")


def test_asymmetry_is_averaged_not_assumed():
    """Feed both Δ(A→B) and Δ(B→A) with an added directional (serial-order) bias.
    The solver treats them as independent constraints, so the recovered edge
    displacement equals the average of the two observations, halving the bias —
    rather than honouring only one direction."""
    truth = _ring_map(6)
    fwd_edges = build_pair_graph(truth, strategy="knn", k=3)
    # directional bias of +0.5 deg in x added to the *travel* direction
    bias = np.array([0.5, 0.0])
    fwd_vecs = _vectors_from_truth(fwd_edges, truth) + bias
    rev_edges = [(j, i) for (i, j) in fwd_edges]
    rev_vecs = _vectors_from_truth(rev_edges, truth) + bias  # bias along reverse travel

    edges = fwd_edges + rev_edges
    vecs = np.vstack([fwd_vecs, rev_vecs])
    est, _ = embed_displacement_lsq(edges, vecs, len(truth), anchors={0: tuple(truth[0])})

    # With both directions, the symmetric part of the bias cancels in the LSQ
    # normal equations: recovered error should be well under the raw 0.5 bias.
    err_both = map_error(est, truth)

    est_fwd, _ = embed_displacement_lsq(fwd_edges, fwd_vecs, len(truth),
                                        anchors={0: tuple(truth[0])})
    err_fwd = map_error(est_fwd, truth)

    assert err_both["median"] < err_fwd["median"], (
        "averaging both directions must reduce directional-bias error "
        f"(both={err_both['median']:.3f} vs fwd-only={err_fwd['median']:.3f})"
    )
    print(f"[test_asymmetry_is_averaged_not_assumed] ok "
          f"(both={err_both['median']:.3f}° < fwd={err_fwd['median']:.3f}°)")


def test_pair_graphs_are_connected():
    truth = _ring_map(15)
    for strat, kw in [
        ("knn+struts", dict(k=3, n_long=4)),
        ("spanning+redundancy", dict(k=3)),
        ("spanning", {}),
        ("complete", {}),
    ]:
        edges = build_pair_graph(truth, strategy=strat, **kw)
        labels = _component_labels(edges, len(truth))
        ncomp = labels.max() + 1
        assert ncomp == 1, f"{strat} produced {ncomp} components (must be connected)"
        # no duplicate / self edges
        assert all(i < j for (i, j) in edges), f"{strat} edges not normalized i<j"
        assert len(set(edges)) == len(edges), f"{strat} has duplicate edges"
    print("[test_pair_graphs_are_connected] ok")


def test_pure_knn_disconnected_gets_bridged():
    """k=1 knn on a ring can fragment; the builder must stitch it to 1 component."""
    truth = _ring_map(12)
    edges = build_pair_graph(truth, strategy="knn", k=1, n_long=0)
    labels = _component_labels(edges, len(truth))
    assert labels.max() + 1 == 1, "knn k=1 must be bridged to a single component"
    print("[test_pure_knn_disconnected_gets_bridged] ok")


def test_procrustes_recovers_known_transform():
    truth = _ring_map(10)
    theta = np.deg2rad(37.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    s = 2.3
    t = np.array([4.0, -1.5])
    transformed = (s * (truth @ R.T)) + t
    # recover truth from transformed using 3 anchors
    anchors = {1: tuple(truth[1]), 5: tuple(truth[5]), 9: tuple(truth[9])}
    aligned, tf = align_procrustes(transformed, anchors)
    err = map_error(aligned, truth)
    assert err["max"] < 1e-6, f"Procrustes should invert a similarity, max={err['max']}"
    assert tf["rmse_anchors"] < 1e-6
    print("[test_procrustes_recovers_known_transform] ok")


def test_mds_under_noise_beats_chance():
    """Sanity: with modest noise, MDS error is small relative to the map extent."""
    rng = np.random.default_rng(0)
    truth = _ring_map(12)
    edges = build_pair_graph(truth, strategy="knn+struts", k=4, n_long=6)
    dists = _distances_from_truth(edges, truth) + rng.normal(0, 0.3, len(edges))
    est, _ = embed_mds(edges, np.abs(dists), len(truth), method="smacof", n_init=3, seed=2)
    anchors = {0: tuple(truth[0]), 4: tuple(truth[4]), 8: tuple(truth[8])}
    aligned, _ = align_procrustes(est, anchors)
    err = map_error(aligned, truth)
    extent = np.ptp(truth, axis=0).max()
    assert err["median"] < 0.25 * extent, (
        f"MDS median error {err['median']:.2f}° too large vs extent {extent:.1f}°")
    print(f"[test_mds_under_noise_beats_chance] ok (median={err['median']:.2f}°)")


# ---------------------------------------------------------------------------

def main() -> int:
    test_lsq_noiseless_exact()
    test_lsq_one_anchor_enough()
    test_lsq_no_anchor_is_translation_only()
    test_mds_noiseless_recovers_shape()
    test_mds_smacof_matches_classical_noiseless()
    test_disconnected_graph_is_flagged()
    test_asymmetry_is_averaged_not_assumed()
    test_pair_graphs_are_connected()
    test_pure_knn_disconnected_gets_bridged()
    test_procrustes_recovers_known_transform()
    test_mds_under_noise_beats_chance()
    print("\n[relative_map_test] todos los tests pasaron ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
