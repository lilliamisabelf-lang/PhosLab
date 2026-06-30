"""Relative (paired-stimulation) phosphene map recovery.

Pure-math core for the *paired* mapping method: each trial stimulates two
electrodes A → B and the participant draws a directed line A→B. That line gives
either a **displacement vector** Δ(A,B) = pos(B) − pos(A) (preferred) or just a
**scalar distance** |Δ| (MDS baseline). From many pairs we recover the map.

Two estimators (compared head-to-head in scripts/analysis/simulate_pair_count.py):

- ``embed_displacement_lsq`` — least-squares graph embedding from displacement
  vectors. Linear, closed-form per axis, recovers the map up to a single global
  **translation** (orientation/scale come free from the vectors). Needs only one
  anchor to pin translation.

- ``embed_mds`` — classical / SMACOF MDS from the scalar distance matrix. Sparse
  graphs are densified with graph shortest-paths (Isomap-style). Recovers the map
  up to a **similarity** transform (rotation + reflection + translation, and scale
  if the metric is not absolute), so it needs ≥3 non-collinear anchors to orient.

Both estimates are brought into the absolute (degrees) frame with
``align_procrustes`` against a handful of anchor electrodes whose absolute
positions were measured with the existing single-phosphene method.

Everything here is a pure function of its arguments: numpy at runtime, scipy for
the graph shortest-path used by the MDS densifier. No I/O, no global state, safe
to import from a notebook. Conventions match the rest of the pipeline: degrees,
origin at fixation, +x right, +y up.
"""

from __future__ import annotations

import numpy as np

# scipy is already a hard dependency of the project (dynaphos / stats.ellipse).
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components


# ---------------------------------------------------------------------------
# Pair-graph construction
# ---------------------------------------------------------------------------

def build_pair_graph(
    coords_deg,
    strategy: str = "knn+struts",
    k: int = 4,
    n_long: int = 6,
    seed: int | None = 0,
):
    """Choose which electrode pairs to probe.

    The recovered map's quality depends on the *topology* of the pair-graph, not
    just the pair count: a graph must be **connected** (otherwise components float
    independently) and ideally **rigid** (otherwise it folds/accordions under
    noise). This function builds an undirected edge list from the predicted
    (atlas) positions.

    Args:
        coords_deg: (N, 2) predicted positions used only to decide proximity.
        strategy: one of
            - ``"complete"``           every pair (N·(N−1)/2 edges).
            - ``"knn"``                each node to its k nearest neighbours.
            - ``"knn+struts"``         knn plus ``n_long`` random long-range edges
                                       to control global drift (recommended).
            - ``"spanning"``           a single nearest-neighbour spanning tree
                                       (N−1 edges, the bare minimum, no redundancy).
            - ``"spanning+redundancy"`` spanning tree plus knn for averaging.
        k: neighbours per node for the knn-based strategies.
        n_long: number of long-range "strut" edges for ``knn+struts``.
        seed: RNG seed for the random strut selection (reproducible).

    Returns:
        list[tuple[int, int]] of edges with i < j, de-duplicated.
    """
    coords = np.asarray(coords_deg, dtype=np.float64)
    n = coords.shape[0]
    if n < 2:
        return []

    def _undirected(pairs):
        out = set()
        for i, j in pairs:
            if i == j:
                continue
            out.add((i, j) if i < j else (j, i))
        return sorted(out)

    # Pairwise euclidean distances in predicted space.
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.hypot(diff[..., 0], diff[..., 1])

    if strategy == "complete":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    if strategy in ("knn", "knn+struts", "spanning+redundancy"):
        kk = max(1, min(int(k), n - 1))
        knn_edges = []
        for i in range(n):
            # nearest kk excluding self
            order = np.argsort(dmat[i])
            nbrs = [j for j in order if j != i][:kk]
            knn_edges.extend((i, j) for j in nbrs)
        edges = set(_undirected(knn_edges))

        if strategy in ("knn", "knn+struts"):
            if strategy == "knn+struts" and n_long > 0:
                edges |= _strut_edges(dmat, n_long, seed)
            # A pure-knn graph can still be disconnected; stitch components.
            edges |= _bridge_components(dmat, edges, n)
            return sorted(edges)

        # spanning+redundancy: guarantee a spanning tree then add knn.
        edges |= set(_spanning_tree(dmat, n))
        edges |= _bridge_components(dmat, edges, n)
        return sorted(edges)

    if strategy == "spanning":
        return _spanning_tree(dmat, n)

    raise ValueError(f"unknown pair-graph strategy: {strategy!r}")


def _spanning_tree(dmat, n):
    """Prim's algorithm on the dense distance matrix → minimum spanning tree."""
    if n < 2:
        return []
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    edges = []
    # best[j] = (dist, i) cheapest edge connecting unvisited j to the tree.
    best_dist = dmat[0].copy()
    best_src = np.zeros(n, dtype=int)
    best_dist[0] = np.inf
    for _ in range(n - 1):
        j = int(np.argmin(np.where(in_tree, np.inf, best_dist)))
        i = int(best_src[j])
        edges.append((i, j) if i < j else (j, i))
        in_tree[j] = True
        # relax
        closer = dmat[j] < best_dist
        closer &= ~in_tree
        best_dist[closer] = dmat[j][closer]
        best_src[closer] = j
    return sorted(set(edges))


def _strut_edges(dmat, n_long, seed):
    """Pick ``n_long`` long-range edges (top quartile of distances) at random."""
    n = dmat.shape[0]
    rng = np.random.default_rng(seed)
    iu, ju = np.triu_indices(n, k=1)
    d = dmat[iu, ju]
    if d.size == 0:
        return set()
    thresh = np.quantile(d, 0.75)
    cand = np.where(d >= thresh)[0]
    if cand.size == 0:
        cand = np.arange(d.size)
    pick = rng.choice(cand, size=min(int(n_long), cand.size), replace=False)
    return {(int(iu[p]), int(ju[p])) for p in pick}


def _bridge_components(dmat, edges, n):
    """If ``edges`` leaves the graph disconnected, add the shortest cross-component
    edge repeatedly until a single component remains. Returns only the added
    edges."""
    added = set()
    cur = set(edges)
    while True:
        labels = _component_labels(cur, n)
        ncomp = labels.max() + 1 if n else 0
        if ncomp <= 1:
            return added
        # find shortest edge joining two different components
        best = None
        for a in range(n):
            for b in range(a + 1, n):
                if labels[a] != labels[b]:
                    if best is None or dmat[a, b] < best[0]:
                        best = (dmat[a, b], a, b)
        if best is None:
            return added  # shouldn't happen for n >= 2
        _, a, b = best
        e = (a, b) if a < b else (b, a)
        added.add(e)
        cur.add(e)


def _component_labels(edges, n):
    if n == 0:
        return np.zeros(0, dtype=int)
    rows, cols, data = [], [], []
    for i, j in edges:
        rows += [i, j]
        cols += [j, i]
        data += [1.0, 1.0]
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    return labels


# ---------------------------------------------------------------------------
# Estimator 1 — displacement least-squares graph embedding
# ---------------------------------------------------------------------------

def embed_displacement_lsq(edges, vectors_obs, n_points, anchors=None):
    """Recover positions from observed displacement vectors.

    Solves, independently per axis, the linear least-squares problem

        min_p  Σ_(i,j)∈E  ‖(p_j − p_i) − Δ_obs(i,j)‖²

    The edge-incidence (difference) operator B maps positions to edge
    displacements; we solve ``B p ≈ Δ`` per axis. B has a one-dimensional null
    space per connected component (a constant shift), so the solution is fixed by
    pinning anchors (or, with no anchors, by zero-centering each component — the
    map is then correct up to translation only).

    Args:
        edges: list of (i, j) with the SAME orientation used to record Δ.
        vectors_obs: (E, 2) observed Δ(i→j) in degrees, row r matches edges[r].
        n_points: N, total electrode count (some may be unconnected → NaN out).
        anchors: optional dict {index: (x_deg, y_deg)} of known absolute
            positions. If given, the result is shifted so anchors match in a
            least-squares sense (per connected component that contains ≥1 anchor).

    Returns:
        (coords_est (N,2) float, info dict). ``info['n_components']`` and
        ``info['component_labels']`` expose connectivity; disconnected nodes with
        no edges are returned as NaN.
    """
    edges = list(edges)
    E = len(edges)
    vectors_obs = np.asarray(vectors_obs, dtype=np.float64).reshape(E, 2)
    n = int(n_points)

    if E == 0:
        coords = np.full((n, 2), np.nan)
        return coords, {"n_components": n, "component_labels": np.arange(n)}

    # Build incidence operator B (E × N): row r has −1 at i, +1 at j.
    B = np.zeros((E, n), dtype=np.float64)
    for r, (i, j) in enumerate(edges):
        B[r, i] = -1.0
        B[r, j] = +1.0

    labels = _component_labels(edges, n)
    ncomp = int(labels.max()) + 1 if n else 0

    coords = np.full((n, 2), np.nan, dtype=np.float64)

    # Solve per connected component so each component's gauge (translation) is
    # handled independently. Nodes touched by no edge stay NaN.
    touched = np.zeros(n, dtype=bool)
    for r, (i, j) in enumerate(edges):
        touched[i] = touched[j] = True

    for c in range(ncomp):
        nodes = np.where((labels == c) & touched)[0]
        if nodes.size == 0:
            continue
        node_pos = {node: idx for idx, node in enumerate(nodes)}
        rows = [r for r, (i, j) in enumerate(edges)
                if labels[i] == c and labels[j] == c]
        if not rows:
            continue
        Bc = np.zeros((len(rows), nodes.size), dtype=np.float64)
        dc = np.zeros((len(rows), 2), dtype=np.float64)
        for rr, r in enumerate(rows):
            i, j = edges[r]
            Bc[rr, node_pos[i]] = -1.0
            Bc[rr, node_pos[j]] = +1.0
            dc[rr] = vectors_obs[r]
        # Pin the gauge: append a row that fixes the component centroid to 0.
        # This makes the system full-rank; absolute placement is restored later
        # by the anchor alignment step.
        pin = np.ones((1, nodes.size), dtype=np.float64)
        A = np.vstack([Bc, pin])
        rhs = np.vstack([dc, np.zeros((1, 2))])
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        coords[nodes] = sol

    info = {"n_components": ncomp, "component_labels": labels}

    if anchors:
        coords = _shift_to_anchors(coords, labels, anchors)

    return coords, info


def _shift_to_anchors(coords, labels, anchors):
    """Translate each connected component so its anchors match absolute positions
    (least-squares mean shift). Components without an anchor are left as-is
    (centroid-zeroed)."""
    coords = coords.copy()
    anchor_idx = np.array(sorted(anchors.keys()), dtype=int)
    for c in np.unique(labels):
        comp_anchors = [a for a in anchor_idx if labels[a] == c
                        and not np.any(np.isnan(coords[a]))]
        if not comp_anchors:
            continue
        est = coords[comp_anchors]
        tgt = np.array([anchors[a] for a in comp_anchors], dtype=np.float64)
        shift = np.mean(tgt - est, axis=0)
        comp_nodes = np.where(labels == c)[0]
        coords[comp_nodes] += shift
    return coords


# ---------------------------------------------------------------------------
# Estimator 2 — MDS from the scalar distance matrix (baseline)
# ---------------------------------------------------------------------------

def embed_mds(edges, distances_obs, n_points, method="smacof",
              n_init=4, max_iter=300, seed=0, return_trace=False):
    """Recover positions from observed scalar distances (MDS baseline).

    The pair-graph is usually sparse, so the missing entries of the N×N distance
    matrix are filled with **graph shortest-path** distances (Isomap-style) before
    running MDS. The result is correct only up to a similarity transform
    (rotation + reflection + translation; scale too if distances aren't absolute),
    which ``align_procrustes`` resolves with ≥3 anchors.

    Args:
        edges: list of (i, j).
        distances_obs: (E,) observed |Δ(i,j)| in degrees, row r matches edges[r].
        n_points: N.
        method: ``"smacof"`` (stress-majorization, robust) or ``"classical"``
            (eigendecomposition of the double-centered matrix, fast).
        n_init: SMACOF restarts (best stress kept).
        max_iter: SMACOF iterations per restart.
        seed: RNG seed for SMACOF init.
        return_trace: if True (SMACOF only), info also carries ``stress_trace``,
            the per-iteration stress of the best restart (for convergence plots).

    Returns:
        (coords_est (N,2) float, info dict with ``stress`` and ``n_components``).
    """
    edges = list(edges)
    distances_obs = np.asarray(distances_obs, dtype=np.float64).ravel()
    n = int(n_points)

    if n == 0:
        return np.zeros((0, 2)), {"stress": 0.0, "n_components": 0}

    # Sparse graph of measured distances → dense geodesic distance matrix.
    rows, cols, data = [], [], []
    for r, (i, j) in enumerate(edges):
        d = float(distances_obs[r])
        rows += [i, j]
        cols += [j, i]
        data += [d, d]
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    ncomp, labels = connected_components(graph, directed=False)

    D = shortest_path(graph, method="D", directed=False)
    # Disconnected pairs come back as inf; fill with the largest finite distance
    # so MDS doesn't blow up. (Disconnection is reported via ncomp regardless.)
    finite = D[np.isfinite(D)]
    fill = (finite.max() * 1.5) if finite.size else 1.0
    D = np.where(np.isfinite(D), D, fill)
    np.fill_diagonal(D, 0.0)

    trace = None
    if method == "classical":
        coords = _classical_mds(D)
        stress = _mds_stress(D, coords)
    elif method == "smacof":
        coords, stress, trace = _smacof(
            D, n_init=n_init, max_iter=max_iter, seed=seed)
    else:
        raise ValueError(f"unknown MDS method: {method!r}")

    info = {"stress": float(stress), "n_components": int(ncomp),
            "component_labels": labels}
    if return_trace and trace is not None:
        info["stress_trace"] = trace
    return coords, info


def _classical_mds(D, n_dims=2):
    """Classical (Torgerson) MDS: double-center −½D², take top eigenvectors."""
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    # Symmetrize against numerical drift before eigendecomposition.
    B = (B + B.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1][:n_dims]
    L = np.clip(eigvals[order], 0.0, None)
    return eigvecs[:, order] * np.sqrt(L)


def _smacof(D, n_dims=2, n_init=4, max_iter=300, eps=1e-9, seed=0):
    """SMACOF (Scaling by MAjorizing a COmplicated Function). Minimizes raw
    stress Σ_{i<j} (‖x_i − x_j‖ − D_ij)² by iterative majorization. Best of
    ``n_init`` random restarts (plus one classical-MDS warm start) is returned."""
    n = D.shape[0]
    rng = np.random.default_rng(seed)
    best_coords, best_stress, best_trace = None, np.inf, []

    inits = [_classical_mds(D, n_dims)]
    for _ in range(max(0, n_init - 1)):
        inits.append(rng.standard_normal((n, n_dims)) * (D.mean() + eps))

    for X in inits:
        X = X.astype(np.float64).copy()
        prev = np.inf
        trace = []
        for _ in range(max_iter):
            Dx = _pairwise(X)
            stress = _mds_stress(D, X, Dx=Dx)
            trace.append(stress)
            if abs(prev - stress) < 1e-10:
                break
            prev = stress
            # Guttman transform
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(Dx > eps, D / Dx, 0.0)
            Bm = -ratio
            np.fill_diagonal(Bm, 0.0)
            Bm[np.diag_indices(n)] = -Bm.sum(axis=1)
            X = (Bm @ X) / n
        if stress < best_stress:
            best_stress, best_coords, best_trace = stress, X, trace

    return best_coords, best_stress, best_trace


def _pairwise(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def _mds_stress(D, X, Dx=None):
    if Dx is None:
        Dx = _pairwise(X)
    iu = np.triu_indices(D.shape[0], k=1)
    return float(np.sum((Dx[iu] - D[iu]) ** 2))


# ---------------------------------------------------------------------------
# Anchor alignment (Procrustes) — bring an estimate into the absolute frame
# ---------------------------------------------------------------------------

def align_procrustes(coords_est, anchors, allow_scale=True, allow_reflection=True):
    """Least-squares similarity alignment of an estimate onto anchor positions.

    Finds rotation R (optionally with reflection), scale s, translation t that
    minimise Σ‖s·R·est_a + t − true_a‖² over the anchor electrodes ``a``, then
    applies that transform to ALL points. Use this to orient an MDS estimate
    (needs ≥3 non-collinear anchors) or to scale/rotate a displacement-LSQ
    estimate if you don't fully trust the screen calibration.

    Args:
        coords_est: (N, 2) estimate (NaN rows are carried through untouched).
        anchors: dict {index: (x_deg, y_deg)} of known absolute positions.
        allow_scale: if False, s is fixed to 1 (rigid alignment).
        allow_reflection: if False, reflections are rejected (proper rotation).

    Returns:
        (coords_aligned (N,2), transform dict with R, s, t, rmse_anchors).
    """
    coords_est = np.asarray(coords_est, dtype=np.float64)
    idx = np.array(sorted(anchors.keys()), dtype=int)
    if idx.size < 1:
        raise ValueError("align_procrustes needs at least one anchor")

    src = coords_est[idx]
    dst = np.array([anchors[i] for i in idx], dtype=np.float64)
    if np.any(np.isnan(src)):
        raise ValueError("anchor electrode has no estimated position (NaN)")

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d

    # Single anchor: translation only (R=I, s=1).
    if idx.size == 1:
        R = np.eye(2)
        s = 1.0
        t = mu_d - mu_s
    else:
        H = src_c.T @ dst_c
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if not allow_reflection and np.linalg.det(R) < 0:
            Vt2 = Vt.copy()
            Vt2[-1] *= -1
            R = Vt2.T @ U.T
        if allow_scale:
            var_s = (src_c ** 2).sum()
            s = (S.sum() / var_s) if var_s > 1e-12 else 1.0
        else:
            s = 1.0
        t = mu_d - s * (R @ mu_s)

    aligned = np.full_like(coords_est, np.nan)
    ok = ~np.any(np.isnan(coords_est), axis=1)
    aligned[ok] = (s * (coords_est[ok] @ R.T)) + t

    res = (s * (src @ R.T) + t) - dst
    rmse = float(np.sqrt((res ** 2).sum(axis=1).mean()))
    return aligned, {"R": R, "s": float(s), "t": t, "rmse_anchors": rmse}


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def map_error(coords_est, coords_true):
    """Per-point and aggregate localization error (degrees), after alignment.

    Args:
        coords_est: (N, 2) recovered positions (may contain NaN rows).
        coords_true: (N, 2) ground-truth positions.

    Returns:
        dict with per_point (N,) euclidean error (NaN where unrecovered),
        mean / median / p95 / max over the recovered points, and n_recovered.
    """
    est = np.asarray(coords_est, dtype=np.float64)
    true = np.asarray(coords_true, dtype=np.float64)
    err = np.sqrt(((est - true) ** 2).sum(axis=1))
    valid = ~np.isnan(err)
    ev = err[valid]
    if ev.size == 0:
        return {"per_point": err, "mean": float("nan"), "median": float("nan"),
                "p95": float("nan"), "max": float("nan"), "n_recovered": 0,
                "n_total": int(true.shape[0])}
    return {
        "per_point": err,
        "mean": float(np.mean(ev)),
        "median": float(np.median(ev)),
        "p95": float(np.percentile(ev, 95)),
        "max": float(np.max(ev)),
        "n_recovered": int(ev.size),
        "n_total": int(true.shape[0]),
    }
