import time
import numpy as np
from scipy.cluster.hierarchy import fcluster
import torch
import heapq
from scipy import sparse
import matplotlib.pyplot as plt

VERBOSE = False

def _reindex_labels_np(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reindex an integer label map into a compact range [0, K-1].

    Parameters
    ----------
    L : np.ndarray
        Integer array of arbitrary, possibly non-contiguous labels.

    Returns
    -------
    np.ndarray
        Label map with labels remapped to [0, K-1], same shape as `L`.
    np.ndarray
        Lookup table such that `remap[old_label] = new_label`.
    """
    L = L.astype(np.int64, copy=False)
    uniq = np.unique(L)
    remap = np.full(uniq.max() + 1, -1, dtype=np.int64)
    remap[uniq] = np.arange(uniq.size, dtype=np.int64)
    return remap[L], remap


def _build_rag_np(L):
    """
    Build a Region Adjacency Graph (RAG) from a 2D label map.

    Parameters
    ----------
    L : np.ndarray
        2D array of integer region labels.

    Returns
    -------
    np.ndarray
        Array of shape (E, 2) with undirected edges (u, v) between
        neighboring, distinct regions.
    """
    start = time.time()
    H, W = L.shape
    a = L[:, :-1].ravel()
    b = L[:, 1:].ravel()
    m = a != b
    e1 = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)[m]
    a = L[:-1, :].ravel()
    b = L[1:, :].ravel()
    m = a != b
    e2 = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)[m]
    end = time.time()
    if VERBOSE:
        print(f"Time to build RAG: {end - start:.8f} s")
    return np.unique(np.vstack([e1, e2]), axis=0)


def compute_region_attention(attention, superpixels, normalize=True):
    """
    Aggregate an attention map into per-region attention values.

    Parameters
    ----------
    attention : np.ndarray
        Either a pixel-wise attention map of shape (H, W[, C]) or an
        already per-region vector of shape (N,).
    superpixels : np.ndarray
        2D integer superpixel map of shape (H, W).
    normalize : bool, optional
        If True, min-max normalize the resulting region scores to [0, 1].

    Returns
    -------
    np.ndarray
        1D array of length N with attention aggregated per region.
    """
    sp = superpixels.astype(np.int32)
    N = int(sp.max()) + 1

    # Already per-region
    att = np.asarray(attention)
    if att.ndim == 1 and att.shape[0] == N:
        out = att.astype(np.float32, copy=False)
        if normalize:
            lo, hi = float(out.min()), float(out.max())
            out = (out - lo) / (hi - lo + 1e-6)
        return out

    # Pixel-wise map (H,W[,C])
    if att.ndim == 3:
        att = att.mean(axis=2)
    assert att.shape == sp.shape, "attention_map must have same HxW as superpixels"

    lab = sp.ravel()
    vals = att.astype(np.float32).ravel()
    counts = np.bincount(lab, minlength=N).astype(np.float32)
    sums = np.bincount(lab, weights=vals, minlength=N).astype(np.float32)
    out = sums / (counts + 1e-6)

    if normalize:
        lo, hi = float(out.min()), float(out.max())
        out = (out - lo) / (hi - lo + 1e-6)

    return out.astype(np.float32)


def _edge_cost_torch_phaseA(edges: torch.Tensor, state: dict, hp: dict) -> torch.Tensor:
    """
    Compute pairwise edge costs for phase A of the hierarchical merge.

    This phase optionally combines deep/color features and positional
    features, plus an attention prior and hard object gating.

    Parameters
    ----------
    edges : torch.Tensor
        Tensor of shape (E, 2) with integer indices (u, v).
    state : dict
        Dictionary holding current region statistics (features, sizes,
        attention, priors, etc.).
    hp : dict
        Hyper-parameters controlling the cost (e.g. `w_pos`, `att_weight`).

    Returns
    -------
    torch.Tensor
        1D tensor of length E with edge scores.
    """
    mu = state["mu"]
    att = state["att"]
    prior = state["prior"]

    u, v = edges[:, 0], edges[:, 1]

    w_pos = hp.get("w_pos", 0.0)
    xy_start_idx = hp.get("xy_start_idx", None)

    if xy_start_idx is not None and xy_start_idx > 0:
        diff_color_deep = (mu[u, :15] -
                           mu[v, :15]) / 15

        diff_squared_color_deep = torch.sum(
            diff_color_deep**2, dim=1)

        diff_xy = (mu[u, xy_start_idx:] -
                   mu[v, xy_start_idx:]) / (mu.shape[1] - xy_start_idx)
        diff_squared_xy = torch.sum(
            diff_xy**2, dim=1)*w_pos * np.sqrt(state["remaining"] / state["N0"])

        diff_squared = diff_squared_color_deep + diff_squared_xy
        diff2 = torch.sqrt(diff_squared)
    else:
        diff2 = torch.sqrt(torch.sum((mu[u] - mu[v])**2, dim=1))

    score = diff2

    if att is not None and hp["att_weight"] > 0:
        au, av = att[u], att[v]
        if hp["att_pair_mode"] == "mean":
            a = 0.5 * (au + av)
        elif hp["att_pair_mode"] == "min":
            a = torch.minimum(au, av)
        else:  # "max"
            a = torch.maximum(au, av)
        score = score + hp["att_weight"] * a

    if prior is not None and hp["hard_gate"]:
        forbid = (prior[u] != prior[v])
        score = torch.where(forbid, torch.full_like(
            score, float("inf")), score)

    return score


def _edge_cost_torch_phaseB(edges: torch.Tensor, state: dict, hp: dict) -> torch.Tensor:
    """
    Compute pairwise edge costs for phase B of the hierarchical merge.

    This phase uses a Ward-like criterion combining feature distance
    and region sizes, optionally modulated by attention.

    Parameters
    ----------
    edges : torch.Tensor
        Tensor of shape (E, 2) with integer indices (u, v).
    state : dict
        Dictionary holding current region statistics (features, sizes,
        attention, priors, etc.).
    hp : dict
        Hyper-parameters controlling the cost.

    Returns
    -------
    torch.Tensor
        1D tensor of length E with edge scores.
    """
    mu = state["mu"]
    size = state["size"]
    att = state["att"]

    u, v = edges[:, 0], edges[:, 1]
    xy_start_idx = hp.get("xy_start_idx", None)

    if xy_start_idx is not None and xy_start_idx > 0:
        diff_color_deep = mu[u, :xy_start_idx] - mu[v, :xy_start_idx]
        diff_squared_color_deep = torch.sum(
            diff_color_deep**2, dim=1)

        diff2 = torch.sqrt(diff_squared_color_deep)
    else:
        diff2 = torch.sqrt(torch.sum((mu[u] - mu[v])**2, dim=1))

    su = size[u].clamp_min(1e-6)
    sv = size[v].clamp_min(1e-6)

    ward = (su * sv) / (su + sv) * diff2
    score = ward

    if att is not None and hp["att_weight"] > 0:
        au, av = att[u], att[v]
        if hp["att_pair_mode"] == "mean":
            a = 0.5 * (au + av)
        elif hp["att_pair_mode"] == "min":
            a = torch.minimum(au, av)
        else:  # "max"
            a = torch.maximum(au, av)
        score = score + hp["att_weight"] * a

    return score


def normalize_features_min_max(features):
    """
    Apply per-channel min-max normalization to feature vectors.

    Parameters
    ----------
    features : np.ndarray
        Array of shape (N, D) with feature vectors.

    Returns
    -------
    np.ndarray
        Array of same shape with each channel scaled to [0, 1].
    """

    feats_np = np.asarray(features, dtype=np.float32)
    if feats_np.ndim != 2:
        raise ValueError("features must be of shape (N, D)")

    min_vals = feats_np.min(axis=0, keepdims=True)
    max_vals = feats_np.max(axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals = np.where(range_vals < 1e-6, 1.0, range_vals)
    normalized = (feats_np - min_vals) / range_vals
    normalized = normalized.astype(np.float32)

    return normalized


def _init_adjacency(edges_np, N):
    """
    Build an adjacency list from an array of edges.

    Parameters
    ----------
    edges_np : np.ndarray
        Array of shape (E, 2) with integer pairs (u, v).
    N : int
        Number of nodes.

    Returns
    -------
    list[set[int]]
        List of length N where each entry is the set of neighbors.
    """
    adj = [set() for _ in range(N)]
    for a, b in edges_np:
        a = int(a)
        b = int(b)
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _fuse_regions(keep, kill, state, new_obj_for_keep=None):
    """
    Merge region `kill` into region `keep` and update all state.

    This updates region statistics (features, sizes, attention),
    object-level priors, and the adjacency structure in-place.

    Parameters
    ----------
    keep : int
        Index of the surviving region.
    kill : int
        Index of the region to be removed.
    state : dict
        Mutable state containing region statistics and adjacency.
    new_obj_for_keep : int, optional
        New object id for `keep` when object-level priors are used.
    """
    mu = state["mu"]
    size = state["size"]
    att = state["att"]
    prior = state["prior"]
    obj_id = state["obj_id"]
    obj_size = state["obj_size"]
    adj = state["adj"]
    active = state["active"]
    att_update = state["att_update"]

    sk = float(size[keep].item())
    sl = float(size[kill].item())
    w = size[keep] + size[kill]
    mu[keep] = (size[keep] * mu[keep] + size[kill]
                * mu[kill]) / w.clamp_min(1e-6)
    size[keep] = w

    if att is not None:
        if att_update == "max":
            att[keep] = torch.maximum(att[keep], att[kill])
        else:
            att[keep] = (sk * att[keep] + sl * att[kill]) / max(sk + sl, 1e-6)

    if prior is not None and new_obj_for_keep is not None:
        old_keep = int(obj_id[keep].item())
        old_kill = int(obj_id[kill].item())
        if obj_size is not None:
            obj_size[old_kill] -= sl
            if old_keep != new_obj_for_keep:
                obj_size[old_keep] -= sk
                obj_size[new_obj_for_keep] += sk
            obj_size[new_obj_for_keep] += sl
        obj_id[keep] = int(new_obj_for_keep)

    nbrs = (adj[keep] | adj[kill]) - {keep, kill}
    for t in adj[kill]:
        adj[t].discard(kill)
        if t != keep:
            adj[t].add(keep)
    adj[keep] = set(nbrs)
    adj[kill].clear()
    active[kill] = False


def _push_edges_for_keep(keep, phase, state, heaps, hp):
    """
    Recompute and push all incident edges of `keep` into the appropriate heap.

    Depending on the phase and merge mode, edges are directed to
    different priority queues used during the hierarchy construction.

    Parameters
    ----------
    keep : int
        Index of the region whose neighborhood changed.
    phase : str
        Phase identifier, e.g. "A", "B" or "C".
    state : dict
        Current merge state.
    heaps : dict
        Dictionary of heaps (priority queues) used by the algorithm.
    hp : dict
        Hyper-parameters for the edge costs.
    """
    if not state["adj"][keep]:
        return
    obj_id = state["obj_id"]
    obj_size = state["obj_size"]
    mode = hp["object_merge_mode"]
    BIG = hp["BIG"]
    nbrs = sorted(state["adj"][keep])
    new_edges = torch.tensor([[min(keep, t), max(keep, t)]
                             for t in nbrs], dtype=torch.int64)

    if state["prior"] is None or mode == "strict":

        scores = _edge_cost_torch_phaseA(new_edges, state, hp).tolist()
        for s, (u, v) in zip(scores, new_edges.tolist()):
            if np.isfinite(s):
                heapq.heappush(heaps["strict"], (float(s), int(u), int(v)))
        return

    if phase == "A":
        scores = _edge_cost_torch_phaseA(new_edges, state, hp).tolist()
        for s, (u, v) in zip(scores, new_edges.tolist()):
            if np.isfinite(s) and (obj_id[u] == obj_id[v]).item():
                heapq.heappush(heaps["intra"], (float(s), int(u), int(v)))
    elif phase == "B":
        scores = _edge_cost_torch_phaseB(new_edges, state, hp).tolist()
        for s, (u, v) in zip(scores, new_edges.tolist()):
            if np.isfinite(s):
                heapq.heappush(heaps["final"], (float(s), int(u), int(v)))


# ---------- Heaps seeding ----------


def _seed_heaps_strict(edges, edges_np, state, heaps, hp):
    """
    Initialize the heap for strict merging without object priors.

    Parameters
    ----------
    edges : torch.Tensor
        Tensor of shape (E, 2) with edge indices.
    edges_np : np.ndarray
        Numpy version of the edges for iteration.
    state : dict
        Current merge state.
    heaps : dict
        Heaps (priority queues) where edges are pushed.
    hp : dict
        Hyper-parameters for the edge costs.
    """
    scores = _edge_cost_torch_phaseA(edges, state, hp).tolist()
    for s, (u, v) in zip(scores, edges_np):
        if np.isfinite(s):
            heapq.heappush(heaps["strict"], (float(s), int(u), int(v)))


def _seed_heaps_relaxed(edges, state, heaps, hp):
    """
    Initialize the heaps for relaxed merging with object priors.

    Edges inside the same object go to the intra-object heap, and
    edges crossing objects go to the cross-object heap.

    Parameters
    ----------
    edges : torch.Tensor
        Tensor of shape (E, 2) with edge indices.
    state : dict
        Current merge state (must include `obj_id`).
    heaps : dict
        Heaps (priority queues) where edges are pushed.
    hp : dict
        Hyper-parameters for the edge costs.
    """
    scores = _edge_cost_torch_phaseA(edges, state, hp).tolist()
    obj_id = state["obj_id"]
    obj_size = state["obj_size"]
    U = edges[:, 0].tolist()
    V = edges[:, 1].tolist()
    for s, u, v in zip(scores, U, V):
        if (obj_id[u] == obj_id[v]).item():
            heapq.heappush(heaps["intra"], (float(s), int(u), int(v)))
        else:
            heapq.heappush(heaps["cross"], (float(s), int(u), int(v)))


def _phase_strict(state, heaps, hp, merge_tree):
    """
    Run the single-phase strict hierarchical merging.

    Merges regions while respecting hard object gating (if any) and
    records the merge tree. This is used when no object priors or
    a strict merge mode is requested.

    Parameters
    ----------
    state : dict
        Merge state that will be updated in-place.
    heaps : dict
        Heaps storing candidate edges.
    hp : dict
        Hyper-parameters controlling merging.
    merge_tree : list
        List to which (keep, kill) merge pairs are appended.
    """
    adj = state["adj"]
    active = state["active"]
    prior = state["prior"]
    remaining = state["remaining"]
    while heaps["strict"] and remaining > 1:
        s, u, v = heapq.heappop(heaps["strict"])
        if not (active[u] and active[v]) or (v not in adj[u]):
            continue
        keep, kill = (u, v) if u < v else (v, u)
        if prior is not None and hp["hard_gate"] and prior[keep] != prior[kill]:
            continue
        merge_tree.append((int(keep), int(kill)))
        _fuse_regions(keep, kill, state, new_obj_for_keep=None)
        remaining -= 1
        _push_edges_for_keep(keep, phase="C", state=state, heaps=heaps, hp=hp)
    state["remaining"] = remaining


def _phase_A_intra(state, heaps, hp, merge_tree):
    """
    Phase A of relaxed merging: intra-object merges only.

    During this phase, merges are restricted to regions that share the
    same object id, allowing refinement inside each object before
    cross-object merging.

    Parameters
    ----------
    state : dict
        Merge state, updated in-place.
    heaps : dict
        Heaps containing intra-object candidate edges.
    hp : dict
        Hyper-parameters controlling merging.
    merge_tree : list
        List to which (keep, kill) merge pairs are appended.
    """
    adj = state["adj"]
    active = state["active"]
    obj_id = state["obj_id"]
    remaining = state["remaining"]
    while heaps["intra"] and remaining > 1:
        s, u, v = heapq.heappop(heaps["intra"])
        if not (active[u] and active[v]) or (v not in adj[u]):
            continue
        if (obj_id[u] != obj_id[v]).item():
            continue
        keep, kill = (u, v) if u < v else (v, u)
        merge_tree.append((int(keep), int(kill)))
        _fuse_regions(keep, kill, state,
                      new_obj_for_keep=int(obj_id[keep].item()))
        remaining -= 1
        state["remaining"] = remaining
        _push_edges_for_keep(keep, phase="A", state=state, heaps=heaps, hp=hp)
    state["remaining"] = remaining


def _phase_B_seed_relaxed(state, heaps, hp, merge_tree):
    """
    Phase B of relaxed merging: cross-object and global merges.

    Seeds a heap of cross-object edges based on the current active
    regions and then repeatedly merges the best available pair until
    only one region (or no valid merges) remains.

    Parameters
    ----------
    state : dict
        Merge state, updated in-place.
    heaps : dict
        Heaps containing candidate edges for phase B.
    hp : dict
        Hyper-parameters controlling merging.
    merge_tree : list
        List to which (keep, kill) merge pairs are appended.
    """
    adj = state["adj"]
    active = state["active"]
    obj_id = state["obj_id"]
    remaining = state["remaining"]
    N0 = state["N0"]

    for u in range(N0):
        if not active[u]:
            continue
        for v in list(adj[u]):
            if v <= u:
                continue
            if obj_id is not None and (obj_id[u] == obj_id[v]).item():
                continue
            uv = torch.tensor([[min(u, v), max(u, v)]], dtype=torch.int64)
            s = _edge_cost_torch_phaseB(uv, state, hp)[0].item()
            if np.isfinite(s):
                heapq.heappush(heaps["final"], (float(s), int(u), int(v)))

    while remaining > 1:
        merged = False
        while heaps["final"]:
            s, u, v = heapq.heappop(heaps["final"])
            if not (active[u] and active[v]) or (v not in adj[u]):
                continue
            if obj_id is not None and (obj_id[u] == obj_id[v]).item():
                continue
            keep, kill = (u, v) if u < v else (v, u)
            merge_tree.append((int(keep), int(kill)))
            new_oid = int(obj_id[keep].item()) if obj_id is not None else None
            _fuse_regions(keep, kill, state, new_obj_for_keep=new_oid)
            remaining -= 1
            merged = True
            _push_edges_for_keep(
                keep, phase="B", state=state, heaps=heaps, hp=hp)
            break

        if not merged and remaining > 1:
            best_score = float("inf")
            best_u = None
            best_v = None

            for u in range(N0):
                if not active[u]:
                    continue
                for v in list(adj[u]):
                    if v <= u or not active[v]:
                        continue
                    if obj_id is not None and (obj_id[u] == obj_id[v]).item():
                        continue
                    uv = torch.tensor(
                        [[min(u, v), max(u, v)]], dtype=torch.int64)
                    s = _edge_cost_torch_phaseB(uv, state, hp)[0].item()
                    if np.isfinite(s) and s < best_score:
                        best_score = s
                        best_u, best_v = u, v

            if best_u is None:
                break

            keep, kill = (best_u, best_v) if best_u < best_v else (
                best_v, best_u)
            merge_tree.append((int(keep), int(kill)))
            new_oid = int(obj_id[keep].item()) if obj_id is not None else None
            _fuse_regions(keep, kill, state, new_obj_for_keep=new_oid)
            remaining -= 1
            _push_edges_for_keep(
                keep, phase="B", state=state, heaps=heaps, hp=hp)

    state["remaining"] = remaining


def hierarchical_merge_rag(
    features_np,
    sp_map_np,
    prior_labels_np=None,
    hard_gate=True,
    object_merge_mode="strict",   # "strict" | "relaxed"
    att_region_np=None,
    w_att=0.0,
    att_pair_mode="max",
    att_update="mean",            # "mean" | "max"
    w_pos=0.0,
    xy_start_idx=None,
):
    """
    Build a hierarchical region merge tree from a superpixel RAG.

    This function constructs a hierarchy over superpixels by repeatedly
    merging adjacent regions according to feature, position, attention
    and optional object-level priors.

    Parameters
    ----------
    features_np : np.ndarray
        Array of shape (N_regions, D) with region-level features.
    sp_map_np : np.ndarray
        2D integer superpixel map of shape (H, W).
    prior_labels_np : np.ndarray, optional
        1D array of length N_regions with object ids for each region.
    hard_gate : bool, optional
        If True, forbids merges across different priors in strict mode.
    object_merge_mode : {"strict", "relaxed"}, optional
        Merge strategy when priors are present.
    att_region_np : np.ndarray, optional
        Per-region attention scores of length N_regions.
    w_att : float, optional
        Weight of the attention term in edge costs.
    att_pair_mode : {"mean", "min", "max"}, optional
        How to combine attention values between two regions.
    att_update : {"mean", "max"}, optional
        How to update region attention after merges.
    w_pos : float, optional
        Weight of the positional component in the feature distance.
    xy_start_idx : int, optional
        Index in the feature vector where XY positional channels start.

    Returns
    -------
    dict
        Dictionary describing the hierarchy, including:
        - "merge_tree": list of (keep, kill) merges in order
        - "N0": initial number of regions
        - "remap": mapping from original labels to compact labels
        - optionally "index_change_phase" in relaxed mode.
    """
    L0, remap = _reindex_labels_np(sp_map_np)
    N0 = int(L0.max()) + 1

    mu = torch.from_numpy(features_np.astype(np.float32))
    size = torch.bincount(torch.from_numpy(L0.ravel()), minlength=N0).float()
    if mu.shape[0] != N0:
        raise ValueError("features_np and sp_map_np disagree on region count")

    prior = None
    if prior_labels_np is not None:
        prior = torch.from_numpy(prior_labels_np.astype(np.int64))
        if prior.numel() != N0:
            raise ValueError(
                "prior_labels length must match number of regions")

    att = None
    if att_region_np is not None:
        a = np.asarray(att_region_np, dtype=np.float32)
        if a.ndim != 1 or a.shape[0] != N0:
            raise ValueError("att_region_np must be shape [N_regions]")
        att = torch.from_numpy(a)

    edges_np = _build_rag_np(L0)
    if edges_np.size == 0:
        return {"merge_tree": [], "N0": N0, "remap": remap}
    edges = torch.from_numpy(edges_np.astype(np.int64))

    adj = _init_adjacency(edges_np, N0)
    active = [True] * N0
    merge_tree = []

    obj_id = None
    obj_size = None
    if prior is not None:
        max_oid = int(prior.max().item()) + 1
        obj_id = prior.clone()
        obj_size = torch.bincount(
            obj_id, weights=size, minlength=max_oid).float()

    heaps = {"strict": [], "intra": [], "cross": [], "final": []}
    state = {
        "mu": mu, "size": size, "att": att, "prior": prior,
        "obj_id": obj_id, "obj_size": obj_size, "adj": adj,
        "active": active, "att_update": att_update,
        "N0": N0, "remaining": N0, "L0": L0
    }
    hp = {
        "att_weight": float(w_att), "att_pair_mode": att_pair_mode,
        "object_merge_mode": object_merge_mode, "hard_gate": bool(hard_gate),
        "BIG": 1e6,
        "w_pos": float(w_pos),
        "xy_start_idx": int(xy_start_idx) if xy_start_idx is not None else None
    }

    if prior is None or object_merge_mode == "strict":
        _seed_heaps_strict(edges, edges_np, state, heaps, hp)
        _phase_strict(state, heaps, hp, merge_tree)
        return {"merge_tree": merge_tree, "N0": N0, "remap": remap}

    _seed_heaps_relaxed(edges, state, heaps, hp)
    _phase_A_intra(state, heaps, hp, merge_tree)

    index_change_phase = sum(state["active"])

    _phase_B_seed_relaxed(state, heaps, hp, merge_tree)

    return {"merge_tree": merge_tree, "N0": N0, "remap": remap, "index_change_phase": index_change_phase}


def get_intermediate_clusters(Z_or_H, superpixels, n_clusters):
    """
    Obtain an intermediate partition from a hierarchy or linkage.

    If `Z_or_H` is the custom hierarchy dict returned by
    `hierarchical_merge_rag`, this reconstructs a superpixel map with
    exactly `n_clusters` regions. Otherwise, it assumes a SciPy
    linkage matrix and applies standard flat clustering.

    Parameters
    ----------
    Z_or_H : dict or np.ndarray
        Either a hierarchy dict or a SciPy linkage matrix.
    superpixels : np.ndarray
        2D integer superpixel map used as the base partition.
    n_clusters : int
        Desired number of regions in the output.

    Returns
    -------
    np.ndarray
        2D array of labels with `n_clusters` distinct regions.
    """
    K = int(n_clusters)
    base = Z_or_H.get("base_sp_map", superpixels)
    start_time = time.time()
    sp_k = _labels_from_merges(base, Z_or_H, K)
    end_time = time.time()
    if VERBOSE:
        print(
        f"Time to get intermediate clusters: {end_time - start_time:.2f} s")
    return sp_k



def reassign_superpixels_to_clusters(superpixels, labels):
    """
    Map superpixel labels to new cluster labels.

    Parameters
    ----------
    superpixels : np.ndarray
        2D integer superpixel map.
    labels : np.ndarray
        1D array where `labels[sp_id]` gives the new label for superpixel `sp_id`.

    Returns
    -------
    np.ndarray
        2D array where each pixel takes the cluster label of its superpixel.
    """
    return labels[superpixels]


def compute_attention_per_region(attention, superpixels, object_map=None, mode="superpixel", normalize=True, threshold=None):
    """
    Compute per-pixel attention derived from region or object attention.

    Depending on `mode`, attention is aggregated per superpixel or per
    object, then optionally normalized and thresholded and broadcast
    back to the pixel grid.

    Parameters
    ----------
    attention : np.ndarray
        Pixel-wise or region-wise attention input.
    superpixels : np.ndarray
        2D integer superpixel map.
    object_map : np.ndarray, optional
        2D integer map of object ids, required when `mode="object"`.
    mode : {"superpixel", "object"}, optional
        Scope over which attention is pooled.
    normalize : bool, optional
        If True, normalize aggregated scores to [0, 1].
    threshold : float, optional
        If set, binarize the result at this threshold.

    Returns
    -------
    np.ndarray
        2D float map of attention values aligned with `superpixels`.
    """
    if mode not in ("superpixel", "object"):
        raise ValueError("mode must be 'superpixel' or 'object'")

    if mode == "superpixel":
        out = compute_region_attention(
            attention, superpixels, normalize=normalize)
    else:
        if object_map is None:
            raise ValueError("object_map is required when mode='object'")

        sp = superpixels.astype(np.int32)
        H, W = sp.shape
        N = int(sp.max()) + 1

        sp2obj = associate_superpixels_to_objects(superpixels, object_map)
        n_objs = int(sp2obj.max()) + 1

        att = np.asarray(attention)

        if att.ndim == 1 and att.shape[0] == N:
            counts_r = np.bincount(sp.ravel(), minlength=N).astype(np.float32)
            sums_o = np.bincount(sp2obj, weights=att.astype(
                np.float32) * counts_r, minlength=n_objs).astype(np.float32)
            cnts_o = np.bincount(sp2obj, weights=counts_r,
                                 minlength=n_objs).astype(np.float32)
            obj_att = sums_o / (cnts_o + 1e-6)
        else:
            if att.ndim == 3:
                att = att.mean(axis=2)
            if att.shape != (H, W):
                raise ValueError(
                    "attention_map must match HxW or be length N_regions")
            vals = att.astype(np.float32).ravel()
            labs = object_map.astype(np.int32).ravel()
            cnts_o = np.bincount(labs, minlength=n_objs).astype(np.float32)
            sums_o = np.bincount(labs, weights=vals,
                                 minlength=n_objs).astype(np.float32)
            obj_att = sums_o / (cnts_o + 1e-6)

        if normalize:
            lo, hi = float(obj_att.min()), float(obj_att.max())
            obj_att = (obj_att - lo) / (hi - lo + 1e-6)

        out = obj_att[sp2obj].astype(np.float32)

    if threshold is not None:
        t = float(threshold)
        out = out.astype(np.float32, copy=True)
        out[out < t] = 0.0
        out[out > t] = 1.0

    return out


def _labels_from_merges(sp_map_np, H, K):
    """
    Reconstruct a label map from a merge tree at a given scale K.

    Parameters
    ----------
    sp_map_np : np.ndarray
        2D base superpixel map.
    H : dict
        Hierarchy dict containing "merge_tree", "N0" and "remap".
    K : int
        Desired number of regions in the output.

    Returns
    -------
    np.ndarray
        2D integer label map with at most K regions.
    """
    L0 = H["remap"][sp_map_np.astype(np.int64)]
    N0 = H["N0"]
    merges = np.asarray(H["merge_tree"], dtype=np.int64)
    need = max(0, N0 - int(K))
    take = min(need, merges.shape[0])
    parent = np.arange(N0, dtype=np.int64)
    for i in range(take):
        keep, kill = merges[i]
        parent[kill] = keep

    def find(a):
        while parent[a] != parent[parent[a]]:
            parent[a] = parent[parent[a]]
        return parent[a]
    flat = L0.ravel()
    out = np.empty_like(flat)
    for i in range(flat.size):
        out[i] = find(flat[i])
    uniq, inv = np.unique(out, return_inverse=True)
    return inv.reshape(sp_map_np.shape).astype(np.int32)


def associate_superpixels_to_objects(superpixels, object_map):
    """
    Associate each superpixel with the object it most overlaps.

    This builds a sparse overlap matrix between superpixels and objects
    and assigns each superpixel to the object with maximum overlap.

    Parameters
    ----------
    superpixels : np.ndarray
        2D integer superpixel map.
    object_map : np.ndarray
        2D integer object map of the same shape.

    Returns
    -------
    np.ndarray
        1D array of length N_superpixels with object id per superpixel.
    """
    sp_flat = superpixels.ravel().astype(np.int32)
    obj_flat = object_map.ravel().astype(np.int32)

    N = sp_flat.max() + 1
    n_objs = obj_flat.max() + 1
    data = np.ones_like(sp_flat, dtype=np.uint32)
    M = sparse.coo_matrix((data, (sp_flat, obj_flat)),
                          shape=(N, n_objs)).tocsr()
    superpixel_to_object = M.argmax(axis=1).A1.astype(object_map.dtype)
    return superpixel_to_object


def extract_features_by_type(pixel_features=None, inputs=None):
    """
    Extract deep, color and XY feature tensors from network inputs.

    The function expects a `(C, H, W)` or `(1, C, H, W)` tensor for
    `pixel_features`, where the last 5 channels are reserved and not
    used as deep features. When `inputs` is provided, it is assumed
    to contain LAB color channels followed by XY coordinates.

    Parameters
    ----------
    pixel_features : torch.Tensor or np.ndarray, optional
        Feature tensor from a backbone network.
    inputs : torch.Tensor or np.ndarray, optional
        Input tensor containing color and XY channels.

    Returns
    -------
    tuple
        `(features_deep, features_color, features_xy)` where each entry
        is a numpy array of shape (H, W, C) or None.
    """
    features_deep = None
    features_color = None
    features_xy = None

    if pixel_features is None:
        raise ValueError(
            "pixel_features est requis pour features_type='deep'")
    feat_tensor = pixel_features[0] if pixel_features.ndim == 4 else pixel_features
    feat_tensor = feat_tensor[:-5]
    features_deep = feat_tensor.permute(1, 2, 0).cpu().numpy()
    features_deep = features_deep / 5

    if inputs is not None:
        if isinstance(inputs, torch.Tensor):
            inputs_tensor = inputs[0] if inputs.ndim == 4 else inputs
            features_color = inputs_tensor[:-
                                           2].permute(1, 2, 0).cpu().numpy()
            features_xy = inputs_tensor[-2:].permute(1, 2, 0).cpu().numpy()
        else:
            features_color = inputs[..., :-2]  # LAB
            features_xy = inputs[..., -2:]  # XY

    return features_deep, features_color, features_xy


def _compute_feature_parts_and_xy_start(
    features_deep, features_color, features_xy, labels, counts
):
    """
    Compute region-level pooled features and locate XY channels.

    Features are pooled over superpixels using simple averages and
    min-max normalized per channel. The index at which XY features
    start in the concatenated feature vector is also returned.

    Parameters
    ----------
    features_deep : np.ndarray or None
        Pixel-level deep features of shape (H, W, C_deep).
    features_color : np.ndarray or None
        Pixel-level color features of shape (H, W, C_color).
    features_xy : np.ndarray or None
        Pixel-level XY features of shape (H, W, C_xy).
    labels : np.ndarray
        1D array of region ids, same length as flattened image.
    counts : np.ndarray
        1D array of region sizes.

    Returns
    -------
    list[np.ndarray]
        List of feature matrices to be concatenated along channel dim.
    int or None
        Starting index of XY channels in the concatenated features.
    """
    N = int(labels.max()) + 1
    feature_parts = []
    xy_start_idx = None

    if features_deep is not None:
        _, _, c_deep = features_deep.shape
        channels_deep = [
            np.bincount(
                labels, weights=features_deep[..., i_c].ravel(), minlength=N
            )
            for i_c in range(c_deep)
        ]
        mean_deep = np.stack(channels_deep, axis=1) / (counts[:, None] + 1e-6)
        mean_deep = mean_deep.astype(np.float32)
        mean_deep = normalize_features_min_max(mean_deep)
        feature_parts.append(mean_deep)

    if features_color is not None:
        _, _, c_color = features_color.shape
        channels_color = [
            np.bincount(
                labels, weights=features_color[..., i_c].ravel(), minlength=N
            )
            for i_c in range(c_color)
        ]
        mean_color = np.stack(channels_color, axis=1) / \
            (counts[:, None] + 1e-6)
        mean_color = mean_color.astype(np.float32)
        mean_color = normalize_features_min_max(mean_color)
        feature_parts.append(mean_color)

    if features_xy is not None:
        xy_start_idx = sum(
            f.shape[1] if f is not None else 0 for f in feature_parts
        )

        _, _, c_xy = features_xy.shape
        channels_xy = [
            np.bincount(
                labels, weights=features_xy[..., i_c].ravel(), minlength=N
            )
            for i_c in range(c_xy)
        ]
        mean_xy = np.stack(channels_xy, axis=1) / (counts[:, None] + 1e-6)
        mean_xy = mean_xy.astype(np.float32)
        mean_xy = normalize_features_min_max(mean_xy)
        feature_parts.append(mean_xy)

    if len(feature_parts) == 0:
        raise ValueError("Au moins un type de features doit être fourni")

    return feature_parts, xy_start_idx


def get_global_hierarchy(superpixel_img,
                         pixel_features=None,
                         w_pos: float = 0.0,
                         inputs=None,
                         object_map=None,
                         object_merge_mode: str = "strict",
                         attention_map=None,
                         w_att: float = 2.0,
                         attention_pair_mode: str = "max",
                         attention_update: str = "mean",
                         attention_scope="object",
                         threshold_attention=None,
                         ):

    features_deep, features_color, features_xy = extract_features_by_type(
        pixel_features=pixel_features,
        inputs=inputs,
    )

    sp_conn = superpixel_img.astype(np.int32)

    labels = sp_conn.ravel().astype(np.int32)
    N = labels.max() + 1
    counts = np.bincount(labels, minlength=N).astype(np.float32)

    feature_parts, xy_start_idx = _compute_feature_parts_and_xy_start(
        features_deep, features_color, features_xy, labels, counts
    )

    feats = np.concatenate(feature_parts, axis=1).astype(np.float32)
    if object_map is not None:
        sp2obj = associate_superpixels_to_objects(sp_conn, object_map)
    else:
        sp2obj = None

    att_region = None
    attn_region_map = None
    if attention_map is not None and w_att > 0:
        att_region = compute_attention_per_region(
            attention_map, sp_conn, normalize=True, object_map=object_map, mode=attention_scope, threshold=threshold_attention)
        attn_region_map = reassign_superpixels_to_clusters(
            superpixel_img, att_region)

    H = hierarchical_merge_rag(
        features_np=feats,
        sp_map_np=sp_conn,
        prior_labels_np=sp2obj,
        hard_gate=True,
        object_merge_mode=object_merge_mode,
        att_region_np=att_region,
        w_att=float(w_att),
        att_pair_mode=attention_pair_mode,
        att_update=attention_update,
        w_pos=float(w_pos),
        xy_start_idx=xy_start_idx,
    )

    H["base_sp_map"] = sp_conn
    H["object_merge_mode"] = object_merge_mode
    H["attention_weight"] = float(w_att)
    H["attention_pair_mode"] = attention_pair_mode
    H["attention_update"] = attention_update
    H["attention_scope"] = attention_scope
    H["attn_region_map"] = attn_region_map
    return H
