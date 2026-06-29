"""
Oracle — three uses of the same trained BibleGAT.

Background
──────────
Transformers are Graph Neural Networks (arxiv 2506.22084).  The BibleGAT we
trained is therefore a transformer over the Biblical graph.  Its learned
attention weights and reconstruction head generalise beyond the Bible to any
content projected into the 8-dim Biblical coordinate space.

Three uses of the same weights
──────────────────────────────

1. next_action(current, candidates) → action probabilities
   The GAT attention mechanism computes α_{current→candidate} for every
   candidate.  These weights ARE a learned policy: "given where I am in the
   information space, how much should I attend to each candidate?"
   Same as next-token prediction in a language model, but over navigation
   candidates instead of vocabulary tokens.

2. alignment_score(content_vec) → [0, 1]
   The GNN was trained to reconstruct masked nodes from their Biblical
   neighbours.  Running a new content vector through this same procedure
   (mask it, reconstruct from k-nearest Biblical verses, measure NMSE)
   gives a reconstruction likelihood = how well the Biblical manifold
   "understands" this content.
     1.0  =  perfectly consistent with the Biblical world model
     ~0.37 =  no better than predicting the mean (random baseline)
     0.0  =  completely alien

3. novelty_score(content_vec) → [0, 1]
   Flipped logit of the reconstruction head.  Instead of measuring
   how well the manifold fits the content, measure how POORLY it fits.
   High novelty = this information is genuinely outside everything
   the world model has seen.  Useful for discovery and anomaly detection.

   novelty = 1 - alignment   (exact complement)

The key property: same weights, inference-time only, no retraining needed.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from config import MODELS_DIR, GNN_LAYERS, GNN_HEADS, GNN_DROPOUT, W2V_VECTOR_SIZE
from graph.gnn import BibleGAT

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_model: BibleGAT | None = None
_gnn_embeddings: dict[int, np.ndarray] = {}
_verse_id_list: list[int] = []
_faiss_index = None
_proj_coef: np.ndarray | None = None    # (8, 8): W2V_norm @ coef ≈ GNN_norm
_proj_intercept: np.ndarray | None = None  # (8,)

# Calibrated thresholds — derived from BibleGAT's own reconstruction distribution
# over all 38,927 verses.  Set by _calibrate() at init time.
# Temporary defaults let callers import without crashing before init().
alignment_floor: float = 0.3   # 10th percentile — below = outside prior
alignment_high:  float = 0.5   # 50th percentile — solidly within manifold


def init(gnn_embeddings: dict, verse_id_list: list, faiss_index) -> None:
    global _model, _gnn_embeddings, _verse_id_list, _faiss_index
    global _proj_coef, _proj_intercept

    dim = W2V_VECTOR_SIZE
    _model = BibleGAT(dim, dim, GNN_LAYERS, GNN_HEADS, GNN_DROPOUT)
    _model.load_state_dict(
        torch.load(MODELS_DIR / "gnn_weights.pt", weights_only=True)
    )
    _model.eval()

    _gnn_embeddings = gnn_embeddings
    _verse_id_list  = verse_id_list
    _faiss_index    = faiss_index

    # Load W2V→GNN projection for converting W2V-space inputs to GNN space
    import pickle
    proj_path = MODELS_DIR / "w2v_to_gnn_projection.pkl"
    if proj_path.exists():
        with open(proj_path, "rb") as f:
            proj = pickle.load(f)
        _proj_coef      = proj["coef"]       # (8, 8)
        _proj_intercept = proj["intercept"]  # (8,)
        print(f"[oracle] BibleGAT loaded — layers={GNN_LAYERS} heads={GNN_HEADS} dim={dim}  "
              f"(W2V→GNN projection loaded)")
    else:
        print(f"[oracle] BibleGAT loaded — layers={GNN_LAYERS} heads={GNN_HEADS} dim={dim}  "
              f"WARNING: projection not found, oracle will accept GNN-space input only")

    # Calibrate alignment thresholds from the oracle's own reconstruction distribution
    _calibrate()


# ---------------------------------------------------------------------------
# Calibration — derive thresholds from the model's own reconstruction stats
# ---------------------------------------------------------------------------

def _calibrate(k_neighbors: int = 5) -> None:
    """
    Compute alignment_floor and alignment_high from BibleGAT's reconstruction
    distribution over all known verse embeddings (GNN space, no projection needed).

    Method
    ──────
    1. Batch FAISS search: find k neighbors for every verse in one call.
    2. Loop: for each verse, mask it, reconstruct from its neighbors, compute
       exp(-NMSE).  Input is already in GNN space — no W2V projection.
    3. alignment_floor = 10th percentile  (below = genuinely outside prior)
       alignment_high  = 50th percentile  (solidly within manifold)

    Prints a calibration report so operators can see the real distribution.
    """
    global alignment_floor, alignment_high

    if not _gnn_embeddings or _faiss_index is None or _model is None:
        return

    verse_ids = list(_gnn_embeddings.keys())
    n = len(verse_ids)
    dim = W2V_VECTOR_SIZE

    print(f"[oracle] calibrating thresholds over {n} verses …", flush=True)

    # Build matrix of all GNN embeddings in verse_id order (already L2-norm'd in FAISS)
    mat = np.stack([
        _gnn_embeddings[vid].astype(np.float32) for vid in verse_ids
    ])  # (n, dim)

    # L2-normalise rows (same as what FAISS stores)
    row_norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat_n = mat / (row_norms + 1e-9)

    # Batch FAISS search — k+1 because the verse matches itself at rank 0
    scores_mat, idx_mat = _faiss_index.search(mat_n, k_neighbors + 1)
    # idx_mat: (n, k+1); first column is typically the verse itself (sim ≈ 1.0)

    alignment_scores = np.zeros(n, dtype=np.float32)

    _model.eval()
    with torch.no_grad():
        for i in range(n):
            content_vec = mat_n[i]

            # Collect neighbors — skip the self-match (rank 0 when sim > 0.99)
            nbr_vecs = []
            for rank, idx in enumerate(idx_mat[i]):
                if idx < 0:
                    continue
                nbr_id  = _verse_id_list[int(idx)]
                if nbr_id == verse_ids[i] and rank == 0:
                    continue   # skip self
                nbr_vecs.append(_gnn_embeddings[nbr_id].astype(np.float32))
                if len(nbr_vecs) == k_neighbors:
                    break

            if not nbr_vecs:
                alignment_scores[i] = 0.0
                continue

            # Mini-graph: node 0 = masked content, nodes 1..k = neighbors
            masked = np.zeros(dim, dtype=np.float32)
            all_vecs = np.stack([masked] + nbr_vecs).astype(np.float32)
            x = torch.tensor(all_vecs)
            k = len(nbr_vecs)
            src = torch.arange(1, k + 1, dtype=torch.long)
            dst = torch.zeros(k, dtype=torch.long)
            edge_index = torch.stack([src, dst])

            target = torch.tensor(content_vec, dtype=torch.float32)
            z      = _model(x, edge_index)
            mse    = F.mse_loss(z[0], target).item()
            var    = float(target.var().item())
            nmse   = mse / (var + 1e-9)
            alignment_scores[i] = math.exp(-nmse)

    alignment_floor = float(np.percentile(alignment_scores, 10))
    alignment_high  = float(np.percentile(alignment_scores, 50))

    print(
        f"[oracle] calibration complete — "
        f"min={alignment_scores.min():.4f}  "
        f"p10={alignment_floor:.4f}  "
        f"p50={alignment_high:.4f}  "
        f"p90={np.percentile(alignment_scores, 90):.4f}  "
        f"max={alignment_scores.max():.4f}"
    )


# ---------------------------------------------------------------------------
# Internal: project W2V → GNN space
# ---------------------------------------------------------------------------

def _w2v_to_gnn(vec: np.ndarray) -> np.ndarray:
    """Project an L2-normalised W2V vector into GNN space.

    All oracle public functions accept W2V-space vectors and call this
    internally, keeping the oracle interface space-agnostic for callers.
    """
    from sklearn.preprocessing import normalize as sk_norm
    vec_n = sk_norm(vec.reshape(1, -1).astype(np.float32), norm="l2").flatten()
    if _proj_coef is not None:
        projected = vec_n @ _proj_coef + _proj_intercept
        norm = np.linalg.norm(projected)
        return projected / (norm + 1e-9)
    return vec_n   # pass-through if projection not loaded


# ---------------------------------------------------------------------------
# 1. Next action — attention as action policy
# ---------------------------------------------------------------------------

def next_action(
    current_vec: np.ndarray,
    candidate_vecs: list[np.ndarray],
) -> list[float]:
    """
    Rank candidates by the GNN attention score from the current position.

    Accepts W2V-space vectors; projects to GNN space internally before
    running the GAT attention mechanism.

    Parameters
    ----------
    current_vec      : 8-dim L2-normalised W2V position vector
    candidate_vecs   : list of 8-dim W2V vectors (possible next positions)

    Returns
    -------
    list[float] of length len(candidate_vecs), summing to 1.0
    """
    if _model is None:
        raise RuntimeError("Oracle not initialised — call oracle.init() first.")
    if not candidate_vecs:
        return []

    # Project to GNN space — the attention weights were trained in GNN space
    current_gnn    = _w2v_to_gnn(current_vec)
    candidates_gnn = [_w2v_to_gnn(v) for v in candidate_vecs]

    k = len(candidates_gnn)
    all_vecs = np.stack([current_gnn] + candidates_gnn).astype(np.float32)
    x = torch.tensor(all_vecs)

    # Directed edges: current (node 0) → each candidate (nodes 1..k)
    src = torch.zeros(k, dtype=torch.long)
    dst = torch.arange(1, k + 1, dtype=torch.long)
    edge_index = torch.stack([src, dst])

    with torch.no_grad():
        layer = _model.layers[0]
        _, (_, alpha) = layer(x, edge_index, return_attention_weights=True)
        # alpha: (k, n_heads) — with 1 head this is (k, 1)
        weights = alpha.squeeze(-1).cpu().numpy()   # (k,)

    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if total > 0:
        weights /= total
    else:
        weights = np.ones(k, dtype=np.float32) / k

    return weights.tolist()


# ---------------------------------------------------------------------------
# 2. Alignment score — reconstruction likelihood
# ---------------------------------------------------------------------------

def alignment_score(
    content_vec: np.ndarray,
    k_neighbors: int = 5,
) -> float:
    """
    Measure how well the Biblical manifold explains this content.

    Accepts a W2V-space vector; projects to GNN space internally.
    Method: mask the content node, reconstruct from its k-nearest Biblical
    GNN verse embeddings, compute NMSE.  Converts to probability via exp(-NMSE).

    Returns
    -------
    float in [0, 1]:
      1.0   = perfect reconstruction = maximally consistent with world model
      ~0.37 = no better than mean prediction (random baseline)
      → 0   = completely alien to the Biblical manifold
    """
    if _model is None:
        raise RuntimeError("Oracle not initialised.")

    # Project W2V → GNN space (reconstruction and neighbors must share the same space)
    content_gnn = _w2v_to_gnn(content_vec)

    neighbor_vecs = _get_biblical_neighbors(content_gnn, k_neighbors)
    if not neighbor_vecs:
        return 0.0

    # Graph: node 0 = content (masked to 0); nodes 1..k = Biblical neighbors
    masked = np.zeros_like(content_gnn, dtype=np.float32)
    all_vecs = np.stack([masked] + neighbor_vecs).astype(np.float32)
    x = torch.tensor(all_vecs)

    # Edges: neighbors → content (they provide the reconstruction signal)
    n = len(neighbor_vecs)
    src = torch.arange(1, n + 1, dtype=torch.long)
    dst = torch.zeros(n, dtype=torch.long)
    edge_index = torch.stack([src, dst])

    target = torch.tensor(content_gnn, dtype=torch.float32)

    with torch.no_grad():
        z = _model(x, edge_index)
        mse  = F.mse_loss(z[0], target).item()
        var  = float(target.var().item())
        nmse = mse / (var + 1e-9)

    return float(math.exp(-nmse))   # 1.0 = perfect, ~0.37 = mean-level


# ---------------------------------------------------------------------------
# 3. Novelty score — flipped logit
# ---------------------------------------------------------------------------

def novelty_score(
    content_vec: np.ndarray,
    k_neighbors: int = 5,
) -> float:
    """
    Measure how far outside the known manifold this content is.

    This is the flipped reconstruction head: instead of rewarding correct
    prediction, we measure the FAILURE to predict — i.e. genuine novelty.

    Returns
    -------
    float in [0, 1]:
      1.0  = maximally novel (world model has no context for this at all)
      0.0  = perfectly predicted = zero novelty (seen something just like it)
    """
    return 1.0 - alignment_score(content_vec, k_neighbors)


# ---------------------------------------------------------------------------
# Combined scoring (convenience)
# ---------------------------------------------------------------------------

def score(content_vec: np.ndarray, k_neighbors: int = 5) -> dict:
    """
    Return alignment, novelty, and nearest Biblical concepts in one call.
    """
    from sklearn.preprocessing import normalize
    neighbors = _get_biblical_neighbors_with_meta(content_vec, k=k_neighbors)
    a = alignment_score(content_vec, k_neighbors)
    return {
        "alignment": round(a, 4),
        "novelty":   round(1.0 - a, 4),
        "nearest_biblical_concepts": neighbors,
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _get_biblical_neighbors(vec: np.ndarray, k: int) -> list[np.ndarray]:
    from sklearn.preprocessing import normalize as sk_norm
    q = sk_norm(vec.reshape(1, -1).astype(np.float32), norm="l2")
    _, indices = _faiss_index.search(q, k)
    result = []
    for idx in indices[0]:
        if idx >= 0:
            vid = _verse_id_list[int(idx)]
            result.append(_gnn_embeddings[vid].astype(np.float32))
    return result


def _get_biblical_neighbors_with_meta(vec: np.ndarray, k: int) -> list[dict]:
    from services import store
    from sklearn.preprocessing import normalize as sk_norm
    q = sk_norm(vec.reshape(1, -1).astype(np.float32), norm="l2")
    scores, indices = _faiss_index.search(q, k)
    result = []
    for sim, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            vid = _verse_id_list[int(idx)]
            m = store.get_meta(vid)
            result.append({"ref": m["ref"], "text": m["text"][:120],
                           "cosine_sim": round(float(sim), 4)})
    return result
