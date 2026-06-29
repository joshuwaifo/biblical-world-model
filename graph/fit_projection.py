"""
Fit a linear W2V → GNN projection from all 38,927 verse pairs.

The FAISS index is built on L2-normalised GNN embeddings.  Any new text
(query, retrieved content, arbitrary bytes-as-text) must be projected into
GNN space before searching.  This script learns the optimal linear map.

Method
──────
Ridge regression: minimise  ||W_n @ P + b - G_n||_F^2 + alpha * ||P||_F^2
where W_n, G_n are L2-normalised W2V and GNN verse vectors respectively.

After fitting, the projection is evaluated on a held-out 10% split and
saved to models/w2v_to_gnn_projection.pkl as {"coef": P, "intercept": b}.

Run:  python graph/fit_projection.py
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from config import MODELS_DIR

W2V_PATH  = MODELS_DIR / "verse_vectors.pkl"
GNN_PATH  = MODELS_DIR / "gnn_embeddings.pkl"
PROJ_PATH = MODELS_DIR / "w2v_to_gnn_projection.pkl"


def fit() -> None:
    with open(W2V_PATH, "rb") as f:
        w2v_vecs: dict = pickle.load(f)
    with open(GNN_PATH, "rb") as f:
        gnn_vecs: dict = pickle.load(f)

    common = sorted(set(w2v_vecs) & set(gnn_vecs))
    W = np.stack([w2v_vecs[v] for v in common]).astype(np.float32)
    G = np.stack([gnn_vecs[v]  for v in common]).astype(np.float32)

    # L2-normalise both — FAISS uses cosine, so direction is what matters
    W_n = normalize(W, norm="l2")
    G_n = normalize(G, norm="l2")

    # Evaluation split
    W_tr, W_te, G_tr, G_te = train_test_split(W_n, G_n, test_size=0.1, random_state=0)

    # Sweep alpha to find best cosine alignment on held-out set
    best_alpha, best_cos = 1e-3, -1.0
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1]:
        reg = Ridge(alpha=alpha, fit_intercept=True)
        reg.fit(W_tr, G_tr)
        pred = normalize(reg.predict(W_te), norm="l2")
        cos  = float((pred * G_te).sum(axis=1).mean())
        print(f"  alpha={alpha:.0e}  projected cosine={cos:.6f}")
        if cos > best_cos:
            best_cos, best_alpha = cos, alpha

    # Refit on ALL data with best alpha
    print(f"\nBest alpha={best_alpha:.0e} — refitting on all {len(common):,} verses")
    reg = Ridge(alpha=best_alpha, fit_intercept=True)
    reg.fit(W_n, G_n)

    # Final evaluation
    pred_all = normalize(reg.predict(W_n), norm="l2")
    cos_all  = float((pred_all * G_n).sum(axis=1).mean())
    print(f"Final projected cosine (all verses): {cos_all:.6f}")

    projection = {
        "coef":      reg.coef_.T.astype(np.float32),   # (8, 8): W @ coef + b ≈ G
        "intercept": reg.intercept_.astype(np.float32), # (8,)
        "alpha":     best_alpha,
        "n_verses":  len(common),
        "mean_projected_cosine": cos_all,
    }

    with open(PROJ_PATH, "wb") as f:
        pickle.dump(projection, f)

    print(f"\nProjection saved → {PROJ_PATH}")
    print(f"  shape: coef={projection['coef'].shape}  "
          f"intercept={projection['intercept'].shape}")


if __name__ == "__main__":
    fit()
