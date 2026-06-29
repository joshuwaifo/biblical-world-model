"""
Build and persist FAISS indices over verse embeddings.

Run:  python index/faiss_store.py

Produces:
  models/verse_embeddings.faiss  — IndexFlatIP over L2-normalised GNN 8-dim vectors
  models/verse_id_map.pkl        — list[int] mapping FAISS row → verse_id (GNN)
  models/verse_w2v.faiss         — IndexFlatIP over L2-normalised W2V 8-dim vectors
  models/verse_w2v_id_map.pkl    — list[int] mapping FAISS row → verse_id (W2V)

Two indices, two spaces:
  GNN FAISS  — oracle internal operations (alignment, novelty, next_action)
  W2V FAISS  — semantic search (user queries, navigator position tracking)
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import faiss
from sklearn.preprocessing import normalize

from config import (MODELS_DIR, FAISS_INDEX_PATH, VERSE_ID_MAP_PATH,
                    W2V_FAISS_INDEX_PATH, W2V_VERSE_ID_MAP_PATH)

GNN_EMBEDDINGS_PATH = MODELS_DIR / "gnn_embeddings.pkl"
W2V_VECTORS_PATH    = MODELS_DIR / "verse_vectors.pkl"


def build() -> None:
    """GNN-space FAISS (for oracle alignment/novelty/next_action)."""
    with open(GNN_EMBEDDINGS_PATH, "rb") as f:
        embeddings: dict[int, np.ndarray] = pickle.load(f)

    verse_ids = sorted(embeddings.keys())
    mat = np.stack([embeddings[vid] for vid in verse_ids]).astype(np.float32)
    mat = normalize(mat, norm="l2")

    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(VERSE_ID_MAP_PATH, "wb") as f:
        pickle.dump(verse_ids, f)

    print(f"GNN FAISS    → {FAISS_INDEX_PATH}  ({index.ntotal:,} vectors, dim={dim})")
    print(f"GNN id map   → {VERSE_ID_MAP_PATH}")


def build_w2v() -> None:
    """W2V-space FAISS (for semantic text search and navigator position tracking)."""
    with open(W2V_VECTORS_PATH, "rb") as f:
        w2v_vecs: dict[int, np.ndarray] = pickle.load(f)

    verse_ids = sorted(w2v_vecs.keys())
    mat = np.stack([w2v_vecs[vid] for vid in verse_ids]).astype(np.float32)
    mat = normalize(mat, norm="l2")

    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    faiss.write_index(index, str(W2V_FAISS_INDEX_PATH))
    with open(W2V_VERSE_ID_MAP_PATH, "wb") as f:
        pickle.dump(verse_ids, f)

    print(f"W2V FAISS    → {W2V_FAISS_INDEX_PATH}  ({index.ntotal:,} vectors, dim={dim})")
    print(f"W2V id map   → {W2V_VERSE_ID_MAP_PATH}")


if __name__ == "__main__":
    build()
    build_w2v()
