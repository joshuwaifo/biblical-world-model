"""
Convert the Bible graph (NetworkX) to a PyTorch Geometric Data object.

Starting from the most irreducible representation:
  - Homogeneous graph (verse nodes only; entities added at Step 2)
  - Node features = 8-dim Word2Vec mean-pool vectors
  - Edge index = PRECEDES edges only (canonical sequential order)

Run:  python graph/pyg_data.py
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch_geometric.data import Data

from config import MODELS_DIR

VERSE_VECTORS_PATH = MODELS_DIR / "verse_vectors.pkl"
GRAPH_PATH         = MODELS_DIR / "bible_graph.gpickle"
PYG_DATA_PATH      = MODELS_DIR / "pyg_data.pt"


def build() -> Data:
    # Load Word2Vec verse vectors
    with open(VERSE_VECTORS_PATH, "rb") as f:
        verse_vectors: dict[int, np.ndarray] = pickle.load(f)

    # Load graph
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    # Keep only verse nodes, ordered by verse id (stable ordering)
    verse_nodes = sorted(
        [n for n in G.nodes() if n[0] == "verse"],
        key=lambda n: n[1],
    )
    verse_id_to_idx = {n[1]: i for i, n in enumerate(verse_nodes)}
    n = len(verse_nodes)

    # Node feature matrix  X : (n, d)
    d = next(iter(verse_vectors.values())).shape[0]
    X = np.zeros((n, d), dtype=np.float32)
    for i, (_, vid) in enumerate(verse_nodes):
        if vid in verse_vectors:
            X[i] = verse_vectors[vid]
    x = torch.tensor(X)

    # Edge index from graph edges (verse→verse only)
    src_list, dst_list = [], []
    for (st, si), (dt, di), attrs in G.edges(data=True):
        if st == "verse" and dt == "verse":
            if si in verse_id_to_idx and di in verse_id_to_idx:
                src_list.append(verse_id_to_idx[si])
                dst_list.append(verse_id_to_idx[di])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = n
    data.verse_ids = torch.tensor([vid for _, vid in verse_nodes], dtype=torch.long)

    torch.save(data, PYG_DATA_PATH)
    print(f"PyG data → {PYG_DATA_PATH}")
    print(f"  nodes:          {data.num_nodes:,}")
    print(f"  node features:  {d}")
    print(f"  edges:          {edge_index.shape[1]:,}")

    # Save the id→index map for retrieval later
    with open(MODELS_DIR / "verse_id_to_idx.pkl", "wb") as f:
        pickle.dump(verse_id_to_idx, f)

    return data


if __name__ == "__main__":
    build()
