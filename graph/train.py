"""
Self-supervised GNN training with meaningful metrics and double-descent-aware
checkpointing.

Training signal: masked feature reconstruction (MLM-style for graphs).
  - Randomly mask MASK_FRAC of verse node features (zero them out).
  - Train the GNN to reconstruct the original features from unmasked neighbours.
  - Loss: MSE between GNN output and original features on masked nodes only.
  - This forces genuine multi-hop graph reasoning; the task cannot be solved
    from the initial features alone.

Link prediction used for validation only (held-out edges not seen during train).

Metrics recorded at every epoch
────────────────────────────────
  train_loss        MSE on masked-node reconstruction
  val_perplexity    exp((BCE_pos + BCE_neg) / 2) on held-out edges
                    anchors: 1.0 = perfect  |  2.0 = coin-flip (random)
  kc_gzip           gzip(embeddings) / raw_bytes  (lower = more structured)
  participation     (Σλ)² / Σ(λ²) on PCA eigenvalues  (higher = more axes used)

Checkpointing
─────────────
  Every epoch: weights + embeddings saved to models/checkpoints/epoch_{N}/
  End of run:  best checkpoint (lowest val_perplexity) restored to
               models/gnn_weights.pt and models/gnn_embeddings.pkl
  Full history: models/training_history.pkl — inspect for double descent

Run:  python graph/train.py
"""

import sys
import gzip
import json
import pickle
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from config import (
    MODELS_DIR,
    GNN_LAYERS, GNN_HEADS, GNN_OUT_DIM, GNN_DROPOUT,
    GNN_EPOCHS, GNN_LR,
)
from graph.gnn import BibleGAT

PYG_DATA_PATH       = MODELS_DIR / "pyg_data.pt"
GNN_WEIGHTS_PATH    = MODELS_DIR / "gnn_weights.pt"
GNN_EMBEDDINGS_PATH = MODELS_DIR / "gnn_embeddings.pkl"
HISTORY_PATH        = MODELS_DIR / "training_history.pkl"
CKPT_DIR            = MODELS_DIR / "checkpoints"

MASK_FRAC = 0.15   # fraction of verse nodes whose features are masked each step


# ---------------------------------------------------------------------------
# Node split — hold out val_frac of nodes for reconstruction validation
# ---------------------------------------------------------------------------

def split_nodes(n: int, val_frac: float = 0.1):
    perm    = torch.randperm(n)
    n_val   = max(1, int(n * val_frac))
    val_idx   = perm[:n_val]
    train_mask = torch.ones(n, dtype=torch.bool)
    train_mask[val_idx] = False
    return train_mask, val_idx


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def mask_features(x: torch.Tensor, frac: float):
    """
    Zero out frac of node feature vectors.
    Returns (x_masked, mask_idx) where mask_idx is the indices of masked nodes.
    """
    n = x.shape[0]
    n_mask   = max(1, int(n * frac))
    mask_idx = torch.randperm(n)[:n_mask]
    x_masked = x.clone()
    x_masked[mask_idx] = 0.0
    return x_masked, mask_idx


# ---------------------------------------------------------------------------
# Loss (train: reconstruction MSE; val: link-prediction BCE)
# ---------------------------------------------------------------------------

def reconstruction_loss(z: torch.Tensor, x_orig: torch.Tensor,
                        mask_idx: torch.Tensor) -> torch.Tensor:
    """MSE between GNN output and original features on masked nodes only."""
    return F.mse_loss(z[mask_idx], x_orig[mask_idx])


def _bce_loss(z, pos_edges, neg_edges):
    pos = (z[pos_edges[0]] * z[pos_edges[1]]).sum(-1)
    neg = (z[neg_edges[0]] * z[neg_edges[1]]).sum(-1)
    return F.binary_cross_entropy_with_logits(
        pos, torch.ones_like(pos)
    ) + F.binary_cross_entropy_with_logits(
        neg, torch.zeros_like(neg)
    )


def val_perplexity(
    z: torch.Tensor,
    x_orig: torch.Tensor,
    val_idx: torch.Tensor,
) -> float:
    """
    exp( NMSE ) where NMSE = MSE_val / Var(x_orig).

    Normalised MSE anchors:
      perfect reconstruction → NMSE = 0   → PP = 1.0
      predicting the mean    → NMSE = 1   → PP = e ≈ 2.72
      worse than mean        → NMSE > 1   → PP > e
    """
    with torch.no_grad():
        mse  = F.mse_loss(z[val_idx], x_orig[val_idx]).item()
    var  = x_orig.var().item()
    nmse = mse / (var + 1e-9)
    return float(np.exp(nmse))


# ---------------------------------------------------------------------------
# KC proxies
# ---------------------------------------------------------------------------

def kc_gzip(z: np.ndarray) -> float:
    """
    Compression ratio of the flat embedding matrix via gzip.
    Lower = more structure (better KC approximation).
    """
    raw   = z.astype(np.float32).tobytes()
    cmp   = gzip.compress(raw, compresslevel=9)
    return len(cmp) / len(raw)


def participation_ratio(z: np.ndarray) -> float:
    """
    (Σλ)² / Σ(λ²) over PCA eigenvalues of z.
    Range: [1, d].  Higher = more axes genuinely active.
    """
    pca = PCA(n_components=z.shape[1])
    pca.fit(z)
    lam = pca.explained_variance_
    return float(lam.sum() ** 2 / (lam ** 2).sum())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    data: Data = torch.load(PYG_DATA_PATH, weights_only=False)
    in_dim  = data.x.shape[1]
    out_dim = in_dim

    torch.manual_seed(0)
    train_mask, val_idx = split_nodes(data.num_nodes)
    n_val = val_idx.shape[0]

    print(f"Graph  nodes={data.num_nodes:,}  edges={data.edge_index.shape[1]:,}  "
          f"val_nodes={n_val:,}")
    print(f"GNN    layers={GNN_LAYERS}  heads={GNN_HEADS}  "
          f"dim={out_dim}  lr={GNN_LR}  epochs={GNN_EPOCHS}")
    print(f"Task   masked feature reconstruction  mask_frac={MASK_FRAC}")
    print(f"Val    NMSE-perplexity  anchors: 1.0=perfect  e≈2.72=mean-predict\n")

    model = BibleGAT(in_dim, out_dim, GNN_LAYERS, GNN_HEADS, GNN_DROPOUT)
    opt   = torch.optim.Adam(model.parameters(), lr=GNN_LR)

    x = data.x
    edge_index = data.edge_index

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    for epoch in range(1, GNN_EPOCHS + 1):
        # ── Train: masked reconstruction ──
        model.train()
        opt.zero_grad()
        x_masked, mask_idx = mask_features(x, MASK_FRAC)
        z    = model(x_masked, edge_index)
        loss = reconstruction_loss(z, x, mask_idx)
        loss.backward()
        opt.step()

        # ── Validate: reconstruct held-out nodes from unmasked neighbours ──
        model.eval()
        with torch.no_grad():
            # Mask val nodes so they must be recovered from graph context
            x_val_masked        = x.clone()
            x_val_masked[val_idx] = 0.0
            z_full = model(x_val_masked, edge_index)

        z_np   = z_full.numpy()
        v_perp = val_perplexity(z_full, x, val_idx)
        kc       = kc_gzip(z_np)
        part     = participation_ratio(z_np)

        # ── Checkpoint ──
        ckpt_path = CKPT_DIR / f"epoch_{epoch:04d}"
        ckpt_path.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt_path / "weights.pt")
        verse_ids = data.verse_ids.numpy()
        emb_dict  = {int(vid): z_np[i] for i, vid in enumerate(verse_ids)}
        with open(ckpt_path / "embeddings.pkl", "wb") as f:
            pickle.dump(emb_dict, f)

        record = {
            "epoch":          epoch,
            "train_loss":     round(loss.item(), 6),
            "val_perplexity": round(v_perp,      6),
            "kc_gzip":        round(kc,           6),
            "participation":  round(part,         4),
            "ckpt_path":      str(ckpt_path),
        }
        history.append(record)

        print(f"  epoch {epoch:3d}/{GNN_EPOCHS}"
              f"  loss={loss.item():.4f}"
              f"  val_perplexity={v_perp:.4f}"
              f"  kc_gzip={kc:.4f}"
              f"  participation={part:.2f}/{out_dim}")

    # ── Persist full history ──
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history, f)
    print(f"\nFull history → {HISTORY_PATH}")

    # ── Double-descent-aware best selection ──
    best = min(history, key=lambda r: r["val_perplexity"])
    print(f"\nBest epoch = {best['epoch']}  "
          f"val_perplexity={best['val_perplexity']}  "
          f"kc_gzip={best['kc_gzip']}  "
          f"participation={best['participation']}")

    best_ckpt = Path(best["ckpt_path"])
    shutil.copy(best_ckpt / "weights.pt",    GNN_WEIGHTS_PATH)
    shutil.copy(best_ckpt / "embeddings.pkl", GNN_EMBEDDINGS_PATH)
    print(f"Best checkpoint restored → {GNN_WEIGHTS_PATH}")

    # ── Print full history for double-descent inspection ──
    print("\nFull validation curve (for double-descent inspection):")
    print(f"  {'epoch':>5}  {'train_loss':>10}  {'val_perp':>10}  "
          f"{'kc_gzip':>8}  {'partic':>6}")
    for r in history:
        marker = " ← best" if r["epoch"] == best["epoch"] else ""
        print(f"  {r['epoch']:5d}  {r['train_loss']:10.4f}  "
              f"{r['val_perplexity']:10.4f}  {r['kc_gzip']:8.4f}  "
              f"{r['participation']:6.2f}{marker}")


if __name__ == "__main__":
    train()
