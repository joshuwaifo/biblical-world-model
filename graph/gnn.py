"""
Graph Attention Network (GAT) for the Bible graph.

Starting from the most irreducible architecture:
  - 1 layer      (single message-passing step)
  - 1 head       (basic attention, no multi-head)
  - dim_out = dim_in  (no projection; same dimensionality as Word2Vec)
  - no dropout, no residual, no normalisation

Expand only when verification shows the current geometry has saturated:
  layers: 1 → 2 → 3
  heads:  1 → 2 → 4
  dim:    keep equal to W2V_VECTOR_SIZE unless geometric evidence demands more
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class BibleGAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_layers: int = 1,
        n_heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # Each GATConv with concat=False averages over heads → out_dim preserved
            self.layers.append(
                GATConv(in_dim, out_dim, heads=n_heads, concat=False,
                        dropout=dropout, add_self_loops=True)
            )
        self.activation = nn.ELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:   # no activation on final layer
                x = self.activation(x)
        return x
