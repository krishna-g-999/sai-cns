"""A Graph Isomorphism Network (GIN) in pure PyTorch, no torch-geometric.

GIN is a strong, standard message-passing architecture. Each layer updates an atom's vector from the
sum of its neighbours' vectors passed through a small MLP; after several layers the per-atom vectors
are pooled into one molecule vector and read out to the endpoint. Message aggregation is done with
`index_add_`, so no external graph library is needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def global_mean_pool(h, batch, num_graphs):
    out = torch.zeros(num_graphs, h.size(1), device=h.device)
    out.index_add_(0, batch, h)
    counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1).unsqueeze(1).float()
    return out / counts


class GINLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            agg = torch.zeros_like(x)
        else:
            src, dst = edge_index[0], edge_index[1]
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, x[src])  # sum neighbour messages into each node
        return self.mlp((1 + self.eps) * x + agg)


class GIN(nn.Module):
    def __init__(self, node_dim, hidden=64, n_layers=3, out_dim=1, dropout=0.2):
        super().__init__()
        self.input = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList([GINLayer(hidden) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(n_layers)])
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, x, edge_index, batch, num_graphs):
        h = torch.relu(self.input(x))
        for layer, norm in zip(self.layers, self.norms):
            h = torch.relu(norm(layer(h, edge_index)))
        return self.readout(global_mean_pool(h, batch, num_graphs)).squeeze(-1)


def collate(graphs, y):
    """Merge a list of (x, edge_index) graphs into one batched graph with a node->graph index."""
    xs, eis, batch, offset = [], [], [], 0
    for i, (x, ei) in enumerate(graphs):
        xs.append(x)
        if ei.numel():
            eis.append(ei + offset)
        batch.append(torch.full((x.size(0),), i, dtype=torch.long))
        offset += x.size(0)
    X = torch.cat(xs, dim=0)
    EI = torch.cat(eis, dim=1) if eis else torch.zeros((2, 0), dtype=torch.long)
    B = torch.cat(batch, dim=0)
    return X, EI, B, torch.tensor(y, dtype=torch.float32)
