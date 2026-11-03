############################################################################
# Edege-claisification GNN: do wo hits come from the same tructh particle? #
############################################################################


from __future__ import annotations
import torch
from torch import nn
from torch_geometric.nn import MessagePassing


def mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """Two-layer perceptron."""
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

class EdgeMPLayer(MessagePassing):
    """One round of message passing that uses the edge features.

    Each hit receives the mean of messages from its neighbours. A message is
    computed from the two hit embeddings and the (signed) edge features. The
    hit embedding is then updated with a residual connection.
    """

    def __init__(self, hidden: int, edge_dim: int) -> None:
        super().__init__(aggr="mean")
        self.message_mlp = mlp(2 * hidden + edge_dim, hidden, hidden)
        self.update_mlp = mlp(2 * hidden, hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        aggregated = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return self.norm(h + self.update_mlp(torch.cat([h, aggregated], dim=-1)))

    def message(self, h_i, h_j, edge_attr):  # noqa: D102
        return self.message_mlp(torch.cat([h_i, h_j, edge_attr], dim=-1))

class EdgeClassifier(nn.Module):
    """GNN returning one logit per edge (apply a sigmoid to get a probability).

    The final score is built from symmetric functions of the two end hits
    (sum and absolute difference of their embeddings, absolute edge features),
    so the edges (i, j) and (j, i) always get the same logit.
    """

    def __init__(self, node_dim: int = 4, edge_dim: int = 5, hidden: int = 32, n_layers: int = 3):
        super().__init__()
        self.node_encoder = mlp(node_dim, hidden, hidden)
        self.layers = nn.ModuleList([EdgeMPLayer(hidden, edge_dim) for _ in range(n_layers)])
        self.head = mlp(2 * hidden + edge_dim, hidden, 1)

    def forward(self, data) -> torch.Tensor:
        h = self.node_encoder(data.x)
        for layer in self.layers:
            h = layer(h, data.edge_index, data.edge_attr)
        src, dst = data.edge_index
        features = torch.cat(
            [h[src] + h[dst], (h[src] - h[dst]).abs(), data.edge_attr.abs()], dim=-1
        )
        return self.head(features).squeeze(-1)
