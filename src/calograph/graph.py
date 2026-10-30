###############################################################################
#                                                                             #
# Build a hit graph (with truth edge labels) from a calorimeter event.        #
###############################################################################


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from calograph.generator import DetectorConfig, Event

@dataclass
class HitGraph:
    """ Graph of Hots. N = number of hits, E = number of (directd) edges."""
    node_features: np.ndarray   # (N, 4) float32: x, y, z, time (scaled)
    edge_index: np.ndarray      # (2, E) int64: source and target hit indices
    edge_features: np.ndarray   # (E, 5) float32: dx, dy, dz, distance, dt (scaled)
    edge_label: np.ndarray      # (E,) float32: 1 if both hits share a truth particle
    pos: np.ndarray             # (N, 3) float32: x, y, z in cm
    particle_id: np.ndarray     # (N,) int64: truth particle of each hit

    @property
    def n_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

def _symmetrize(src: np.darray, dst: np.darray) -> np.darray:
    """ Return unique undirected edges as directed pairs in both directions."""
    keep = src != dst  # no self-loops
    edges = np.stack([src[keep], dst[keep]]).astype(np.int64)
    if edges.shape[1] == 0:
        return edges
    edges = np.concatenate([edges, edges[::-1]], axis=1)
    return np.unique(edges, axis=1)
def knn_edges(points: np.ndarray, k: int) -> np.ndarray:
    """Connect each point to its k nearest neighbours (made undirected)."""
    n = points.shape[0]
    if n < 2:
        return np.empty((2, 0), dtype=np.int64)
    k = min(k, n - 1)
    _, idx = cKDTree(points).query(points, k=k + 1)  # column 0 is the point itself
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].ravel()
    return _symmetrize(src, dst)

def radius_edges(points: np.ndarray, radius: float) -> np.ndarray:
    """Connect every pair of points closer than radius (made undirected)."""
    if points.shape[0] < 2:
        return np.empty((2, 0), dtype=np.int64)
    pairs = cKDTree(points).query_pairs(r=radius, output_type="ndarray")
    return _symmetrize(pairs[:, 0], pairs[:, 1])

def build_graph(
    event: Event,
    k: int = 8,
    radius_cm: float | None = None,
    cfg: DetectorConfig | None = None,
) -> HitGraph:
    """Build the hit graph of an event.

    Edges come from the k nearest neighbours in (x, y, z), or from all pairs
    closer than radius_cm if it is given. Time is kept as a feature only,
    because its noise is comparable to the Moliere radius in this toy detector.
    The edge label is 1 if both hits come from the same truth particle.
    """
    if radius_cm is None and k < 1:
        raise ValueError("k must be at least 1")
    if radius_cm is not None and radius_cm <= 0:
        raise ValueError("radius_cm must be positive")
    cfg = cfg or DetectorConfig()

    pos = np.column_stack([event.x, event.y, event.z])
    if radius_cm is None:
        edge_index = knn_edges(pos, k)
    else:
        edge_index = radius_edges(pos, radius_cm)
    src, dst = edge_index

    length = cfg.moliere_cm                              # lateral scale
    depth = cfg.depth_max_x0 * cfg.X0_cm                 # detector depth in cm
    t_scale = depth / cfg.speed_cm_per_ns                # time to cross the detector

    node_features = np.column_stack(
        [event.x / length, event.y / length, event.z / depth, event.time / t_scale]
    ).astype(np.float32)

    delta = pos[dst] - pos[src]
    dt_cm = (event.time[dst] - event.time[src]) * cfg.speed_cm_per_ns
    edge_features = np.column_stack(
        [delta / length, np.linalg.norm(delta, axis=1) / length, dt_cm / length]
    ).astype(np.float32)

    edge_label = (event.particle_id[src] == event.particle_id[dst]).astype(np.float32)

    return HitGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        edge_label=edge_label,
        pos=pos.astype(np.float32),
        particle_id=event.particle_id.astype(np.int64),
    )

def to_pyg(graph: HitGraph):
    """Convert a HitGraph to a torch_geometric.data.Data object."""
    import torch
    from torch_geometric.data import Data

    return Data(
        x=torch.from_numpy(graph.node_features),
        edge_index=torch.from_numpy(graph.edge_index),
        edge_attr=torch.from_numpy(graph.edge_features),
        edge_label=torch.from_numpy(graph.edge_label),
        pos=torch.from_numpy(graph.pos),
        particle_id=torch.from_numpy(graph.particle_id),
    )

