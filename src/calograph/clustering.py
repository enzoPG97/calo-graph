"""Turn edge score into hit clusters and measure how well they match the truth."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def cluster_hits(
    edge_index: np.ndarray, scores: np.ndarray, n_nodes: int, threshold: float = 0.0
) -> np.ndarray:
    """Keep edges with score > threshold and return the connected-component label of each hit.

    The scores are logits, so threshold 0 means probability 0.5. A hit with no
    kept edge forms a cluster of its own.
    """
    keep = np.asarray(scores) > threshold
    src, dst = np.asarray(edge_index)[:, keep]
    adjacency = coo_matrix((np.ones(src.shape[0]), (src, dst)), shape=(n_nodes, n_nodes))
    _, labels = connected_components(adjacency, directed=False)
    return labels

def cluster_metrics(
    truth: np.ndarray, predicted: np.ndarray, min_hits: int = 10, min_fraction: float = 0.1
) -> dict[str, float]:
    """Compare predicted clusters with truth particles (one value per event).

    purity      fraction of hits that belong to the majority truth particle of their cluster
    efficiency  fraction of hits that sit in the best predicted cluster of their particle
    merged      True if a cluster of at least min_hits hits has at least min_fraction of its
                hits from each of two or more particles
    split       True if a particle has at least min_fraction of its hits in each of two or
                more clusters of at least min_hits hits
    n_clusters  number of predicted clusters with at least min_hits hits
    """
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("truth and predicted must be non-empty arrays of the same length")

    _, t_index = np.unique(truth, return_inverse=True)
    _, p_index = np.unique(predicted, return_inverse=True)
    table = np.zeros((t_index.max() + 1, p_index.max() + 1), dtype=np.int64)
    np.add.at(table, (t_index, p_index), 1)  # table[particle, cluster] = number of hits

    n = truth.size
    cluster_size = table.sum(axis=0)
    particle_size = table.sum(axis=1)
    big = cluster_size >= min_hits

    share_of_cluster = table / cluster_size                      # per column
    share_of_particle = table / particle_size[:, None]           # per row
    merged = np.any(big & ((share_of_cluster >= min_fraction).sum(axis=0) >= 2))
    split = np.any(((share_of_particle >= min_fraction) & big).sum(axis=1) >= 2)

    return {
        "purity": float(table.max(axis=0).sum() / n),
        "efficiency": float(table.max(axis=1).sum() / n),
        "merged": bool(merged),
        "split": bool(split),
        "n_clusters": int(big.sum()),
    }
