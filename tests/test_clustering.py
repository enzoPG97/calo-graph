import numpy as np
import pytest

from calograph.clustering import cluster_hits, cluster_metrics


def test_cluster_hits_finds_connected_components():
    edge_index = np.array([[0, 1, 2, 3, 1, 2], [1, 0, 3, 2, 2, 1]])  # 0-1, 2-3, 1-2 (both directions)
    scores = np.array([5.0, 5.0, 5.0, 5.0, -5.0, -5.0])               # the 1-2 link is rejected
    labels = cluster_hits(edge_index, scores, n_nodes=4)
    assert labels[0] == labels[1] and labels[2] == labels[3]
    assert labels[0] != labels[2]
    # keeping the 1-2 link joins everything
    assert len(set(cluster_hits(edge_index, np.full(6, 5.0), n_nodes=4))) == 1


def test_cluster_hits_without_kept_edges_gives_singletons():
    edge_index = np.array([[0, 1], [1, 0]])
    labels = cluster_hits(edge_index, np.array([-1.0, -1.0]), n_nodes=3)
    assert len(set(labels)) == 3


def test_perfect_clustering_ignores_label_names():
    truth = np.array([0, 0, 0, 1, 1, 1])
    predicted = np.array([7, 7, 7, 3, 3, 3])
    m = cluster_metrics(truth, predicted, min_hits=2)
    assert m["purity"] == 1.0 and m["efficiency"] == 1.0
    assert not m["merged"] and not m["split"] and m["n_clusters"] == 2


def test_one_giant_cluster_is_merged():
    truth = np.array([0, 0, 0, 1, 1, 1])
    m = cluster_metrics(truth, np.zeros(6, dtype=int), min_hits=2)
    assert m["purity"] == pytest.approx(0.5)
    assert m["efficiency"] == 1.0
    assert m["merged"] and not m["split"]


def test_singletons_are_pure_but_inefficient():
    truth = np.array([0, 0, 0, 1, 1, 1])
    m = cluster_metrics(truth, np.arange(6), min_hits=2)
    assert m["purity"] == 1.0
    assert m["efficiency"] == pytest.approx(2 / 6)
    assert m["n_clusters"] == 0 and not m["merged"] and not m["split"]


def test_split_particle_is_detected():
    truth = np.zeros(6, dtype=int)
    m = cluster_metrics(truth, np.array([0, 0, 0, 1, 1, 1]), min_hits=2)
    assert m["purity"] == 1.0
    assert m["efficiency"] == pytest.approx(0.5)
    assert m["split"] and not m["merged"]


def test_merge_needs_a_minimum_share_of_the_cluster():
    # 20-hit cluster: 1 foreign hit (5%) is not a merge, 3 foreign hits (15%) are
    few = cluster_metrics(np.array([0] * 19 + [1] * 1 + [1] * 5),
                          np.array([0] * 20 + [1] * 5), min_hits=5, min_fraction=0.1)
    many = cluster_metrics(np.array([0] * 17 + [1] * 3 + [1] * 5),
                           np.array([0] * 20 + [1] * 5), min_hits=5, min_fraction=0.1)
    assert not few["merged"]
    assert many["merged"]


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        cluster_metrics(np.array([0, 1]), np.array([0]))
    with pytest.raises(ValueError):
        cluster_metrics(np.array([]), np.array([]))
