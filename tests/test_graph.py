import numpy as np
import pytest

from calograph import generate_pair_event
from calograph.generator import Event
from calograph.graph import build_graph, knn_edges, radius_edges


def _event(sep=6.0, seed=0):
    return generate_pair_event(sep, seed=seed)


def test_shapes_and_dtypes():
    ev = _event()
    g = build_graph(ev, k=6)
    assert g.node_features.shape == (ev.n_hits, 4)
    assert g.edge_index.shape == (2, g.n_edges)
    assert g.edge_features.shape == (g.n_edges, 5)
    assert g.edge_label.shape == (g.n_edges,)
    assert g.edge_index.dtype == np.int64
    assert g.edge_label.dtype == np.float32
    assert g.node_features.dtype == np.float32
    assert g.edge_index.min() >= 0
    assert g.edge_index.max() < ev.n_hits


def test_no_self_loops_no_duplicates_and_undirected():
    g = build_graph(_event(), k=6)
    src, dst = g.edge_index
    assert not np.any(src == dst)
    pairs = set(zip(src.tolist(), dst.tolist()))
    assert len(pairs) == g.n_edges                       # no duplicate edges
    assert all((j, i) in pairs for i, j in pairs)        # every edge has its reverse


def test_every_node_has_at_least_k_neighbours():
    k = 5
    g = build_graph(_event(), k=k)
    degree = np.bincount(g.edge_index[0], minlength=g.n_nodes)
    assert degree.min() >= k


def test_edge_labels_match_truth_ids():
    ev = _event(sep=3.0)
    g = build_graph(ev, k=6)
    src, dst = g.edge_index
    expected = (ev.particle_id[src] == ev.particle_id[dst]).astype(np.float32)
    np.testing.assert_array_equal(g.edge_label, expected)


def test_separated_showers_have_mostly_same_particle_edges():
    sep_far = build_graph(_event(sep=15.0), k=6).edge_label.mean()
    sep_zero = build_graph(_event(sep=0.0), k=6).edge_label.mean()
    assert sep_far > 0.99
    assert sep_zero < 0.8
    assert sep_zero < sep_far


def test_radius_graph_respects_radius():
    ev = _event()
    r = 1.0
    g = build_graph(ev, radius_cm=r)
    src, dst = g.edge_index
    dist = np.linalg.norm(g.pos[dst] - g.pos[src], axis=1)
    assert g.n_edges > 0
    assert dist.max() <= r + 1e-5


def test_tiny_events_do_not_crash():
    one = np.zeros(1)
    ev = Event(x=one, y=one, z=one, energy=one, time=one,
               layer=np.zeros(1, dtype=int), particle_id=np.zeros(1, dtype=int))
    g = build_graph(ev, k=4)
    assert g.n_nodes == 1 and g.n_edges == 0
    assert radius_edges(np.zeros((1, 3)), 1.0).shape == (2, 0)
    assert knn_edges(np.zeros((0, 3)), 3).shape == (2, 0)


def test_invalid_arguments_raise():
    ev = _event()
    with pytest.raises(ValueError):
        build_graph(ev, k=0)
    with pytest.raises(ValueError):
        build_graph(ev, radius_cm=-1.0)


def test_is_reproducible():
    a = build_graph(_event(seed=4), k=6)
    b = build_graph(_event(seed=4), k=6)
    np.testing.assert_array_equal(a.edge_index, b.edge_index)
    np.testing.assert_array_equal(a.edge_label, b.edge_label)


def test_to_pyg_conversion():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from calograph.graph import to_pyg

    g = build_graph(_event(), k=6)
    data = to_pyg(g)
    assert data.num_nodes == g.n_nodes
    assert data.edge_index.shape == (2, g.n_edges)
    assert data.edge_index.dtype == torch.int64
    assert data.edge_attr.shape == (g.n_edges, 5)
    assert data.edge_label.shape == (g.n_edges,)
    assert data.validate()
