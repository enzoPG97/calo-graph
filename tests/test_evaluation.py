import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from calograph.dataset import make_dataset  # noqa: E402
from calograph.evaluation import (evaluate_by_separation, keep_all_edges,  # noqa: E402
                                  select_threshold)

SMALL = dict(energy_range_gev=(3.0, 4.0))


def oracle(data):
    """Perfect scorer: +10 for same-particle edges, -10 otherwise."""
    return (data.edge_label * 2 - 1) * 10


def test_make_dataset_respects_min_separation():
    graphs = make_dataset(6, seed=0, min_sep_cm=2.0, max_sep_cm=3.0, **SMALL)
    seps = torch.cat([g.separation_cm for g in graphs])
    assert ((seps >= 2.0) & (seps <= 3.0)).all()


def test_keep_all_edges_scores_every_edge():
    data = make_dataset(1, seed=0, **SMALL)[0]
    assert keep_all_edges(data).shape == (data.edge_index.shape[1],)


def test_oracle_scores_give_perfect_edges_and_pure_clusters():
    rows = evaluate_by_separation(oracle, bin_edges=(0.0, 3.0, 6.0), events_per_bin=3,
                                  seed=1, **SMALL)
    assert [(r["sep_lo"], r["sep_hi"]) for r in rows] == [(0.0, 3.0), (3.0, 6.0)]
    for r in rows:
        assert r["n_events"] == 3
        assert r["auc"] == pytest.approx(1.0)
        assert r["purity"] == pytest.approx(1.0)
        assert r["merge_rate"] == 0.0
        assert 0.0 <= r["efficiency"] <= 1.0


def test_keeping_every_edge_merges_overlapping_showers():
    rows = evaluate_by_separation(keep_all_edges, bin_edges=(0.0, 1.0), events_per_bin=3,
                                  seed=1, **SMALL)
    assert rows[0]["merge_rate"] == 1.0
    assert rows[0]["purity"] < 0.7


def test_evaluation_is_reproducible():
    kwargs = dict(bin_edges=(0.0, 2.0), events_per_bin=2, seed=5, **SMALL)
    assert evaluate_by_separation(oracle, **kwargs) == evaluate_by_separation(oracle, **kwargs)


def test_select_threshold_picks_loosest_threshold_meeting_the_purity_target():
    graphs = make_dataset(4, seed=2, max_sep_cm=1.0, **SMALL)
    best, table = select_threshold(oracle, graphs)
    assert best == 0.0                              # oracle is pure already at threshold 0
    assert table[0.0][0] == pytest.approx(1.0)
    # constant scores of 1: below threshold 1 everything merges, from threshold 1 up
    # every edge is dropped (strict >) and all hits are singletons, which are pure
    best_const, _ = select_threshold(keep_all_edges, graphs)
    assert best_const == 1.0


def test_select_threshold_falls_back_to_strictest_when_target_unreachable():
    graphs = make_dataset(2, seed=3, max_sep_cm=1.0, **SMALL)
    best, _ = select_threshold(keep_all_edges, graphs, min_purity=1.01, thresholds=(0.0, 1.0, 2.0))
    assert best == 2.0
