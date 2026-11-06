import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch  # noqa: E402

from calograph.dataset import make_dataset  # noqa: E402
from calograph.metrics import edge_auc  # noqa: E402
from calograph.model import EdgeMLP  # noqa: E402
from calograph.train import TrainConfig, evaluate, load_model, train  # noqa: E402


def _tiny_config(tmp_path, **kwargs):
    base = dict(n_train=6, n_val=3, epochs=2, batch_size=3, hidden=8, n_layers=2,
                mlp_hidden=8, out_dir=str(tmp_path), seed=0)
    base.update(kwargs)
    return TrainConfig(**base)


def test_edge_auc_known_cases():
    labels = [0, 0, 1, 1]
    assert edge_auc(labels, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert edge_auc(labels, [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)
    assert edge_auc(labels, [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)   # ties: half credit
    assert edge_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert np.isnan(edge_auc([1, 1, 1], [0.1, 0.2, 0.3]))                 # one class only


def test_dataset_is_reproducible_and_in_range():
    a = make_dataset(4, seed=1, max_sep_cm=5.0, energy_range_gev=(2.0, 4.0))
    b = make_dataset(4, seed=1, max_sep_cm=5.0, energy_range_gev=(2.0, 4.0))
    assert len(a) == 4
    for da, db in zip(a, b):
        assert torch.equal(da.edge_index, db.edge_index)
        assert torch.equal(da.edge_label, db.edge_label)
    seps = torch.cat([d.separation_cm for d in a])
    assert ((seps >= 0) & (seps <= 5.0)).all()


def test_different_seeds_give_different_events():
    a = make_dataset(3, seed=1, energy_range_gev=(2.0, 4.0))
    b = make_dataset(3, seed=2, energy_range_gev=(2.0, 4.0))
    seps_a = [d.separation_cm.item() for d in a]
    seps_b = [d.separation_cm.item() for d in b]
    assert seps_a != seps_b


def test_batching_keeps_one_separation_per_graph():
    graphs = make_dataset(3, seed=0, energy_range_gev=(2.0, 3.0))
    batch = Batch.from_data_list(graphs)
    assert batch.separation_cm.shape == (3,)


def test_baseline_output_shape_and_symmetry():
    data = make_dataset(1, seed=0, energy_range_gev=(2.0, 3.0))[0]
    out = EdgeMLP(hidden=8).eval()(data)
    assert out.shape == (data.edge_index.shape[1],)
    src, dst = data.edge_index
    lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(src, dst))}
    reverse = torch.tensor([lookup[(int(j), int(i))] for i, j in zip(src, dst)])
    assert torch.allclose(out, out[reverse], atol=1e-5)


def test_config_from_json_and_unknown_key(tmp_path):
    good = tmp_path / "ok.json"
    good.write_text(json.dumps({"model": "mlp", "epochs": 3}))
    cfg = TrainConfig.from_json(good)
    assert cfg.model == "mlp" and cfg.epochs == 3 and cfg.k == 8   # others keep defaults
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"epochz": 3}))
    with pytest.raises(TypeError):
        TrainConfig.from_json(bad)


@pytest.mark.parametrize("model", ["gnn", "mlp"])
def test_training_runs_and_saves_a_loadable_checkpoint(tmp_path, model):
    cfg = _tiny_config(tmp_path, model=model)
    history = train(cfg, verbose=False)
    assert len(history) == 2
    assert all(np.isfinite(row["val_loss"]) for row in history)
    assert (tmp_path / f"{model}_best.pt").exists()
    assert (tmp_path / f"{model}_history.json").exists()
    loaded, loaded_cfg = load_model(tmp_path / f"{model}_best.pt")
    assert loaded_cfg == cfg


def test_training_is_reproducible(tmp_path):
    a = train(_tiny_config(tmp_path / "a"), verbose=False)
    b = train(_tiny_config(tmp_path / "b"), verbose=False)
    assert [r["val_loss"] for r in a] == pytest.approx([r["val_loss"] for r in b])


def test_unknown_model_name_raises(tmp_path):
    with pytest.raises(ValueError):
        train(_tiny_config(tmp_path, model="transformer"), verbose=False)


def test_evaluate_reports_sensible_metrics():
    from torch_geometric.loader import DataLoader
    from calograph.model import EdgeClassifier

    torch.manual_seed(0)
    loader = DataLoader(make_dataset(4, seed=3, energy_range_gev=(2.0, 3.0)), batch_size=2)
    metrics = evaluate(EdgeClassifier(hidden=8, n_layers=2), loader)
    assert set(metrics) == {"loss", "accuracy", "auc"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["auc"] <= 1.0
