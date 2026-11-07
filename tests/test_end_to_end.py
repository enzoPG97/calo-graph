import json
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from calograph import evaluation, train  # noqa: E402


def _run(monkeypatch, module, *argv):
    """Call module.main() as if `python -m calograph.<module> <argv>` had been typed."""
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    module.main()


def test_train_then_evaluate_from_the_command_line(tmp_path, monkeypatch, capsys):
    config = tmp_path / "tiny.json"
    config.write_text(json.dumps({
        "n_train": 4, "n_val": 3, "epochs": 1, "batch_size": 2, "hidden": 8,
        "n_layers": 2, "mlp_hidden": 8, "out_dir": str(tmp_path / "ck"),
    }))
    for model in ("gnn", "mlp"):
        _run(monkeypatch, train, "--config", str(config), "--model", model)
        assert (tmp_path / "ck" / f"{model}_best.pt").exists()

    out = tmp_path / "results" / "evaluation.json"
    _run(monkeypatch, evaluation, "--checkpoints", str(tmp_path / "ck"),
         "--out", str(out), "--events-per-bin", "2")

    results = json.loads(out.read_text())
    assert set(results) == {"meta", "gnn", "mlp", "graph"}
    assert set(results["meta"]["thresholds"]) == {"gnn", "mlp"}
    assert len(results["gnn"]) == len(evaluation.DEFAULT_BINS) - 1
    assert "thresholds chosen" in capsys.readouterr().out
