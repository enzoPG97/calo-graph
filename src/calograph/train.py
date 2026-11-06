"""Training loop for the edge classifiers (GNN and MLP baseline).

Usage:
    python -m calograph.train --config configs/default.json
    python -m calograph.train --config configs/default.json --model mlp
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from calograph.dataset import make_dataset
from calograph.metrics import edge_auc
from calograph.model import EdgeClassifier, EdgeMLP


@dataclass
class TrainConfig:
    model: str = "gnn"          # "gnn" or "mlp"
    n_train: int = 300
    n_val: int = 100
    max_sep_cm: float = 6.0
    k: int = 8
    hidden: int = 32            # GNN width
    n_layers: int = 3           # GNN message-passing rounds
    mlp_hidden: int = 96        # baseline width (about the same number of parameters)
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-3
    seed: int = 0
    out_dir: str = "checkpoints"

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        """Load a config; unknown keys raise a TypeError."""
        return cls(**json.loads(Path(path).read_text()))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg: TrainConfig) -> torch.nn.Module:
    if cfg.model == "gnn":
        return EdgeClassifier(hidden=cfg.hidden, n_layers=cfg.n_layers)
    if cfg.model == "mlp":
        return EdgeMLP(hidden=cfg.mlp_hidden)
    raise ValueError(f"unknown model {cfg.model!r}, expected 'gnn' or 'mlp'")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader) -> dict[str, float]:
    """Loss, accuracy and AUC over all edges of a data loader."""
    model.eval()
    logits, labels = [], []
    for batch in loader:
        logits.append(model(batch))
        labels.append(batch.edge_label)
    logits, labels = torch.cat(logits), torch.cat(labels)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    accuracy = ((logits > 0).float() == labels).float().mean().item()
    auc = edge_auc(labels.numpy(), logits.numpy())
    return {"loss": loss, "accuracy": accuracy, "auc": auc}


def train(cfg: TrainConfig, verbose: bool = True) -> list[dict[str, float]]:
    """Train a model, save the best checkpoint (by validation loss), return the history."""
    set_seed(cfg.seed)
    train_seed, val_seed = (int(s) for s in np.random.SeedSequence(cfg.seed).generate_state(2))
    train_data = make_dataset(cfg.n_train, train_seed, cfg.max_sep_cm, k=cfg.k)
    val_data = make_dataset(cfg.n_val, val_seed, cfg.max_sep_cm, k=cfg.k)

    shuffle_rng = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
                              generator=shuffle_rng)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size)

    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model={cfg.model}  parameters={n_params}  "
              f"train events={cfg.n_train}  val events={cfg.n_val}")

    history, best_loss = [], float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.edge_label)
            loss.backward()
            optimizer.step()
            total += loss.item() * logits.numel()
            count += logits.numel()
        val = evaluate(model, val_loader)
        row = {"epoch": epoch, "train_loss": total / count,
               "val_loss": val["loss"], "val_accuracy": val["accuracy"], "val_auc": val["auc"]}
        history.append(row)
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            torch.save({"model_state": model.state_dict(), "config": asdict(cfg),
                        "epoch": epoch, "val": val}, out_dir / f"{cfg.model}_best.pt")
        if verbose:
            print(f"epoch {epoch:3d}  train loss {row['train_loss']:.4f}  "
                  f"val loss {row['val_loss']:.4f}  acc {row['val_accuracy']:.3f}  "
                  f"auc {row['val_auc']:.3f}")

    (out_dir / f"{cfg.model}_history.json").write_text(json.dumps(history, indent=2))
    return history


def load_model(path: str | Path) -> tuple[torch.nn.Module, TrainConfig]:
    """Load a checkpoint written by train(). Returns the model (eval mode) and its config."""
    checkpoint = torch.load(path)
    cfg = TrainConfig(**checkpoint["config"])
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="JSON config file")
    parser.add_argument("--model", choices=["gnn", "mlp"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = TrainConfig.from_json(args.config) if args.config else TrainConfig()
    overrides = {"model": args.model, "epochs": args.epochs, "seed": args.seed,
                 "out_dir": args.out_dir}
    cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    train(cfg)


if __name__ == "__main__":
    main()
