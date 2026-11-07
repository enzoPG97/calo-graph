"""Evaluate edge classifiers as a function of shower separation.

Usage:
    python -m calograph.evaluation --checkpoints checkpoints --out results/evaluation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from calograph.clustering import cluster_hits, cluster_metrics
from calograph.dataset import make_dataset
from calograph.metrics import edge_auc
from calograph.train import load_model

DEFAULT_BINS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0,8.0)
THRESHOLD_GRID = tuple(float(t) for t in np.arange(0.0, 8.5, 0.5))


def select_threshold(
    score_fn, graphs, min_purity: float = 0.95, thresholds=THRESHOLD_GRID
) -> tuple[float, dict[float, tuple[float, float]]]:
    """Choose the working point: the loosest logit threshold with mean purity >= min_purity.

    Use validation graphs, never the test events. Connected components are
    fragile (one surviving cross-particle edge merges two showers), so the
    threshold has to be far stricter than logit 0 (probability 0.5). Fixing a
    purity target and comparing efficiency is a fair way to compare models;
    averaging purity and efficiency is not, because one giant cluster scores
    well on it. If no threshold on the grid reaches the target, the strictest
    one is returned. Also returns {threshold: (mean purity, mean efficiency)}.
    """
    with torch.no_grad():
        scores = [score_fn(d).numpy() for d in graphs]
    table = {}
    for t in thresholds:
        metrics = [
            cluster_metrics(d.particle_id.numpy(),
                            cluster_hits(d.edge_index.numpy(), sc, d.num_nodes, t))
            for d, sc in zip(graphs, scores)
        ]
        table[t] = (float(np.mean([m["purity"] for m in metrics])),
                    float(np.mean([m["efficiency"] for m in metrics])))
    passing = [t for t in thresholds if table[t][0] >= min_purity]
    return (min(passing) if passing else max(thresholds)), table

def evaluate_by_separation(
    score_fn,
    bin_edges=DEFAULT_BINS,
    events_per_bin: int = 40,
    seed: int = 2026,
    k: int = 8,
    threshold: float = 0.0,
    energy_range_gev: tuple[float, float] = (3.0, 10.0),
) -> list[dict[str, float]]:
    """Edge AUC and cluster metrics in bins of separation, on freshly generated events.

    score_fn(data) must return one logit per edge. Each bin gets its own
    reproducible events, with separation uniform inside the bin.
    """
    rows = []
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        graphs = make_dataset(events_per_bin, seed + i, max_sep_cm=hi, min_sep_cm=lo,
                              energy_range_gev=energy_range_gev, k=k)
        all_labels, all_scores, per_event = [], [], []
        for data in graphs:
            with torch.no_grad():
                scores = score_fn(data).numpy()
            all_labels.append(data.edge_label.numpy())
            all_scores.append(scores)
            predicted = cluster_hits(data.edge_index.numpy(), scores, data.num_nodes, threshold)
            per_event.append(cluster_metrics(data.particle_id.numpy(), predicted))
        rows.append({
            "sep_lo": lo, "sep_hi": hi, "n_events": len(graphs),
            "auc": edge_auc(np.concatenate(all_labels), np.concatenate(all_scores)),
            "purity": float(np.mean([m["purity"] for m in per_event])),
            "efficiency": float(np.mean([m["efficiency"] for m in per_event])),
            "merge_rate": float(np.mean([m["merged"] for m in per_event])),
            "split_rate": float(np.mean([m["split"] for m in per_event])),
            "mean_clusters": float(np.mean([m["n_clusters"] for m in per_event])),
        })
    return rows

def keep_all_edges(data) -> torch.Tensor:
    """Reference 'model' that keeps every edge: clusters are the connected pieces of the kNN graph."""
    return torch.ones(data.edge_index.shape[1])

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--out", default="results/evaluation.json")
    parser.add_argument("--events-per-bin", type=int, default=40)
    parser.add_argument("--test-seed", type=int, default=2026)
    args = parser.parse_args()

    results = {"meta": {"bins": list(DEFAULT_BINS), "events_per_bin": args.events_per_bin,
                        "test_seed": args.test_seed, "thresholds": {}}}
    for name in ("gnn", "mlp"):
        model, cfg = load_model(Path(args.checkpoints) / f"{name}_best.pt")
        # Same validation events the model was early-stopped on (see train.py).
        val_seed = int(np.random.SeedSequence(cfg.seed).generate_state(2)[1])
        val_graphs = make_dataset(cfg.n_val, val_seed, cfg.max_sep_cm, k=cfg.k)
        threshold, _ = select_threshold(model, val_graphs)
        results["meta"]["thresholds"][name] = threshold
        results[name] = evaluate_by_separation(
            model, events_per_bin=args.events_per_bin, seed=args.test_seed,
            k=cfg.k, threshold=threshold)
    results["graph"] = evaluate_by_separation(
        keep_all_edges, events_per_bin=args.events_per_bin, seed=args.test_seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("thresholds chosen on validation events:", results["meta"]["thresholds"])
    header = (f"{'sep [cm]':>9} | {'model':>5} | {'AUC':>5} {'purity':>7} {'effic.':>7} "
              f"{'merge':>6} {'split':>6} {'clusters':>9}")
    print(header)
    print("-" * len(header))
    for i, (lo, hi) in enumerate(zip(DEFAULT_BINS[:-1], DEFAULT_BINS[1:])):
        for name in ("gnn", "mlp", "graph"):
            r = results[name][i]
            auc = "  -  " if name == "graph" else f"{r['auc']:.3f}"
            print(f"{lo:>4g}-{hi:<4g} | {name:>5} | {auc:>5} {r['purity']:>7.3f} "
                  f"{r['efficiency']:>7.3f} {r['merge_rate']:>6.2f} {r['split_rate']:>6.2f} "
                  f"{r['mean_clusters']:>9.2f}")


if __name__ == "__main__":
    main()


