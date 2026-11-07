"""Plot results/evaluation.json (written by python -m calograph.evaluation)."""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

source = sys.argv[1] if len(sys.argv) > 1 else "results/evaluation.json"
target = sys.argv[2] if len(sys.argv) > 2 else "docs/results.png"
results = json.load(open(source))
thresholds = results["meta"]["thresholds"]

style = {"gnn": ("tab:red", "o", "GNN"), "mlp": ("tab:blue", "s", "MLP baseline"),
         "graph": ("gray", "^", "keep all edges")}


def centres(name):
    return [(r["sep_lo"] + r["sep_hi"]) / 2 for r in results[name]]


fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
panels = [("auc", "Edge AUC", ("gnn", "mlp")),
          ("efficiency", "Cluster efficiency", ("gnn", "mlp", "graph")),
          ("purity", "Cluster purity", ("gnn", "mlp", "graph")),
          ("merge_rate", "Fraction of events with merged showers", ("gnn", "mlp", "graph"))]
for ax, (key, title, names) in zip(axes.flat, panels):
    for name in names:
        colour, marker, label = style[name]
        ax.plot(centres(name), [r[key] for r in results[name]], marker=marker,
                color=colour, label=label)
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel("shower separation [cm]")
axes[0, 0].legend()
fig.suptitle(
    f"Test events, {results['meta']['events_per_bin']} per bin; thresholds (logit) "
    f"GNN {thresholds['gnn']:g}, MLP {thresholds['mlp']:g}", fontsize=10)
fig.tight_layout()
fig.savefig(target, dpi=150, facecolor="white")
