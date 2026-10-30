import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from calograph import generate_pair_event
from calograph.graph import build_graph

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
for ax, sep in zip(axes, [0, 2.0, 4.0, 8.0]):
    ev = generate_pair_event(sep, seed=3, angle_rad=0.0)
    g = build_graph(ev, k=6)
    src, dst = g.edge_index
    segments = [((g.pos[i, 2], g.pos[i, 0]), (g.pos[j, 2], g.pos[j, 0]))
                for i, j in zip(src, dst) if i < j]
    labels = [g.edge_label[e] for e, (i, j) in enumerate(zip(src, dst)) if i < j]
    colors = ["lightgray" if lab == 1 else "red" for lab in labels]
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=0.4))
    ax.scatter(g.pos[:, 2], g.pos[:, 0], c=g.particle_id, s=4, cmap="coolwarm", zorder=3)
    ax.set_title(f"sep = {sep} cm, {100 * (1 - g.edge_label.mean()):.0f}% cross edges")
    ax.set_xlabel("z [cm]")
    ax.autoscale()
axes[0].set_ylabel("x [cm]")
fig.tight_layout()
fig.savefig("docs/graph_example.png", dpi=150, facecolor="white")
