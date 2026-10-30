import matplotlib.pyplot as plt

from calograph import generate_pair_event, mixing_fraction

separations = [0.0, 2.0, 4.0, 8.0]
fig, axes = plt.subplots(1, len(separations), figsize=(14, 3.5), sharey=True)
for ax, sep in zip(axes, separations):
    ev = generate_pair_event(sep, seed=3, angle_rad=0.0)
    ax.scatter(ev.z, ev.x, c=ev.particle_id, s=5, cmap="coolwarm")
    ax.set_title(f"sep = {sep} cm\nmixing = {mixing_fraction(ev):.2f}")
    ax.set_xlabel("z [cm]")
axes[0].set_ylabel("x [cm]")
fig.tight_layout()
fig.savefig("docs/separation_example.png", dpi=150)
