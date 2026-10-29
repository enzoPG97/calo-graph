import matplotlib.pyplot as plt

from calograph import generate_event

ev = generate_event(n_particles=2, seed=7)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(ev.z, ev.x, c=ev.particle_id, s=6, cmap="coolwarm")
ax[0].set_xlabel("z [cm]"); ax[0].set_ylabel("x [cm]"); ax[0].set_title("Side view")
ax[1].scatter(ev.x, ev.y, c=ev.particle_id, s=6, cmap="coolwarm")
ax[1].set_xlabel("x [cm]"); ax[1].set_ylabel("y [cm]"); ax[1].set_title("Front view")
fig.tight_layout()
fig.savefig("event.png", dpi=150)
