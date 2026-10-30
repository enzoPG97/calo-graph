##COde for toy layered-calorimeter shower generator
""" Shower shapes follow a simplified gamma-distribution logitudinal profile and a Gaussian transverse profile ( for more details see PDG review " Passage of particles through matter", Electromagnetic cascades).
All constants are toy values and are not a representation of real materials or detectors.
"""

from __future__ import annotations
from dataclasses import dataclass, fields
import numpy as np
from scipy.spatial import cKDTree ## This class provides an index into a set of k-dimensional points which can be used to rapidly look up the nearest neighbors of any point.


@dataclass(frozen=True)
class DetectorConfig:
    """ TOy detector and shower parameters (lengths in cm, energy in GeV)."""
    X0_cm: float = 1.12 #radiation length from DAMPE calorimeter
    moliere_cm: float = 2.259 # moliere radius from DAMPE calorimeter
    critical_energy_gev: float = 10.5e-3 # Crtiical energy for eletrons in BGO see PDG BGO values for more details
    b: float = 0.5 # gamma distribution rate parameter, See PDg values betwee0.4 and 0.7 quite constant
    n_layers: int=40
    layer_thickness_X0: float=14 # layer thickness in units of X0
    hits_per_gev: float = 20.0         # hits generated per GeV of energy
    time_sigma_ns: float = 0.05
    speed_cm_per_ns: float = 30.0

    @property
    def depth_max_x0(self) -> float:
        return self.n_layers * self.layer_thickness_X0

    @property
    def transverse_sigma_cm(self) -> float:
        # 2D Gaussian with 90% of the hits inside one Moliere radius
        return self.moliere_cm / np.sqrt(2.0 * np.log(10.0))

@dataclass
class Event:
    """Hits of one event. All arrays have the same length (one entry per hit)."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    energy: np.ndarray
    time: np.ndarray
    layer: np.ndarray
    particle_id: np.ndarray  # truth: which particle made the hit

    def __post_init__(self) ->None:
        lengths = {f.name: len(getattr(self,f.name)) for f in fields(self)}
        if len(set(lengths.values())) !=1:
            raise ValueError(f"all hit arrays must have the same lenght, got {lengths}")
    @property
    def n_hits(self) -> int:
        return int(self.energy.shape[0])


def shower_alpha(energy_gev: float, cfg: DetectorConfig) -> float:
    """Shape parameter of the longitudinal gamma distribution."""
    t_max = max(np.log(energy_gev / cfg.critical_energy_gev) - 0.5, 0.0)
    return 1.0 + cfg.b * t_max


def generate_shower(
    energy_gev: float,
    x_cm: float,
    y_cm: float,
    t0_ns: float,
    particle_id: int,
    cfg: DetectorConfig,
    rng: np.random.Generator,
) -> Event:
    """Generate the hits of one electron shower along +z."""
    if energy_gev <= 0:
        raise ValueError("energy_gev must be positive")

    n_total = max(1, int(round(cfg.hits_per_gev * energy_gev)))
    alpha = shower_alpha(energy_gev, cfg)

    depth_x0 = rng.gamma(shape=alpha, scale=1.0 / cfg.b, size=n_total)
    depth_x0 = depth_x0[depth_x0 < cfg.depth_max_x0]  # hits past the last layer leak out
    n = depth_x0.shape[0]

    layer = np.floor(depth_x0 / cfg.layer_thickness_X0).astype(np.int64)
    z = (layer + 0.5) * cfg.layer_thickness_X0 * cfg.X0_cm

    sigma = cfg.transverse_sigma_cm
    x = x_cm + rng.normal(0.0, sigma, size=n)
    y = y_cm + rng.normal(0.0, sigma, size=n)

    time = t0_ns + z / cfg.speed_cm_per_ns + rng.normal(0.0, cfg.time_sigma_ns, size=n)
    # Equal energy per generated hit; leaked hits are lost, so the sum can be < E
    energy = np.full(n, energy_gev / n_total)

    return Event(
        x=x, y=y, z=z, energy=energy, time=time, layer=layer,
        particle_id=np.full(n, particle_id, dtype=np.int64),
    )


def generate_event(
    n_particles: int = 2,
    seed: int | None = None,
    cfg: DetectorConfig | None = None,
    energy_range_gev: tuple[float, float] = (5.0, 50.0),
    window_cm: float = 20.0,
) -> Event:
    """Generate an event with several showers at random positions.

    Particle IDs are 0 .. n_particles-1. The same seed gives the same event.
    """
    cfg = cfg or DetectorConfig()
    rng = np.random.default_rng(seed)

    showers = []
    for pid in range(n_particles):
        energy = rng.uniform(*energy_range_gev)
        x_cm, y_cm = rng.uniform(-window_cm / 2, window_cm / 2, size=2)
        showers.append(generate_shower(energy, x_cm, y_cm, 0.0, pid, cfg, rng))
    
    #replace merged = {...} with a function that concatenate the hits of several shower into one event.
    """
    merged = {
        f.name: np.concatenate([getattr(s, f.name) for s in showers])
        for f in fields(Event)
    }
    return Event(**merged)
    """
    return merge_showers(showers)
    
def merge_showers(showers: list[Event]) -> Event:
    """ Concatenate the hits of several showers into one event."""

    return Event(
            **{f.name: np.concatenate([getattr(s, f.name) for s in showers])
           for f in fields(Event)}
    )

def generate_pair_event(
    separation_cm: float,
    energies_gev: tuple[float, float] = (20.0, 20.0),
    seed: int | None = None,
    cfg: DetectorConfig | None = None,
    angle_rad: float | None = None,
) -> Event:
    """Two showers whose axes are separation_cm apart (truth IDs 0 and 1).

    The direction of the offset in the x-y plane is random unless angle_rad
    is given. The pair is centred on the origin.
    """
    if separation_cm < 0:
        raise ValueError("separation_cm must be >= 0")
    cfg = cfg or DetectorConfig()
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.0, 2.0 * np.pi) if angle_rad is None else angle_rad
    dx = 0.5 * separation_cm * np.cos(angle)
    dy = 0.5 * separation_cm * np.sin(angle)
    return merge_showers([
        generate_shower(energies_gev[0], -dx, -dy, 0.0, 0, cfg, rng),
        generate_shower(energies_gev[1], +dx, +dy, 0.0, 1, cfg, rng),
    ])


def truth_summary(event: Event) -> dict[int, dict[str, float]]:
    """Number of hits and deposited energy for each truth particle."""
    out = {}
    for pid in np.unique(event.particle_id):
        mask = event.particle_id == pid
        out[int(pid)] = {
            "n_hits": int(mask.sum()),
            "energy_gev": float(event.energy[mask].sum()),
        }
    return out


def mixing_fraction(event: Event) -> float:
    """Fraction of hits whose nearest other hit belongs to a different particle.

    0 means the showers are fully separated; about 0.5 means fully overlapping
    showers of similar size.
    """
    if event.n_hits < 2:
        return 0.0
    points = np.column_stack([event.x, event.y, event.z])
    _, idx = cKDTree(points).query(points, k=2)  # column 0 is the hit itself
    neighbour = idx[:, 1]
    return float(np.mean(event.particle_id[neighbour] != event.particle_id))
