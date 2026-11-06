"""Training data: many pair events at random separations, as PyG graphs."""

from __future__ import annotations

import numpy as np
import torch

from calograph.generator import DetectorConfig, generate_pair_event
from calograph.graph import build_graph, to_pyg


def make_dataset(
    n_events: int,
    seed: int,
    max_sep_cm: float = 6.0,
    energy_range_gev: tuple[float, float] = (3.0, 10.0),
    k: int = 8,
    cfg: DetectorConfig | None = None,
) -> list:
    """Generate n_events two-shower events with separations uniform in [0, max_sep_cm].

    Each graph stores its separation in `separation_cm` (shape [1], so a batch
    gets one value per graph). The same seed always gives the same dataset.
    """
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(n_events):
        sep = float(rng.uniform(0.0, max_sep_cm))
        energies = tuple(float(e) for e in rng.uniform(*energy_range_gev, size=2))
        event_seed = int(rng.integers(0, 2**31 - 1))
        event = generate_pair_event(sep, energies_gev=energies, seed=event_seed, cfg=cfg)
        data = to_pyg(build_graph(event, k=k, cfg=cfg))
        data.separation_cm = torch.tensor([sep], dtype=torch.float32)
        graphs.append(data)
    return graphs
