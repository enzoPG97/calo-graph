## test coe ton check things work
import numpy as np
import pytest

from calograph import DetectorConfig, generate_event, generate_shower

def test_same_seed_is_possible():
    a = generate_event(n_particles=3, seed=42)
    b= generate_event(n_particles=3, seed=42)
    for name in ("x", "y", "z", "energy", "time", "particle_id"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))

def test_different_seed_differs():
    a = generate_event(seed=1)
    b = generate_event(seed=2)
    assert a.n_hits != b.n_hits or not np.allclose(a.x, b.x[: a.n_hits])


def test_arrays_have_same_length_and_valid_layers():
    cfg = DetectorConfig()
    ev = generate_event(n_particles=3, seed=0, cfg=cfg)
    n = ev.n_hits
    assert n > 0
    for name in ("x", "y", "z", "energy", "time", "layer", "particle_id"):
        assert getattr(ev, name).shape == (n,)
    assert ev.layer.min() >= 0
    assert ev.layer.max() < cfg.n_layers


def test_particle_ids_are_the_expected_set():
    ev = generate_event(n_particles=4, seed=3)
    assert set(np.unique(ev.particle_id)) == {0, 1, 2, 3}


def test_energy_never_exceeds_incident_energy():
    cfg = DetectorConfig()
    rng = np.random.default_rng(0)
    sh = generate_shower(10.0, 0.0, 0.0, 0.0, 0, cfg, rng)
    assert sh.energy.sum() <= 10.0 + 1e-9
    assert sh.energy.sum() > 0.8 * 10.0  # most of a 10 GeV shower is contained


def test_mean_depth_matches_gamma_mean():
    # Deep detector so truncation does not bias the mean.
    cfg = DetectorConfig(n_layers=200, hits_per_gev=100.0)
    rng = np.random.default_rng(1)
    sh = generate_shower(20.0, 0.0, 0.0, 0.0, 0, cfg, rng)
    depth_x0 = (sh.layer + 0.5) * cfg.layer_thickness_X0
    alpha = 1.0 + cfg.b * (np.log(20.0 / cfg.critical_energy_gev) - 0.5)
    assert depth_x0.mean() == pytest.approx(alpha / cfg.b, abs=0.5)


def test_ninety_percent_of_hits_within_moliere_radius():
    cfg = DetectorConfig(hits_per_gev=100.0)
    rng = np.random.default_rng(2)
    sh = generate_shower(50.0, 0.0, 0.0, 0.0, 0, cfg, rng)
    r = np.hypot(sh.x, sh.y)
    assert (r < cfg.moliere_cm).mean() == pytest.approx(0.90, abs=0.03)


def test_hit_time_increases_with_depth():
    cfg = DetectorConfig(time_sigma_ns=0.0)
    rng = np.random.default_rng(3)
    sh = generate_shower(10.0, 0.0, 0.0, 0.0, 0, cfg, rng)
    order = np.argsort(sh.z)
    assert np.all(np.diff(sh.time[order]) >= 0)


def test_non_positive_energy_raises():
    with pytest.raises(ValueError):
        generate_shower(0.0, 0.0, 0.0, 0.0, 0, DetectorConfig(), np.random.default_rng(0))
