import numpy as np
import pytest

from calograph import (Event, generate_pair_event, mixing_fraction,
                       truth_summary)


def test_event_rejects_inconsistent_lengths():
    a = np.zeros(3)
    with pytest.raises(ValueError):
        Event(x=a, y=a, z=a, energy=a, time=a,
              layer=a.astype(int), particle_id=np.zeros(2, dtype=int))


def test_pair_event_has_two_truth_ids_and_is_reproducible():
    a = generate_pair_event(4.0, seed=5)
    b = generate_pair_event(4.0, seed=5)
    assert set(np.unique(a.particle_id)) == {0, 1}
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.particle_id, b.particle_id)


def test_negative_separation_raises():
    with pytest.raises(ValueError):
        generate_pair_event(-1.0)


def test_shower_axes_are_separated_by_requested_distance():
    sep = 10.0
    ev = generate_pair_event(sep, seed=1, angle_rad=0.0)
    x0 = ev.x[ev.particle_id == 0].mean()
    x1 = ev.x[ev.particle_id == 1].mean()
    assert x1 - x0 == pytest.approx(sep, abs=0.3)
    assert ev.y.mean() == pytest.approx(0.0, abs=0.3)


def test_truth_energy_bookkeeping():
    energies = (20.0, 35.0)
    ev = generate_pair_event(6.0, energies_gev=energies, seed=2)
    summary = truth_summary(ev)
    assert set(summary) == {0, 1}
    for pid, incident in enumerate(energies):
        deposited = summary[pid]["energy_gev"]
        assert deposited <= incident + 1e-9   # nothing is created from nothing
        assert deposited > 0.8 * incident     # most of the shower is contained
    assert sum(s["n_hits"] for s in summary.values()) == ev.n_hits


def _mean_mixing(sep, seeds=range(5)):
    return float(np.mean(
        [mixing_fraction(generate_pair_event(sep, seed=s)) for s in seeds]
    ))


def test_mixing_fraction_decreases_with_separation():
    m0, m2, m15 = _mean_mixing(0.0), _mean_mixing(2.0), _mean_mixing(15.0)
    assert m0 > 0.3
    assert m0 > m2 > m15
    assert m15 < 0.02


def test_mixing_fraction_single_particle_is_zero():
    ev = generate_pair_event(5.0, seed=0)
    names = ("x", "y", "z", "energy", "time", "layer", "particle_id")
    single = Event(**{k: getattr(ev, k)[ev.particle_id == 0] for k in names})
    assert mixing_fraction(single) == 0.0
