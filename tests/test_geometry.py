import numpy as np
import pytest

from aegis.geometry import dome_slots, segment_min_distance, segment_point_distance


@pytest.mark.parametrize("n", [1, 5, 24, 60])
def test_slots_lie_on_the_upper_shell(n):
    slots = dome_slots(n, 40.0, min_elevation_deg=8.0)
    assert slots.shape == (n, 3)
    np.testing.assert_allclose(np.linalg.norm(slots, axis=1), 40.0)
    assert (slots[:, 2] >= 40.0 * np.sin(np.radians(8.0)) - 1e-9).all()


def test_slots_are_evenly_spread():
    slots = dome_slots(24, 40.0)
    dist = np.linalg.norm(slots[:, None] - slots[None], axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = dist.min(axis=1)
    # Equal-area cells: no drone crowds another, no big gaps.
    assert nearest.min() > 0.6 * nearest.mean()
    assert nearest.max() < 1.4 * nearest.mean()


def test_swept_distance_catches_crossings_between_ticks():
    # Head-on pass: endpoints are 6 m apart before and after, but they meet mid-tick.
    a0, a1 = np.array([0.0, 0, 0]), np.array([3.0, 0, 0])
    b0, b1 = np.array([3.0, 0.5, 0]), np.array([0.0, 0.5, 0])
    assert segment_min_distance(a0, a1, b0, b1) == pytest.approx(0.5)


def test_point_distance():
    assert segment_point_distance(np.array([-5.0, 2, 0]), np.array([5.0, 2, 0]), np.zeros(3)) == pytest.approx(2.0)
