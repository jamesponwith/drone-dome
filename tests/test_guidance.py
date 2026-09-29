import numpy as np
import pytest

from aegis.guidance import intercept_time, steer


def test_head_on_intercept():
    # 100 m apart, closing at 20 + 30 m/s.
    t = intercept_time(np.zeros(3), 30.0, np.array([100.0, 0, 0]), np.array([-20.0, 0, 0]))
    assert t == pytest.approx(2.0)


def test_crossing_intercept_point_is_reachable():
    p, speed = np.zeros(3), 30.0
    q, v = np.array([80.0, -60.0, 20.0]), np.array([0.0, 25.0, 0.0])
    t = intercept_time(p, speed, q, v)
    assert t is not None
    assert np.linalg.norm(q + v * t - p) == pytest.approx(speed * t)


def test_cannot_catch_faster_receding_target():
    assert intercept_time(np.zeros(3), 30.0, np.array([50.0, 0, 0]), np.array([40.0, 0, 0])) is None


def test_steer_respects_limits_and_settles_on_slot():
    pos, vel, aim = np.zeros(3), np.zeros(3), np.array([30.0, 10.0, 20.0])
    for _ in range(200):
        new = steer(pos, vel, aim, 30.0, 45.0, 0.05, arrive=True)
        assert np.linalg.norm(new) <= 30.0 + 1e-9
        assert np.linalg.norm(new - vel) <= 45.0 * 0.05 + 1e-9
        vel = new
        pos = pos + vel * 0.05
    assert np.linalg.norm(pos - aim) < 0.1
    assert np.linalg.norm(vel) < 0.5
