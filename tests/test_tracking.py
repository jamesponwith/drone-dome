import numpy as np

from aegis.config import DomeConfig
from aegis.sim import Threat
from aegis.threat import assess
from aegis.tracking import Detection, Tracker

DT = 0.05


def _fly(threat, n, rng, tracker=None, p_detect=1.0):
    tracker = tracker or Tracker()
    ids = set()
    t = 0.0
    for _ in range(n):
        t += DT
        threat.pos = threat.pos + threat.velocity(t) * DT
        dets = []
        if rng.random() < p_detect:
            sigma = 0.3 + 0.004 * np.linalg.norm(threat.pos)
            dets = [Detection(threat.pos + rng.normal(0, sigma, 3), sigma)]
        tracks = tracker.step(dets, DT)
        ids |= {tr.id for tr in tracks if tr.confirmed}
    return tracker, ids, t


def test_tracker_converges_on_straight_flyer():
    rng = np.random.default_rng(0)
    th = Threat(0, np.array([250.0, 0, 40]), np.array([-25.0, 3, -4]), 0.0, True)
    tracker, ids, _ = _fly(th, 40, rng)
    (tr,) = tracker.tracks
    assert len(ids) == 1
    assert np.linalg.norm(tr.pos - th.pos) < 3.0
    assert np.linalg.norm(tr.vel - th.base_vel) < 3.0


def test_imm_holds_a_weaving_target_better_than_single_model():
    def breaks(make_tracker):
        total = 0
        for seed in range(5):
            rng = np.random.default_rng(seed)
            th = Threat(0, np.array([250.0, 0, 40]), np.array([-25.0, 0, -4]), 0.0, True, 10.0, 0.5)
            _, ids, _ = _fly(th, 180, rng, tracker=make_tracker(), p_detect=0.97)
            total += len(ids) - 1
        return total

    imm = breaks(Tracker)
    cruise_only = breaks(lambda: Tracker(accel_sigmas=(4.0,)))
    assert imm <= 2
    assert cruise_only > 3 * max(imm, 1)


def test_clutter_free_tracks_do_not_merge():
    rng = np.random.default_rng(1)
    tracker = Tracker()
    a = Threat(0, np.array([200.0, 20, 30]), np.array([-20.0, 0, 0]), 0.0, True)
    b = Threat(1, np.array([200.0, -20, 30]), np.array([-20.0, 0, 0]), 0.0, True)
    for _ in range(40):
        a.pos = a.pos + a.base_vel * DT
        b.pos = b.pos + b.base_vel * DT
        dets = [Detection(p + rng.normal(0, 1.0, 3), 1.0) for p in (a.pos, b.pos)]
        tracker.step(dets, DT)
    assert len([tr for tr in tracker.tracks if tr.confirmed]) == 2


def test_assessment_separates_inbound_from_passing():
    cfg = DomeConfig()
    rng = np.random.default_rng(2)
    inbound = Threat(0, np.array([250.0, 0, 40]), np.array([-25.0, 0, -4]), 0.0, True)
    passing = Threat(1, np.array([250.0, 0, 40]), np.array([-25.0, 12, 0]), 0.0, False)
    for th, expect in [(inbound, True), (passing, False)]:
        tracker, _, _ = _fly(th, 30, rng)
        a = assess(tracker.tracks[0], cfg.dome_radius, cfg.core_radius, cfg.hostile_margin)
        assert a.hostile is expect
        if expect:
            assert 0 < a.time_to_breach < np.inf
            assert a.cpa_distance < cfg.dome_radius
