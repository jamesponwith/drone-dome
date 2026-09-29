import pytest

from aegis.cli import run_scenario


@pytest.mark.parametrize("name", ["probe", "raid", "surround", "ambush"])
def test_dome_holds(name):
    s = run_scenario(name, seed=3, drones=24).summary()
    assert s["captured"] == s["hostile"]
    assert s["leaked"] == 0
    assert s["benign_captured"] == 0


def test_saturation_spends_every_drone_before_leaking():
    s = run_scenario("saturation", seed=7, drones=24).summary()
    assert s["hostile"] > 24
    assert s["drones_remaining"] == 0
    # No drone wasted: every one that was spent took something down.
    assert s["captured"] == s["drones_spent"]
