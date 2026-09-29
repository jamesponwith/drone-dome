"""Attack scenarios: when and how inbound objects appear."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Spawn:
    t: float
    pos: np.ndarray
    vel: np.ndarray
    hostile: bool  # ground truth: is its path aimed into the dome?
    weave_amp: float = 0.0  # m/s of lateral velocity
    weave_freq: float = 0.0  # Hz


def _inbound(
    rng: np.random.Generator,
    t: float,
    speed: float,
    az: float | None = None,
    el_deg: tuple[float, float] = (4.0, 35.0),
    miss: float = 0.0,
    weave: float = 0.0,
) -> Spawn:
    az = rng.uniform(0.0, 2 * math.pi) if az is None else az
    el = math.radians(rng.uniform(*el_deg))
    r = rng.uniform(240.0, 270.0)
    start = r * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    aim = np.array([rng.normal(0.0, 1.2), rng.normal(0.0, 1.2), rng.uniform(1.5, 4.0)])
    if miss:
        lateral = np.array([-math.sin(az), math.cos(az), 0.0])
        aim = aim + lateral * miss * rng.choice([-1.0, 1.0]) + np.array([0.0, 0.0, 20.0])
    heading = (aim - start) / np.linalg.norm(aim - start)
    return Spawn(
        t=t,
        pos=start,
        vel=heading * speed,
        hostile=not miss,
        weave_amp=weave,
        weave_freq=rng.uniform(0.3, 0.6) if weave else 0.0,
    )


def _passerby(rng, t):
    return _inbound(rng, t, rng.uniform(15, 25), miss=rng.uniform(95, 150))


def probe(rng):
    """A few isolated hostiles plus some traffic that should be ignored."""
    return [
        _inbound(rng, 3.0, 22.0),
        _passerby(rng, 5.0),
        _inbound(rng, 7.0, 27.0),
        _passerby(rng, 9.0),
        _inbound(rng, 11.0, 25.0, weave=10.0),
    ]


def raid(rng):
    """Eight simultaneous hostiles from one sector."""
    axis = rng.uniform(0.0, 2 * math.pi)
    return [
        _inbound(rng, 4.0 + rng.uniform(0, 0.6), rng.uniform(22, 32), az=axis + rng.uniform(-0.6, 0.6))
        for _ in range(8)
    ]


def surround(rng):
    """Twelve hostiles from every direction at once, plus two passers-by."""
    n = 12
    spawns = [
        _inbound(rng, 4.0, rng.uniform(20, 30), az=2 * math.pi * k / n + rng.uniform(-0.15, 0.15))
        for k in range(n)
    ]
    return spawns + [_passerby(rng, 3.0), _passerby(rng, 6.0)]


def ambush(rng):
    """Fast, low, back-to-back threats down one lane: faster than the interceptors."""
    az = rng.uniform(0.0, 2 * math.pi)
    return [
        _inbound(rng, 3.0 + 0.8 * k, 40.0, az=az + rng.uniform(-0.1, 0.1), el_deg=(3.0, 6.0))
        for k in range(5)
    ]


def saturation(rng):
    """More threats than drones, mixed speeds and weaving. Shows how the dome degrades."""
    spawns = [
        _inbound(
            rng,
            rng.uniform(3.0, 15.0),
            rng.uniform(18, 40),
            weave=10.0 if rng.random() < 0.3 else 0.0,
        )
        for _ in range(36)
    ]
    return spawns + [_passerby(rng, rng.uniform(3.0, 15.0)) for _ in range(4)]


SCENARIOS: dict[str, Callable[[np.random.Generator], list[Spawn]]] = {
    "probe": probe,
    "raid": raid,
    "surround": surround,
    "ambush": ambush,
    "saturation": saturation,
}
