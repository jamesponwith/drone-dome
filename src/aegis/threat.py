"""Threat assessment: will this track hit the dome, and how soon?"""

import math
from dataclasses import dataclass

import numpy as np

from aegis.tracking import Track


@dataclass(frozen=True)
class Assessment:
    track_id: int
    hostile: bool
    speed: float
    cpa_distance: float  # closest predicted approach to the asset
    time_to_cpa: float
    time_to_breach: float  # until it crosses the dome shell; 0 if inside, inf if never
    time_to_core: float  # until it reaches the protected core; inf if never
    priority: float

    @property
    def deadline(self) -> float:
        """Latest time an intercept is still useful."""
        return self.time_to_core if math.isfinite(self.time_to_core) else self.time_to_cpa + 1.0


def time_to_sphere(p: np.ndarray, v: np.ndarray, radius: float) -> float:
    """Time until straight-line motion from p enters the origin-centered sphere."""
    c = float(p @ p) - radius**2
    if c <= 0.0:
        return 0.0
    a = float(v @ v)
    b = 2.0 * float(p @ v)
    disc = b * b - 4.0 * a * c
    if a < 1e-9 or disc < 0.0:
        return math.inf
    t = (-b - math.sqrt(disc)) / (2.0 * a)
    return t if t >= 0.0 else math.inf


def _cpa(p: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """(time, distance) of closest approach to the origin along straight-line motion."""
    speed2 = float(v @ v)
    t = -float(p @ v) / speed2 if speed2 > 1e-6 else 0.0
    return t, float(np.linalg.norm(p + v * max(t, 0.0)))


def assess(
    track: Track,
    dome_radius: float,
    core_radius: float,
    margin: float,
    was_hostile: bool = False,
) -> Assessment:
    """Classify a track and time its arrival.

    Both the instantaneous and the smoothed velocity are projected forward and
    the more dangerous prediction wins, so a target weaving toward the asset
    is judged by its mean path instead of its current swerve. Once hostile, a
    track keeps that status until it is clearly headed away (CPA beyond twice
    the dome radius or already past), which stops fire control from flapping.
    """
    p = track.pos
    candidates = [track.vel] + ([track.mean_vel] if track.mean_vel is not None else [])
    t_cpa, cpa, v = min(((*_cpa(p, v), v) for v in candidates), key=lambda c: c[1])
    limit = 2.0 * dome_radius if was_hostile else dome_radius + margin
    hostile = t_cpa > 0.0 and cpa < limit

    t_breach = time_to_sphere(p, v, dome_radius)
    t_core = time_to_sphere(p, v, core_radius)
    urgency = t_core if math.isfinite(t_core) else t_breach + 5.0
    return Assessment(
        track_id=track.id,
        hostile=hostile,
        speed=float(np.linalg.norm(v)),
        cpa_distance=cpa,
        time_to_cpa=t_cpa,
        time_to_breach=t_breach,
        time_to_core=t_core,
        priority=1.0 / max(urgency, 0.1) if hostile else 0.0,
    )
