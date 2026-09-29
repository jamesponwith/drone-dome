"""Intercept geometry and a speed/acceleration-limited steering law."""

import math

import numpy as np


def intercept_time(p: np.ndarray, speed: float, q: np.ndarray, v: np.ndarray) -> float | None:
    """Earliest t > 0 at which a pursuer at p flying `speed` can meet a target at q moving at v.

    Solves |q + v t - p| = speed * t. Returns None when the target outruns the pursuer.
    """
    d = q - p
    c = float(d @ d)
    if c < 1e-9:
        return 0.0
    a = float(v @ v) - speed**2
    b = 2.0 * float(d @ v)
    if abs(a) < 1e-9:
        return -c / b if b < 0.0 else None
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    roots = [t for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)) if t > 0.0]
    return min(roots) if roots else None


def lead_intercept(p: np.ndarray, speed: float, track) -> tuple[float, np.ndarray] | None:
    """(time, point) to meet a track, extrapolating with the horizon-appropriate velocity.

    The velocity used depends on the answer, so iterate a few times to a fixed point.
    """
    v = track.vel
    t = None
    for _ in range(3):
        t = intercept_time(p, speed, track.pos, v)
        if t is None:
            return None
        v = track.predicted_velocity(t)
    return t, track.pos + v * t


def steer(
    pos: np.ndarray,
    vel: np.ndarray,
    aim: np.ndarray,
    max_speed: float,
    max_accel: float,
    dt: float,
    arrive: bool,
) -> np.ndarray:
    """New velocity heading for `aim`.

    arrive=True brakes to a stop on the aim point (station keeping);
    arrive=False flies through it at full speed (intercept).
    """
    to = aim - pos
    dist = float(np.linalg.norm(to))
    if dist < 1e-6:
        v_des = np.zeros(3)
    else:
        speed = max_speed
        if arrive:
            speed = min(speed, math.sqrt(1.2 * max_accel * dist), dist / (2 * dt))
        v_des = to / dist * speed

    dv = v_des - vel
    dv_norm = float(np.linalg.norm(dv))
    limit = max_accel * dt
    if dv_norm > limit:
        dv *= limit / dv_norm
    new_vel = vel + dv
    sp = float(np.linalg.norm(new_vel))
    return new_vel * (max_speed / sp) if sp > max_speed else new_vel
