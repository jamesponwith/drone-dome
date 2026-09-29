"""Shell layout and swept-collision geometry."""

import numpy as np


def dome_slots(n: int, radius: float, min_elevation_deg: float = 8.0) -> np.ndarray:
    """Spread n points evenly over a hemispherical cap using a Fibonacci lattice.

    Sampling uniformly in z gives equal area per point on a sphere, so the
    shell has no dense poles or sparse gaps however many drones remain.
    """
    if n <= 0:
        return np.empty((0, 3))
    z_min = np.sin(np.radians(min_elevation_deg))
    i = np.arange(n)
    z = z_min + (1.0 - z_min) * (i + 0.5) / n
    r = np.sqrt(1.0 - z * z)
    theta = np.pi * (3.0 - np.sqrt(5.0)) * i
    return radius * np.column_stack([r * np.cos(theta), r * np.sin(theta), z])


def segment_min_distance(a0, a1, b0, b1) -> float:
    """Closest approach of two points moving linearly over the same interval.

    Checking only end-of-tick positions misses fast crossings (60 m/s closing
    speed covers 3 m per 50 ms tick, twice the capture radius).
    """
    r0 = a0 - b0
    dr = (a1 - a0) - (b1 - b0)
    denom = float(dr @ dr)
    t = 0.0 if denom < 1e-12 else float(np.clip(-(r0 @ dr) / denom, 0.0, 1.0))
    return float(np.linalg.norm(r0 + t * dr))


def segment_point_distance(a0, a1, p) -> float:
    return segment_min_distance(a0, a1, p, p)
