"""Tunable parameters for the dome, its sensor, and the simulation clock."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DomeConfig:
    n_drones: int = 24
    dome_radius: float = 40.0  # m, radius of the drone shell around the asset
    core_radius: float = 6.0  # m, the protected asset; anything reaching it is a leak
    min_elevation_deg: float = 8.0  # lowest ring of the shell
    drone_max_speed: float = 30.0  # m/s
    drone_max_accel: float = 45.0  # m/s^2
    capture_radius: float = 1.5  # m, net/contact capture envelope
    reserve: int = 4  # drones that stay on the shell unless a threat is critical
    engage_range: float = 180.0  # m, max distance from the asset for an intercept point
    salvo_breach_time: float = 4.0  # s, threats breaching sooner than this get two interceptors
    hostile_margin: float = 5.0  # m, CPA slack added to the dome radius when classifying
    classify_vel_sigma: float = 4.0  # m/s, velocity must be this well known before classifying


@dataclass(frozen=True)
class SensorConfig:
    max_range: float = 320.0  # m
    p_detect: float = 0.97  # per-scan detection probability
    base_sigma: float = 0.3  # m, position noise floor
    sigma_per_m: float = 0.004  # m of noise added per m of range


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.05  # s, also the radar scan period
    duration: float = 45.0  # s, hard stop
    record_every: int = 2  # ticks between replay frames
    seed: int = 7
