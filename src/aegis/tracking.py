"""Multi-target tracker: constant-velocity Kalman filters with gated global association."""

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

_I3 = np.eye(3)
_H = np.hstack([_I3, np.zeros((3, 3))])
_NO_MATCH = 1e9


@dataclass
class Detection:
    pos: np.ndarray
    sigma: float


def _cv_model(dt: float, accel_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    F = np.eye(6)
    F[:3, 3:] = dt * _I3
    Q = accel_sigma**2 * np.block(
        [[dt**4 / 4 * _I3, dt**3 / 2 * _I3], [dt**3 / 2 * _I3, dt**2 * _I3]]
    )
    return F, Q


class Track:
    """Interacting Multiple Model (IMM) track.

    Runs one constant-velocity Kalman filter per motion model (e.g. cruise and
    maneuver, differing only in process noise) and blends them by how well each
    explains the measurements. Straight flyers get tight estimates from the
    cruise model; weaving ones stay in gate thanks to the maneuver model.
    """

    def __init__(
        self,
        id: int,
        x: np.ndarray,
        P: np.ndarray,
        accel_sigmas: tuple[float, ...],
        transition: np.ndarray,
    ):
        self.id = id
        self.accel_sigmas = accel_sigmas
        self.transition = transition  # transition[i, j] = P(model j next | model i now)
        n = len(accel_sigmas)
        self.xs = [x.copy() for _ in range(n)]
        self.Ps = [P.copy() for _ in range(n)]
        self.mu = np.full(n, 1.0 / n)
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.mean_vel: np.ndarray | None = None  # smoothed velocity; follows a weaver's mean path
        self._combine()

    @property
    def pos(self) -> np.ndarray:
        return self.x[:3]

    @property
    def vel(self) -> np.ndarray:
        return self.x[3:]

    @property
    def vel_sigma(self) -> float:
        """RMS 1-sigma velocity uncertainty per axis, m/s."""
        return float(np.sqrt(np.trace(self.P[3:, 3:]) / 3.0))

    @property
    def maneuvering(self) -> float:
        """Probability the target is in the highest-noise (maneuver) model."""
        return float(self.mu[-1])

    def _combine(self) -> None:
        self.x = sum(m * x for m, x in zip(self.mu, self.xs))
        self.P = sum(
            m * (P + np.outer(x - self.x, x - self.x)) for m, x, P in zip(self.mu, self.xs, self.Ps)
        )

    def smooth_velocity(self, dt: float, tau: float) -> None:
        """Low-pass the velocity estimate; started once the track is confirmed."""
        if self.mean_vel is None:
            self.mean_vel = self.vel.copy()
        else:
            self.mean_vel = self.mean_vel + (self.vel - self.mean_vel) * min(1.0, dt / tau)

    def predicted_velocity(self, horizon: float, blend_time: float = 1.0) -> np.ndarray:
        """Velocity to extrapolate with over `horizon` seconds.

        Instantaneous velocity is right for the next instant; the smoothed mean
        is right for a weaver over several seconds. Blend between them.
        """
        if self.mean_vel is None:
            return self.vel
        w = min(1.0, max(0.0, horizon / blend_time))
        return (1.0 - w) * self.vel + w * self.mean_vel

    def predict(self, dt: float) -> None:
        c = self.transition.T @ self.mu  # predicted model probabilities
        mix = self.transition * self.mu[:, None] / np.maximum(c, 1e-12)[None, :]
        xs, Ps = [], []
        for j, accel_sigma in enumerate(self.accel_sigmas):
            x0 = sum(mix[i, j] * self.xs[i] for i in range(len(self.xs)))
            P0 = sum(
                mix[i, j] * (self.Ps[i] + np.outer(self.xs[i] - x0, self.xs[i] - x0))
                for i in range(len(self.xs))
            )
            F, Q = _cv_model(dt, accel_sigma)
            xs.append(F @ x0)
            Ps.append(F @ P0 @ F.T + Q)
        self.xs, self.Ps, self.mu = xs, Ps, c
        self._combine()

    def innovation(self, det: Detection) -> tuple[np.ndarray, np.ndarray]:
        y = det.pos - _H @ self.x
        S = _H @ self.P @ _H.T + det.sigma**2 * _I3
        return y, S

    def update(self, det: Detection) -> None:
        R = det.sigma**2 * _I3
        likelihood = np.empty(len(self.xs))
        for j, (x, P) in enumerate(zip(self.xs, self.Ps)):
            y = det.pos - _H @ x
            S = _H @ P @ _H.T + R
            S_inv = np.linalg.inv(S)
            K = P @ _H.T @ S_inv
            IKH = np.eye(6) - K @ _H
            self.xs[j] = x + K @ y
            self.Ps[j] = IKH @ P @ IKH.T + K @ R @ K.T  # Joseph form
            likelihood[j] = np.exp(-0.5 * float(y @ S_inv @ y)) / np.sqrt(np.linalg.det(S))
        mu = self.mu * likelihood
        total = mu.sum()
        self.mu = mu / total if total > 1e-300 else self.mu
        self._combine()


class Tracker:
    def __init__(
        self,
        accel_sigmas: tuple[float, ...] = (4.0, 40.0),  # cruise, maneuver (m/s^2)
        switch_prob: float = 0.04,  # per-scan chance of changing motion model
        gate: float = 16.27,  # chi-square, 3 dof, 99.9%
        confirm_hits: int = 3,
        max_misses: int = 6,
        init_vel_sigma: float = 60.0,
        smoothing_tau: float = 1.0,  # s, time constant of Track.mean_vel
        smoothing_start_sigma: float = 4.0,  # m/s, don't seed mean_vel from an unsettled estimate
    ):
        n = len(accel_sigmas)
        self.accel_sigmas = accel_sigmas
        self.transition = np.full((n, n), switch_prob / max(n - 1, 1))
        np.fill_diagonal(self.transition, 1.0 - switch_prob if n > 1 else 1.0)
        self.gate = gate
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.init_vel_sigma = init_vel_sigma
        self.smoothing_tau = smoothing_tau
        self.smoothing_start_sigma = smoothing_start_sigma
        self.tracks: list[Track] = []
        self._ids = itertools.count(1)

    def step(self, detections: list[Detection], dt: float) -> list[Track]:
        for tr in self.tracks:
            tr.predict(dt)

        matched_tracks: set[int] = set()
        used_dets: set[int] = set()
        if self.tracks and detections:
            cost = np.full((len(self.tracks), len(detections)), _NO_MATCH)
            for i, tr in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    y, S = tr.innovation(det)
                    d2 = float(y @ np.linalg.solve(S, y))
                    if d2 < self.gate:
                        cost[i, j] = d2
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] >= _NO_MATCH:
                    continue
                tr = self.tracks[i]
                tr.update(detections[j])
                tr.hits += 1
                tr.misses = 0
                tr.confirmed = tr.confirmed or tr.hits >= self.confirm_hits
                matched_tracks.add(i)
                used_dets.add(j)

        for i, tr in enumerate(self.tracks):
            if i not in matched_tracks:
                tr.misses += 1
            if tr.mean_vel is not None or (tr.confirmed and tr.vel_sigma < self.smoothing_start_sigma):
                tr.smooth_velocity(dt, self.smoothing_tau)
        # Tentative tracks get one coast; confirmed ones ride out short dropouts.
        self.tracks = [
            tr for tr in self.tracks if tr.misses <= (self.max_misses if tr.confirmed else 1)
        ]

        for j, det in enumerate(detections):
            if j not in used_dets:
                P = np.diag([det.sigma**2] * 3 + [self.init_vel_sigma**2] * 3)
                x = np.concatenate([det.pos, np.zeros(3)])
                self.tracks.append(Track(next(self._ids), x, P, self.accel_sigmas, self.transition))
        return self.tracks
