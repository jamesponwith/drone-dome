"""World simulation: truth motion, sensing, tracking, command, and outcome bookkeeping."""

import math
from dataclasses import dataclass, field

import numpy as np

from aegis.command import Commander, Drone, DroneState
from aegis.config import DomeConfig, SensorConfig, SimConfig
from aegis.geometry import segment_min_distance, segment_point_distance
from aegis.guidance import lead_intercept, steer
from aegis.scenarios import Spawn
from aegis.sensor import Radar
from aegis.tracking import Tracker

_PAD_RADIUS = 3.0  # m, drones launch from a ring around the asset
_SPENT_SINK_RATE = 4.0  # m/s, a spent drone descends with its catch
_MIN_ALTITUDE = 0.3
_OVERSHOOT_RANGE = 12.0  # m


@dataclass
class Threat:
    id: int
    pos: np.ndarray
    base_vel: np.ndarray
    spawn_t: float
    hostile: bool
    weave_amp: float = 0.0
    weave_freq: float = 0.0
    fate: str = "inbound"  # inbound | captured | leaked | departed

    @property
    def alive(self) -> bool:
        return self.fate == "inbound"

    def velocity(self, t: float) -> np.ndarray:
        if not self.weave_amp:
            return self.base_vel
        axis = np.cross(self.base_vel, [0.0, 0.0, 1.0])
        axis /= np.linalg.norm(axis)
        return self.base_vel + axis * self.weave_amp * math.sin(
            2 * math.pi * self.weave_freq * (t - self.spawn_t)
        )


@dataclass
class Event:
    t: float
    level: str
    msg: str
    pos: list[float] | None = None


@dataclass
class Simulation:
    spawns: list[Spawn]
    dome: DomeConfig = field(default_factory=DomeConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.sim.seed)
        self.radar = Radar(self.sensor, self.rng)
        self.tracker = Tracker()
        self.commander = Commander(self.dome, self.log)
        self.t = 0.0
        self.tick_count = 0
        self.events: list[Event] = []
        self.frames: list[dict] = []
        self.threats: list[Threat] = []
        self._pending = sorted(self.spawns, key=lambda s: s.t)
        n = self.dome.n_drones
        self.drones = [
            Drone(
                i,
                np.array(
                    [
                        _PAD_RADIUS * math.cos(2 * math.pi * i / n),
                        _PAD_RADIUS * math.sin(2 * math.pi * i / n),
                        _MIN_ALTITUDE,
                    ]
                ),
            )
            for i in range(n)
        ]
        self.log(0.0, f"AEGIS online. {n} drones launching to a {self.dome.dome_radius:.0f} m shell", "shell")
        self.commander.tick(0.0, [], self.drones)

    def log(self, t: float, msg: str, level: str = "info", pos=None) -> None:
        self.events.append(Event(round(t, 2), level, msg, pos))

    @property
    def finished(self) -> bool:
        if self.t >= self.sim.duration:
            return True
        settled = not self._pending and not any(th.alive for th in self.threats)
        return settled and self.t > 3.0 and self._last_activity + 3.0 < self.t

    def run(self) -> "Simulation":
        self._last_activity = 0.0
        self._record()
        while not self.finished:
            self.step()
        return self

    def step(self) -> None:
        dt = self.sim.dt
        self.t += dt
        self.tick_count += 1

        while self._pending and self._pending[0].t <= self.t:
            s = self._pending.pop(0)
            self.threats.append(
                Threat(len(self.threats), s.pos.copy(), s.vel, s.t, s.hostile, s.weave_amp, s.weave_freq)
            )

        live = [th for th in self.threats if th.alive]
        th_prev = {th.id: th.pos.copy() for th in live}
        for th in live:
            th.pos = th.pos + th.velocity(self.t) * dt

        dr_prev = {d.id: d.pos.copy() for d in self.drones}
        self._fly_drones(dt)
        self._resolve_contacts(live, th_prev, dr_prev)

        detections = self.radar.scan([th.pos for th in self.threats if th.alive])
        tracks = self.tracker.step(detections, dt)
        self.commander.tick(self.t, tracks, self.drones)

        if any(th.alive for th in self.threats) or any(
            d.state == DroneState.INTERCEPT for d in self.drones
        ):
            self._last_activity = self.t
        if self.tick_count % self.sim.record_every == 0:
            self._record()

    def _fly_drones(self, dt: float) -> None:
        cfg = self.dome
        tracks = {tr.id: tr for tr in self.tracker.tracks}
        for d in self.drones:
            if d.state == DroneState.SPENT:
                d.vel = np.array([0.0, 0.0, -_SPENT_SINK_RATE if d.pos[2] > 0 else 0.0])
            elif d.state == DroneState.INTERCEPT and d.target in tracks:
                tr = tracks[d.target]
                rel = tr.pos - d.pos
                if rel @ rel < _OVERSHOOT_RANGE**2 and rel @ (tr.vel - d.vel) > 0.0:
                    d.overshot = True  # range is opening at close quarters: we flew past it
                solution = lead_intercept(d.pos, cfg.drone_max_speed, tr)
                if solution is None:
                    d.infeasible_for += dt
                    aim = tr.pos  # pure pursuit until the geometry improves
                else:
                    d.infeasible_for = 0.0
                    aim = solution[1]
                d.vel = steer(d.pos, d.vel, aim, cfg.drone_max_speed, cfg.drone_max_accel, dt, arrive=False)
            else:
                aim = d.slot if d.slot is not None else d.pos
                d.vel = steer(d.pos, d.vel, aim, cfg.drone_max_speed, cfg.drone_max_accel, dt, arrive=True)
            d.pos = d.pos + d.vel * dt
            d.pos[2] = max(d.pos[2], 0.0 if d.state == DroneState.SPENT else _MIN_ALTITUDE)

    def _resolve_contacts(self, live, th_prev, dr_prev) -> None:
        cfg = self.dome
        for th in live:
            for d in self.drones:
                if d.state == DroneState.SPENT:
                    continue
                if segment_min_distance(dr_prev[d.id], d.pos, th_prev[th.id], th.pos) <= cfg.capture_radius:
                    how = "intercepted" if d.state == DroneState.INTERCEPT else "body-blocked"
                    th.fate = "captured"
                    d.state = DroneState.SPENT
                    d.target = None
                    d.slot = None
                    note = "" if th.hostile else " (was not a threat)"
                    self.log(
                        self.t,
                        f"{d.callsign} {how} object at {np.linalg.norm(th.pos):.0f} m{note}",
                        "kill" if th.hostile else "alert",
                        [round(float(c), 2) for c in th.pos],
                    )
                    break
            if not th.alive:
                continue
            if segment_point_distance(th_prev[th.id], th.pos, np.zeros(3)) <= cfg.core_radius:
                th.fate = "leaked"
                self.log(self.t, "IMPACT: an object reached the protected core", "alert", [0.0, 0.0, 0.0])
            elif np.linalg.norm(th.pos) > self.sensor.max_range + 50 and th.pos @ th.velocity(self.t) > 0:
                th.fate = "departed"

    def _record(self) -> None:
        r = lambda v: [round(float(c), 2) for c in v]  # noqa: E731
        state_code = {DroneState.STATION: 0, DroneState.INTERCEPT: 1, DroneState.SPENT: 2}
        assessments = self.commander.assessments
        self.frames.append(
            {
                "t": round(self.t, 2),
                "d": [r(d.pos) + [state_code[d.state]] for d in self.drones],
                "th": [[th.id] + r(th.pos) + [int(th.hostile)] for th in self.threats if th.alive],
                "tr": [
                    [tr.id] + r(tr.pos) + [int(assessments[tr.id].hostile) if tr.id in assessments else -1]
                    for tr in self.tracker.tracks
                    if tr.confirmed
                ],
                "en": [[d.id, d.target] for d in self.drones if d.state == DroneState.INTERCEPT],
            }
        )

    def summary(self) -> dict:
        hostile = [th for th in self.threats if th.hostile]
        benign = [th for th in self.threats if not th.hostile]
        count = lambda group, fate: sum(th.fate == fate for th in group)  # noqa: E731
        return {
            "hostile": len(hostile),
            "captured": count(hostile, "captured"),
            "leaked": count(hostile, "leaked"),
            "evaded": count(hostile, "departed"),  # flew through the dome but missed the core
            "unresolved": count(hostile, "inbound"),
            "benign": len(benign),
            "benign_captured": count(benign, "captured"),
            "drones_spent": sum(d.state == DroneState.SPENT for d in self.drones),
            "drones_remaining": sum(d.state != DroneState.SPENT for d in self.drones),
            "duration": round(self.t, 2),
        }
