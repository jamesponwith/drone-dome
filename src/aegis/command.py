"""Fire control: classify tracks, assign interceptors, and keep the shell sealed."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from scipy.optimize import linear_sum_assignment

from aegis.config import DomeConfig
from aegis.geometry import dome_slots
from aegis.guidance import lead_intercept
from aegis.threat import Assessment, assess
from aegis.tracking import Track

_INFEASIBLE = 1e9
_GIVE_UP_AFTER = 1.0  # s of no intercept solution before an interceptor is recalled
_TURNAROUND = 0.8  # s a drone that missed needs before it can be re-tasked


class DroneState(StrEnum):
    STATION = "station"  # holding a slot on the shell
    INTERCEPT = "intercept"  # launched at a track
    SPENT = "spent"  # captured something; out of the fight


@dataclass
class Drone:
    id: int
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    state: DroneState = DroneState.STATION
    slot: np.ndarray | None = None
    target: int | None = None
    infeasible_for: float = 0.0
    overshot: bool = False  # flew past its target without capturing it
    ready_at: float = 0.0  # earliest time it can be assigned again

    @property
    def callsign(self) -> str:
        return f"D{self.id:02d}"


Log = Callable[[float, str, str], None]


class Commander:
    def __init__(self, cfg: DomeConfig, log: Log):
        self.cfg = cfg
        self.log = log
        self.engagements: dict[int, set[int]] = {}  # track id -> drone ids
        self.assessments: dict[int, Assessment] = {}
        self._classified: dict[int, bool] = {}
        self._shell: frozenset[int] | None = None

    def tick(self, t: float, tracks: list[Track], drones: list[Drone]) -> None:
        by_id = {tr.id: tr for tr in tracks}
        self.assessments = {}
        for tr in tracks:
            if not tr.confirmed or tr.vel_sigma > self.cfg.classify_vel_sigma:
                continue
            self.assessments[tr.id] = assess(
                tr,
                self.cfg.dome_radius,
                self.cfg.core_radius,
                self.cfg.hostile_margin,
                was_hostile=bool(self._classified.get(tr.id)),
            )
        self._announce(t)
        self._recall(t, by_id, drones)
        self._assign(t, by_id, drones)
        self._heal(t, drones)

    def _announce(self, t: float) -> None:
        for tid, a in self.assessments.items():
            if self._classified.get(tid) == a.hostile:
                continue
            first = tid not in self._classified
            self._classified[tid] = a.hostile
            if a.hostile:
                eta = (
                    f"shell breach in {a.time_to_breach:.1f} s"
                    if math.isfinite(a.time_to_breach)
                    else "grazing the shell"
                )
                self.log(
                    t,
                    f"TRACK {tid} HOSTILE: {a.speed:.0f} m/s, CPA {a.cpa_distance:.1f} m, {eta}",
                    "hostile",
                )
            elif first:
                self.log(t, f"TRACK {tid} passing: CPA {a.cpa_distance:.0f} m, no action", "info")
            else:
                self.log(t, f"TRACK {tid} reclassified non-threatening", "info")

    def _recall(self, t: float, by_id: dict[int, Track], drones: list[Drone]) -> None:
        for tid in list(self.engagements):
            live = {d for d in self.engagements[tid] if drones[d].state == DroneState.INTERCEPT}
            a = self.assessments.get(tid)
            if tid not in by_id:
                reason = "track dropped"
            elif a is not None and not a.hostile:
                reason = "no longer a threat"
            else:
                reason = None
            for did in list(live):
                d = drones[did]
                if reason is not None:
                    continue
                if d.overshot:
                    why = "missed"
                elif d.infeasible_for > _GIVE_UP_AFTER:
                    why = "cannot catch"
                else:
                    continue
                # Freeing the drone lets _assign re-attack with whoever is best placed now.
                self._return(d)
                d.ready_at = t + _TURNAROUND
                live.discard(did)
                self.log(t, f"{d.callsign} {why} TRACK {tid}, re-tasking", "info")
            if reason is not None:
                for did in live:
                    self._return(drones[did])
                if live:
                    names = ", ".join(drones[d].callsign for d in sorted(live))
                    self.log(t, f"TRACK {tid} {reason}; {names} returning to shell", "info")
                live = set()
            if live:
                self.engagements[tid] = live
            else:
                del self.engagements[tid]

    @staticmethod
    def _return(d: Drone) -> None:
        d.state = DroneState.STATION
        d.target = None
        d.infeasible_for = 0.0
        d.overshot = False

    def _assign(self, t: float, by_id: dict[int, Track], drones: list[Drone]) -> None:
        cfg = self.cfg
        # A coasting track has just been captured or dropped by the radar; launching
        # at it would send a drone after a ghost.
        hostile = sorted(
            (a for a in self.assessments.values() if a.hostile and by_id[a.track_id].misses == 0),
            key=lambda a: -a.priority,
        )
        needs: list[Assessment] = []
        for a in hostile:
            want = 2 if a.time_to_breach < cfg.salvo_breach_time else 1
            needs += [a] * max(0, want - len(self.engagements.get(a.track_id, ())))
        free = [d for d in drones if d.state == DroneState.STATION and d.ready_at <= t]
        if not needs or not free:
            return

        # The reserve keeps the shell from being stripped bare, but it is spent
        # before letting an unanswered threat through the shell.
        critical = any(
            a.time_to_breach < cfg.salvo_breach_time and a.track_id not in self.engagements
            for a in needs
        )
        limit = len(free) if critical else max(0, len(free) - cfg.reserve)
        if limit == 0:
            return

        cost = np.full((len(free), len(needs)), _INFEASIBLE)
        for i, d in enumerate(free):
            for j, a in enumerate(needs):
                tr = by_id[a.track_id]
                solution = lead_intercept(d.pos, cfg.drone_max_speed, tr)
                if solution is None:
                    continue
                t_int, point = solution
                if t_int > a.deadline or np.linalg.norm(point) > cfg.engage_range:
                    continue
                cost[i, j] = t_int

        rows, cols = linear_sum_assignment(cost)
        # needs is in priority order, so sorting on column keeps the most urgent.
        picks = sorted((j, i) for i, j in zip(rows, cols) if cost[i, j] < _INFEASIBLE)[:limit]
        for j, i in picks:
            d, a = free[i], needs[j]
            d.state = DroneState.INTERCEPT
            d.target = a.track_id
            d.slot = None
            self.engagements.setdefault(a.track_id, set()).add(d.id)
            self.log(
                t,
                f"{d.callsign} -> TRACK {a.track_id}, intercept in {cost[i, j]:.1f} s",
                "engage",
            )

    def _heal(self, t: float, drones: list[Drone]) -> None:
        """Re-spread the drones still on station so the shell has no holes."""
        shell = [d for d in drones if d.state == DroneState.STATION]
        members = frozenset(d.id for d in shell)
        if members == self._shell:
            return
        grew = self._shell is not None and len(members) > len(self._shell)
        shrank = self._shell is not None and len(members) < len(self._shell)
        self._shell = members
        if not shell:
            self.log(t, "SHELL EMPTY: no drones on station", "alert")
            return
        slots = dome_slots(len(shell), self.cfg.dome_radius, self.cfg.min_elevation_deg)
        positions = np.array([d.pos for d in shell])
        dist = np.linalg.norm(positions[:, None, :] - slots[None, :, :], axis=2)
        for i, j in zip(*linear_sum_assignment(dist)):
            shell[i].slot = slots[j]
        if grew or shrank:
            self.log(t, f"Shell re-formed with {len(shell)} drones", "shell")
