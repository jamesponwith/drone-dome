"""Command line entry point: run a scenario, narrate it, export a replay."""

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from aegis.config import DomeConfig, SimConfig
from aegis.replay import to_replay, write_html
from aegis.scenarios import SCENARIOS
from aegis.sim import Simulation

_TAGS = {
    "hostile": "!!",
    "engage": "->",
    "kill": "XX",
    "alert": "**",
    "shell": "()",
    "info": "  ",
}


def run_scenario(name: str, seed: int, drones: int) -> Simulation:
    spawns = SCENARIOS[name](np.random.default_rng(seed))
    dome = dataclasses.replace(DomeConfig(), n_drones=drones)
    return Simulation(spawns, dome=dome, sim=dataclasses.replace(SimConfig(), seed=seed)).run()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="aegis", description="Drone-dome interception simulator")
    parser.add_argument("scenario", nargs="?", default="raid", choices=[*SCENARIOS, "all"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--drones", type=int, default=DomeConfig.n_drones)
    parser.add_argument("--out", type=Path, default=Path("replays"), help="directory for HTML replays")
    parser.add_argument("--quiet", action="store_true", help="summary only, no event log")
    args = parser.parse_args(argv)

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in names:
        sim = run_scenario(name, args.seed, args.drones)
        print(f"\n=== {name.upper()} (seed {args.seed}, {args.drones} drones) ===")
        if not args.quiet:
            for e in sim.events:
                print(f"[T+{e.t:05.2f}] {_TAGS.get(e.level, '  ')} {e.msg}")
        s = sim.summary()
        print(
            f"--- hostiles {s['hostile']}: captured {s['captured']}, leaked {s['leaked']}"
            f"{', evaded ' + str(s['evaded']) if s['evaded'] else ''}"
            f"{', unresolved ' + str(s['unresolved']) if s['unresolved'] else ''}"
            f" | benign {s['benign']} (captured {s['benign_captured']})"
            f" | drones spent {s['drones_spent']}, remaining {s['drones_remaining']}"
        )
        path = write_html(to_replay(sim, name), args.out / f"{name}.html")
        print(f"--- replay: {path}")
