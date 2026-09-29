"""Package a finished simulation into a self-contained HTML replay."""

import dataclasses
import json
from importlib.resources import files
from pathlib import Path

from aegis.sim import Simulation

_PLACEHOLDER = "/*__REPLAY__*/null"


def to_replay(sim: Simulation, scenario: str) -> dict:
    return {
        "scenario": scenario,
        "dt": sim.sim.dt * sim.sim.record_every,
        "dome": dataclasses.asdict(sim.dome),
        "sensor_range": sim.sensor.max_range,
        "summary": sim.summary(),
        "events": [dataclasses.asdict(e) for e in sim.events],
        "frames": sim.frames,
    }


def write_html(replay: dict, path: Path) -> Path:
    template = files("aegis").joinpath("viewer.html").read_text()
    html = template.replace(_PLACEHOLDER, json.dumps(replay, separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path
