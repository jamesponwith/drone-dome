# AEGIS: drone dome

A simulation of a fleet of mini drones that forms a protective shell around an
asset and intercepts inbound objects before they reach it. Drones hold slots on
a hemispherical shell. When a radar track is classified as hostile, the best
placed drone leaves the shell to capture it (net/contact), and the rest of the
shell re-spaces itself to close the gap.

```
uv sync
uv run aegis raid                # narrate one scenario, write replays/raid.html
uv run aegis all --quiet         # every scenario, summaries only
uv run aegis surround --seed 3 --drones 32
uv run pytest
```

Open `replays/<scenario>.html` in a browser for a 3D replay (drag to orbit,
scroll to zoom, scrub the timeline). Append `#t=8.5` to the URL to open at a
given moment.

Live replays: https://jamesponwith.github.io/drone-dome/

## Pipeline

Each 50 ms tick runs sense → track → assess → assign → guide → heal:

| Stage | Module | What it does |
|---|---|---|
| Sense | `sensor.py` | Radar reports with range-dependent noise and 3% missed scans. Friendly drones are filtered upstream (IFF). |
| Track | `tracking.py` | IMM filter per track: a cruise model and a maneuver model blended by likelihood, so straight flyers get tight estimates and weaving targets stay in the association gate. Global nearest-neighbor association (Hungarian) with a chi-square gate; tracks are confirmed after 3 hits and coast through short dropouts. |
| Assess | `threat.py` | Projects both the instantaneous and the smoothed mean velocity and takes the more dangerous prediction. Hostile if the path's closest approach falls inside the shell. Classification waits until the velocity estimate has settled, and once a track is hostile, hysteresis keeps it hostile. |
| Assign | `command.py` | Hungarian assignment of free shell drones to hostile tracks, minimizing time to intercept. An intercept only counts if it lands before the threat reaches the core and inside the engagement range. Threats close to breaching get a two-drone salvo. A reserve stays on the shell unless a threat would otherwise go unanswered. |
| Guide | `guidance.py` | Lead-pursuit to the predicted intercept point, with a speed- and acceleration-limited steering law. Long-horizon aim uses the smoothed velocity; terminal aim uses the instantaneous one. |
| Detect miss | `sim.py` / `command.py` | If the range to target starts opening at close quarters, that's a miss. The drone is freed after a short turnaround so fire control can re-attack with whichever drone is best placed. |
| Heal | `command.py`, `geometry.py` | Whenever the set of on-station drones changes, lay out a fresh Fibonacci lattice (equal area per drone) and assign drones to slots by minimum travel (Hungarian). |

Capture is checked with swept-segment closest approach, not end-of-tick positions: at 60 m/s closing speed, a single tick covers twice the capture radius.

## Scenarios

| Name | What it throws at the dome |
|---|---|
| `probe` | Three spaced hostiles (one weaving) plus benign traffic that should be ignored |
| `raid` | Eight simultaneous hostiles from one sector |
| `surround` | Twelve hostiles from every direction at once, plus passers-by |
| `ambush` | Five 40 m/s low-altitude threats down one lane, faster than the drones |
| `saturation` | 36 hostiles against 24 drones, mixed speeds, 30% weaving |

Sweep over 16 seeds each for `probe`, `raid`, `surround`, `ambush`: 448/448
hostiles captured, 0 leaks, 0 benign objects touched. Under `saturation`, every
drone makes exactly one capture before the shell runs out. That is the
designed failure mode: the dome degrades by running out of drones, not by
wasting them.

## Known limits / next steps

- Terminal accuracy against hard maneuvering targets is the weakest link. A
  weaver can take several attempts; defense in depth covers for it, but
  augmented proportional navigation with a target-acceleration estimate would
  cut drone spend.
- There's no drone-to-drone collision avoidance in flight. Shell slots are
  spaced apart, but interceptors fly straight through.
- Drones are single-use. A recover-and-rearm cycle (return to pad, reload net,
  relaunch) would turn the magazine limit into a throughput limit.
- Only one sensor, at the asset. Distributed sensing and track fusion would add
  coverage at low altitude.

## Deploying to GitHub Pages

`.github/workflows/deploy.yml` regenerates every scenario replay
(`uv run aegis all --out _site`), adds the landing page from
`site/index.html`, and deploys the result to GitHub Pages on every push to
`main`. It is served as a project page at
`https://jamesponwith.github.io/drone-dome/`.
