"""One generate run, Start (+ Beer Near) to saved trail, map and report.

Shared by ``autotrail generate`` and the web GUI (``autotrail gui``). The
generator itself lives in :mod:`autotrail.generate`; this module only turns
user settings into its options, runs it, scores the winner on more runs and
writes the files.

The Beer Near is normally the user's pick (that's where the car and cooler
go). Given one, the leg back to Start is sized from the real walking distance
between the two, not from ``return_miles``. With no Beer Near, the generator
picks one about ``return_miles`` from Start, as before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .graph import StreetGraph
from .pack import HareProfile, pace_from_min_per_mile
from .pack import runners as make_runners
from .trail import METRES_PER_MILE, Trail

LonLat = tuple[float, float]


def min_per_mile(text: str) -> float:
    """'7:30' -> 7.5."""
    m, _, s = str(text).strip().partition(":")
    return int(m) + int(s or 0) / 60


@dataclass
class RunSettings:
    miles: float = 5.0             # true trail, Start to Beer Near
    return_leg: bool = True        # lay a hared leg back to Start
    return_miles: float = 1.0      # only used when the Beer Near is auto-picked
    seed: int = 0
    tries: int = 300               # route attempts
    runs: int = 20                 # simulated runs for the final report
    runners: int = 12
    fastest: str = "7:30"
    slowest: str = "9:30"
    hare_pace: str = "10:15"
    head_start_min: float = 15.0

    def pack(self):
        return make_runners(self.runners, min_per_mile(self.fastest), min_per_mile(self.slowest))

    def hare(self) -> HareProfile:
        return HareProfile(pace_from_min_per_mile(min_per_mile(self.hare_pace)), self.head_start_min * 60.0)


@dataclass
class RunResult:
    trail: Trail
    card: object
    results: list
    hare: HareProfile
    report: str
    walk_back_mi: Optional[float]          # shortest walk Beer Near -> Start (manual Beer Near only)
    files: dict[str, Path] = field(default_factory=dict)


def walk_m(g: StreetGraph, a, b) -> float:
    """Shortest walking distance from node a to node b (inf if unreachable)."""
    import networkx as nx

    if a == b:
        return 0.0
    try:
        return float(nx.dijkstra_path_length(g.G, a, b, weight="length"))
    except nx.NetworkXNoPath:
        return math.inf


def check_ends(g: StreetGraph, start, end, miles: float) -> tuple[float, Optional[str]]:
    """Walking distance Start -> Beer Near in miles, and a problem message if the pair can't work."""
    d = walk_m(g, start, end) / METRES_PER_MILE
    if math.isinf(d):
        return d, "The Beer Near can't be reached on foot from Start on this map."
    if start == end:
        return d, "Start and Beer Near snap to the same corner. Put the Beer Near somewhere else."
    if d > 0.8 * miles:
        return d, (f"The Beer Near is a {d:.1f} mi walk from Start, too far for a {miles:g} mi trail "
                   f"(it needs to be under {0.8 * miles:.1f} mi). Move it closer or make the trail longer.")
    return d, None


def preview_walk_mi(g: StreetGraph, start: LonLat, end: LonLat, miles: float) -> tuple[float, Optional[str]]:
    """Like :func:`check_ends` for two clicks, without changing the graph (to within a block end)."""
    a, b = g.snap_to_street(*start), g.snap_to_street(*end)
    ax, ay = g.to_metres(a["lon"], a["lat"])
    bx, by = g.to_metres(b["lon"], b["lat"])
    if math.hypot(ax - bx, ay - by) < 15.0:
        return 0.0, "Start and Beer Near are on the same spot. Put the Beer Near somewhere else."

    def ends(p):  # (node, metres from the click) for both ends of its street
        u, v, k = p["edge"]
        return [(u, p["offset_m"]), (v, g.edge_length(u, v, k) - p["offset_m"])]

    if set(a["edge"][:2]) == set(b["edge"][:2]):  # same block
        d = abs(a["offset_m"] - (b["offset_m"] if a["edge"][0] == b["edge"][0]
                                 else g.edge_length(*b["edge"]) - b["offset_m"]))
    else:
        d = min(da + walk_m(g, na, nb) + db for na, da in ends(a) for nb, db in ends(b))
    d /= METRES_PER_MILE
    if math.isinf(d):
        return d, "The Beer Near can't be reached on foot from Start on this map."
    if d > 0.8 * miles:
        return d, (f"The Beer Near is a {d:.1f} mi walk from Start, too far for a {miles:g} mi trail "
                   f"(it needs to be under {0.8 * miles:.1f} mi). Move it closer or make the trail longer.")
    return d, None


def run(g: StreetGraph, start: LonLat, end: Optional[LonLat], s: RunSettings, *,
        out: Path, map_out: Optional[Path] = None, gpx: Optional[Path] = None,
        tiles: str = "streets", progress: Callable[[str], None] | None = None) -> RunResult:
    """Generate, score and save one trail. ``start``/``end`` are (lon, lat)."""
    from .generate import GenerateOptions, generate
    from .score import format_report, score_results
    from .sim import simulate_many
    from .viz import save_replay_map

    say = progress or (lambda msg: None)
    # Ends go exactly where they were clicked: the nearest point on the nearest street.
    s_node = g.node_at(*start)
    e_node = g.node_at(*end) if end is not None else None
    if e_node is None and not s.return_leg:
        raise ValueError("Without a leg back to Start, pick a Beer Near.")

    walk_back = None
    back_mi = s.return_miles
    if e_node is not None:
        walk_back, problem = check_ends(g, s_node, e_node, s.miles)
        if problem:
            raise ValueError(problem)
        say(f"Beer Near is a {walk_back:.2f} mi walk from Start")
        # The search area and the longest allowed leg back both scale with this.
        back_mi = max(walk_back, 0.25)

    pack, hare = s.pack(), s.hare()
    opts = GenerateOptions(miles=s.miles, return_miles=back_mi, tries=s.tries, exact_ends=True)
    res = generate(g, s_node, e_node, opts, pack=pack, hare=hare, seed=s.seed, progress=say)
    trail = res.trail
    if not s.return_leg:
        trail.return_path = []
        trail.marks = [m for m in trail.marks if m.params.get("leg") != "return"]
    trail.meta.update({"name": f"Autotrail {s.miles:g} mi (seed {s.seed})",
                       "beer_near": "picked" if e_node is not None else "auto"})
    say(f"Scoring the winner on {s.runs} more runs")
    results = simulate_many(trail, g, pack, runs=s.runs, seed=s.seed)
    card = score_results(results, s.miles * METRES_PER_MILE, hare=hare)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trail.save(out, g)
    files = {"trail": out}
    if gpx:
        Path(gpx).write_text(trail.to_gpx(g), encoding="utf-8")
        files["gpx"] = Path(gpx)
    map_out = Path(map_out or out.with_suffix(".html"))
    save_replay_map(trail, g, results[0], map_out, card=card, tiles=tiles, hare=hare)
    files["map"] = map_out
    return RunResult(trail, card, results, hare, format_report(card, results), walk_back, files)
