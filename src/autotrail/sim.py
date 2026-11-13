"""Tier 1 pack simulator: fast rule-based hounds on the street graph.

The simulation is event driven. A hound's plan is a short list of legs (run
along a track from one distance to another, or wait), and the only events are
"reached the end of the plan": a mark, a corner, a check, the give-up point,
the end of a probe. On-On calls interrupt the plans of hounds that hear them.

Everything a hound runs is a ``Track``: the true trail, a false branch, or a
probe (a short run down an unmarked street and back). Tracks start at a node,
so a hound's position is always (track, metres along it).

Rules (see the design doc, "Tier 1"):
  * Running: follow the trail; each blob is spotted with ``detect_prob``. More
    than ``give_up_m`` without a mark: go back to the last mark and look again.
  * Corners: if the trail goes straight on, hounds follow it. If it turns, a
    hound spots the blob just past the corner with ``detect_prob``; otherwise
    it runs straight on, finds nothing, and comes back. A hound that saw
    another hound take the corner the right way within ``follow_s`` seconds
    just follows (this is why the back of the pack runs close to true trail).
  * Checks: hounds fan out over the directions the pack hasn't tried yet
    (shared memory). Probes run ``check_search_m`` out and back; falses run
    to the F mark and back. The hound that finds ``confirm_marks`` marks on
    the true trail calls On-On; every hound at that check within
    ``call_radius_m`` (straight line) runs to the caller. A solved check costs
    late arrivals nothing: they go straight on.
  * Done: reached the Beer Near.

``CheckPolicy`` and ``CallModel`` are the extension points for smarter hounds
and noise-limited calling (phase 5).
"""

from __future__ import annotations

import bisect
import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from .graph import StreetGraph
from .pack import HoundProfile, default_pack
from .trail import BranchPurpose, MarkKind, Trail

MAX_SIM_S = 4 * 3600.0
PROBE_MAX_M = 400.0


# ---------------------------------------------------------------- geometry

class Geometry:
    """Cached metric geometry of graph edges and node bearings."""

    def __init__(self, graph: StreetGraph):
        self.g = graph
        self._edge: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._node_xy: dict = {}

    def node_xy(self, n) -> tuple[float, float]:
        p = self._node_xy.get(n)
        if p is None:
            p = self._node_xy[n] = self.g.to_metres(*self.g.node_lonlat(n))
        return p

    def bearing(self, a, b) -> float:
        ax, ay = self.node_xy(a)
        bx, by = self.node_xy(b)
        return math.atan2(by - ay, bx - ax)

    def edge_poly(self, e: tuple):
        """(cum distance scaled to graph length, x, y, lon, lat) along edge e."""
        hit = self._edge.get(e)
        if hit is None:
            line = self.g.edge_line(*e)
            lon, lat = (np.asarray(a, dtype=float) for a in line.xy)
            x, y = self.g._fwd.transform(lon, lat)
            x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            seg = np.hypot(np.diff(x), np.diff(y))
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            length = self.g.edge_length(*e)
            cum = cum * (length / cum[-1]) if cum[-1] > 0 else np.linspace(0, length, len(cum))
            hit = self._edge[e] = (cum, x, y, lon, lat)
        return hit


def turn_angle(geo: Geometry, prev, node, nxt) -> float:
    """Absolute turn in radians going prev -> node -> nxt (0 = straight on)."""
    a = geo.bearing(prev, node)
    b = geo.bearing(node, nxt)
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


class Track:
    """A run of edges from a node, measured in metres from its start."""

    def __init__(self, tid: int, kind: str, edges: list[tuple], geo: Geometry,
                 max_len: float | None = None, label: str = ""):
        self.tid, self.kind, self.edges, self.label = tid, kind, edges, label
        lengths = [geo.g.edge_length(*e) for e in edges]
        self.cum = np.concatenate([[0.0], np.cumsum(lengths)]) if edges else np.array([0.0])
        full = float(self.cum[-1])
        self.length = min(full, max_len) if max_len is not None else full
        self.start_node = edges[0][0] if edges else None
        parts_s, parts = [], []
        for i, e in enumerate(edges):
            cum, x, y, lon, lat = geo.edge_poly(e)
            sl = slice(0 if i == 0 else 1, None)
            parts_s.append(cum[sl] + self.cum[i])
            parts.append(np.column_stack([x, y, lon, lat])[sl])
        if parts:
            self._s = np.concatenate(parts_s)
            self._p = np.concatenate(parts)
        else:
            self._s = np.array([0.0])
            self._p = np.zeros((1, 4))

    def xy(self, s):
        s = np.asarray(s, dtype=float)
        return np.interp(s, self._s, self._p[:, 0]), np.interp(s, self._s, self._p[:, 1])

    def lonlat(self, s):
        s = np.asarray(s, dtype=float)
        return np.interp(s, self._s, self._p[:, 2]), np.interp(s, self._s, self._p[:, 3])

    def node_s(self) -> list[float]:
        return [float(c) for c in self.cum]


# ---------------------------------------------------------------- compiled trail

@dataclass
class Option:
    kind: str              # "true", "false", "probe"
    nb: object             # neighbour node the option heads to
    track: Track | None = None
    false_s: float = 0.0   # where the F mark is on a false branch


@dataclass
class Check:
    idx: int
    s: float
    node: object
    gap_end: float                      # first mark after the check on the true trail
    options: list[Option] = field(default_factory=list)
    # per-run state
    solved: bool = False
    solved_t: float | None = None
    claimed: set = field(default_factory=set)
    dead: set = field(default_factory=set)
    arrivals: dict = field(default_factory=dict)  # hound idx -> first arrival time

    def reset(self) -> None:
        self.solved, self.solved_t = False, None
        self.claimed, self.dead, self.arrivals = set(), set(), {}


@dataclass
class Junction:
    s: float
    node: object
    straight_correct: bool
    visible: bool              # a mark within sight_m past the corner
    wrong: list                # neighbour nodes that are not the trail
    straightest_wrong: object  # where a hound that misses the turn goes
    last_t: float | None = None  # per run: when a hound last went the right way here


class TrailModel:
    """A Trail compiled for fast simulation."""

    def __init__(self, trail: Trail, graph: StreetGraph, sight_m: float = 40.0):
        self.trail, self.g = trail, graph
        self.geo = Geometry(graph)
        self.warnings: list[str] = []
        self._next_tid = 0
        self._probes: dict[tuple, Track] = {}

        tp = [(e.u, e.v, e.key) for e in trail.true_path]
        self.true = self._track("true", tp, label="true trail")
        self.L = self.true.length
        cum = self.true.cum
        node_s = {}  # node -> first s on the true path
        for i, e in enumerate(tp):
            node_s.setdefault(e[0], float(cum[i]))
        node_s.setdefault(tp[-1][1], float(cum[-1]))

        fwd_idx: dict[tuple, list[int]] = {}
        rev_idx: dict[tuple, list[int]] = {}
        for i, e in enumerate(tp):
            fwd_idx.setdefault(e, []).append(i)
            rev_idx.setdefault((e[1], e[0]), []).append(i)

        def true_s(m) -> float | None:
            at = m.at
            if at.edge is not None:
                e = (at.edge.u, at.edge.v, at.edge.key)
                if e in fwd_idx:
                    return float(cum[fwd_idx[e][0]] + at.offset_m)
                if (e[0], e[1]) in rev_idx:
                    i = rev_idx[(e[0], e[1])][0]
                    return float(cum[i] + max(0.0, graph.edge_length(*tp[i]) - at.offset_m))
                return None
            return node_s.get(at.node)

        # Branches, and marks on them.
        branch_tracks: list[tuple[Track, object]] = []
        branch_edge: dict[tuple, tuple[Track, int]] = {}
        for b in trail.branches:
            if b.purpose != BranchPurpose.FALSE:
                self.warnings.append(f"{b.purpose.value.title()} branches are phase 2; ignored by the sim")
                continue
            be = [(e.u, e.v, e.key) for e in b.edges]
            t = self._track("false", be, label=b.label or "false")
            branch_tracks.append((t, b.from_node))
            for i, e in enumerate(be):
                branch_edge.setdefault(e, (t, i))

        blobs: list[float] = []
        branch_blobs: dict[int, list[float]] = {}
        branch_false: dict[int, float] = {}
        check_s: list[float] = []
        skipped: dict[str, int] = {}
        for m in trail.marks:
            if m.params.get("leg") == "return":
                continue
            s = true_s(m)
            on_branch = None
            if s is None and m.at.edge is not None:
                e = (m.at.edge.u, m.at.edge.v, m.at.edge.key)
                if e in branch_edge:
                    t, i = branch_edge[e]
                    on_branch = (t, float(t.cum[i] + m.at.offset_m))
            if m.kind in (MarkKind.BLOB, MarkKind.ARROW):
                if s is not None:
                    blobs.append(s)
                elif on_branch:
                    branch_blobs.setdefault(on_branch[0].tid, []).append(on_branch[1])
            elif m.kind == MarkKind.CHECK and s is not None:
                check_s.append(s)
            elif m.kind == MarkKind.FALSE and on_branch:
                t, bs = on_branch
                branch_false[t.tid] = min(branch_false.get(t.tid, math.inf), bs)
            elif m.kind == MarkKind.BEER:
                continue
            else:
                skipped[m.kind.value] = skipped.get(m.kind.value, 0) + 1
        for k, n in skipped.items():
            self.warnings.append(f"{n} {k} mark(s) not simulated yet (phase 2 marks, or off the trail)")
        self.blobs = sorted(blobs)
        self.branch_blobs = {k: sorted(v) for k, v in branch_blobs.items()}

        # Checks snap to the nearest node of the true path.
        nodes_along = [(float(cum[i]), e[0]) for i, e in enumerate(tp)][1:]
        self.checks: list[Check] = []
        used_nodes = set()
        for s in sorted(check_s):
            if not nodes_along:
                break
            cs, cn = min(nodes_along, key=lambda t: abs(t[0] - s))
            if cn in used_nodes:
                continue
            used_nodes.add(cn)
            i = bisect.bisect_right(self.blobs, cs + 0.5)
            gap_end = self.blobs[i] if i < len(self.blobs) else self.L
            self.checks.append(Check(len(self.checks), cs, cn, gap_end))
        self.check_at = {c.s: c for c in self.checks}

        # Options at each check.
        idx_at_s = {float(cum[i]): i for i in range(len(tp))}
        for c in self.checks:
            i = idx_at_s[c.s]
            prev, nxt = tp[i - 1][0], tp[i][1]
            opts = [Option("true", nxt)]
            false_nbs = set()
            for t, frm in branch_tracks:
                if frm == c.node:
                    fs = branch_false.get(t.tid, t.length)
                    opts.append(Option("false", t.edges[0][1], t, min(fs, t.length)))
                    false_nbs.add(t.edges[0][1])
            for nb in self._neighbours(c.node):
                if nb in (prev, nxt) or nb in false_nbs:
                    continue
                opts.append(Option("probe", nb))
            c.options = opts
        for t, frm in branch_tracks:
            if frm not in used_nodes:
                self.warnings.append(f"false '{t.label}' doesn't start at a check; ignored by the sim")

        # Corners along the true path (not at checks).
        self.junctions: list[Junction] = []
        for i in range(1, len(tp)):
            node = tp[i][0]
            s = float(cum[i])
            if s in self.check_at:
                continue
            prev, nxt = tp[i - 1][0], tp[i][1]
            nbs = [nb for nb in self._neighbours(node) if nb != prev]
            wrong = [nb for nb in nbs if nb != nxt]
            if not wrong:
                continue
            straightest = min(nbs, key=lambda nb: turn_angle(self.geo, prev, node, nb))
            j = bisect.bisect_right(self.blobs, s + 0.01)
            visible = j < len(self.blobs) and self.blobs[j] - s <= sight_m
            sw = min(wrong, key=lambda nb: turn_angle(self.geo, prev, node, nb))
            self.junctions.append(Junction(s, node, straightest == nxt, visible, wrong, sw))
        self.junction_s = [j.s for j in self.junctions]
        self.check_s = [c.s for c in self.checks]
        if not self.blobs:
            self.warnings.append("no blobs on the true trail: hounds will get lost (add marks first)")

    # -- helpers
    def _track(self, kind, edges, max_len=None, label="") -> Track:
        t = Track(self._next_tid, kind, edges, self.geo, max_len=max_len, label=label)
        self._next_tid += 1
        return t

    def _neighbours(self, node) -> list:
        G = self.g.G
        return [nb for nb in G.successors(node) if nb != node]

    def probe(self, node, nb) -> Track:
        """Unmarked run from node via nb, going straight on at each corner."""
        key = (node, nb)
        t = self._probes.get(key)
        if t is None:
            G = self.g.G
            edges = []
            total = 0.0
            prev, cur = node, nb
            seen = {node}
            while True:
                e = (prev, cur, self.g.best_key(prev, cur))
                edges.append(e)
                total += self.g.edge_length(*e)
                seen.add(cur)
                if total >= PROBE_MAX_M:
                    break
                nxts = [n for n in G.successors(cur) if n != prev and n != cur and n not in seen]
                if not nxts:
                    break
                p = prev
                prev, cur = cur, min(nxts, key=lambda n: turn_angle(self.geo, p, cur, n))
            t = self._probes[key] = self._track("probe", edges, label=f"probe {node}->{nb}")
        return t

    def reset(self) -> None:
        for c in self.checks:
            c.reset()
        for j in self.junctions:
            j.last_t = None


# ---------------------------------------------------------------- extension points

class CheckPolicy(Protocol):
    def choose(self, hound: "Hound", check: Check, candidates: list[int], rng: random.Random) -> int: ...


class RandomUnexplored:
    """Tier 1: any direction the pack hasn't claimed yet, at random."""

    def choose(self, hound, check, candidates, rng):
        fresh = [i for i in candidates if i not in check.claimed]
        return rng.choice(fresh or candidates)


class CallModel(Protocol):
    def hears(self, caller_xy: tuple[float, float], hound_xy: tuple[float, float]) -> bool: ...


@dataclass
class RadiusCall:
    """Tier 1: everyone within a straight-line radius hears On-On."""
    radius_m: float = 150.0

    def hears(self, caller_xy, hound_xy):
        return math.hypot(caller_xy[0] - hound_xy[0], caller_xy[1] - hound_xy[1]) <= self.radius_m


# ---------------------------------------------------------------- hounds and results

@dataclass
class Leg:
    track: Track
    s0: float
    s1: float
    t0: float
    t1: float


@dataclass
class Hound:
    idx: int
    prof: HoundProfile
    pace: float
    track: Track
    s: float = 0.0
    t: float = 0.0
    legs: list[Leg] = field(default_factory=list)
    plan_i: int = 0
    plan_start: tuple = None
    version: int = 0
    last_seen: float = 0.0
    careful: bool = False
    check: Check | None = None
    tried: set = field(default_factory=set)
    found: int = 0
    heading: bool = False
    opt: int = 0
    done_t: float | None = None
    events: list[tuple[float, str]] = field(default_factory=list)

    def distance(self) -> float:
        return sum(abs(l.s1 - l.s0) for l in self.legs)

    def pos_at(self, t: float) -> tuple[Track, float]:
        for l in reversed(self.legs[self.plan_i:]):
            if l.t0 <= t:
                if t >= l.t1 or l.t1 == l.t0:
                    return l.track, l.s1
                return l.track, l.s0 + (l.s1 - l.s0) * (t - l.t0) / (l.t1 - l.t0)
        return self.plan_start


@dataclass
class HoundResult:
    name: str
    pace_s_per_m: float
    role: str
    finish_s: float | None
    distance_m: float
    extra_m: float


@dataclass
class CheckResult:
    idx: int
    s: float
    node: object
    first_arrival_s: float | None
    last_arrival_s: float | None
    solved_s: float | None
    front_loss_s: float | None     # how long the first hound there lost
    gap_in_s: float | None         # first to last arrival at the check
    gap_out_s: float | None        # first to last past the first mark after it


@dataclass
class SimResult:
    true_trail_m: float
    hounds: list[HoundResult]
    checks: list[CheckResult]
    gap_profile: list[tuple[float, float]]  # (metres along trail, front-to-back seconds)
    seed: int
    raw: list[Hound] = field(default_factory=list, repr=False)

    @property
    def finishers(self) -> list[HoundResult]:
        return [h for h in self.hounds if h.finish_s is not None]


# ---------------------------------------------------------------- engine

class Simulation:
    def __init__(self, model: TrailModel, pack: list[HoundProfile], seed: int = 0,
                 policy: CheckPolicy | None = None, calls: CallModel | None = None):
        self.m, self.pack, self.seed = model, pack, seed
        self.policy = policy or RandomUnexplored()
        self.calls = calls or RadiusCall()
        self.rng = random.Random(seed)
        self.heap: list = []
        self._seq = 0

    # -- plumbing
    def go(self, h: Hound, legs: list[tuple], then: Callable[[Hound], None]) -> None:
        """Queue legs [(track, s0, s1) or (track, s, s, wait_s)] and call ``then`` at the end."""
        h.plan_i = len(h.legs)
        h.plan_start = (h.track, h.s)
        t = h.t
        for spec in legs:
            track, s0, s1 = spec[:3]
            dt = abs(s1 - s0) * h.pace + (spec[3] if len(spec) > 3 else 0.0)
            h.legs.append(Leg(track, s0, s1, t, t + dt))
            t += dt
        if legs:
            last = legs[-1]
            h.track, h.s = last[0], last[2]
        h.t = t
        h.version += 1
        self._seq += 1
        heapq.heappush(self.heap, (t, self._seq, h.idx, h.version, then))

    def interrupt(self, h: Hound, t: float) -> None:
        keep = []
        for l in h.legs[h.plan_i:]:
            if l.t1 <= t:
                keep.append(l)
            elif l.t0 < t:
                f = (t - l.t0) / (l.t1 - l.t0)
                keep.append(Leg(l.track, l.s0, l.s0 + (l.s1 - l.s0) * f, l.t0, t))
        del h.legs[h.plan_i:]
        h.legs.extend(keep)
        h.track, h.s = (keep[-1].track, keep[-1].s1) if keep else h.plan_start
        h.t = t
        h.version += 1

    def log(self, h: Hound, what: str) -> None:
        h.events.append((h.t, what))

    # -- running
    def run(self, h: Hound) -> None:
        m, p = self.m, h.prof
        T, s = h.track, h.s
        if T is m.true:
            stops = [(m.L, self.at_end)]
            i = bisect.bisect_right(m.blobs, s + 1e-6)
            if i < len(m.blobs):
                stops.append((m.blobs[i], self.at_blob))
            i = bisect.bisect_right(m.check_s, s + 1e-6)
            if i < len(m.check_s):
                stops.append((m.check_s[i], self.at_check))
            i = bisect.bisect_right(m.junction_s, s + 1e-6)
            if i < len(m.junction_s):
                stops.append((m.junction_s[i], self.at_junction))
            if h.check is not None:
                # Search out from the check; each mark found buys another give_up_m.
                reach = max(p.check_search_m, h.check.gap_end - h.check.s + 20.0)
                limit = max(h.check.s + reach, h.last_seen + p.give_up_m)
            else:
                limit = h.last_seen + p.give_up_m
            if limit > s + 1e-6:
                stops.append((limit, self.at_give_up))
            else:
                stops.append((s, self.at_give_up))
        else:  # a false branch
            c = h.check
            opt = c.options[h.opt]
            stops = [(opt.false_s, self.at_false)]
            bl = m.branch_blobs.get(T.tid, [])
            i = bisect.bisect_right(bl, s + 1e-6)
            if i < len(bl) and bl[i] < opt.false_s:
                stops.append((bl[i], self.at_blob))
        # Ties: checks and the end beat corners and blobs at the same spot.
        order = {self.at_end: 0, self.at_check: 1, self.at_false: 1, self.at_blob: 2,
                 self.at_junction: 3, self.at_give_up: 4}
        stop, fn = min(stops, key=lambda x: (x[0], order[x[1]]))
        self.go(h, [(T, s, stop)], fn)

    def detects(self, h: Hound) -> bool:
        if h.careful:
            return True
        return self.rng.random() < h.prof.detect_prob

    def at_blob(self, h: Hound) -> None:
        if self.detects(h):
            h.careful = False
            if h.track is self.m.true:
                h.last_seen = h.s
            if h.check is not None and not h.heading:
                h.found += 1
                if h.found >= h.prof.confirm_marks:
                    self.call_on_on(h)
                    return
        self.run(h)

    def at_end(self, h: Hound) -> None:
        h.done_t = h.t
        h.check = None
        self.log(h, "beer near")

    def at_give_up(self, h: Hound) -> None:
        if h.check is not None:
            # Searched this direction from the check without enough marks.
            self.log(h, "nothing this way")
            self.go(h, [(h.track, h.s, h.check.s)], self.choose)
            return
        if h.careful:
            # Looked again and still nothing: a gap in the marking. Push on.
            h.last_seen = h.s
            h.careful = False
            self.run(h)
            return
        self.log(h, "lost, back to last mark")
        h.careful = True
        self.go(h, [(h.track, h.s, h.last_seen)], self.run)

    # -- corners
    def at_junction(self, h: Hound) -> None:
        j = self.m.junctions[bisect.bisect_left(self.m.junction_s, h.s - 1e-6)]
        following = j.last_t is not None and h.t - j.last_t <= h.prof.follow_s
        if j.straight_correct or following or (j.visible and self.detects(h)):
            j.last_t = h.t
            h.careful = False
            self.run(h)
            return
        self.wrong_turn(h, j, j.straightest_wrong, {j.straightest_wrong},
                        max(30.0, h.prof.give_up_m - (j.s - h.last_seen)))

    def wrong_turn(self, h: Hound, j: Junction, nb, tried: set, length: float) -> None:
        pr = self.m.probe(j.node, nb)
        out = min(length, pr.length)
        self.log(h, "wrong turn")

        def back(hh: Hound, j=j, tried=tried) -> None:
            hh.track, hh.s = self.m.true, j.s
            left = [n for n in j.wrong if n not in tried]
            following = j.last_t is not None and hh.t - j.last_t <= hh.prof.follow_s
            if (following or (j.visible and self.detects(hh)) or not left
                    or self.rng.random() < 1.0 / (len(left) + 1)):
                j.last_t = hh.t
                hh.careful = True
                self.run(hh)
                return
            nb2 = self.rng.choice(left)
            self.wrong_turn(hh, j, nb2, tried | {nb2}, hh.prof.check_search_m)

        self.go(h, [(pr, 0.0, out), (pr, out, 0.0)], back)

    # -- checks
    def at_check(self, h: Hound) -> None:
        c = self.m.check_at[h.s] if h.s in self.m.check_at else min(
            self.m.checks, key=lambda c: abs(c.s - h.s))
        c.arrivals.setdefault(h.idx, h.t)
        h.last_seen = c.s
        h.careful = False
        if c.solved:
            self.run(h)
            return
        self.log(h, f"check {c.idx + 1}")
        h.check, h.tried, h.found = c, set(), 0
        self.go(h, [(self.m.true, c.s, c.s, h.prof.check_pause_s)], self.choose)

    def choose(self, h: Hound) -> None:
        c = h.check
        h.track, h.s, h.found, h.heading = self.m.true, c.s, 0, False
        if c.solved:
            h.check = None
            h.last_seen = c.s
            self.run(h)
            return
        n = len(c.options)
        cands = [i for i in range(n) if i not in c.dead and i not in h.tried]
        if not cands:
            h.tried = set()
            cands = [i for i in range(n) if i not in c.dead] or [0]
        i = self.policy.choose(h, c, cands, self.rng)
        c.claimed.add(i)
        h.tried.add(i)
        h.opt = i
        opt = c.options[i]
        if opt.kind == "true":
            h.last_seen = c.s
            self.run(h)
        elif opt.kind == "false":
            h.track, h.s = opt.track, 0.0
            self.run(h)
        else:
            pr = self.m.probe(c.node, opt.nb)
            out = min(h.prof.check_search_m, pr.length)

            def back(hh: Hound, i=i) -> None:
                hh.check.dead.add(i)
                self.go(hh, [(self.m.true, hh.check.s, hh.check.s, hh.prof.check_pause_s / 2)], self.choose)

            self.go(h, [(pr, 0.0, out), (pr, out, 0.0)], back)

    def at_false(self, h: Hound) -> None:
        c = h.check
        c.dead.add(h.opt)
        self.log(h, "false")
        self.go(h, [(h.track, h.s, 0.0)], self.choose)

    def call_on_on(self, h: Hound) -> None:
        c, t = h.check, h.t
        on_true = h.track is self.m.true
        self.log(h, "On-On!" if on_true else "On-On! (down a false)")
        if on_true:
            c.solved, c.solved_t = True, t
            h.check = None
        cx, cy = (float(v) for v in h.track.xy(h.s))
        target_track, target_s = h.track, h.s
        for k in self.hounds:
            if k is h or k.check is not c or k.done_t is not None or k.heading:
                continue
            kt, ks = k.pos_at(t)
            kx, ky = (float(v) for v in kt.xy(ks))
            if not self.calls.hears((cx, cy), (kx, ky)):
                continue
            self.interrupt(k, t)
            legs = []
            if k.track is self.m.true:
                if on_true and k.s >= c.s:
                    # Already heading out the right way.
                    k.check = None
                    k.last_seen = max(k.last_seen, c.s)
                    self.run(k)
                    continue
                legs.append((self.m.true, k.s, c.s))
            else:
                legs.append((k.track, k.s, 0.0))
            start = c.s if on_true else 0.0
            legs.append((target_track, start, target_s))
            k.heading = True
            if not on_true:
                k.opt = h.opt

            def arrive(kk: Hound, on_true=on_true, ts=target_s) -> None:
                kk.heading = False
                if on_true:
                    kk.check = None
                    kk.last_seen = ts
                else:
                    kk.found = -10**6  # following a call: don't call again
                self.run(kk)

            self.go(k, legs, arrive)
        if on_true:
            self.run(h)
        else:
            h.found = -10**6  # don't call again on this branch
            self.run(h)

    # -- main loop
    def simulate(self) -> SimResult:
        m = self.m
        m.reset()
        self.hounds = []
        for i, p in enumerate(self.pack):
            pace = p.pace_s_per_m * (1 + self.rng.uniform(-p.pace_jitter, p.pace_jitter))
            h = Hound(i, p, pace, m.true, plan_start=(m.true, 0.0))
            self.hounds.append(h)
            self.run(h)
        while self.heap:
            t, _, hi, ver, fn = heapq.heappop(self.heap)
            h = self.hounds[hi]
            if ver != h.version:
                continue
            if t > MAX_SIM_S:
                break
            fn(h)
        return self._result()

    def _result(self) -> SimResult:
        m = self.m
        hounds = []
        for h in self.hounds:
            d = h.distance()
            hounds.append(HoundResult(h.prof.name, h.pace, h.prof.role, h.done_t, d, d - m.L))
        # First passage times along the true trail.
        sample_s = np.arange(0.0, m.L, 100.0)
        extra = sorted({c.s for c in m.checks} | {c.gap_end for c in m.checks})
        targets = np.array(sorted(set(sample_s.tolist()) | set(extra)))
        fp = np.vstack([first_passage(h, m.true, targets) for h in self.hounds])
        col = {float(s): i for i, s in enumerate(targets)}

        def spread(s) -> float | None:
            v = fp[:, col[float(s)]]
            v = v[~np.isnan(v)]
            return float(v.max() - v.min()) if len(v) >= 2 else None

        checks = []
        for c in m.checks:
            arr = sorted(c.arrivals.values())
            first = arr[0] if arr else None
            checks.append(CheckResult(
                c.idx, c.s, c.node, first, arr[-1] if arr else None, c.solved_t,
                (c.solved_t - first) if (c.solved_t is not None and first is not None) else None,
                spread(c.s), spread(c.gap_end)))
        gap_profile = [(float(s), g) for s in sample_s if (g := spread(s)) is not None]
        return SimResult(m.L, hounds, checks, gap_profile, self.seed, raw=self.hounds)


def first_passage(h: Hound, track: Track, targets: np.ndarray) -> np.ndarray:
    """Earliest time the hound is at each target distance along ``track`` (NaN if never)."""
    out = np.full(len(targets), np.nan)
    for l in h.legs:
        if l.track is not track or l.s0 == l.s1:
            continue
        lo, hi = min(l.s0, l.s1), max(l.s0, l.s1)
        a, b = np.searchsorted(targets, lo, "left"), np.searchsorted(targets, hi, "right")
        if a >= b:
            continue
        idx = np.arange(a, b)
        idx = idx[np.isnan(out[idx])]
        if len(idx):
            out[idx] = l.t0 + (targets[idx] - l.s0) / (l.s1 - l.s0) * (l.t1 - l.t0)
    return out


def sample_tracks(result: SimResult, dt: float = 5.0):
    """Positions of every hound every ``dt`` seconds: (times, [n_hounds, n_times, 2] lon/lat, x/y)."""
    hs = result.raw
    t_end = max((l.t1 for h in hs for l in h.legs), default=0.0)
    times = np.arange(0.0, t_end + dt, dt)
    ll = np.full((len(hs), len(times), 2), np.nan)
    xy = np.full((len(hs), len(times), 2), np.nan)
    for i, h in enumerate(hs):
        for l in h.legs:
            a, b = np.searchsorted(times, l.t0, "left"), np.searchsorted(times, l.t1, "right")
            if a >= b:
                continue
            tt = times[a:b]
            f = (tt - l.t0) / (l.t1 - l.t0) if l.t1 > l.t0 else np.zeros_like(tt)
            s = l.s0 + (l.s1 - l.s0) * f
            ll[i, a:b, 0], ll[i, a:b, 1] = l.track.lonlat(s)
            xy[i, a:b, 0], xy[i, a:b, 1] = l.track.xy(s)
        # Hold the last position (at the Beer Near) to the end.
        last = np.where(~np.isnan(ll[i, :, 0]))[0]
        if len(last):
            ll[i, last[-1]:] = ll[i, last[-1]]
            xy[i, last[-1]:] = xy[i, last[-1]]
    return times, ll, xy


def simulate(trail: Trail, graph: StreetGraph, pack: list[HoundProfile] | None = None, *,
             seed: int = 0, model: TrailModel | None = None, **kw) -> SimResult:
    """One run of the pack on a trail."""
    model = model or TrailModel(trail, graph)
    return Simulation(model, pack or default_pack(), seed=seed, **kw).simulate()


def simulate_many(trail: Trail, graph: StreetGraph, pack: list[HoundProfile] | None = None, *,
                  runs: int = 20, seed: int = 0, model: TrailModel | None = None, **kw) -> list[SimResult]:
    """``runs`` seeded runs (seed, seed+1, ...) sharing one compiled trail."""
    model = model or TrailModel(trail, graph)
    pack = pack or default_pack()
    return [Simulation(model, pack, seed=seed + i, **kw).simulate() for i in range(runs)]
