"""Tier 1 pack simulator: rule-based hounds on the street graph.

The simulation is event driven. A hound's plan is a short list of legs (run
along a track from one distance to another, or wait), and the only events are
"reached the end of the plan": a mark, a corner, a check, the give-up point,
the end of a probe. Calls interrupt the plans of hounds that hear them.

Everything a hound runs is a ``Track``: the true trail, a false or a
whichy-way's wrong leg, or a probe (a run down an unmarked street and back).
Tracks start at a node, so a hound's position is always (track, metres along).

Rules (Brian's hash rules; see ``autotrail.rules`` for "straight on"):
  * Running: follow the blobs; each is spotted with ``detect_prob`` (hares hide
    them, and nobody sees one down a side street). More than ``give_up_m``
    without a mark: go back to the last mark and look again.
  * Corners: checks, arrows, whichy-ways and F marks are bold and never missed.
    At a corner with none of them, a hound goes straight on. If trail really
    turned there (poor marking) it runs out of marks, comes back, and tries
    the other ways.
  * Checks: the first hound picks a direction right away, the next a
    different one, and so on until every direction is taken; later hounds
    wait for a call. Probes run ``check_search_m`` out and back. Marks found
    are called "On one", "On two", then "On-On" at the third
    (``confirm_marks``); a whichy-way needs only one. On-On solves the check:
    hounds within ``call_radius_m`` and every hound waiting at the check go
    to the caller, and the check is marked so late arrivals go straight on.
    A false with three marks fools hounds the same way, until someone finds
    the F; then the false is scratched and the check is open again.
  * Done: reached the Beer Near. The return leg isn't simulated yet.

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
from .rules import COVER_M, Geometry
from .trail import BranchPurpose, MarkKind, Trail

MAX_SIM_S = 4 * 3600.0
PROBE_MAX_M = 400.0
CALLS = {1: "On one", 2: "On two"}


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


# ---------------------------------------------------------------- compiled trail

@dataclass
class Option:
    kind: str              # "true", "false", "probe"
    nb: object             # neighbour node the option heads to
    track: Track | None = None
    false_s: float = 0.0   # where the F mark is on a false or wrong leg


@dataclass
class Check:
    idx: int
    s: float
    node: object
    gap_end: float                      # first mark after the check on the true trail
    kind: str = "check"                 # "check" or "whichy"
    confirm: int | None = None          # marks needed to call On-On (None: the hound's own)
    fake: bool = False                  # trail goes straight on
    options: list[Option] = field(default_factory=list)
    # per-run state
    solved_t: float | None = None
    pointer: int | None = None          # the way the check is marked (after an On-On)
    claimed: dict = field(default_factory=dict)   # option -> number of hounds on it
    dead: set = field(default_factory=set)
    waiters: list = field(default_factory=list)
    arrivals: dict = field(default_factory=dict)  # hound idx -> first arrival time

    @property
    def solved(self) -> bool:
        return self.solved_t is not None

    @property
    def label(self) -> str:
        return ("whichy-way " if self.kind == "whichy" else "check ") + str(self.idx + 1)

    def reset(self) -> None:
        self.solved_t, self.pointer = None, None
        self.claimed, self.dead, self.waiters, self.arrivals = {}, set(), [], {}


@dataclass
class Junction:
    """A corner on the true trail that matters: an arrow, or an unmarked turn."""
    s: float
    node: object
    nxt: object
    arrow: bool
    straight: object     # where an unmarked corner sends a hound (None: unclear)
    ways: list           # every way on except back


class TrailModel:
    """A Trail compiled for fast simulation."""

    def __init__(self, trail: Trail, graph: StreetGraph):
        self.trail, self.g = trail, graph
        self.geo = Geometry(graph)
        self.warnings: list[str] = []
        self._next_tid = 0
        self._probes: dict[tuple, Track] = {}

        tp = [(e.u, e.v, e.key) for e in trail.true_path]
        self.true = self._track("true", tp, label="true trail")
        self.L = self.true.length
        cum = self.true.cum
        nodes_along = [(float(cum[i]), e[0], i) for i, e in enumerate(tp)][1:]
        node_s = {}  # node -> first s on the true path
        turn_s = {}  # (node, next node) -> s: tells apart two passes through one node
        for s, n, i in nodes_along:
            node_s.setdefault(n, s)
            turn_s.setdefault((n, tp[i][1]), s)

        fwd_idx: dict[tuple, int] = {}
        rev_idx: dict[tuple, int] = {}
        for i, e in enumerate(tp):
            fwd_idx.setdefault(e, i)
            rev_idx.setdefault((e[1], e[0]), i)

        def true_s(m) -> float | None:
            at = m.at
            if at.edge is not None:
                e = (at.edge.u, at.edge.v, at.edge.key)
                if e in fwd_idx:
                    return float(cum[fwd_idx[e]] + at.offset_m)
                if (e[0], e[1]) in rev_idx:
                    i = rev_idx[(e[0], e[1])]
                    return float(cum[i] + max(0.0, graph.edge_length(*tp[i]) - at.offset_m))
                return None
            to = m.params.get("to")
            if to is not None and (at.node, to) in turn_s:
                return turn_s[(at.node, to)]
            return node_s.get(at.node)

        def snap_node(s: float):
            """Nearest corner node along the true path: (s, node, edge index leaving it)."""
            return min(nodes_along, key=lambda t: abs(t[0] - s)) if nodes_along else None

        # Side legs (falses and whichy-way wrong legs), and marks on them.
        side: list[tuple[Track, object]] = []
        side_edge: dict[tuple, tuple[Track, int]] = {}
        for b in trail.branches:
            if b.purpose != BranchPurpose.FALSE:
                self.warnings.append(f"{b.purpose.value.title()} branches are phase 2; ignored by the sim")
                continue
            be = [(e.u, e.v, e.key) for e in b.edges]
            t = self._track("false", be, label=b.label or "false")
            side.append((t, b.from_node))
            for i, e in enumerate(be):
                side_edge.setdefault(e, (t, i))

        blobs: list[float] = []
        side_blobs: dict[int, list[float]] = {}
        side_f: dict[int, float] = {}
        check_specs: list[tuple[float, str, dict]] = []
        arrow_s: set[float] = set()
        skipped: dict[str, int] = {}
        for m in trail.marks:
            if m.params.get("leg") == "return":
                continue
            s = true_s(m)
            on_side = None
            if s is None and m.at.edge is not None:
                e = (m.at.edge.u, m.at.edge.v, m.at.edge.key)
                if e in side_edge:
                    t, i = side_edge[e]
                    on_side = (t, float(t.cum[i] + m.at.offset_m))
            if m.kind == MarkKind.BLOB:
                if s is not None:
                    blobs.append(s)
                elif on_side:
                    side_blobs.setdefault(on_side[0].tid, []).append(on_side[1])
            elif m.kind == MarkKind.ARROW and s is not None:
                snapped = snap_node(s)
                if snapped and abs(snapped[0] - s) <= 30.0:
                    arrow_s.add(snapped[0])
                else:
                    blobs.append(s)
            elif m.kind in (MarkKind.CHECK, MarkKind.WHICHY_WAY) and s is not None:
                check_specs.append((s, "whichy" if m.kind == MarkKind.WHICHY_WAY else "check", m.params))
            elif m.kind == MarkKind.FALSE and on_side:
                t, bs = on_side
                side_f[t.tid] = min(side_f.get(t.tid, math.inf), bs)
            elif m.kind == MarkKind.BEER:
                continue
            else:
                skipped[m.kind.value] = skipped.get(m.kind.value, 0) + 1
        for k, n in skipped.items():
            self.warnings.append(f"{n} {k} mark(s) not simulated yet (phase 2 marks, or off the trail)")
        self.blobs = sorted(blobs)
        self.side_blobs = {k: sorted(v) for k, v in side_blobs.items()}

        # Checks and whichy-ways snap to the nearest corner of the true path.
        self.checks: list[Check] = []
        used = set()
        for s, kind, params in sorted(check_specs, key=lambda t: t[0]):
            snapped = snap_node(s)
            if snapped is None or snapped[1] in used:
                continue
            cs, cn, i = snapped
            used.add(cn)
            j = bisect.bisect_right(self.blobs, cs + 0.5)
            gap_end = self.blobs[j] if j < len(self.blobs) else self.L
            prev, nxt = tp[i - 1][0], tp[i][1]
            c = Check(0, cs, cn, gap_end, kind=kind, confirm=1 if kind == "whichy" else None,
                      fake=self.geo.straight_on(prev, cn) == nxt)
            opts = [Option("true", nxt)]
            side_nbs = set()
            for t, frm in side:
                if frm == cn:
                    opts.append(Option("false", t.edges[0][1], t, min(side_f.get(t.tid, t.length), t.length)))
                    side_nbs.add(t.edges[0][1])
            if kind == "check":
                for nb in self.geo.ways_on(prev, cn):
                    if nb != nxt and nb not in side_nbs:
                        opts.append(Option("probe", nb))
            c.options = opts
            self.checks.append(c)
        self.checks.sort(key=lambda c: c.s)
        for k, c in enumerate(self.checks):
            c.idx = k
        self.check_at = {c.s: c for c in self.checks}
        self.check_s = [c.s for c in self.checks]
        for t, frm in side:
            if frm not in used:
                self.warnings.append(f"'{t.label}' doesn't start at a check or whichy-way; ignored by the sim")
        # The hare runs every false and wrong leg out to its F and back.
        self.side_detours = [(c.s, 2 * o.false_s) for c in self.checks for o in c.options if o.kind == "false"]

        # Corners that matter: arrows (marks to count) and unmarked turns (where hounds go wrong).
        self.junctions: list[Junction] = []
        self.unmarked_turns: list[float] = []
        self.arrow_nodes: list = []
        last_mark = -math.inf
        for s, node, i in nodes_along:
            if s in self.check_at:
                last_mark = s
                continue
            prev, nxt = tp[i - 1][0], tp[i][1]
            ways = self.geo.ways_on(prev, node)
            arrow = s in arrow_s
            straight = self.geo.straight_on(prev, node) if ways else None
            if arrow:
                self.arrow_nodes.append(node)
                last_mark = s
            elif straight == nxt or not ways or s - last_mark <= COVER_M:
                continue  # straight on, or the same street corner as the last mark
            else:
                self.unmarked_turns.append(s)
            self.junctions.append(Junction(s, node, nxt, arrow, straight, ways))
        self.junction_s = [j.s for j in self.junctions]
        if self.unmarked_turns:
            self.warnings.append(f"{len(self.unmarked_turns)} turn(s) with no check, arrow or whichy-way: "
                                 "hounds will go straight on there")
        if not self.blobs:
            self.warnings.append("no blobs on the true trail: hounds will get lost (add marks first)")

    # -- helpers
    def _track(self, kind, edges, max_len=None, label="") -> Track:
        t = Track(self._next_tid, kind, edges, self.geo, max_len=max_len, label=label)
        self._next_tid += 1
        return t

    def probe(self, node, nb) -> Track:
        """Unmarked run from node via nb, going straight on at each corner."""
        key = (node, nb)
        t = self._probes.get(key)
        if t is None:
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
                ways = [n for n in self.geo.ways_on(prev, cur) if n not in seen]
                if not ways:
                    break
                nxt = self.geo.straight_on(prev, cur)
                if nxt is None or nxt in seen:
                    nxt = min(ways, key=lambda n, p=prev, c=cur: self.geo.deviation(p, c, n))
                prev, cur = cur, nxt
            t = self._probes[key] = self._track("probe", edges, label=f"probe {node}->{nb}")
        return t

    def reset(self) -> None:
        for c in self.checks:
            c.reset()


# ---------------------------------------------------------------- extension points

class CheckPolicy(Protocol):
    def choose(self, hound: "Hound", check: Check, free: list[int], rng: random.Random) -> int | None:
        """Pick a direction from the untaken ones, or None to wait for a call."""


class FirstHoundsCheck:
    """Tier 1: each hound takes a direction nobody has; once all are taken, wait for a call."""

    def choose(self, hound, check, free, rng):
        return rng.choice(free) if free else None


class CallModel(Protocol):
    def hears(self, caller_xy: tuple[float, float], hound_xy: tuple[float, float]) -> bool: ...


@dataclass
class RadiusCall:
    """Tier 1: everyone within a straight-line radius hears a call."""
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
    waiting: bool = False
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
    kind: str
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
    gap_profile: list[tuple[float, float]]    # (metres along trail, front-to-back seconds)
    front_profile: list[tuple[float, float]]  # (metres along trail, when the first hound got there)
    side_detours: list[tuple[float, float]]   # (metres along trail, metres the hare runs out and back)
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
        self.policy = policy or FirstHoundsCheck()
        self.calls = calls or RadiusCall()
        self.rng = random.Random(seed)
        self.heap: list = []
        self._seq = 0
        self.now = 0.0

    # -- plumbing
    def go(self, h: Hound, legs: list[tuple], then: Callable[[Hound], None]) -> None:
        """Queue legs [(track, s0, s1) or (track, s, s, wait_s)] and call ``then`` at the end."""
        h.waiting = False
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
        if h.waiting:
            # Standing at the check since h.t: record the wait.
            if t > h.t:
                h.legs.append(Leg(h.track, h.s, h.s, h.t, t))
            h.plan_i = len(h.legs)
            h.plan_start = (h.track, h.s)
            h.t, h.waiting = t, False
            h.version += 1
            return
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
                # Searching out from a check; each mark found buys another give_up_m.
                reach = max(p.check_search_m, h.check.gap_end - h.check.s + 20.0)
                limit = max(h.check.s + reach, h.last_seen + p.give_up_m)
            else:
                limit = h.last_seen + p.give_up_m
            stops.append((max(limit, s), self.at_give_up))
        else:  # a false or a whichy-way's wrong leg
            opt = h.check.options[h.opt]
            stops = [(opt.false_s, self.at_false)]
            bl = m.side_blobs.get(T.tid, [])
            i = bisect.bisect_right(bl, s + 1e-6)
            if i < len(bl) and bl[i] < opt.false_s:
                stops.append((bl[i], self.at_blob))
        order = {self.at_end: 0, self.at_check: 1, self.at_false: 1, self.at_junction: 2,
                 self.at_blob: 3, self.at_give_up: 4}
        stop, fn = min(stops, key=lambda x: (x[0], order[x[1]]))
        self.go(h, [(T, s, stop)], fn)

    def confirm_needed(self, h: Hound) -> int:
        c = h.check
        return c.confirm if c is not None and c.confirm is not None else h.prof.confirm_marks

    def found_mark(self, h: Hound) -> bool:
        """Count a mark found past a check. Returns True if the hound called On-On."""
        if h.check is None or h.heading:
            return False
        h.found += 1
        need = self.confirm_needed(h)
        if h.found >= need:
            self.call_on_on(h)
            return True
        if h.found > 0:
            self.log(h, CALLS.get(h.found, f"On {h.found}"))
        return False

    def at_blob(self, h: Hound) -> None:
        if h.careful or self.rng.random() < h.prof.detect_prob:
            h.careful = False
            if h.track is self.m.true:
                h.last_seen = h.s
            if self.found_mark(h):
                return
        self.run(h)

    def at_end(self, h: Hound) -> None:
        h.done_t = h.t
        self.log(h, "beer near")
        if h.check is not None and not h.heading:
            # Reached the Beer Near from a check before the third mark: that solves it.
            self.call_on_on(h)
        h.check = None

    def at_give_up(self, h: Hound) -> None:
        c = h.check
        if c is not None:
            # Searched this way from the check without enough marks.
            self.log(h, "nothing this way")
            self.release(c, h.opt)
            self.go(h, [(h.track, h.s, c.s)], self.choose)
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
        if j.arrow:
            h.last_seen = h.s
            h.careful = False
            if self.found_mark(h):
                return
            self.run(h)
            return
        # An unmarked corner where trail turns: go straight on (or guess, if there's no straight).
        way = j.straight if j.straight is not None else self.rng.choice(j.ways)
        if way == j.nxt:
            self.run(h)
            return
        self.wrong_turn(h, j, way, {way}, max(30.0, h.prof.give_up_m - (j.s - h.last_seen)))

    def wrong_turn(self, h: Hound, j: Junction, nb, tried: set, length: float) -> None:
        pr = self.m.probe(j.node, nb)
        out = min(length, pr.length)
        self.log(h, "no marks this way")

        def back(hh: Hound, j=j, tried=tried) -> None:
            hh.track, hh.s = self.m.true, j.s
            left = [n for n in j.ways if n not in tried and n != j.nxt]
            if not left or self.rng.random() < 1.0 / (len(left) + 1):
                hh.careful = True
                self.run(hh)
                return
            nb2 = self.rng.choice(left)
            self.wrong_turn(hh, j, nb2, tried | {nb2}, hh.prof.check_search_m)

        self.go(h, [(pr, 0.0, out), (pr, out, 0.0)], back)

    # -- checks
    def at_check(self, h: Hound) -> None:
        c = self.m.check_at[h.s]
        if h.check is not None and h.check is not c and not h.heading and h.track is self.m.true:
            # Missed a blob but reached the next check: that's trail, so the last check is solved.
            self.call_on_on(h, resume=False)
        c.arrivals.setdefault(h.idx, h.t)
        h.last_seen = c.s
        h.careful = False
        h.check, h.tried, h.found = c, set(), 0
        if c.pointer is not None:
            self.follow_pointer(h)
            return
        self.log(h, c.label)
        self.go(h, [(self.m.true, c.s, c.s, h.prof.check_pause_s)], self.choose)

    def follow_pointer(self, h: Hound) -> None:
        """The check is marked: go the way it points."""
        c = h.check
        h.opt = c.pointer
        opt = c.options[c.pointer]
        h.found = -10**6  # following, not checking: no calls
        if opt.kind == "true":
            h.check = None
            h.track, h.s, h.last_seen = self.m.true, c.s, c.s
        else:
            h.track, h.s = opt.track, 0.0
        self.run(h)

    def release(self, c: Check, i: int) -> None:
        """A hound is done with direction i; if nobody else is on it, a waiter can take it."""
        n = c.claimed.get(i, 0) - 1
        if n > 0:
            c.claimed[i] = n
        else:
            c.claimed.pop(i, None)
        if c.waiters and i not in c.dead and i not in c.claimed:
            w = c.waiters.pop(0)
            self.interrupt(w, self.now)
            self.choose(w)

    def choose(self, h: Hound) -> None:
        c = h.check
        h.track, h.s, h.heading = self.m.true, c.s, False
        h.found = 0
        if c.pointer is not None:
            self.follow_pointer(h)
            return
        n = len(c.options)
        free = [i for i in range(n) if i not in c.dead and i not in c.claimed and i not in h.tried]
        if not free and all(i in c.dead or i in h.tried for i in range(n)):
            # Tried everything open to me (maybe missed a blob): start over on what's not scratched.
            h.tried = set()
            free = [i for i in range(n) if i not in c.dead and i not in c.claimed]
        i = self.policy.choose(h, c, free, self.rng)
        if i is None:
            h.waiting = True
            c.waiters.append(h)
            return
        c.claimed[i] = c.claimed.get(i, 0) + 1
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
                self.release(hh.check, i)
                self.go(hh, [(self.m.true, hh.check.s, hh.check.s, hh.prof.check_pause_s / 2)], self.choose)

            self.go(h, [(pr, 0.0, out), (pr, out, 0.0)], back)

    def at_false(self, h: Hound) -> None:
        c = h.check
        i = h.opt
        if i not in c.dead:
            self.log(h, "F! back to the check")
        c.dead.add(i)
        if c.pointer == i:
            c.pointer = None  # scratch the arrow at the check
        self.release(c, i)
        self.go(h, [(h.track, h.s, 0.0)], self.choose)

    def call_on_on(self, h: Hound, resume: bool = True) -> None:
        """h calls On-On for its check. With ``resume`` False, the caller moves h on itself."""
        c, t = h.check, h.t
        on_true = h.track is self.m.true
        self.log(h, f"On-On! ({c.label})" if on_true else f"On-On! (down a false, {c.label})")
        h.found = -10**6  # one call is enough
        if c.solved:
            # Already solved and marked: the pack has gone; nobody follows this one.
            if on_true:
                h.check = None
            if resume and h.done_t is None:
                self.run(h)
            return
        c.pointer = h.opt
        if on_true:
            c.solved_t = t
            c.claimed.clear()
            h.check = None
        cx, cy = (float(v) for v in h.track.xy(h.s))
        target_track, target_s = h.track, h.s
        waiters = list(c.waiters)
        c.waiters.clear()
        for k in self.hounds:
            if k is h or k.check is not c or k.done_t is not None or k.heading:
                continue
            if k not in waiters:
                kt, ks = k.pos_at(t)
                kx, ky = (float(v) for v in kt.xy(ks))
                if not self.calls.hears((cx, cy), (kx, ky)):
                    continue
            self.interrupt(k, t)
            if k.track is self.m.true and on_true and k.s >= c.s:
                # Already out the right way.
                k.check = None
                k.last_seen = max(k.last_seen, c.s)
                k.found = -10**6
                self.run(k)
                continue
            if k not in waiters and k.opt in c.claimed:
                self.release(c, k.opt)  # nobody is waiting now, so this wakes no one
            if k.track is target_track and k.s >= target_s:
                legs = []
            elif k.track is target_track:
                legs = [(k.track, k.s, target_s)]
            else:
                legs = [(k.track, k.s, 0.0)] if k.track is not self.m.true else [(self.m.true, k.s, c.s)]
                legs.append((target_track, c.s if on_true else 0.0, target_s))
            k.heading = True
            k.opt = h.opt

            def arrive(kk: Hound, on_true=on_true, ts=target_s) -> None:
                kk.heading = False
                kk.found = -10**6  # following a call: don't call again
                if on_true:
                    kk.check = None
                    kk.last_seen = ts
                self.run(kk)

            if legs:
                self.go(k, legs, arrive)
            else:
                arrive(k)
        if resume and h.done_t is None:
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
            self.now = t
            fn(h)
        for h in self.hounds:
            if h.waiting and h.done_t is None:
                self.interrupt(h, max(self.now, h.t))
        return self._result()

    def _result(self) -> SimResult:
        m = self.m
        hounds = []
        for h in self.hounds:
            d = h.distance()
            hounds.append(HoundResult(h.prof.name, h.pace, h.prof.role, h.done_t, d, d - m.L))
        sample_s = np.arange(0.0, m.L, 100.0)
        extra = sorted({c.s for c in m.checks} | {c.gap_end for c in m.checks} | {m.L})
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
                c.idx, c.s, c.node, c.kind, first, arr[-1] if arr else None, c.solved_t,
                (c.solved_t - first) if (c.solved_t is not None and first is not None) else None,
                spread(c.s), spread(c.gap_end)))
        gap_profile = [(float(s), g) for s in sample_s if (g := spread(s)) is not None]
        front = []
        for i, s in enumerate(targets):
            v = fp[:, i]
            v = v[~np.isnan(v)]
            if len(v):
                front.append((float(s), float(v.min())))
        return SimResult(m.L, hounds, checks, gap_profile, front, list(m.side_detours), self.seed,
                         raw=self.hounds)


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
