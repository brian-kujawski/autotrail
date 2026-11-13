"""One-click trails: a route search, then mark placement, following real hash rules.

Stage 1 (route): a random-restart weighted walk from Start that bends toward
the Beer Near as its distance budget runs down. It never revisits a corner and
never comes within ``keep_apart_m`` of trail it laid earlier (no crossing, no
running back down the other sidewalk). It closes with a shortest path that
obeys the same rule. The leg back to Start is the shortest path that keeps
clear of the trail. Without a Beer Near, one is picked about ``return_miles``
from Start.

Stage 2 (marks): every turn gets a check or an arrow (hounds go straight on at
unmarked corners; a jog of under 20 m is straight on). A share of the turns
are checks, at least ``min_check_gap_m`` apart. To keep hounds honest, some
checks and whichy-ways have straight on as the answer: any stretch of
``dull_m`` with no check or whichy-way may get one at a crossing, and now and
then a run of short blocks gets a whichy-way at every crossing (straight,
straight, straight... then the turn). No ratio is targeted. A whichy-way or two
may replace an arrow; its wrong leg ends in an F after one mark. Zero to two
falses hang off checks: a couple of blocks of close, obvious marks, then an F.
Blobs every ~50 m, with a short unmarked gap after each check. The hare runs
the trail plus every false and wrong leg out and back, so those cost the hare
time; the score checks the pack doesn't catch the hare.

Each candidate is simulated and the best total score wins. Hill-climbing the
marks is phase 2.
"""

from __future__ import annotations

import bisect
import heapq
import math
import random
from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np

from .graph import StreetGraph
from .pack import HareProfile, HoundProfile, default_pack
from .rules import COVER_M, STRAIGHT_DEG, Footprint, Geometry, is_sidewalk, jog_corners
from .score import DistanceFit, ScoreCard, score_results
from .sight import PackTiming, SightField, SightLines, check_looks_at_hare
from .sim import SimResult, TrailModel, simulate_many
from .trail import METRES_PER_MILE, Branch, BranchPurpose, EdgePos, EdgeRef, Mark, MarkKind, Trail


@dataclass
class GenerateOptions:
    miles: float = 5.0                # true trail, Start to Beer Near
    return_miles: float = 1.0         # roughly how far the Beer Near is from Start
    tries: int = 60                   # route attempts
    candidates: int = 4               # best routes that get marks and a sim
    placements: int = 5               # mark layouts tried per route
    runs: int = 8                     # sim runs per candidate
    keep_apart_m: float = 35.0        # trail never comes this close to itself
    blob_every_m: float = 50.0
    check_share: float = 0.5          # share of turns that get a check (the rest get arrows)
    min_check_gap_m: float = 300.0
    dull_m: float = 400.0             # a stretch this long with no check or whichy-way is dull
    fill_share: float = 0.75          # chance a dull stretch gets a straight-on check or whichy-way
    straight_whichy_share: float = 0.3  # share of those that are whichy-ways
    fill_gap_m: float = 150.0         # a straight-on check or whichy-way keeps this far from others
    whichy_run_share: float = 0.15    # chance of a run of straight whichy-ways ending in a turn
    check_gap_m: tuple[float, float] = (30.0, 70.0)  # unmarked stretch after a check
    falses: tuple[int, int] = (0, 2)
    false_len_m: tuple[float, float] = (150.0, 300.0)
    false_mark_every_m: float = 25.0
    whichy_ways: tuple[int, int] = (0, 1)
    whichy_leg_m: tuple[float, float] = (50.0, 100.0)
    exact_ends: bool = False          # Start/Beer Near were picked by hand: keep them unless a dead end
    turn_cost_m: float = 120.0        # route search: leaving a corner other than straight on costs this
    straight_run_m: float = 400.0     # route search: a straight longer than this...
    straight_cost: float = 3.0        # ...costs this much extra per metre past it (long straights spread the pack)
    sight_cost: float = 3.0           # each metre the pack could see the hare run costs this many extra
    risky_check_share: float = 0.2    # chance a turn gets a check when checkers would look at the hare


# ---------------------------------------------------------------- local network

class Network:
    """Adjacency lists for the part of the graph within reach of Start."""

    def __init__(self, g: StreetGraph, centre, radius_m: float, streets_only: bool = False):
        self.g = g
        G = g.G
        reach = self._reach(G, centre, radius_m)
        self.adj: dict = {n: [] for n in reach}
        self.radj: dict = {n: [] for n in reach}
        for u in reach:
            for v, keys in G[u].items():
                if v == u or v not in reach:
                    continue
                ks = [k for k, d in keys.items() if not (streets_only and is_sidewalk(d))]
                if not ks:
                    continue
                k = min(ks, key=lambda k: keys[k]["length"])
                d = keys[k]
                length = float(d["length"])
                # Alleys and parking lots are fair game for a hash, just not first choice.
                # Sidewalks are kept (where OSM maps them separately, the street itself is
                # often left out of the walking network) but a centreline is preferred.
                cost = length * (1.3 if _hw(d) == "service" else 1.5 if is_sidewalk(d) else 1.0)
                self.adj[u].append((v, k, length, cost))
                self.radj[v].append((u, k, length, cost))
        self.nodes = {n for n in reach if self.adj[n]}

    @staticmethod
    def _reach(G, src, cutoff):
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            for v, keys in G[u].items():
                nd = d + min(float(e["length"]) for e in keys.values())
                if nd <= cutoff and nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    def dijkstra(self, src, cutoff=math.inf, reverse=False):
        adj = self.radj if reverse else self.adj
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, k, l, _ in adj.get(u, ()):
                nd = d + l
                if nd <= cutoff and nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    def clear_path(self, a, b, fp: Footprint, tag: str, pos0: float, banned: set, *,
                   max_len: float = math.inf, rng: random.Random | None = None, noise: float = 0.0,
                   came_from=None, turn_cost_m: float = 0.0, max_expand: int = 80000,
                   sight: SightField | None = None, sight_cost: float = 0.0,
                   straight_run_m: float = math.inf, straight_cost: float = 0.0,
                   run0: float = 0.0) -> list[tuple] | None:
        """A* a -> b that avoids ``banned`` nodes and keeps clear of the footprint.

        The heuristic is straight-line distance. Leaving a corner any way but
        straight on costs ``turn_cost_m`` extra (each such turn needs a mark),
        so routes run along streets instead of staircasing. With ``noise`` > 0
        each street costs up to (1 + noise) times its length, drawn at random,
        so repeated searches find different, still sensible, routes. With a
        ``sight`` field, each metre the pack could see the hare run costs
        ``sight_cost`` metres extra. Each metre of a straight run past
        ``straight_run_m`` costs ``straight_cost`` extra (the run is tracked
        along the best path to each state, so this is a heuristic nudge).
        """
        if a == b:
            return []
        geo = fp.geo
        bx, by = geo.node_xy(b)

        def h(n):
            x, y = geo.node_xy(n)
            return math.hypot(x - bx, y - by)

        factor: dict = {}
        s0 = (a, came_from)
        cost = {s0: 0.0}
        real = {s0: 0.0}
        run = {s0: run0}
        back = {}
        pq = [(h(a), 0.0, s0)]
        expanded = 0
        goal = None
        while pq:
            _, d, st = heapq.heappop(pq)
            u, p = st
            if u == b:
                goal = st
                break
            if d > cost.get(st, math.inf):
                continue
            expanded += 1
            if expanded > max_expand:
                return None
            ru = real[st]
            straight = geo.straight_on(p, u) if ((turn_cost_m or straight_cost) and p is not None) else None
            for v, k, l, c in self.adj.get(u, ()):
                if v == p or (v in banned and v != b):
                    continue
                rl = ru + l
                if rl > max_len:
                    continue
                f = 1.0
                if noise and rng is not None:
                    f = factor.get((u, v))
                    if f is None:
                        f = factor[(u, v)] = factor[(v, u)] = 1.0 + noise * rng.random()
                nd = d + c * f
                if turn_cost_m and p is not None and v != straight and len(self.adj.get(u, ())) > 2:
                    nd += turn_cost_m
                rn = l
                if straight_cost:
                    # A straightaway is what the pack sees, not what the marks say: any way on within
                    # STRAIGHT_DEG of the way in keeps the run going (a crossing with a parallel
                    # sidewalk or median has no single "straight on", but the street is still straight).
                    rn = run[st] + l if (p is None or geo.deviation(p, u, v) <= STRAIGHT_DEG) else l
                    nd += straight_cost * (max(0.0, rn - straight_run_m) - max(0.0, rn - l - straight_run_m))
                nst = (v, u)
                if nd >= cost.get(nst, math.inf):
                    continue
                if fp.clashes((u, v, k), tag, pos0 + ru):
                    continue
                if sight is not None and sight_cost:
                    nd += sight_cost * sight.exposed_m((u, v, k), pos0 + ru)
                    if nd >= cost.get(nst, math.inf):
                        continue
                cost[nst] = nd
                real[nst] = rl
                run[nst] = rn
                back[nst] = (st, k)
                heapq.heappush(pq, (nd + h(v), nd, nst))
        if goal is None:
            return None
        out, st = [], goal
        while st != s0:
            pst, k = back[st]
            out.append((pst[0], st[0], k))
            st = pst
        out.reverse()
        nodes = [a] + [e[1] for e in out]
        return out if len(set(nodes)) == len(nodes) else None

    def nearest(self, x: float, y: float, geo: Geometry, min_ways: int = 3):
        """The network corner nearest a point in local metres (with at least ``min_ways`` streets)."""
        if not hasattr(self, "_xy"):
            ids = [n for n in self.nodes if len(self.adj[n]) >= min_ways]
            self._ids = ids
            self._xy = np.array([geo.node_xy(n) for n in ids])
        d2 = (self._xy[:, 0] - x) ** 2 + (self._xy[:, 1] - y) ** 2
        return self._ids[int(np.argmin(d2))]


# ---------------------------------------------------------------- stage 1: routes

def _hw(d: dict) -> str:
    hw = d.get("highway")
    return str(hw[0] if isinstance(hw, list) else hw)


def _und(e) -> tuple:
    return (e[0], e[1]) if str(e[0]) <= str(e[1]) else (e[1], e[0])


def path_len(g: StreetGraph, edges) -> float:
    return sum(g.edge_length(*e) for e in edges)


def loop_route(net: Network, geo: Geometry, start, end, target_m: float, rng: random.Random,
               radius_m: float, keep_apart_m: float = 25.0, noise: float = 0.25,
               turn_cost_m: float = 120.0, timing: PackTiming | None = None, sight_cost: float = 0.0,
               straight_run_m: float = math.inf, straight_cost: float = 0.0, stats: dict | None = None):
    """One trail Start -> end shaped as a loop through two or three random waypoints.

    Waypoints sit ``radius_m`` (jittered) out from Start, at increasing angles
    measured from the direction of the Beer Near, so Start -> waypoints -> Beer
    Near -> back to Start goes round once and never has to cross itself. Each
    stretch is an A* search that keeps clear of trail already laid and, given
    ``timing``, shies away from streets the pack could see the hare on.

    Returns (edges, footprint) or None.
    """
    sx, sy = geo.node_xy(start)
    ex, ey = geo.node_xy(end)
    base = math.atan2(ey - sy, ex - sx) if end != start else rng.uniform(0, 2 * math.pi)
    side = rng.choice((-1.0, 1.0))
    n_wp = rng.choice((2, 2, 3))
    angles = sorted(rng.uniform(40, 320) for _ in range(n_wp))
    if any(b - a < 50 for a, b in zip(angles, angles[1:])):
        return None
    wps = []
    for ang in angles:
        r = radius_m * rng.uniform(0.75, 1.25)
        th = base + side * math.radians(ang)
        wp = net.nearest(sx + r * math.cos(th), sy + r * math.sin(th), geo)
        if wp in (start, end) or wp in wps:
            return None
        wps.append(wp)
    fp = Footprint(geo, keep_apart_m)
    sight = SightField(geo, timing) if (timing is not None and sight_cost) else None
    edges: list[tuple] = []
    visited = {start}
    walked = 0.0
    stops = wps + [end]
    for i, goal in enumerate(stops):
        left = target_m * 1.25 - walked
        kw = dict(max_len=left, rng=rng, noise=noise, came_from=edges[-1][0] if edges else None,
                  turn_cost_m=turn_cost_m)
        here = edges[-1][1] if edges else start
        banned = visited - ({edges[-1][1]} if edges else set())
        nudges = dict(sight=sight, sight_cost=sight_cost, straight_run_m=straight_run_m, straight_cost=straight_cost)
        tries = [nudges]
        if sight_cost or straight_cost:
            # The nudges make the search wander further before it finds the goal: give it room
            # before falling back to no nudges (they're preferences, never the reason there's no route).
            tries += [nudges | {"max_expand": 400000}, {}]
        seg = None
        for k, extra in enumerate(tries):
            seg = net.clear_path(here, goal, fp, "true", walked, banned, **kw, **extra)
            # The search checks against earlier stretches only; this one mustn't touch itself either.
            if seg is not None and not fp.copy().path_clashes(seg, "true", walked):
                if stats is not None and k:
                    stats["unnudged" if not extra else "wider"] = stats.get("unnudged" if not extra else "wider", 0) + 1
                break
            seg = None
        if seg is None:
            return None
        for e in seg:
            fp.add(e, "true", walked)
            if sight is not None:
                sight.add_edge(e, walked)
            walked += net.g.edge_length(*e)
            visited.add(e[1])
        edges += seg
    return edges, fp


# ---------------------------------------------------------------- stage 2: marks

def _pos(edges: list[tuple], cum: np.ndarray, s: float) -> EdgePos:
    i = min(max(0, bisect.bisect_right(cum, s) - 1), len(edges) - 1)
    return EdgePos(EdgeRef(*edges[i]), float(min(s - cum[i], cum[i + 1] - cum[i])))


def _cum(g: StreetGraph, edges) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum([g.edge_length(*e) for e in edges])])


@dataclass
class Corner:
    i: int            # edge index the trail leaves by
    s: float          # metres along the path
    node: object
    prev: object
    nxt: object
    ways: list        # every way on except back
    straight: object  # where an unmarked corner sends a hound (None: no clear straight on)
    jog: bool = False  # part of a jog (two turns that cancel out within JOG_M): straight on

    @property
    def turn(self) -> bool:
        return not self.jog and self.straight != self.nxt

    @property
    def crossing(self) -> bool:
        """A plain corner trail goes straight through, with another way to check."""
        return not self.jog and not self.turn and len(self.ways) >= 2


def corners(geo: Geometry, edges: list[tuple], cum: np.ndarray) -> list[Corner]:
    """Every point on the path where a hound has a choice."""
    out = []
    jogs = jog_corners(geo, edges, cum)
    for i in range(1, len(edges)):
        node, prev, nxt = edges[i][0], edges[i - 1][0], edges[i][1]
        ways = geo.ways_on(prev, node)
        if len(ways) >= 2 or (ways and ways[0] != nxt):
            out.append(Corner(i, float(cum[i]), node, prev, nxt, ways, geo.straight_on(prev, node), i in jogs))
    return out


def blob_positions(L: float, every_m: float, rng: random.Random,
                   gaps: list[tuple[float, float]] = (), extra: list[float] = ()) -> list[float]:
    """Blob distances along a path: regular spacing, none inside gaps, plus ``extra``."""
    gaps = sorted(gaps)

    def gap_at(s):
        return next((b for a, b in gaps if a - 1e-6 <= s < b - 1e-6), None)

    out = list(extra) + [b for _, b in gaps if b < L]
    s = min(20.0, L / 2)
    while s < L - 10:
        end = gap_at(s)
        if end is not None:
            s = end + every_m * rng.uniform(0.8, 1.2)
            continue
        if all(abs(s - x) > 12.0 for x in out):
            out.append(s)
        s += every_m * rng.uniform(0.8, 1.2)
    return sorted(set(round(x, 2) for x in out if 0 < x < L))


def _side_leg(g: StreetGraph, geo: Geometry, fp: Footprint, node, nb, length_m: float,
              tag: str) -> tuple[list[tuple], float] | None:
    """Straight-on path from node via nb that keeps clear of trail; (edges, usable length)."""
    edges, total = [], 0.0
    prev, cur = node, nb
    seen = {node}
    probe = fp.copy()
    probe.zones.append((*geo.node_xy(node), 2 * fp.r))
    while True:
        e = (prev, cur, g.best_key(prev, cur))
        if probe.clashes(e, tag, total):
            break
        probe.add(e, tag, total)
        edges.append(e)
        total += g.edge_length(*e)
        seen.add(cur)
        if total >= length_m:
            break
        nxt = geo.straight_on(prev, cur)
        if nxt is None or nxt in seen:
            break  # the hare would have to mark a turn on a false: keep it simple
        prev, cur = cur, nxt
    if not edges:
        return None
    return edges, min(length_m, total - 1.0)


def _arrow(node, nxt, **params) -> Mark:
    return Mark(MarkKind.ARROW, EdgePos(node=node), {"to": nxt, **params})


def place_marks(trail: Trail, g: StreetGraph, rng: random.Random,
                opts: GenerateOptions | None = None, fp: Footprint | None = None,
                timing: PackTiming | None = None) -> Trail:
    """Replace the trail's marks and side legs with a layout that follows the hash rules.

    Given ``timing``, a turn where hounds checking the wrong ways would look
    straight at the hare is less likely to get a check (it gets an arrow, and
    maybe a check at the next corner instead).
    """
    o = opts or GenerateOptions()
    geo = Geometry(g)
    tp = [(e.u, e.v, e.key) for e in trail.true_path]
    rp = [(e.u, e.v, e.key) for e in trail.return_path]
    cum = _cum(g, tp)
    L = float(cum[-1])
    if fp is None:
        fp = Footprint(geo, o.keep_apart_m)
        fp.add_path(tp, "true")
        fp.add_path(rp, "return")
    fp = fp.copy()

    cs = corners(geo, tp, cum)
    turns = [c for c in cs if c.turn]
    by_s = {c.i: c.s for c in cs}

    risky = lambda c: False  # noqa: E731
    if timing is not None:
        lines = SightLines(geo, 150.0 + 250.0)
        d = np.arange(0.0, L, 10.0)
        xs, ys = [], []
        for x in d:
            ep = _pos(tp, cum, float(x))
            ec, ex, ey, _, _ = geo.edge_poly((ep.edge.u, ep.edge.v, ep.edge.key))
            xs.append(float(np.interp(ep.offset_m, ec, ex)))
            ys.append(float(np.interp(ep.offset_m, ec, ey)))
        true_xy = np.column_stack([xs, ys])

        memo: dict[int, bool] = {}

        def risky(c):
            if c.i not in memo:
                wrong = [nb for nb in c.ways if nb != c.nxt]
                memo[c.i] = check_looks_at_hare(geo, lines, timing, c.node, wrong, true_xy, d, c.s)
            return memo[c.i]

    # Checks at turns (and, rarely, fake checks on long straights); arrows at the other turns.
    kind: dict[int, str] = {}
    last = -math.inf
    last_mark = -math.inf
    moved: list[Corner] = []  # risky turns that got an arrow: a check may go one corner on
    for c in turns:
        if c.s - last_mark <= COVER_M:
            continue  # the same street corner as the last mark (a sidewalk jog): one mark covers it
        ok = 200.0 <= c.s <= L - 200.0 and c.s - last >= o.min_check_gap_m
        if ok and rng.random() < o.check_share:
            if not risky(c) or rng.random() < o.risky_check_share:
                kind[c.i] = "check"
                last = c.s
            else:
                kind[c.i] = "arrow"
                moved.append(c)
        else:
            kind[c.i] = "arrow"
        last_mark = c.s
    for m in moved:
        # Arrow at the turn, then a check at the next corner (if it's a plain crossing, nearby and safe).
        nxt = next((c for c in cs if c.s > m.s + COVER_M), None)
        if (nxt is None or nxt.turn or nxt.s - m.s > 200.0 or len(nxt.ways) < 2 or nxt.s > L - 200.0
                or any(abs(nxt.s - by_s[i]) < o.min_check_gap_m for i, k in kind.items() if k == "check")
                or risky(nxt)):
            continue
        kind[nxt.i] = "check"
    # Keep hounds honest: some checks and whichy-ways where the answer is straight on.
    # A run of short blocks may get a whichy-way at every crossing, straight, straight... then the turn.
    if rng.random() < o.whichy_run_share:
        _whichy_run(cs, kind, L, rng)
    # Then any dull stretch (no check or whichy-way for ``dull_m``) may get one at a crossing.
    for _ in range(len(cs)):
        events = [0.0] + sorted(by_s[i] for i, k in kind.items() if k != "arrow") + [L]
        dull = [(a, b) for a, b in zip(events, events[1:]) if b - a > o.dull_m]
        if not dull:
            break
        a, b = max(dull, key=lambda t: t[1] - t[0])
        # Somewhere that leaves a stretch shorter than dull_m behind it (and room ahead).
        hi = min(a + o.dull_m, b - o.fill_gap_m)
        ok = [c for c in cs if a + o.fill_gap_m <= c.s <= hi and 100.0 <= c.s <= L - 100.0 and not risky(c)]
        turned = [c for c in ok if c.turn and kind.get(c.i) == "arrow"]
        crossing = [c for c in ok if c.crossing and c.i not in kind]
        if not (turned or crossing) or rng.random() >= o.fill_share:
            # Leave this one dull (the score says so); mark it so we don't try again.
            kind.setdefault(("dull", a), "arrow")
            by_s[("dull", a)] = (a + b) / 2
            continue
        # A turn on the stretch, or a crossing where trail goes straight on: no fixed ratio.
        pool = turned if turned and (not crossing or rng.random() < 0.75) else crossing
        c = rng.choice(pool)
        kind[c.i] = "whichy_pick" if rng.random() < o.straight_whichy_share else "check"
    for k in [k for k in kind if isinstance(k, tuple)]:
        del kind[k]

    marks: list[Mark] = []
    branches: list[Branch] = []
    gaps = []
    extra_blobs: list[float] = []
    by_i = {c.i: c for c in cs}

    # Whichy-ways replace some arrows: two arrows, one wrong leg that ends in an F after one mark.
    # Straight-on whichy-ways (and runs of them) were picked above; if one can't get a wrong leg
    # it becomes a check.
    arrows = [i for i, k in kind.items() if k == "arrow"]
    rng.shuffle(arrows)
    want_w = rng.randint(*o.whichy_ways)
    picked = sorted(i for i, k in kind.items() if k == "whichy_pick")
    for i in picked + arrows:
        if i in arrows and want_w <= 0:
            break
        c = by_i[i]
        for nb in rng.sample(c.ways, len(c.ways)):
            if nb == c.nxt:
                continue
            leg = _side_leg(g, geo, fp, c.node, nb, rng.uniform(*o.whichy_leg_m), "whichy")
            if leg is None or leg[1] < 40.0:
                continue
            edges, f_at = leg
            fp.add_path(edges, "whichy")
            branches.append(Branch(BranchPurpose.FALSE, [EdgeRef(*e) for e in edges],
                                   f"whichy-way wrong leg at {c.s / METRES_PER_MILE:.2f} mi"))
            bc = _cum(g, edges)
            marks.append(Mark(MarkKind.BLOB, _pos(edges, bc, min(20.0, f_at / 2))))
            marks.append(Mark(MarkKind.FALSE, _pos(edges, bc, f_at)))
            if i in arrows:
                want_w -= 1
            kind[i] = "whichy"
            marks.append(Mark(MarkKind.WHICHY_WAY, EdgePos(node=c.node), {"ways": [c.nxt, nb]}
                              | ({"fake": True} if not c.turn else {})))
            # The true way shows its one confirming mark early too.
            first = g.edge_length(*tp[i])
            extra_blobs.append(c.s + min(20.0, first * 0.6))
            gaps.append((c.s, c.s + min(20.0, first * 0.6)))
            break
        if kind[i] == "whichy_pick":
            kind[i] = "check"  # no room for a wrong leg

    # Checks: a short unmarked gap on the true way, maybe a false down another way.
    checks = sorted(i for i, k in kind.items() if k == "check")
    want_f = rng.randint(*o.falses)
    false_at = set(rng.sample(checks, min(want_f, len(checks))))
    nchk = 0
    for i in checks:
        c = by_i[i]
        nchk += 1
        first = g.edge_length(*tp[i])
        gap = max(10.0, min(rng.uniform(*o.check_gap_m), first * 0.8))
        gaps.append((c.s, c.s + gap))
        marks.append(Mark(MarkKind.CHECK, EdgePos(node=c.node),
                          {"check": nchk} | ({"fake": True} if not c.turn else {})))
        if i in false_at:
            others = [nb for nb in c.ways if nb != c.nxt]
            rng.shuffle(others)
            for nb in others:
                leg = _side_leg(g, geo, fp, c.node, nb, rng.uniform(*o.false_len_m), "false")
                if leg is None or leg[1] < 3 * o.false_mark_every_m + 20:
                    continue
                edges, f_at = leg
                fp.add_path(edges, "false")
                branches.append(Branch(BranchPurpose.FALSE, [EdgeRef(*e) for e in edges],
                                       f"false at check {nchk}"))
                bc = _cum(g, edges)
                s = 15.0
                while s < f_at - 10:
                    marks.append(Mark(MarkKind.BLOB, _pos(edges, bc, s)))
                    s += o.false_mark_every_m
                marks.append(Mark(MarkKind.FALSE, _pos(edges, bc, f_at)))
                break

    for i, k in kind.items():
        if k == "arrow":
            c = by_i[i]
            marks.append(_arrow(c.node, c.nxt))

    for s in blob_positions(L, o.blob_every_m, rng, gaps, extra_blobs):
        marks.append(Mark(MarkKind.BLOB, _pos(tp, cum, s)))

    if rp:
        rcum = _cum(g, rp)
        for c in corners(geo, rp, rcum):
            if c.turn:
                marks.append(_arrow(c.node, c.nxt, leg="return"))
        for s in blob_positions(float(rcum[-1]), o.blob_every_m, rng):
            marks.append(Mark(MarkKind.BLOB, _pos(rp, rcum, s), {"leg": "return"}))

    order = {MarkKind.CHECK: 0, MarkKind.WHICHY_WAY: 1, MarkKind.ARROW: 2, MarkKind.FALSE: 3}
    marks.sort(key=lambda m: order.get(m.kind, 9))
    return replace(trail, marks=marks, branches=branches)


def add_marks(trail: Trail, g: StreetGraph, seed: int = 0, opts: GenerateOptions | None = None) -> Trail:
    """Fill in blobs and arrows on a trail that only has checks/falses (e.g. a hand-drawn one).

    Existing marks stay. Turns with no check, arrow or whichy-way get an arrow.
    After each check an unmarked gap of ~50 m is left. Falses with no F get one
    at their far end.
    """
    o = opts or GenerateOptions()
    rng = random.Random(seed)
    geo = Geometry(g)
    tp = [(e.u, e.v, e.key) for e in trail.true_path]
    cum = _cum(g, tp)
    model = TrailModel(trail, g)
    marks = list(trail.marks)
    marked = {c.node for c in model.checks} | set(model.arrow_nodes)
    for c in corners(geo, tp, cum):
        if c.turn and c.node not in marked:
            marks.append(_arrow(c.node, c.nxt, auto=True))
    gaps = []
    for c in model.checks:
        i = bisect.bisect_left(cum, c.s - 1e-6)
        first = g.edge_length(*tp[i]) if i < len(tp) else 0.0
        gaps.append((c.s, c.s + max(5.0, min(50.0, first - 5.0))))
    if not any(m.kind == MarkKind.BLOB and m.params.get("leg") != "return" for m in trail.marks):
        for s in blob_positions(float(cum[-1]), o.blob_every_m, rng, gaps):
            marks.append(Mark(MarkKind.BLOB, _pos(tp, cum, s), {"auto": True}))
    if trail.return_path and not any(m.params.get("leg") == "return" for m in trail.marks):
        rp = [(e.u, e.v, e.key) for e in trail.return_path]
        rcum = _cum(g, rp)
        for c in corners(geo, rp, rcum):
            if c.turn:
                marks.append(_arrow(c.node, c.nxt, leg="return", auto=True))
        for s in blob_positions(float(rcum[-1]), o.blob_every_m, rng):
            marks.append(Mark(MarkKind.BLOB, _pos(rp, rcum, s), {"leg": "return", "auto": True}))
    have_f = {(m.at.edge.u, m.at.edge.v, m.at.edge.key) for m in trail.marks
              if m.kind == MarkKind.FALSE and m.at.edge is not None}
    for b in trail.branches:
        be = [(e.u, e.v, e.key) for e in b.edges]
        if b.purpose == BranchPurpose.FALSE and not have_f.intersection(be):
            last = be[-1]
            marks.append(Mark(MarkKind.FALSE, EdgePos(EdgeRef(*last), max(0.0, g.edge_length(*last) - 1.0)),
                              {"auto": True}))
    return replace(trail, marks=marks)


def needs_marks(trail: Trail, g: StreetGraph) -> bool:
    """True if the trail is missing blobs, or turns without a check/arrow/whichy-way."""
    if not any(m.kind == MarkKind.BLOB and m.params.get("leg") != "return" for m in trail.marks):
        return True
    return bool(TrailModel(trail, g).unmarked_turns)


# ---------------------------------------------------------------- the one click

def _whichy_run(cs: list[Corner], kind: dict, L: float, rng: random.Random,
                min_run: int = 3, block_m: float = 160.0) -> None:
    """A whichy-way at each of a few short-block crossings, all straight, then one at the turn."""
    turns = [c for c in cs if c.turn and kind.get(c.i) == "arrow" and 300.0 <= c.s <= L - 100.0]
    rng.shuffle(turns)
    for t in turns:
        before = [c for c in cs if c.s < t.s and c.i not in kind and not c.jog]
        run: list[Corner] = []
        at = t.s
        for c in reversed(before):
            if at - c.s > block_m or not c.crossing:
                break
            run.append(c)
            at = c.s
        if len(run) < min_run:
            continue
        run = run[:rng.randint(min_run, max(min_run, min(len(run), 6)))]
        if any(kind.get(c.i, "arrow") != "arrow" for c in cs if run[-1].s - 150.0 <= c.s < t.s and c not in run):
            continue  # a check right before the run would spoil it
        for c in run:
            kind[c.i] = "whichy_pick"
        kind[t.i] = "whichy_pick"  # and finally the turn
        return


def vary_layout(o: GenerateOptions, rng: random.Random) -> GenerateOptions:
    """A random mark style around the base options, so the sim can pick what works."""
    return replace(
        o,
        check_share=min(1.0, max(0.2, o.check_share + rng.uniform(-0.25, 0.3))),
        check_gap_m=rng.choice([o.check_gap_m, (50.0, 100.0)]),
        falses=(o.falses[0], rng.randint(o.falses[0], o.falses[1])),
    )


@dataclass
class Generated:
    trail: Trail
    card: ScoreCard
    results: list[SimResult]
    tried: list[tuple[float, float]] = field(default_factory=list)  # (route score, total) per candidate


def generate(g: StreetGraph, start, end=None, opts: GenerateOptions | None = None, *,
             pack: list[HoundProfile] | None = None, hare: HareProfile | None = None,
             seed: int = 0, progress=None) -> Generated:
    """Build a trail Start -> Beer Near -> back to Start and pick the best by simulation."""
    o = opts or GenerateOptions()
    pack = pack or default_pack()
    hare = hare or HareProfile()
    rng = random.Random(seed)
    say = progress or (lambda msg: None)
    geo = Geometry(g)
    target = o.miles * METRES_PER_MILE
    back = o.return_miles * METRES_PER_MILE
    timing = PackTiming.of(hare, pack)

    radius = target * 0.45 + back
    net = Network(g, start, radius)
    say(f"Searching {len(net.nodes):,} intersections within {radius / METRES_PER_MILE:.1f} mi of Start")
    # Start and Beer Near go on real corners (3+ ways), not dead-end stubs or mid-block points:
    # trail leaves Start one way and comes back in another.
    # A hand-picked point mid-block (two ways) is fine: trail leaves one way, comes back the other.
    need = 2 if o.exact_ends else 3
    if len(net.adj.get(start, ())) < need:
        start = net.nearest(*geo.node_xy(start), geo)
    if end is not None and end in net.adj and len(net.adj[end]) < need:
        end = net.nearest(*geo.node_xy(end), geo)

    if end is not None:
        ends = [end]
    else:
        from_start = net.dijkstra(start, cutoff=back * 1.3)
        ring = [n for n, d in from_start.items() if 0.75 * back <= d <= 1.25 * back
                and len(net.adj.get(n, ())) >= 3]
        if not ring:
            ring = [n for n, d in from_start.items() if d >= 0.5 * back] or [max(from_start, key=from_start.get)]
        ends = rng.sample(ring, min(6, len(ring)))

    routes = []
    stats: dict = {}
    # Loop size: perimeter ~ (2 + 2 sin(half the spread)) * radius, times ~1.3 for streets.
    radius = target / 4.6
    per_end = max(1, o.tries // len(ends))
    for e in ends:
        for _ in range(per_end):
            got = loop_route(net, geo, start, e, target, rng, radius, o.keep_apart_m,
                             turn_cost_m=o.turn_cost_m, timing=timing, sight_cost=o.sight_cost,
                             straight_run_m=o.straight_run_m, straight_cost=o.straight_cost, stats=stats)
            if not got:
                continue
            r, fp = got
            length = path_len(g, r)
            radius *= (target / length) ** 0.5  # steer the loop size toward the target
            fit = DistanceFit(target).score(length, []).value
            if fit <= 0:
                continue
            routes.append((fit + rng.random() * 1e-3, r, e, fp))
    if not routes:
        raise RuntimeError("No route found from Start that keeps clear of itself; try another Start, "
                           "a shorter --miles, or another --end")
    routes.sort(key=lambda t: -t[0])

    picked, seen = [], []
    for sc, r, e, fp in routes:
        if e != start:
            fpr = fp.copy()
            fpr.zones += [(*geo.node_xy(start), 2 * o.keep_apart_m), (*geo.node_xy(e), 2 * o.keep_apart_m)]
            banned = {x[0] for x in r} - {start}
            ret = net.clear_path(e, start, fpr, "return", 0.0, banned, max_len=back * 2.0,
                                 came_from=r[-1][0], turn_cost_m=o.turn_cost_m)
            if ret is None:
                continue
            fp = fp.copy()
            fp.add_path(ret, "return")
        else:
            ret = []
        sig = frozenset(_und(x) for x in r)
        if any(len(sig & s) > 0.6 * len(sig) for s in seen):
            continue
        seen.append(sig)
        picked.append((sc, r, e, ret, fp))
        if len(picked) >= o.candidates:
            break
    if not picked:
        raise RuntimeError("Found routes, but none with a leg back to Start that keeps clear of the trail")
    say(f"Tried {len(routes)} routes; simulating marks on the best {len(picked)}")
    if stats.get("unnudged"):
        say(f"Note: {stats['unnudged']} stretch(es) found no route with the long-straight and hare-sight "
            "nudges and were routed without them")

    best: Generated | None = None
    tried = []
    for sc, r, e, ret, fp in picked:
        base = Trail(start=start, end=e, true_path=[EdgeRef(*x) for x in r],
                     return_path=[EdgeRef(*x) for x in ret],
                     meta={"source": "generated", "seed": seed, "target_miles": o.miles,
                           "generated": date.today().isoformat()})
        for p in range(o.placements):
            lo = o if p == 0 else vary_layout(o, rng)
            t = place_marks(base, g, random.Random(rng.random()), lo, fp, timing)
            model = TrailModel(t, g)
            results = simulate_many(t, g, pack, runs=o.runs, seed=rng.randrange(10**6), model=model)
            card = score_results(results, target, hare=hare)
            tried.append((sc, card.total))
            if best is None or card.total > best.card.total:
                best = Generated(t, card, results)
    best.tried = tried
    return best
