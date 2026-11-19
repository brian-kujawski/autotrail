"""One-click trails: a route search, then mark placement, following real hash rules.

Stage 1 (route): a loop from Start through 4-6 waypoints that alternate far
out and close in, then to the Beer Near, each stretch an A* search that never
revisits a corner and never comes within ``keep_apart_m`` of trail laid
earlier (no crossing, no running back down the other sidewalk). Turns are
nearly free (the hare's few seconds to mark them) and straights past
``straight_run_m`` cost extra, so routes twist. Routes are shortlisted on
distance and turns per mile. The leg back to Start is a short path that keeps
clear of the trail. Without a Beer Near, one is picked about ``return_miles``
from Start.

Stage 2 (marks): every turn gets a check or an arrow (hounds go straight on at
unmarked corners; a jog of under 20 m is straight on). A share of the turns
are checks, at least ``min_check_gap_m`` apart. To keep hounds honest, some
checks and whichy-ways have straight on as the answer: any stretch of
``dull_m`` with no check or whichy-way may get one at a crossing, and now and
then a run of short blocks gets a whichy-way at every crossing (straight,
straight, straight... then the turn). No ratio is targeted. Most other turns
get a whichy-way (two arrows on the ground, nothing down the wrong way, so it
costs the hare nothing); ``arrow_share`` of them keep a plain arrow. Zero to two
falses hang off checks: a couple of blocks of close, obvious marks, then an F.
Blobs every ~50 m, with a short unmarked gap after each check. The hare runs
the trail plus every false out and back, so those cost the hare time; the score checks the pack doesn't catch the hare.

Each candidate is simulated and the best total score wins. Hill-climbing the
marks is phase 2.
"""

from __future__ import annotations

import bisect
import copy
import heapq
import math
import random
import time
from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np

from .facts import corner_facts
from .graph import StreetGraph
from .pack import HareProfile, HoundProfile, default_pack
from .parallel import Workers, auto_workers
from .rules import (
    CORRIDOR_M,
    COVER_M,
    STRAIGHT_DEG,
    Footprint,
    Geometry,
    is_sidewalk,
    jog_corners,
    turns_per_mile,
)
from .score import DistanceFit, ScoreCard, score_results
from .sight import PackTiming, SightField, SightLines, check_looks_at_hare
from .sim import SimResult, TrailModel, simulate_many
from .trail import (
    METRES_PER_MILE,
    Branch,
    BranchPurpose,
    EdgePos,
    EdgeRef,
    Mark,
    MarkKind,
    Trail,
)


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
    arrow_share: float = 0.2          # share of turns without a check that keep a plain arrow (the rest: whichy-ways)
    exact_ends: bool = False          # Start/Beer Near were picked by hand: keep them unless a dead end
    # Route search: a turn costs the hare only the ~3 s it takes to bend down and mark it, about
    # 8 m at hare pace. Turns slow the pack far more than the hare, so they aren't avoided.
    turn_cost_m: float = 8.0
    straight_run_m: float = 150.0     # route search: a straight longer than this...
    straight_cost: float = 3.0        # ...costs this much extra per metre past it (long straights spread the pack)
    waypoints: tuple[int, ...] = (4, 5, 6)  # loop points; alternate in and out so the route has room to twist
    target_turns_per_mile: float = 12.0     # shortlist: routes this twisty (or more) count as fully twisty
    sight_cost: float = 3.0           # each metre the pack could see the hare run costs this many extra
    risky_check_share: float = 0.2    # chance a turn gets a check when checkers would look at the hare
    # Freeway cordon (graph.FREEWAY_CORDON_M either side of a freeway): each metre there costs
    # this many metres, sliding with the angle from crossing at right angles to running alongside.
    freeway_cross_cost: float = 2.0
    freeway_along_cost: float = 4.0
    workers: int = 1                  # processes for route attempts and sim runs (0 = one per CPU)


RUN_STEP_M = 25.0  # route search tracks the straight run so far in steps of this
RETURN_STRAIGHT_M = 400.0  # the leg back to Start is nudged off straights longer than this
END_ZONE_M = 70.0          # near Start and the Beer Near, trail may touch the way back (the hare's in view)
MAX_LEN_SHARE = 1.25       # a trail may be at most this times the target (distance fit is 0 there)
WALK_DETOUR = 1.5          # a waypoint's walk from Start may be this many times the crow-flies distance...
WALK_SLACK_M = 200.0       # ...plus this (a grid is up to 1.41x; more means a river or freeway in the way)
WAYPOINT_DRAWS = 6         # tries at each waypoint before giving up the attempt
WAYPOINT_SNAP = 0.35       # a waypoint may sit this share of its radius from the point drawn
SHAPE_DRAWS = 5            # tries to find a loop whose far end is somewhere walkable
KEEP_OUT_SHARE = 0.25      # no waypoint within this share of the loop radius of Start, the Beer Near or between


# ---------------------------------------------------------------- local network

class Network:
    """Adjacency lists for the part of the graph within reach of Start."""

    def __init__(self, g: StreetGraph, centre, radius_m: float, streets_only: bool = False,
                 freeway_cost: tuple[float, float] = (1.0, 1.0)):
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
                # Near a freeway: crossing is fine, hanging about on the service drive isn't.
                share = float(d.get("freeway_share", 0.0) or 0.0)
                if share:
                    cross, along = freeway_cost
                    par = float(d.get("freeway_parallel", 0.0) or 0.0)
                    cost *= 1.0 + share * (cross + (along - cross) * par - 1.0)
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
        straight on costs ``turn_cost_m`` extra: the few seconds the hare takes
        to mark it (each such turn needs a mark). With ``noise`` > 0
        each street costs up to (1 + noise) times its length, drawn at random,
        so repeated searches find different, still sensible, routes. With a
        ``sight`` field, each metre the pack could see the hare run costs
        ``sight_cost`` metres extra. Each metre of a straight run past
        ``straight_run_m`` costs ``straight_cost`` extra (the run so far is part of
        the search state, so this is exact to within ``RUN_STEP_M``).
        """
        if a == b:
            return []
        geo = fp.geo
        bx, by = geo.node_xy(b)

        def h(n):
            x, y = geo.node_xy(n)
            return math.hypot(x - bx, y - by)

        factor: dict = {}
        # Whether a street clashes with trail already laid depends on where along trail it's run
        # only while the stretch just run is short enough to be exempt (see Footprint.clashes):
        # past that, it's the same answer every time the search reaches that street.
        clash_memo: dict = {}
        recent = fp.recent_default
        # The straight-run penalty depends on how long the run so far is, so it's part of the
        # search state (in RUN_STEP_M buckets, capped where the penalty starts: past that, every
        # metre costs the same). Otherwise a cheaper path with a long straight blocks the twisty one.
        nb = int(math.ceil(straight_run_m / RUN_STEP_M)) if (straight_cost and straight_run_m < math.inf) else 0

        def bucket(r):
            return min(int(r // RUN_STEP_M), nb) if nb else 0

        s0 = (a, came_from, bucket(run0), False)
        cost = {s0: 0.0}
        real = {s0: 0.0}
        run = {s0: run0}
        # The run's direction, and a pending hop: a short stub off the axis (crossing to the other
        # side of a boulevard) that doesn't end the straightaway if the next street carries on.
        axis = {s0: geo.heading_in(came_from, a) if came_from is not None else None}
        stub = {s0: 0.0}
        back = {}
        pq = [(h(a), 0.0, s0)]
        expanded = 0
        goal = None
        while pq:
            _, d, st = heapq.heappop(pq)
            u, p = st[0], st[1]
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
                rn, ax, sb = l, None, 0.0
                if straight_cost:
                    # A straightaway is what the pack sees, not what the marks say: any way on within
                    # STRAIGHT_DEG of the way in keeps the run going (a crossing with a parallel
                    # sidewalk or median has no single "straight on", but the street is still straight),
                    # and so does a hop across to the other side of the street (a stub under
                    # CORRIDOR_M, then on the same way): otherwise hopping across a boulevard
                    # would look like breaking up the straight.
                    ax0, sb0, run0_ = axis[st], stub[st], run[st]
                    eh = geo.heading_out(u, v)
                    on_axis = ax0 is not None and _angle(eh, ax0) <= STRAIGHT_DEG
                    if p is None or (not sb0 and geo.deviation(p, u, v) <= STRAIGHT_DEG):
                        rn, ax, prev = run0_ + l, ax0 if ax0 is not None else eh, run0_
                    elif sb0 and on_axis:
                        rn, ax, prev = run0_ + sb0 + l, ax0, run0_
                    elif not sb0 and ax0 is not None and l < CORRIDOR_M:
                        rn, ax, sb, prev = run0_, ax0, l, run0_
                    else:
                        rn, ax, prev = l, geo.heading_in(u, v), 0.0
                    nd += straight_cost * (max(0.0, rn - straight_run_m) - max(0.0, prev - straight_run_m))
                nst = (v, u, bucket(rn), sb > 0)
                if nd >= cost.get(nst, math.inf):
                    continue
                if ru > recent:
                    cl = clash_memo.get((u, v, k))
                    if cl is None:
                        cl = clash_memo[(u, v, k)] = fp.clashes((u, v, k), tag, pos0 + ru)
                else:
                    cl = fp.clashes((u, v, k), tag, pos0 + ru)
                if cl:
                    continue
                if sight is not None and sight_cost:
                    nd += sight_cost * sight.exposed_m((u, v, k), pos0 + ru)
                    if nd >= cost.get(nst, math.inf):
                        continue
                cost[nst] = nd
                real[nst] = rl
                run[nst] = rn
                axis[nst] = ax
                stub[nst] = sb
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

    def nearest(self, x: float, y: float, geo: Geometry, min_ways: int = 3, walkable: bool = False):
        """The network corner nearest a point in local metres (with at least ``min_ways`` streets).

        With ``walkable``, only corners :meth:`set_walks` found a sensible walk from Start.
        """
        if not hasattr(self, "_xy"):
            ids = [n for n in self.nodes if len(self.adj[n]) >= min_ways]
            self._ids = ids
            self._xy = np.array([geo.node_xy(n) for n in ids])
        d2 = (self._xy[:, 0] - x) ** 2 + (self._xy[:, 1] - y) ** 2
        if walkable and getattr(self, "_ok", None) is not None:
            if not self._ok.any():
                return None
            d2 = np.where(self._ok, d2, np.inf)
        return self._ids[int(np.argmin(d2))]

    def set_walks(self, start, geo: Geometry, detour: float = WALK_DETOUR, slack_m: float = WALK_SLACK_M):
        """Walking distance from Start to every corner, and which corners make sensible waypoints:
        reachable, and not much further on foot than as the crow flies (so not across a river,
        a freeway or a rail yard with the nearest crossing far away)."""
        self.walk = self.dijkstra(start)
        self.nearest(0.0, 0.0, geo)  # build the corner index
        sx, sy = geo.node_xy(start)
        crow = np.hypot(self._xy[:, 0] - sx, self._xy[:, 1] - sy)
        walk = np.array([self.walk.get(n, np.inf) for n in self._ids])
        self._ok = walk <= detour * crow + slack_m

    def shortest_m(self, a, b, geo: Geometry, cutoff: float = math.inf) -> float:
        """Plain shortest walk a -> b in metres (no trail to keep clear of); inf past ``cutoff``."""
        if a == b:
            return 0.0
        bx, by = geo.node_xy(b)

        def h(n):
            x, y = geo.node_xy(n)
            return math.hypot(x - bx, y - by)

        dist = {a: 0.0}
        pq = [(h(a), 0.0, a)]
        while pq:
            f, d, u = heapq.heappop(pq)
            if u == b:
                return d
            if f > cutoff:
                return math.inf
            if d > dist[u]:
                continue
            for v, _, l, _ in self.adj.get(u, ()):
                nd = d + l
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd + h(v), nd, v))
        return math.inf


# ---------------------------------------------------------------- stage 1: routes

def _hw(d: dict) -> str:
    hw = d.get("highway")
    return str(hw[0] if isinstance(hw, list) else hw)


def _und(e) -> tuple:
    return (e[0], e[1]) if str(e[0]) <= str(e[1]) else (e[1], e[0])


def _angle(a: float, b: float) -> float:
    """Degrees between two headings (0 to 180)."""
    return abs(math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi))


def path_len(g: StreetGraph, edges) -> float:
    return sum(g.edge_length(*e) for e in edges)


def _tail_run(g: StreetGraph, geo: Geometry, edges: list[tuple]) -> float:
    """Length of the straight the path ends on (so the next stretch carries it on)."""
    run = 0.0
    for i in range(len(edges) - 1, -1, -1):
        run += g.edge_length(*edges[i])
        if i == 0 or geo.deviation(edges[i - 1][0], edges[i][0], edges[i][1]) > STRAIGHT_DEG:
            break
    return run


def reserved_corners(geo: Geometry, reserved: list[tuple], start, end, zone_m: float) -> set:
    """Corners on the way back that the trail may not use: all of them except those within
    ``zone_m`` of Start or the Beer Near (trail may run in on the same street there)."""
    pts = [geo.node_xy(start), geo.node_xy(end)]
    return {n for e in reserved for n in e[:2]
            if all(math.hypot(geo.node_xy(n)[0] - x, geo.node_xy(n)[1] - y) > zone_m for x, y in pts)}


def _seg_dist(px, py, ax, ay, bx, by) -> float:
    """Distance from (px, py) to the segment (ax, ay)-(bx, by)."""
    dx, dy = bx - ax, by - ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


@dataclass
class LoopShape:
    """Where a trail's waypoints go: one loop out from Start and back in to the Beer Near.

    The loop runs from its root (the anchor, on the line from Start to the
    Beer Near) out to a far point in any direction and back, so trail heads
    out one side and comes home the other. Only the order is fixed (one
    way round, never back across itself); the shape is drawn fresh each time:
    long and narrow (a long out-and-back), fat (a bean), bent (a banana),
    fatter at one end than the other, and every point at its own distance.
    """
    ax: float           # anchor: midway between Start and the Beer Near
    ay: float
    heading: float      # anchor -> far point
    length_m: float     # anchor to far point
    width_m: float      # widest point side to side
    bend: float         # sideways bow of the middle, as a share of length (banana / bean)
    skew: float         # one end fatter than the other (-1..1)
    side: float         # +1 / -1: which side trail goes out on
    n: int
    start_deg: float    # sweep from here round to 360 - start_deg (0 = the anchor end)
    outer: int          # which waypoints (odd or even) sit out on the rim; the rest sit in

    def point(self, i: int, rng: random.Random) -> tuple[float, float, float]:
        """A draw for waypoint i (jittered each call): (x, y, how far it is from the loop's middle)."""
        span = 360.0 - 2 * self.start_deg
        slot = span / self.n
        phi = math.radians(self.start_deg + (i + 0.5 + rng.uniform(-0.3, 0.3)) * slot)
        # Alternately out on the rim and pulled in toward the middle (not toward Start), so the
        # route weaves and has distance to spare for twists.
        k = rng.uniform(0.85, 1.15) if i % 2 == self.outer else rng.uniform(0.5, 0.75)
        along = -math.cos(phi) * self.length_m / 2 * k
        across = math.sin(phi) * self.width_m / 2 * k * (1 + self.skew * math.cos(phi))
        across += self.bend * self.length_m * math.sin(phi) ** 2 / 2
        a, s = self.side * across, along + self.length_m / 2
        ux, uy = math.cos(self.heading), math.sin(self.heading)
        x = self.ax + ux * s - uy * a
        y = self.ay + uy * s + ux * a
        return x, y, max(math.hypot(along, across), 0.25 * self.length_m)

    def far_point(self) -> tuple[float, float]:
        return (self.ax + math.cos(self.heading) * self.length_m, self.ay + math.sin(self.heading) * self.length_m)

    def outline(self, steps: int = 48) -> list[tuple[float, float]]:
        """The loop's middle line (no jitter, everything on the rim), for drawing."""
        pts = []
        for j in range(steps + 1):
            phi = math.radians(self.start_deg + j * (360.0 - 2 * self.start_deg) / steps)
            along = -math.cos(phi) * self.length_m / 2
            across = math.sin(phi) * self.width_m / 2 * (1 + self.skew * math.cos(phi))
            across += self.bend * self.length_m * math.sin(phi) ** 2 / 2
            a, s = self.side * across, along + self.length_m / 2
            ux, uy = math.cos(self.heading), math.sin(self.heading)
            pts.append((self.ax + ux * s - uy * a, self.ay + uy * s + ux * a))
        return pts


def loop_shape(rng: random.Random, start_xy, end_xy, radius_m: float, n: int) -> LoopShape:
    """Draw a loop for ``n`` waypoints whose length comes out at roughly 5.5x ``radius_m``.

    The far point can be in any direction. Trail plus the way back has to make one ring that
    never crosses itself; that holds whichever way the loop points, so long as it hangs off the
    right place. Pointing away from the Beer Near (or out to either side), the loop starts
    midway between Start and the Beer Near. Pointing back past the Beer Near, its root slides
    down to the Beer Near: trail heads out past it, goes round, and comes home from beyond.
    """
    sx, sy = start_xy
    ex, ey = end_xy
    heading = rng.uniform(0, 2 * math.pi)
    ratio = rng.uniform(0.15, 0.8)  # width / length: 0.15 is nearly an out-and-back, 0.8 a fat bean
    # The rim is about pi/2 * (length + width); the route weaves in and out of it and twists
    # (about 10% more), so this lands near the old loop's 5.5x radius.
    length = 5.0 * radius_m * rng.uniform(0.85, 1.15) / (math.pi / 2 * (1 + ratio))
    ax, ay = (sx + ex) / 2, (sy + ey) / 2
    toward = math.cos(heading - math.atan2(sy - ey, sx - ex)) if (sx, sy) != (ex, ey) else 0.0
    if toward < 0:  # heading back past the Beer Near: root the loop nearer it (at it, head-on)
        ax, ay = ax + (sx - ex) / 2 * toward, ay + (sy - ey) / 2 * toward
    # Go out on Start's side of the line to the far point, come home on the Beer Near's.
    ux, uy = math.cos(heading), math.sin(heading)
    off = -uy * (sx - ax) + ux * (sy - ay)
    side = math.copysign(1.0, off) if abs(off) > 1.0 else rng.choice((-1.0, 1.0))
    return LoopShape(ax, ay, heading, length, length * ratio, rng.uniform(-0.25, 0.25),
                     rng.uniform(-0.4, 0.4), side, n, 30.0, rng.randrange(2))


def loop_route(net: Network, geo: Geometry, start, end, target_m: float, rng: random.Random,
               radius_m: float, keep_apart_m: float = 25.0, noise: float = 0.25,
               turn_cost_m: float = 8.0, timing: PackTiming | None = None, sight_cost: float = 0.0,
               straight_run_m: float = math.inf, straight_cost: float = 0.0, stats: dict | None = None,
               waypoints: tuple[int, ...] = (4, 5, 6), reserved: list[tuple] = (),
               trace: dict | None = None):
    """One trail Start -> end shaped as a loop through a few random waypoints.

    The waypoints go round one loop in order (see ``LoopShape``): out from
    Start, round a far point, and back in to the Beer Near, so no leg has to cross an earlier one. None sits near Start, the
    Beer Near or the stretch between them: those streets are left for the
    first and last legs and the way back. Each stretch is an A* search that keeps clear of trail already laid and, given
    ``timing``, shies away from streets the pack could see the hare on.

    ``reserved`` is the leg back to Start, planned first: trail keeps clear of
    it (except right at Start and the Beer Near), so there's always a way back.
    ``trace``, if given, is filled in with the waypoints, the path so far and,
    on failure, why (for the GUI's step-through mode).

    Returns (edges, footprint) or None.
    """
    tr = trace if trace is not None else {}
    tr.update(waypoints=[], edges=[], reason=None)
    sx, sy = geo.node_xy(start)
    ex, ey = geo.node_xy(end)
    reserved_nodes = reserved_corners(geo, reserved, start, end, END_ZONE_M)
    n_wp = rng.choice(waypoints)
    walkable = getattr(net, "_ok", None) is not None
    for k in range(SHAPE_DRAWS if walkable else 1):
        # Any direction; if the far end is in a river or past a freeway, draw another.
        shape = loop_shape(rng, (sx, sy), (ex, ey), radius_m, n_wp)
        fx, fy = shape.far_point()
        c = net.nearest(fx, fy, geo, walkable=True) if walkable else None
        if c is None or not walkable:
            continue
        cx, cy = geo.node_xy(c)
        if math.hypot(cx - fx, cy - fy) <= WAYPOINT_SNAP * shape.length_m / 2:
            break
    tr["shape"] = shape
    keep_out = KEEP_OUT_SHARE * radius_m
    wps = []
    for i in range(n_wp):
        wp = None
        for _ in range(WAYPOINT_DRAWS if walkable else 1):
            px, py, r = shape.point(i, rng)
            if _seg_dist(px, py, sx, sy, ex, ey) < keep_out:
                continue  # leave the streets around Start and the Beer Near to the first and last legs
            c = net.nearest(px, py, geo, walkable=walkable)
            if c is None or c in (start, end) or c in wps or c in reserved_nodes:
                continue
            cx, cy = geo.node_xy(c)
            if walkable and math.hypot(cx - px, cy - py) > WAYPOINT_SNAP * r:
                continue  # nothing walkable near the point drawn (water, a park, a freeway): draw again
            wp = c
            break
        if wp is None:
            if walkable:
                continue  # nothing walkable that way (water, a freeway): the loop skips this direction
            tr["reason"] = "two waypoints (or a waypoint and Start, the Beer Near or the way back) on one corner"
            return None
        wps.append(wp)
    if len(wps) < 2:
        tr["reason"] = "fewer than 2 sensible waypoints (water or barriers most ways from Start?)"
        tr["waypoints"] = list(wps)
        return None
    tr["waypoints"] = list(wps)
    if walkable:
        # The plain shortest walk through every waypoint is the least this trail can be. Too long
        # already: give up now, before the slow search that also has to keep clear of the trail.
        budget = target_m * MAX_LEN_SHARE
        least, here = net.walk.get(wps[0], math.inf), wps[0]
        for goal in wps[1:] + [end]:
            if least > budget:
                break
            least += net.shortest_m(here, goal, geo, cutoff=budget - least)
            here = goal
        tr["least_m"] = least
        if least > budget:
            tr["reason"] = ("waypoints too far apart on foot: the shortest walk through them is "
                            + (f"{least / 1609.344:.1f} mi" if least < math.inf else f"over {budget / 1609.344:.1f} mi"))
            return None
    fp = Footprint(geo, keep_apart_m)
    if reserved:
        fp.add_path(reserved, "return")
        # Trail leaves Start and reaches the Beer Near right next to the way back: that close,
        # the pack can see the hare waiting anyway, so the game is over.
        fp.zones += [(sx, sy, END_ZONE_M), (ex, ey, END_ZONE_M)]
    sight = SightField(geo, timing) if (timing is not None and sight_cost) else None
    edges: list[tuple] = []
    visited = {start} | reserved_nodes
    walked = 0.0
    stops = wps + [end]
    for i, goal in enumerate(stops):
        left = target_m * MAX_LEN_SHARE - walked
        kw = dict(max_len=left, rng=rng, noise=noise, came_from=edges[-1][0] if edges else None,
                  turn_cost_m=turn_cost_m, run0=_tail_run(net.g, geo, edges))
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
            # Out of length, or really boxed in? The plain shortest walk through what's left says.
            least, here2 = walked, here
            for goal2 in stops[i:]:
                least += net.shortest_m(here2, goal2, geo)
                here2 = goal2
            if least > target_m * MAX_LEN_SHARE:
                tr["too_long_m"] = least
                tr["reason"] = (f"loop too big: at least {least / METRES_PER_MILE:.1f} mi by the Beer Near "
                                f"(limit {target_m * MAX_LEN_SHARE / METRES_PER_MILE:.1f})")
            else:
                tr["reason"] = (f"no clear way to waypoint {i + 1} of {len(wps)}" if i < len(wps)
                                else "no clear way into the Beer Near")
            return None
        for e in seg:
            fp.add(e, "true", walked)
            if sight is not None:
                sight.add_edge(e, walked)
            walked += net.g.edge_length(*e)
            visited.add(e[1])
        edges += seg
        tr["edges"] = edges
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

    # Whichy-ways: two arrows drawn on the ground, and nothing at all down the wrong way (a
    # hound who takes it finds no mark and comes back, or hears the call). They cost the hare
    # nothing extra, so most turns without a check get one; only ``arrow_share`` keep a plain arrow.
    # Straight-on whichy-ways (and runs of them) were picked above.
    arrows = [i for i, k in kind.items() if k == "arrow"]
    for i in sorted(i for i, k in kind.items() if k == "whichy_pick") + arrows:
        c = by_i[i]
        others = [nb for nb in c.ways if nb != c.nxt]
        if not others or (kind[i] == "arrow" and rng.random() < o.arrow_share):
            if kind[i] == "whichy_pick":
                kind[i] = "check"  # nowhere else to point
            continue
        nb = rng.choice(others)
        kind[i] = "whichy"
        marks.append(Mark(MarkKind.WHICHY_WAY, EdgePos(node=c.node), {"ways": [c.nxt, nb]}
                          | ({"fake": True} if not c.turn else {})))
        # The true way shows its one confirming mark early.
        first = g.edge_length(*tp[i])
        extra_blobs.append(c.s + min(20.0, first * 0.6))
        gaps.append((c.s, c.s + min(20.0, first * 0.6)))

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
             seed: int = 0, progress=None, on_route=None) -> Generated:
    """Build a trail Start -> Beer Near -> back to Start and pick the best by simulation.

    ``on_route(info)`` is called after every route attempt with what was tried (Start, Beer
    Near, waypoints, the path laid, the way back) and how it went; it may block (the GUI's
    step-through mode waits there for a click).
    """
    o = opts or GenerateOptions()
    pack = pack or default_pack()
    hare = hare or HareProfile()
    rng = random.Random(seed)
    say = progress or (lambda msg: None)
    geo = Geometry(g)
    target = o.miles * METRES_PER_MILE
    back = o.return_miles * METRES_PER_MILE
    timing = PackTiming.of(hare, pack)

    net_radius = target * 0.45 + back
    net = Network(g, start, net_radius, freeway_cost=(o.freeway_cross_cost, o.freeway_along_cost))
    say(f"Searching {len(net.nodes):,} intersections within {net_radius / METRES_PER_MILE:.1f} mi of Start")
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

    net.set_walks(start, geo)
    n_workers = o.workers or auto_workers()

    def remote() -> dict:
        # Workers get the map within reach of Start plus a margin (side legs, probes and sight
        # lines run a few hundred metres off the trail), in the same local metres.
        part = g.part(Network._reach(g.G, start, net_radius + PART_MARGIN_M).keys())
        net_r = copy.copy(net)
        net_r.g = part
        return dict(st, g=part, net=net_r, geo=None, local=False)

    st = dict(g=g, net=net, geo=geo, opts=o, start=start, target=target, back=back,
              timing=timing, pack=pack, hare=hare, seed=seed, local=True)
    with Workers(n_workers, st, remote, setup=_worker_setup) as pool:
        if n_workers > 1:
            say(f"Using {n_workers} worker processes")
        best = _search(pool, st, ends, rng, say, on_route)
    return best


# Route attempts run in batches (the same whatever the number of workers, so a seed gives the
# same trail on any PC), and the loop size is steered after each batch. The first batches are
# small, so the first guess at the size gets corrected quickly; after that, ROUTE_BATCH at a time.
ROUTE_BATCH = 12
FIRST_BATCHES = (1, 1, 2, 2, 4, 4, 8, 8)
PART_MARGIN_M = 1000.0  # workers get the map this far past the search area


def _worker_setup(st: dict) -> None:
    st["geo"] = Geometry(st["g"])


def route_footprint(geo: Geometry, keep_apart_m: float, edges, reserved, start, end) -> Footprint:
    """The footprint :func:`loop_route` leaves behind for a route, rebuilt from the route itself."""
    fp = Footprint(geo, keep_apart_m)
    if reserved:
        fp.add_path(reserved, "return")
        fp.zones += [(*geo.node_xy(start), END_ZONE_M), (*geo.node_xy(end), END_ZONE_M)]
    fp.add_path(edges, "true")
    return fp


def _way_back(st: dict, e, rng: random.Random):
    """The leg Beer Near -> Start, planned before the trail so there's always a clear way back."""
    net, geo, o, start = st["net"], st["geo"], st["opts"], st["start"]
    if e == start:
        return []
    fpr = Footprint(geo, o.keep_apart_m)
    kw = dict(max_len=st["back"] * 2.0, rng=rng, noise=0.25)
    # Shorter and simpler than the trail: only the long straights get nudged.
    ret = net.clear_path(e, start, fpr, "return", 0.0, set(), **kw,
                         straight_run_m=max(o.straight_run_m, RETURN_STRAIGHT_M), straight_cost=o.straight_cost)
    return ret if ret is not None else net.clear_path(e, start, fpr, "return", 0.0, set(), **kw)


def _route_attempt(st: dict, job) -> dict:
    """One route attempt (a worker job): the way back, then a loop out to the Beer Near."""
    seed, e, radius = job
    o = st["opts"]
    rng = random.Random(seed)
    t0 = time.time()
    ret = _way_back(st, e, rng)
    trace: dict = {"waypoints": [], "edges": [], "reason": None}
    stats: dict = {}
    got = None
    if ret is None:
        trace["reason"] = "no clear way back from the Beer Near to Start"
    else:
        got = loop_route(st["net"], st["geo"], st["start"], e, st["target"], rng, radius, o.keep_apart_m,
                         turn_cost_m=o.turn_cost_m, timing=st["timing"], sight_cost=o.sight_cost,
                         straight_run_m=o.straight_run_m, straight_cost=o.straight_cost, stats=stats,
                         waypoints=o.waypoints, reserved=ret, trace=trace)
    shape = trace.pop("shape", None)
    trace["outline"] = shape.outline() if shape else []
    return {"ret": ret, "route": got[0] if got else None, "trace": trace, "stats": stats,
            "seconds": time.time() - t0}


def _place_and_score(st: dict, job) -> dict:
    """Lay one mark layout on a route and score it on a few sim runs (a worker job)."""
    r, e, ret, lo, place_seed, sim_seed = job
    g, geo, o, start = st["g"], st["geo"], st["opts"], st["start"]
    base = Trail(start=start, end=e, true_path=[EdgeRef(*x) for x in r],
                 return_path=[EdgeRef(*x) for x in ret],
                 meta={"source": "generated", "seed": st["seed"], "target_miles": o.miles,
                       "generated": date.today().isoformat()})
    fp = route_footprint(geo, o.keep_apart_m, r, ret, start, e)
    t = place_marks(base, g, random.Random(place_seed), lo, fp, st["timing"])
    results = simulate_many(t, g, st["pack"], runs=o.runs, seed=sim_seed, model=TrailModel(t, g))
    card = score_results(results, st["target"], hare=st["hare"])
    # Sim results hold the compiled trail and its graph: only worth keeping in this process.
    return {"trail": t, "card": card, "results": results if st["local"] else None}


def _search(pool: Workers, st: dict, ends: list, rng: random.Random, say, on_route) -> Generated:
    g, geo, o, start, target = st["g"], st["geo"], st["opts"], st["start"], st["target"]
    timing = st["timing"]
    routes = []
    stats: dict = {}
    # Loop size: a first guess (the route weaves in and out, so it's longer than the rim);
    # every batch of attempts steers it toward the target (see below).
    radius = target / 5.5
    twist_ratio: list[float] = []  # route length / shortest walk through its waypoints, per route found
    per_end = max(1, o.tries // len(ends))
    queue = [e for e in ends for _ in range(per_end)]
    n_try, total = 0, len(queue)
    no_way_back = 0
    b0, nb = 0, 0
    while b0 < total:
        size = FIRST_BATCHES[nb] if nb < len(FIRST_BATCHES) else ROUTE_BATCH
        batch = [(rng.randrange(2**32), e, radius) for e in queue[b0:b0 + size]]
        b0, nb = b0 + size, nb + 1
        steer: list[float] = []
        for (_, e, _), out in zip(batch, pool.map(_route_attempt, batch)):
            n_try += 1
            ret, r, trace = out["ret"], out["route"], out["trace"]
            for k, v in out["stats"].items():
                stats[k] = stats.get(k, 0) + v
            if ret is None:
                no_way_back += 1
            # Steer the loop size toward the target on every attempt, not just the ones that work:
            # real streets (rivers, freeways, rail yards, sidewalk networks) make the walk round a
            # loop much longer than its outline, and that differs from city to city.
            est = None
            if r:
                est = path_len(g, r)
                if math.isfinite(trace.get("least_m", math.inf)) and trace["least_m"] > 0:
                    twist_ratio.append(est / trace["least_m"])
            elif trace.get("too_long_m"):
                est = trace["too_long_m"]  # ran out of length partway: at least this long
            elif "least_m" in trace:
                # Never searched (waypoints too far apart on foot, or the search failed): the shortest
                # walk through the waypoints, times how much longer real routes have come out.
                ratio = sorted(twist_ratio)[len(twist_ratio) // 2] if twist_ratio else 1.15
                est = trace["least_m"] * ratio if math.isfinite(trace["least_m"]) else 1.6 * target
            if est:
                steer.append(min(math.log(2.25), max(math.log(0.36), math.log(target / est))))
            info = {"n": n_try, "of": total, "start": start, "end": e, "return": ret or [],
                    "waypoints": trace["waypoints"], "edges": trace["edges"], "ok": False,
                    "outline": trace["outline"], "reason": trace["reason"], "seconds": out["seconds"]}
            if r:
                length = path_len(g, r)
                fit = DistanceFit(target).score(length, []).value
                tpm = turns_per_mile(geo, r, _cum(g, r))
                info.update(length_m=length, turns_per_mile=tpm, fit=fit)
                # Corner facts are worth working out only when someone looks (step-through mode).
                info["facts"] = (lambda r=r: corner_facts(g, r, timing, geo))
                if fit <= 0:
                    info["reason"] = f"{length / METRES_PER_MILE:.2f} mi: too far off {o.miles:g} mi"
                else:
                    # Shortlist on shape too: a twisty route beats a rectangle of the same length.
                    twist = min(1.0, tpm / o.target_turns_per_mile)
                    sc = fit * (0.5 + 0.5 * twist)
                    info.update(ok=True, score=sc)
                    routes.append((sc + rng.random() * 1e-3, r, e, ret))
            if on_route is not None:
                on_route(info)
        if steer:
            # The batch was all tried at one size, so their average says more than any one of them:
            # one attempt moves the size halfway (in log terms) to what it suggests, a batch of 12
            # nearly all the way.
            gain = 0.5 + 0.5 * (1.0 - 1.0 / len(steer))
            radius *= min(1.5, max(0.6, math.exp(gain * sum(steer) / len(steer))))
    if not routes:
        if no_way_back == n_try:
            raise RuntimeError("No clear way back from the Beer Near to Start (within twice the walk); "
                               "try another Beer Near")
        raise RuntimeError("No route found from Start that keeps clear of itself and the way back; "
                           "try another Start, a shorter --miles, or another --end")
    routes.sort(key=lambda t: -t[0])

    picked, seen = [], []
    for sc, r, e, ret in routes:
        sig = frozenset(_und(x) for x in r)
        if any(len(sig & s) > 0.6 * len(sig) for s in seen):
            continue
        seen.append(sig)
        picked.append((sc, r, e, ret))
        if len(picked) >= o.candidates:
            break
    say(f"Tried {len(routes)} routes; simulating marks on the best {len(picked)}")
    if stats.get("unnudged"):
        say(f"Note: {stats['unnudged']} stretch(es) found no route with the long-straight and hare-sight "
            "nudges and were routed without them")

    jobs, scores = [], []
    for sc, r, e, ret in picked:
        for p in range(o.placements):
            lo = o if p == 0 else vary_layout(o, rng)
            jobs.append((r, e, ret, lo, rng.random(), rng.randrange(10**6)))
            scores.append(sc)
    best, best_job, tried = None, None, []
    for sc, job, out in zip(scores, jobs, pool.map(_place_and_score, jobs)):
        tried.append((sc, out["card"].total))
        if best is None or out["card"].total > best["card"].total:
            best, best_job = out, job
    results = best["results"]
    if results is None:  # scored in a worker: rerun the same seeded runs here, on the whole map
        t = best["trail"]
        results = simulate_many(t, g, st["pack"], runs=o.runs, seed=best_job[5], model=TrailModel(t, g))
    gen = Generated(best["trail"], best["card"], results)
    gen.tried = tried
    return gen
