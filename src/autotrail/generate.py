"""One-click trails: a route search, then rule-based mark placement.

Stage 1 (route): a random-restart weighted walk from Start that prefers new
streets, bends toward the Beer Near as its distance budget runs down, and
closes with the shortest path. The best few hundred tries by route-only score
go forward. If no Beer Near is given, one is picked about ``return_miles``
from Start, since the hared leg back is about a mile.

Stage 2 (marks): blobs every ~50 m and just past every corner, a check about
every half mile at an intersection (at least 300 m apart) with a short
unmarked gap after it, and one or two falses of 100 to 200 m down other
streets from each check. Each candidate is simulated and the best total
score wins. Hill-climbing the marks is phase 2.
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
from .pack import HoundProfile, default_pack
from .score import DistanceFit, ScoreCard, score_results
from .sim import Geometry, SimResult, TrailModel, simulate_many, turn_angle
from .trail import METRES_PER_MILE, Branch, BranchPurpose, EdgePos, EdgeRef, Mark, MarkKind, Trail


@dataclass
class GenerateOptions:
    miles: float = 5.0                # true trail, Start to Beer Near
    return_miles: float = 1.0         # hared leg back to Start
    tries: int = 300                  # route attempts
    candidates: int = 4               # best routes that get marks and a sim
    placements: int = 5               # mark layouts tried per route
    runs: int = 8                     # sim runs per candidate
    check_every_m: float = 805.0      # about half a mile
    min_check_gap_m: float = 300.0
    blob_every_m: float = 50.0
    corner_blob_m: float = 15.0
    check_gap_m: tuple[float, float] = (40.0, 100.0)
    falses_per_check: tuple[int, int] = (1, 2)
    false_len_m: tuple[float, float] = (100.0, 200.0)
    false_blobs: int = 1


# ---------------------------------------------------------------- local network

class Network:
    """Adjacency lists for the part of the graph within reach of Start."""

    def __init__(self, g: StreetGraph, centre, radius_m: float):
        self.g = g
        G = g.G
        full = {}
        for n in self._reach(G, centre, radius_m):
            full[n] = None
        self.nodes = full
        self.adj: dict = {n: [] for n in full}
        self.radj: dict = {n: [] for n in full}
        for u in full:
            for v in G.successors(u):
                if v == u or v not in full:
                    continue
                k = g.best_key(u, v)
                length = float(G.edges[u, v, k]["length"])
                virtual = bool(G.edges[u, v, k].get("virtual", False))
                self.adj[u].append((v, k, length, virtual))
                self.radj[v].append((u, k, length, virtual))

    @staticmethod
    def _reach(G, src, cutoff):
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            for v, keys in G[u].items():
                l = min(float(e["length"]) for e in keys.values())
                nd = d + l
                if nd <= cutoff and nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    def dijkstra(self, src, cutoff=math.inf, reverse=False, want_prev=False):
        adj = self.radj if reverse else self.adj
        dist = {src: 0.0}
        prev = {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, k, l, _ in adj.get(u, ()):
                nd = d + l
                if nd <= cutoff and nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = (u, k)
                    heapq.heappush(pq, (nd, v))
        return (dist, prev) if want_prev else dist

    def path_down(self, a, b, dist_to_b: dict) -> list[tuple] | None:
        """Shortest edges a -> b using a precomputed reverse Dijkstra from b (no new search)."""
        if a not in dist_to_b:
            return None
        out, cur = [], a
        while cur != b:
            best = min(((l + dist_to_b[v], v, k) for v, k, l, _ in self.adj[cur] if v in dist_to_b),
                       default=None)
            if best is None:
                return None
            out.append((cur, best[1], best[2]))
            cur = best[1]
            if len(out) > 100000:
                return None
        return out

    def path_edges(self, a, b) -> list[tuple] | None:
        """Shortest edges a -> b inside the network."""
        if a == b:
            return []
        dist, prev = self.dijkstra(a, want_prev=True)
        if b not in dist:
            return None
        out = []
        n = b
        while n != a:
            u, k = prev[n]
            out.append((u, n, k))
            n = u
        return out[::-1]


# ---------------------------------------------------------------- stage 1: routes

def _undirected(e) -> tuple:
    return (e[0], e[1]) if str(e[0]) <= str(e[1]) else (e[1], e[0])


def overlap_m(g: StreetGraph, edges: list[tuple]) -> float:
    """Metres of street run more than once (either direction)."""
    seen, rep = set(), 0.0
    for e in edges:
        k = _undirected(e)
        if k in seen:
            rep += g.edge_length(*e)
        seen.add(k)
    return rep


def walk_route(net: Network, geo: Geometry, start, end, target_m: float, dist_to_end: dict,
               rng: random.Random, avoid: set | None = None, close_slack_m: float = 120.0):
    """One weighted random walk Start -> end of about target_m metres."""
    avoid = avoid or set()
    edges: list[tuple] = []
    visited = {start}
    used: set = set()
    cur, prev, walked = start, None, 0.0
    for _ in range(20000):
        rem = target_m - walked
        d_end = dist_to_end.get(cur, math.inf)
        if rem - d_end < close_slack_m:
            break
        pressure = min(1.0, d_end / max(rem, 1.0)) ** 2
        cands, weights = [], []
        for nb, k, l, virtual in net.adj[cur]:
            de = dist_to_end.get(nb)
            if de is None or walked + l + de > target_m * 1.08:
                continue
            w = math.exp(-4.0 * pressure * (de - d_end) / max(l, 1.0))
            und = _undirected((cur, nb))
            if und in used:
                w *= 0.01        # running a street twice: only if nothing else
            elif nb in visited:
                w *= 0.2         # crossing our own path is fine, now and then
            if nb == prev:
                w *= 0.01        # no U-turns
            if len(net.adj.get(nb, ())) <= 1 and nb != end:
                w *= 0.05        # dead-end stub: we'd have to come straight back
            if prev is not None and turn_angle(geo, prev, cur, nb) < 0.5:
                w *= 1.6         # a little momentum: fewer zig-zags
            if und in avoid:
                w *= 0.15
            if virtual:
                w *= 0.5
            cands.append((nb, k, l))
            weights.append(w)
        if not cands:
            break
        nb, k, l = rng.choices(cands, weights)[0]
        edges.append((cur, nb, k))
        used.add(_undirected((cur, nb)))
        visited.add(nb)
        prev, cur, walked = cur, nb, walked + l
    tail = net.path_down(cur, end, dist_to_end)
    if tail is None:
        return None
    return edges + tail


def route_score(g: StreetGraph, edges: list[tuple], target_m: float) -> float:
    length = sum(g.edge_length(*e) for e in edges)
    fit = DistanceFit(target_m).score(length, []).value
    return fit - 2.0 * overlap_m(g, edges) / max(length, 1.0)


# ---------------------------------------------------------------- stage 2: marks

def _pos(edges: list[tuple], cum: np.ndarray, s: float) -> EdgePos:
    i = min(max(0, bisect.bisect_right(cum, s) - 1), len(edges) - 1)
    return EdgePos(EdgeRef(*edges[i]), float(s - cum[i]))


def _cum(g: StreetGraph, edges) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum([g.edge_length(*e) for e in edges])])


def _corners(g: StreetGraph, edges: list[tuple]) -> list[tuple[int, list]]:
    """(index, other neighbours) for every intersection the path passes through."""
    out = []
    for i in range(1, len(edges)):
        node = edges[i][0]
        prev, nxt = edges[i - 1][0], edges[i][1]
        others = [nb for nb in g.G.successors(node) if nb not in (prev, nxt, node)]
        if others:
            out.append((i, others))
    return out


def blob_positions(g: StreetGraph, edges, every_m: float, corner_m: float, rng: random.Random,
                   gaps: list[tuple[float, float]] = (), skip_nodes: set = frozenset()) -> list[float]:
    """Blob distances along a path: regular spacing, one just past each corner, none in gaps."""
    cum = _cum(g, edges)
    L = float(cum[-1])
    gaps = sorted(gaps)

    def in_gap(s):
        return any(a - 1e-6 <= s < b - 1e-6 for a, b in gaps)

    special = [b for _, b in gaps if b < L]
    for i, _ in _corners(g, edges):
        if edges[i][0] in skip_nodes:
            continue
        s = float(cum[i]) + min(corner_m, g.edge_length(*edges[i]) / 2)
        if not in_gap(s):
            special.append(s)
    regular = []
    s = min(20.0, L / 2)
    while s < L - 10:
        if in_gap(s):
            s = next(b for a, b in gaps if a - 1e-6 <= s < b - 1e-6)
            continue
        regular.append(s)
        s += every_m * rng.uniform(0.8, 1.2)
    keep = [r for r in regular if all(abs(r - x) > corner_m for x in special)]
    return sorted(set(round(x, 2) for x in keep + special))


def _false_branch(g: StreetGraph, geo: Geometry, node, nb, length_m: float,
                  true_nodes: set, true_und: set) -> tuple[list[tuple], float] | None:
    """Straight-on path from node via nb; returns (edges, metres to the F mark)."""
    edges, total = [], 0.0
    prev, cur = node, nb
    seen = {node}
    while True:
        e = (prev, cur, g.best_key(prev, cur))
        if _undirected(e) in true_und:
            break
        edges.append(e)
        total += g.edge_length(*e)
        seen.add(cur)
        if total >= length_m or cur in true_nodes:
            break
        nxts = [n for n in g.G.successors(cur) if n not in seen and n != cur]
        if not nxts:
            break
        p = prev
        prev, cur = cur, min(nxts, key=lambda n: turn_angle(geo, p, cur, n))
    if not edges:
        return None
    f_at = min(length_m, total - 1.0)
    if edges[-1][1] in true_nodes:
        f_at = min(f_at, total - 25.0)
    if f_at < 60.0:
        return None
    return edges, f_at


def place_marks(trail: Trail, g: StreetGraph, rng: random.Random,
                opts: GenerateOptions | None = None) -> Trail:
    """Replace the trail's marks and falses with a rule-based layout."""
    o = opts or GenerateOptions()
    geo = Geometry(g)
    tp = [(e.u, e.v, e.key) for e in trail.true_path]
    cum = _cum(g, tp)
    L = float(cum[-1])
    true_nodes = {e[0] for e in tp} | {tp[-1][1]}
    true_und = {_undirected(e) for e in tp} | {_undirected((e.u, e.v)) for e in trail.return_path}

    # Checks.
    cands = [(float(cum[i]), i, others) for i, others in _corners(g, tp)
             if 400.0 <= cum[i] <= L - 400.0 and g.edge_length(*tp[i]) >= 25.0]
    checks: list[tuple[float, int, list]] = []
    target = o.check_every_m
    while target < L - 400.0 and cands:
        aim = target + rng.uniform(-150, 150)
        ok = [c for c in cands if (not checks or c[0] - checks[-1][0] >= o.min_check_gap_m)
              and abs(c[0] - aim) <= 300.0]
        if ok:
            checks.append(min(ok, key=lambda c: abs(c[0] - aim)))
        target += o.check_every_m

    gaps = []
    marks: list[Mark] = []
    branches: list[Branch] = []
    for ci, (s, i, others) in enumerate(checks):
        first = g.edge_length(*tp[i])
        gap = min(rng.uniform(*o.check_gap_m), first - 5.0)
        gaps.append((s, s + gap))
        node = tp[i][0]
        marks.append(Mark(MarkKind.CHECK, EdgePos(node=node), {"check": ci + 1}))
        nbs = [nb for nb in others if _undirected((node, nb)) not in true_und]
        rng.shuffle(nbs)
        want = rng.randint(*o.falses_per_check)
        made = 0
        for nb in nbs:
            if made >= want:
                break
            fb = _false_branch(g, geo, node, nb, rng.uniform(*o.false_len_m), true_nodes, true_und)
            if fb is None:
                continue
            bedges, f_at = fb
            made += 1
            label = f"false {made} at check {ci + 1}"
            branches.append(Branch(BranchPurpose.FALSE, [EdgeRef(*e) for e in bedges], label))
            bcum = _cum(g, bedges)
            for b in range(o.false_blobs):
                bs = f_at * (b + 1) / (o.false_blobs + 1)
                marks.append(Mark(MarkKind.BLOB, _pos(bedges, bcum, bs)))
            marks.append(Mark(MarkKind.FALSE, _pos(bedges, bcum, f_at)))

    for s in blob_positions(g, tp, o.blob_every_m, o.corner_blob_m, rng, gaps,
                            skip_nodes={tp[i][0] for _, i, _ in checks}):
        marks.append(Mark(MarkKind.BLOB, _pos(tp, cum, s)))

    if trail.return_path:
        rp = [(e.u, e.v, e.key) for e in trail.return_path]
        rcum = _cum(g, rp)
        for s in blob_positions(g, rp, o.blob_every_m, o.corner_blob_m, rng):
            marks.append(Mark(MarkKind.BLOB, _pos(rp, rcum, s), {"leg": "return"}))

    marks.sort(key=lambda m: m.kind != MarkKind.CHECK)
    return replace(trail, marks=marks, branches=branches)


def add_blobs(trail: Trail, g: StreetGraph, seed: int = 0, opts: GenerateOptions | None = None) -> Trail:
    """Fill in blobs on a trail that only has checks/falses (e.g. a hand-drawn one).

    Existing marks stay. After each check an unmarked gap of ~70 m is left.
    """
    o = opts or GenerateOptions()
    rng = random.Random(seed)
    tp = [(e.u, e.v, e.key) for e in trail.true_path]
    cum = _cum(g, tp)
    model = TrailModel(trail, g)
    gaps = []
    for c in model.checks:
        i = bisect.bisect_left(cum, c.s - 1e-6)
        first = g.edge_length(*tp[i]) if i < len(tp) else 0.0
        gaps.append((c.s, c.s + max(5.0, min(70.0, first - 5.0))))
    marks = list(trail.marks)
    for s in blob_positions(g, tp, o.blob_every_m, o.corner_blob_m, rng, gaps,
                            skip_nodes={c.node for c in model.checks}):
        marks.append(Mark(MarkKind.BLOB, _pos(tp, cum, s), {"auto": True}))
    if trail.return_path:
        rp = [(e.u, e.v, e.key) for e in trail.return_path]
        rcum = _cum(g, rp)
        for s in blob_positions(g, rp, o.blob_every_m, o.corner_blob_m, rng):
            marks.append(Mark(MarkKind.BLOB, _pos(rp, rcum, s), {"leg": "return", "auto": True}))
    # Falses without an F mark get one at their far end.
    have_f = {(m.at.edge.u, m.at.edge.v, m.at.edge.key) for m in trail.marks
              if m.kind == MarkKind.FALSE and m.at.edge is not None}
    for b in trail.branches:
        be = [(e.u, e.v, e.key) for e in b.edges]
        if b.purpose == BranchPurpose.FALSE and not have_f.intersection(be):
            last = be[-1]
            marks.append(Mark(MarkKind.FALSE, EdgePos(EdgeRef(*last), max(0.0, g.edge_length(*last) - 1.0)),
                              {"auto": True}))
    return replace(trail, marks=marks)


def has_blobs(trail: Trail) -> bool:
    return any(m.kind in (MarkKind.BLOB, MarkKind.ARROW) and m.params.get("leg") != "return"
               for m in trail.marks)


def vary_layout(o: GenerateOptions, rng: random.Random) -> GenerateOptions:
    """A random mark style around the base options, so the sim can pick what works.

    Styles differ in check spacing, how far the trail resumes after a check,
    and how many falses there are and how convincing (blobs before the F).
    """
    return replace(
        o,
        check_every_m=o.check_every_m * rng.uniform(0.8, 1.15),
        check_gap_m=rng.choice([o.check_gap_m, (60.0, 140.0)]),
        falses_per_check=rng.choice([o.falses_per_check, (2, 3)]),
        false_blobs=rng.choice([o.false_blobs, o.false_blobs + 1]),
    )


# ---------------------------------------------------------------- the one click

@dataclass
class Generated:
    trail: Trail
    card: ScoreCard
    results: list[SimResult]
    tried: list[tuple[float, float]] = field(default_factory=list)  # (route score, total) per candidate


def generate(g: StreetGraph, start, end=None, opts: GenerateOptions | None = None, *,
             pack: list[HoundProfile] | None = None, seed: int = 0, progress=None) -> Generated:
    """Build a trail Start -> Beer Near (-> back to Start) and pick the best by simulation."""
    o = opts or GenerateOptions()
    pack = pack or default_pack()
    rng = random.Random(seed)
    say = progress or (lambda msg: None)
    geo = Geometry(g)
    target = o.miles * METRES_PER_MILE
    back = o.return_miles * METRES_PER_MILE

    radius = target * 0.6 + back
    net = Network(g, start, radius)
    say(f"Searching {len(net.nodes):,} intersections within {radius / METRES_PER_MILE:.1f} mi of Start")

    if end is not None:
        ends = [end]
    else:
        from_start = net.dijkstra(start, cutoff=back * 1.3)
        ring = [n for n, d in from_start.items() if 0.75 * back <= d <= 1.25 * back
                and len(net.adj.get(n, ())) >= 3]
        if not ring:
            ring = [n for n, d in from_start.items() if d >= 0.5 * back] or [max(from_start, key=from_start.get)]
        ends = rng.sample(ring, min(6, len(ring)))
    dist_to_start = net.dijkstra(start, cutoff=back * 3, reverse=True)

    routes = []
    per_end = max(1, o.tries // len(ends))
    for e in ends:
        d_end = net.dijkstra(e, cutoff=target * 1.1, reverse=True)
        if start not in d_end:
            continue
        for _ in range(per_end):
            r = walk_route(net, geo, start, e, target, d_end, rng)
            if r:
                routes.append((route_score(g, r, target), r, e))
    if not routes:
        raise RuntimeError("No route found from Start; is it on the walking network?")
    routes.sort(key=lambda t: -t[0])
    picked, seen = [], set()
    for sc, r, e in routes:
        sig = frozenset(_undirected(x) for x in r)
        if any(len(sig & s) > 0.7 * len(sig) for s in seen):
            continue
        seen.add(sig)
        picked.append((sc, r, e))
        if len(picked) >= o.candidates:
            break
    say(f"Tried {len(routes)} routes; simulating marks on the best {len(picked)}")

    best: Generated | None = None
    tried = []
    for sc, r, e in picked:
        avoid = {_undirected(x) for x in r}
        rets = []
        if e != start:
            for _ in range(20):
                rr = walk_route(net, geo, e, start, back, dist_to_start, rng, avoid=avoid, close_slack_m=80.0)
                if rr:
                    length = sum(g.edge_length(*x) for x in rr)
                    clash = sum(g.edge_length(*x) for x in rr if _undirected(x) in avoid)
                    rets.append((DistanceFit(back).score(length, []).value - 2 * clash / max(length, 1), rr))
            ret = max(rets, key=lambda t: t[0])[1] if rets else (net.path_edges(e, start) or [])
        else:
            ret = []
        base = Trail(start=start, end=e, true_path=[EdgeRef(*x) for x in r],
                     return_path=[EdgeRef(*x) for x in ret],
                     meta={"source": "generated", "seed": seed, "target_miles": o.miles,
                           "generated": date.today().isoformat()})
        for p in range(o.placements):
            lo = o if p == 0 else vary_layout(o, rng)
            t = place_marks(base, g, random.Random(rng.random()), lo)
            model = TrailModel(t, g)
            results = simulate_many(t, g, pack, runs=o.runs, seed=rng.randrange(10**6), model=model)
            card = score_results(results, target)
            tried.append((sc, card.total))
            if best is None or card.total > best.card.total:
                best = Generated(t, card, results)
    best.tried = tried
    return best
