"""Corner facts: what the hare knows at every corner of a finished route.

Mark-pass step 2. Marks used to be laid going forward, one corner at a time,
so a check could invite hounds to lop off a loop the hare hadn't laid yet.
Here the whole route is known: for every corner, and every way out of it that
trail doesn't take, we work out what a hound who went that way would find.

For each wrong way:

* **shortcut**: within ``PROBE_M`` on foot it reaches trail still to come
  (or the way back) and skips at least ``SKIP_RATIO`` times the distance
  walked. A check here invites the pack to lop off trail.
* **way_back**: a shortcut whose far end is the way back from the Beer Near
  (common near Start, where trail leaves right beside it).
* **rejoins**: it reaches trail, but saves nothing much (or it's trail
  already run).
* **dead_end**: everything down it stays within ``PROBE_M``: a cul-de-sac or
  a closed pocket. A safe wrong way for a check, and a place for a false.
* **open**: no trail within ``PROBE_M``. A safe wrong way.
* **back**: it points back the way hounds came (the other sidewalk heading
  back, say). Never a whichy-way's second way.

The walk counts every path type except busy roads (``BUSY``); hounds won't
run down those. Each corner also records whether hounds checking the wrong
ways would see the hare (given pack timing), and how much trail is left.

Nothing here changes marks yet: step 3 uses these facts to place them.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from .graph import StreetGraph
from .rules import PARALLEL_DEG, Geometry
from .sight import PackTiming, SightLines, check_looks_at_hare

PROBE_M = 250.0       # how far a hound on a wrong way gets before it gives up (and how far we look)
SKIP_RATIO = 3.0      # a wrong way is a shortcut if it skips this many times what it costs...
MIN_SKIP_M = 100.0    # ...and at least this much trail (rejoining next to the corner isn't a shortcut)
SEE_TRAIL_M = 20.0    # a hound this close to trail finds it
SAMPLE_M = 8.0        # trail is sampled this often for the "near trail" test
STUB_M = 80.0         # how much of a plain wrong way to draw
BUSY = {"motorway", "motorway_link", "trunk", "trunk_link"}

KINDS = ("shortcut", "way_back", "rejoins", "dead_end", "open", "back", "busy")


@dataclass
class WayFact:
    to: object                  # the first node down this way
    kind: str                   # one of KINDS
    walk_m: float | None = None   # on foot from the corner to where it meets trail
    skip_m: float | None = None   # trail it skips (metres along trail between the corner and that point)
    meets_s: float | None = None  # where it meets trail (metres along; past the Beer Near = the way back)
    path: list = field(default_factory=list)  # nodes from the corner: to the meeting point, or a short stub
    sees_hare: bool = False


@dataclass
class CornerFact:
    i: int
    s: float
    node: object
    prev: object                # the way hounds arrive from
    nxt: object                 # the way trail goes on
    turn: bool
    jog: bool
    left_m: float               # trail still to come, to the Beer Near
    ways: list[WayFact]

    @property
    def shortcut(self) -> bool:
        return any(w.kind in ("shortcut", "way_back") for w in self.ways)

    @property
    def sees_hare(self) -> bool:
        return any(w.sees_hare for w in self.ways)


def _highway(d: dict) -> set:
    h = d.get("highway")
    return {str(x) for x in h} if isinstance(h, list) else {str(h)}


def _busy(d: dict) -> bool:
    return bool(_highway(d) & BUSY)


class _TrailIndex:
    """Trail sampled every ``SAMPLE_M`` with its distance along, for "is this near trail?"."""

    def __init__(self, geo: Geometry, paths: list[tuple[list[tuple], float]]):
        self.cells: dict[tuple, list] = {}
        self.c = SEE_TRAIL_M
        for edges, s0 in paths:
            s = s0
            for e in edges:
                cum, x, y, _, _ = geo.edge_poly(e)
                L = float(cum[-1])
                for d in np.arange(0.0, L + 1e-9, SAMPLE_M) if L > 0 else [0.0]:
                    px, py = float(np.interp(d, cum, x)), float(np.interp(d, cum, y))
                    self.cells.setdefault((int(px // self.c), int(py // self.c)), []).append((px, py, s + float(d)))
                s += L

    def near(self, x: float, y: float) -> list[float]:
        """Where trail passes within ``SEE_TRAIL_M`` of (x, y): metres along at its closest point,
        one per stretch of trail (a corner between two arms of trail is near both)."""
        cx, cy = int(x // self.c), int(y // self.c)
        r2 = self.c * self.c
        hits = []
        for i in (cx - 1, cx, cx + 1):
            for j in (cy - 1, cy, cy + 1):
                for px, py, s in self.cells.get((i, j), ()):
                    d2 = (px - x) ** 2 + (py - y) ** 2
                    if d2 <= r2:
                        hits.append((s, d2))
        hits.sort()
        out, run = [], []
        for h in hits:
            if run and h[0] - run[-1][0] > 2 * SAMPLE_M:
                out.append(min(run, key=lambda t: t[1])[0])
                run = []
            run.append(h)
        if run:
            out.append(min(run, key=lambda t: t[1])[0])
        return out


def _walk(g: StreetGraph, geo: Geometry, idx: _TrailIndex, corner, nb, s: float, d0: float):
    """Bounded walk from ``corner`` out via ``nb`` (never back through the corner).

    Returns (best shortcut or None, best rejoin or None, escaped, prev map). A hit is
    (skip ratio, skip_m, walk_m, meets_s, node).
    """
    G = g.G
    dist = {nb: d0}
    prev = {nb: corner}
    pq = [(d0, nb)]
    best_cut = best_join = None
    escaped = False
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, math.inf):
            continue
        x, y = geo.node_xy(u)
        for t in idx.near(x, y):
            if abs(t - s) <= d + SEE_TRAIL_M:
                continue  # trail right by the corner: you can always get back to that
            skip = t - s
            hit = (skip / max(d, 1.0), skip, d, t, u)
            if skip >= MIN_SKIP_M and skip >= SKIP_RATIO * d:
                if best_cut is None or hit[0] > best_cut[0]:
                    best_cut = hit
            elif best_join is None or d < best_join[2]:
                best_join = hit
        for v, keys in G[u].items():
            if v in (u, corner):
                continue
            ok = [float(kd["length"]) for kd in keys.values() if not _busy(kd)]
            if not ok:
                continue
            nd = d + min(ok)
            if nd > PROBE_M:
                escaped = True
                continue
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return best_cut, best_join, escaped, prev


def _path_to(prev: dict, n) -> list:
    out = [n]
    while n in prev:
        n = prev[n]
        out.append(n)
    return out[::-1]


def _stub(g: StreetGraph, geo: Geometry, corner, nb) -> list:
    """The way a hound runs straight on down a wrong way, for drawing (about ``STUB_M``)."""
    out, total = [corner, nb], g.edge_length(corner, nb, g.best_key(corner, nb))
    p, c = corner, nb
    while total < STUB_M:
        n = geo.straight_on(p, c)
        if n is None or n in out:
            break
        total += g.edge_length(c, n, g.best_key(c, n))
        out.append(n)
        p, c = c, n
    return out


def _back(geo: Geometry, prev, node, nb) -> bool:
    """Does the way node -> nb head back the way hounds came (prev -> node)?"""
    a = geo.heading_in(prev, node) + math.pi
    b = geo.heading_out(node, nb)
    return abs(math.degrees((b - a + math.pi) % (2 * math.pi) - math.pi)) <= PARALLEL_DEG


def corner_facts(g: StreetGraph, true_path: list[tuple], return_path: list[tuple] = (),
                 timing: PackTiming | None = None, geo: Geometry | None = None) -> list[CornerFact]:
    """Facts for every corner on ``true_path`` (edges (u, v, key)), read with the whole route known."""
    from .generate import (  # generate imports this module's users; avoid a cycle
        _cum,
        _pos,
        corners,
    )

    geo = geo or Geometry(g)
    tp = [tuple(e) for e in true_path]
    rp = [tuple(e) for e in return_path]
    cum = _cum(g, tp)
    L = float(cum[-1])
    idx = _TrailIndex(geo, [(tp, 0.0)] + ([(rp, L)] if rp else []))

    looks = None
    if timing is not None:
        lines = SightLines(geo, 150.0 + 250.0)
        d = np.arange(0.0, L, 10.0)
        xy = []
        for x in d:
            ep = _pos(tp, cum, float(x))
            ec, ex, ey, _, _ = geo.edge_poly((ep.edge.u, ep.edge.v, ep.edge.key))
            xy.append((float(np.interp(ep.offset_m, ec, ex)), float(np.interp(ep.offset_m, ec, ey))))
        true_xy = np.array(xy) if xy else np.zeros((0, 2))

        def looks(node, nb, s):
            return check_looks_at_hare(geo, lines, timing, node, [nb], true_xy, d, s)

    out = []
    for c in corners(geo, tp, cum):
        ways = []
        for nb in c.ways:
            if nb == c.nxt:
                continue
            keys = g.G[c.node][nb]
            if all(_busy(kd) for kd in keys.values()):
                ways.append(WayFact(nb, "busy", path=[c.node, nb]))
                continue
            if _back(geo, c.prev, c.node, nb):
                ways.append(WayFact(nb, "back", path=_stub(g, geo, c.node, nb)))
                continue
            d0 = min(float(kd["length"]) for kd in keys.values() if not _busy(kd))
            cut, join, escaped, prev = _walk(g, geo, idx, c.node, nb, c.s, d0)
            if cut is not None:
                _, skip, walk, t, n = cut
                w = WayFact(nb, "way_back" if t > L else "shortcut", walk, skip, t, _path_to(prev, n))
            elif join is not None:
                _, skip, walk, t, n = join
                w = WayFact(nb, "rejoins", walk, skip, t, _path_to(prev, n))
            else:
                w = WayFact(nb, "open" if escaped else "dead_end", path=_stub(g, geo, c.node, nb))
            if looks is not None:
                w.sees_hare = looks(c.node, nb, c.s)
            ways.append(w)
        out.append(CornerFact(c.i, c.s, c.node, c.prev, c.nxt, c.turn, c.jog, L - c.s, ways))
    return out


def summary(facts: list[CornerFact]) -> dict:
    """Counts for a report line: corners, turns, and wrong ways of each kind."""
    n = {k: 0 for k in KINDS}
    for f in facts:
        for w in f.ways:
            n[w.kind] += 1
    return {"corners": len(facts), "turns": sum(f.turn for f in facts),
            "shortcut_corners": sum(f.shortcut for f in facts),
            "sees_hare_corners": sum(f.sees_hare for f in facts), "ways": n}
