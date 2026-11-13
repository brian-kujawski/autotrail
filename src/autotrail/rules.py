"""Hash rules that both the trail generator and the simulated hounds share.

* **Straight on.** At a corner with no mark, a hound assumes trail goes
  straight on. "Straight" is the one way out that is close to the incoming
  direction (within ``STRAIGHT_DEG``) while every other way is clearly off it
  (at least ``CLEAR_DEG`` more). A road bend with only one way on is straight
  too. Anything else (a turn, a fork with two near-straight legs, a T) needs a
  check, an arrow or a whichy-way.
* **Jogs are straight.** Two turns that cancel out (the way out is within
  ``STRAIGHT_DEG`` of the way in) with under ``JOG_M`` between them, such as
  a step onto a sidewalk or across an offset crossing, are just straight on:
  no marks needed, and hounds follow them without thinking.
* **Keep apart.** Trail never crosses or comes near itself, including the
  other sidewalk of a street it already ran: a checking hound could shortcut
  across, and the pack would see the hare. ``Footprint`` keeps track of where
  trail has been and answers "would this edge come within ``radius_m`` of
  trail laid earlier?".
"""

from __future__ import annotations

import math

import numpy as np

from .graph import StreetGraph

COVER_M = 20.0        # one arrow covers a cluster of corners this close (sidewalk jogs, crosswalks)
STRAIGHT_DEG = 20.0   # the straight-on leg is at most this far off the incoming direction
CLEAR_DEG = 20.0      # ...and every other leg is at least this much further off
HEADING_M = 15.0      # direction of a street is measured over its first/last few metres
JOG_M = 20.0          # two turns that cancel out within this distance are just straight on


SIDEWALK_TAGS = {"sidewalk", "crossing", "traffic_island"}


def is_sidewalk(data: dict) -> bool:
    """A separately mapped sidewalk or crosswalk (OSM footway=sidewalk/crossing).

    Hash trail follows streets, and which side of the street doesn't matter, so
    the trail network uses street centrelines and real paths, not sidewalks.
    Otherwise crossing to the other sidewalk looks like a turn, and running back
    down the other side looks like a different street.
    """
    fw = data.get("footway")
    vals = fw if isinstance(fw, list) else [fw]
    return any(str(v) in SIDEWALK_TAGS for v in vals)


class Geometry:
    """Cached metric geometry of graph edges, plus street headings at corners.

    With ``streets_only``, corners only count streets and paths, not sidewalks
    and crosswalks (see ``is_sidewalk``). It is off by default: where OSM maps
    sidewalks separately, the walking network often leaves the street out.
    """

    def __init__(self, graph: StreetGraph, streets_only: bool = False):
        self.g = graph
        self.streets_only = streets_only
        self._edge: dict[tuple, tuple] = {}
        self._node_xy: dict = {}
        self._head: dict = {}
        self._straight: dict = {}

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

    def _key(self, u, v):
        return (u, v, self.g.best_key(u, v))

    def heading_out(self, u, v) -> float:
        """Direction leaving u toward v, measured over the first few metres of the street."""
        key = ("out", u, v)
        if key not in self._head:
            self._head[key] = self._heading_out(u, v)
        return self._head[key]

    def heading_in(self, u, v) -> float:
        """Direction arriving at v from u, over the last few metres of the street."""
        key = ("in", u, v)
        if key not in self._head:
            self._head[key] = self._heading_in(u, v)
        return self._head[key]

    def _heading_out(self, u, v) -> float:
        cum, x, y, _, _ = self.edge_poly(self._key(u, v))
        d = min(HEADING_M, cum[-1])
        px, py = np.interp(d, cum, x), np.interp(d, cum, y)
        if px == x[0] and py == y[0]:
            return self.bearing(u, v)
        return math.atan2(py - y[0], px - x[0])

    def _heading_in(self, u, v) -> float:
        cum, x, y, _, _ = self.edge_poly(self._key(u, v))
        d = max(0.0, cum[-1] - HEADING_M)
        px, py = np.interp(d, cum, x), np.interp(d, cum, y)
        if px == x[-1] and py == y[-1]:
            return self.bearing(u, v)
        return math.atan2(y[-1] - py, x[-1] - px)

    def deviation(self, prev, node, nxt) -> float:
        """Degrees off straight going prev -> node -> nxt (0 = dead straight, 180 = U-turn)."""
        a = self.heading_in(prev, node)
        b = self.heading_out(node, nxt)
        return abs(math.degrees((b - a + math.pi) % (2 * math.pi) - math.pi))

    def is_street(self, u, v) -> bool:
        if not self.streets_only:
            return True
        return any(not is_sidewalk(d) for d in self.g.G[u][v].values())

    def ways_on(self, prev, node) -> list:
        """Every way out of a corner except back the way you came."""
        return [nb for nb in self.g.G.successors(node) if nb not in (prev, node) and self.is_street(node, nb)]

    def straight_on(self, prev, node):
        """Where a hound goes from an unmarked corner, or None if there's no clear straight on."""
        key = (prev, node)
        if key not in self._straight:
            self._straight[key] = self._straight_on(prev, node)
        return self._straight[key]

    def _straight_on(self, prev, node):
        ways = self.ways_on(prev, node)
        if not ways:
            return None
        if len(ways) == 1:
            return ways[0]
        devs = sorted((self.deviation(prev, node, nb), nb) for nb in ways)
        if devs[0][0] <= STRAIGHT_DEG and devs[1][0] - devs[0][0] >= CLEAR_DEG:
            return devs[0][1]
        return None

    def is_corner(self, prev, node) -> bool:
        return len(self.ways_on(prev, node)) >= 2


def jog_corners(geo: Geometry, edges: list[tuple], cum) -> set[int]:
    """Indices i (corner = ``edges[i][0]``) that sit inside a jog along a path.

    A jog starts at a turn and ends at a later corner within ``JOG_M`` where
    the way out lines up again with the way in (within ``STRAIGHT_DEG``).
    """
    out: set[int] = set()
    n = len(edges)
    for i in range(1, n):
        if i in out:
            continue
        prev, node, nxt = edges[i - 1][0], edges[i][0], edges[i][1]
        if geo.deviation(prev, node, nxt) <= STRAIGHT_DEG:
            continue
        a = geo.heading_in(prev, node)
        for j in range(i + 1, n):
            if cum[j] - cum[i] > JOG_M:
                break
            b = geo.heading_out(edges[j][0], edges[j][1])
            if abs(math.degrees((b - a + math.pi) % (2 * math.pi) - math.pi)) <= STRAIGHT_DEG:
                out.update(range(i, j + 1))
                break
    return out


def turn_angle(geo: Geometry, prev, node, nxt) -> float:
    """Absolute turn in radians going prev -> node -> nxt (0 = straight on)."""
    return math.radians(geo.deviation(prev, node, nxt))


class Footprint:
    """Where trail has been, for the keep-apart rule.

    Points are sampled every ``step_m`` along each edge added and kept in a grid
    of ``radius_m`` cells. Each point carries a tag (which leg) and its
    distance along that leg, so the caller can exempt the stretch just run
    (a corner is always close to the street you came in on).
    """

    def __init__(self, geo: Geometry, radius_m: float = 25.0, step_m: float = 8.0):
        self.geo, self.r, self.step = geo, radius_m, step_m
        self.cells: dict[tuple[int, int], list[tuple[float, float, str, float]]] = {}
        self.zones: list[tuple[float, float, float]] = []  # (x, y, r): ignore points in here

    def _samples(self, e: tuple):
        """[(metres along, x, y)] every ``step_m`` along the edge (cached on the Geometry)."""
        cache = self.geo.__dict__.setdefault("_samples", {})
        key = (e, self.step)
        hit = cache.get(key)
        if hit is None:
            cum, x, y, _, _ = self.geo.edge_poly(e)
            n = max(2, int(cum[-1] // self.step) + 1)
            d = np.linspace(0.0, cum[-1], n)
            hit = cache[key] = list(zip(d.tolist(), np.interp(d, cum, x).tolist(), np.interp(d, cum, y).tolist()))
        return hit

    def _cell(self, x, y):
        return (int(math.floor(x / self.r)), int(math.floor(y / self.r)))

    def add(self, e: tuple, tag: str, pos0: float) -> None:
        for di, x, y in self._samples(e):
            self.cells.setdefault(self._cell(x, y), []).append((x, y, tag, pos0 + di))

    def add_path(self, edges, tag: str, pos0: float = 0.0) -> None:
        pos = pos0
        for e in edges:
            self.add(e, tag, pos)
            pos += self.geo.g.edge_length(*e)

    def in_zone(self, x, y) -> bool:
        return any((x - zx) ** 2 + (y - zy) ** 2 <= zr * zr for zx, zy, zr in self.zones)

    def clashes(self, e: tuple, tag: str, pos0: float, recent_m: float = 50.0) -> bool:
        """Would edge e (starting ``pos0`` metres along leg ``tag``) come too close to trail?

        Points of the same leg laid in the last ``recent_m`` metres don't count.
        """
        r2 = self.r * self.r
        for di, x, y in self._samples(e):
            here = pos0 + di
            cx, cy = self._cell(x, y)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for px, py, ptag, ppos in self.cells.get((cx + dx, cy + dy), ()):
                        if ptag == tag and ppos >= here - recent_m - 1e-6:
                            continue
                        if (px - x) ** 2 + (py - y) ** 2 <= r2:
                            if self.zones and self.in_zone(x, y):
                                continue
                            return True
        return False

    def node_clear(self, n, tag: str, pos: float, recent_m: float = 50.0) -> bool:
        """Is node n clear of earlier trail (ignoring the recent stretch of leg ``tag``)?"""
        x, y = self.geo.node_xy(n)
        if self.zones and self.in_zone(x, y):
            return True
        cx, cy = self._cell(x, y)
        r2 = self.r * self.r
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for px, py, ptag, ppos in self.cells.get((cx + dx, cy + dy), ()):
                    if ptag == tag and ppos >= pos - recent_m - 1e-6:
                        continue
                    if (px - x) ** 2 + (py - y) ** 2 <= r2:
                        return False
        return True

    def path_clashes(self, edges, tag: str, pos0: float = 0.0, recent_m: float = 50.0) -> bool:
        """Check a whole path, adding it as it goes (on a copy-free trial basis: caller decides)."""
        pos = pos0
        for e in edges:
            if self.clashes(e, tag, pos, recent_m):
                return True
            self.add(e, tag, pos)
            pos += self.geo.g.edge_length(*e)
        return False

    def copy(self) -> "Footprint":
        fp = Footprint(self.geo, self.r, self.step)
        fp.cells = {k: list(v) for k, v in self.cells.items()}
        fp.zones = list(self.zones)
        return fp
