"""Sightlines: can a hound see the hare?

Hares hide their marks, but not themselves. A hound sees down the street it is
on, both ways, until the street bends or ends; standing at a corner it sees
down every street from that corner. Buildings block everything else.

A sightline is a "ray": from a corner, out along one street and straight on
through the corners after it (the way a hound goes at an unmarked corner),
stopping where there's no clear straight on. A hound sees the hare when both
sit on one ray, within ``SIGHT_M`` of each other along it, and the ray between
them stays within ``BEND_M`` of a straight line (a dogleg breaks the view).

The hare is "on" a ray when within ``STREET_HALF_M`` of its centreline, so a
dash straight across a street is in view only for a few seconds, while running
away down the street a hound is looking along stays in view the whole way.

Only the hare matters here: hounds seeing other hounds on another part of
trail is fine (it pulls the pack together).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .rules import Geometry

SIGHT_M = 250.0       # how far down a straight street a hound can spot the hare at night
STREET_HALF_M = 12.0  # half a street's width: closer than this to the centreline is "on" it
BEND_M = 10.0         # the street between them may wander this far off a straight line
CORNER_M = 15.0       # this close to a corner, a hound looks down every street from it
NEAR_M = 50.0         # sightings this close count in full...
FAR_WEIGHT = 0.3      # ...fading to this much at SIGHT_M


def sighting_weight(d: float) -> float:
    """How much a second in view counts: 1 up close, ``FAR_WEIGHT`` at ``SIGHT_M``."""
    if d <= NEAR_M:
        return 1.0
    return max(FAR_WEIGHT, 1.0 - (1.0 - FAR_WEIGHT) * (d - NEAR_M) / (SIGHT_M - NEAR_M))


class SightLines:
    def __init__(self, geo: Geometry, sight_m: float = SIGHT_M):
        self.geo, self.g, self.sight_m = geo, geo.g, sight_m
        self._rays: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def ray(self, node, nb) -> tuple[np.ndarray, np.ndarray, tuple]:
        """(cum metres, [n, 2] x/y, bounding box) from ``node`` out via ``nb`` and straight on."""
        key = (node, nb)
        hit = self._rays.get(key)
        if hit is not None:
            return hit
        cums, pts, total = [], [], 0.0
        prev, cur, seen = node, nb, {node}
        while True:
            cum, x, y, _, _ = self.geo.edge_poly((prev, cur, self.g.best_key(prev, cur)))
            sl = slice(0 if not pts else 1, None)
            cums.append(cum[sl] + total)
            pts.append(np.column_stack([x, y])[sl])
            total += float(cum[-1])
            seen.add(cur)
            if total >= self.sight_m + 2 * CORNER_M:
                break
            nxt = self.geo.straight_on(prev, cur)
            if nxt is None or nxt in seen:
                break
            prev, cur = cur, nxt
        cum, p = np.concatenate(cums), np.concatenate(pts)
        lo, hi = p.min(axis=0) - STREET_HALF_M, p.max(axis=0) + STREET_HALF_M
        hit = self._rays[key] = (cum, p, (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])))
        return hit

    def rays_from(self, u, v, offset: float, length: float) -> list[tuple]:
        """Rays a hound sees from ``offset`` metres along edge u -> v."""
        keys = [(u, v), (v, u)]
        for n, d in ((u, offset), (v, length - offset)):
            if d <= CORNER_M:
                keys += [(n, nb) for nb in self.g.G.successors(n) if nb != n]
        out, done = [], set()
        for k in keys:
            if k not in done:
                done.add(k)
                out.append(self.ray(*k))
        return out

    def seen(self, u, v, offset: float, length: float, hound_xy, hare_xy) -> bool:
        """Can a hound at ``offset`` along u -> v (at ``hound_xy``) see the hare at ``hare_xy``?"""
        hx, hy = hare_xy
        dx, dy = hound_xy
        if (hx - dx) ** 2 + (hy - dy) ** 2 > self.sight_m ** 2:
            return False
        for cum, p, (x0, y0, x1, y1) in self.rays_from(u, v, offset, length):
            if not (x0 <= hx <= x1 and y0 <= hy <= y1):
                continue
            a_hare, off = _project(cum, p, hx, hy)
            if off > STREET_HALF_M:
                continue
            a_dog, _ = _project(cum, p, dx, dy)
            if abs(a_hare - a_dog) > self.sight_m:
                continue
            lo, hi = sorted((a_dog, a_hare))
            inner = p[(cum > lo) & (cum < hi)]
            if len(inner) and _max_off_chord(inner, np.interp(lo, cum, p[:, 0]), np.interp(lo, cum, p[:, 1]),
                                             np.interp(hi, cum, p[:, 0]), np.interp(hi, cum, p[:, 1])) > BEND_M:
                continue
            return True
        return False


def _project(cum: np.ndarray, p: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """(metres along the polyline, distance off it) of the nearest point to (x, y)."""
    if len(p) == 1:
        return 0.0, float(np.hypot(p[0, 0] - x, p[0, 1] - y))
    a, b = p[:-1], p[1:]
    ab = b - a
    l2 = (ab ** 2).sum(axis=1)
    t = np.where(l2 > 0, ((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) / np.where(l2 > 0, l2, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    qx, qy = a[:, 0] + t * ab[:, 0], a[:, 1] + t * ab[:, 1]
    d = np.hypot(qx - x, qy - y)
    i = int(np.argmin(d))
    return float(cum[i] + t[i] * (cum[i + 1] - cum[i])), float(d[i])


def _max_off_chord(pts: np.ndarray, x0, y0, x1, y1) -> float:
    vx, vy = x1 - x0, y1 - y0
    n = float(np.hypot(vx, vy))
    if n == 0:
        return float(np.hypot(pts[:, 0] - x0, pts[:, 1] - y0).max())
    return float(np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx).max() / n)


@dataclass
class Sighting:
    t0: float            # pack clock, seconds
    t1: float
    exposure_s: float    # seconds in view, weighted by distance
    closest_m: float
    hare_s: float        # where the hare was, metres along the true trail
    hound: str           # who saw it first


def hare_sightings(result, pace_s_per_m: float, head_start_s: float, dt: float = 3.0,
                   sight: SightLines | None = None, cap_s: float = 120.0) -> list[Sighting]:
    """Every stretch of time some hound had the hare in view during one sim run.

    Once ``cap_s`` weighted seconds of sightings are found the trail is plainly
    blown, so the search stops there (the list is then incomplete).
    """
    from .sim import hare_legs, legs_at

    m = result.model
    if m is None:
        return []
    sight = sight or sight_lines_for(m)
    hl = hare_legs(m, pace_s_per_m, head_start_s)
    t_end = min(hl[-1].t1, max((l.t1 for h in result.raw for l in h.legs), default=0.0))
    if t_end <= 0:
        return []
    times = np.arange(0.0, t_end, dt)
    h_tracks, h_s = legs_at(hl, times)
    hare_xy = np.full((len(times), 2), np.nan)
    hare_true_s = np.zeros(len(times))
    side_at = {id(o.track): c.s for c in m.checks for o in c.options if o.track is not None}
    for i, (tr, s) in enumerate(zip(h_tracks, h_s)):
        if tr is None:
            continue
        hare_xy[i] = [float(v) for v in tr.xy(s)]
        hare_true_s[i] = float(s) if tr is m.true else side_at.get(id(tr), 0.0)

    best = np.zeros(len(times))      # weight of the best view at each time
    dist = np.full(len(times), np.inf)
    who: list = [None] * len(times)
    for h in result.raw:
        tracks, s = legs_at(h.legs, times)
        xy = np.full((len(times), 2), np.nan)
        groups: dict[int, tuple] = {}
        for i, t in enumerate(tracks):
            if t is not None:
                groups.setdefault(id(t), (t, []))[1].append(i)
        for tr, idx in groups.values():
            idx = np.asarray(idx)
            x, y = tr.xy(s[idx])
            xy[idx, 0], xy[idx, 1] = x, y
        d = np.hypot(xy[:, 0] - hare_xy[:, 0], xy[:, 1] - hare_xy[:, 1])
        for i in np.where(d <= sight.sight_m)[0]:
            if best[i] >= 1.0:
                continue
            tr = tracks[i]
            si = float(s[i])
            k = min(len(tr.edges) - 1, max(0, int(np.searchsorted(tr.cum, si, "right")) - 1))
            u, v, _ = tr.edges[k]
            length = float(tr.cum[k + 1] - tr.cum[k])
            if sight.seen(u, v, si - float(tr.cum[k]), length, xy[i], hare_xy[i]):
                w = sighting_weight(float(d[i]))
                if w > best[i]:
                    best[i], who[i] = w, h.prof.name
                dist[i] = min(dist[i], float(d[i]))
        if best.sum() * dt >= cap_s:
            break

    out: list[Sighting] = []
    i, n = 0, len(times)
    while i < n:
        if best[i] <= 0:
            i += 1
            continue
        j = i
        while j + 1 < n and best[j + 1] > 0:
            j += 1
        sl = slice(i, j + 1)
        out.append(Sighting(float(times[i]), float(times[j] + dt), float(best[sl].sum() * dt),
                            float(dist[sl].min()), float(hare_true_s[i]), who[i]))
        i = j + 1
    return out


def sight_lines_for(model) -> SightLines:
    """One SightLines per compiled trail (rays are cached on it)."""
    sl = model.__dict__.get("_sight")
    if sl is None:
        sl = model.__dict__["_sight"] = SightLines(model.geo)
    return sl


# ---------------------------------------------------------------- planning (generator nudges)

@dataclass(frozen=True)
class PackTiming:
    """Rough clocks for planning: where the pack is while the hare lays each bit of trail.

    Checks and wrong turns slow the pack, so the effective paces are a little
    slower than running pace (``front_slack``, ``back_slack``); ``margin_m``
    widens every window to allow for hounds probing out from checks.
    """
    hare_pace: float          # seconds per metre
    head_start_s: float
    front_pace: float
    back_pace: float
    front_slack: float = 1.05
    back_slack: float = 1.2
    margin_m: float = 150.0

    @classmethod
    def of(cls, hare, pack) -> PackTiming:
        paces = [p.pace_s_per_m for p in pack]
        return cls(hare.pace_s_per_m, hare.head_start_s, min(paces), max(paces))

    def pack_window(self, hare_pos: float) -> tuple[float, float] | None:
        """Stretch of earlier trail the pack covers while the hare is ``hare_pos`` metres along."""
        t = hare_pos * self.hare_pace - self.head_start_s
        if t <= 0:
            return None  # the pack hasn't left Start yet
        return (t / (self.back_pace * self.back_slack) - self.margin_m,
                t / (self.front_pace * self.front_slack) + self.margin_m)

    def hare_window(self, s: float, linger_s: float = 120.0) -> tuple[float, float]:
        """Stretch of trail the hare runs while the pack is at a check ``s`` metres along."""
        t0 = s * self.front_pace * self.front_slack
        t1 = s * self.back_pace * self.back_slack + linger_s
        return ((t0 + self.head_start_s) / self.hare_pace - 2 * self.margin_m,
                (t1 + self.head_start_s) / self.hare_pace)


def _resample(cum: np.ndarray, p: np.ndarray, upto: float, step: float) -> np.ndarray:
    d = np.arange(0.0, min(upto, float(cum[-1])) + 1e-6, step)
    return np.column_stack([np.interp(d, cum, p[:, 0]), np.interp(d, cum, p[:, 1])])


class SightField:
    """Everything the pack could see from trail laid so far, tagged by how far along it is.

    The route search asks how much of a new street would be in view of the
    pack when the hare runs it, and charges for those metres.
    """

    def __init__(self, geo: Geometry, timing: PackTiming, sight_m: float = SIGHT_M, step_m: float = 8.0):
        self.geo, self.timing, self.step = geo, timing, step_m
        self.lines = SightLines(geo, sight_m)
        self.sight_m = sight_m
        self.r = STREET_HALF_M
        self.cells: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
        self._done: set = set()

    def _cell(self, x, y):
        return (int(math.floor(x / self.r)), int(math.floor(y / self.r)))

    def add_node(self, n, pos: float) -> None:
        if (n, round(pos)) in self._done:
            return
        self._done.add((n, round(pos)))
        for nb in self.geo.g.G.successors(n):
            if nb == n:
                continue
            cum, p, _ = self.lines.ray(n, nb)
            for x, y in _resample(cum, p, self.sight_m, self.step):
                self.cells.setdefault(self._cell(x, y), []).append((float(x), float(y), pos))

    def add_edge(self, e: tuple, pos: float) -> None:
        self.add_node(e[0], pos)
        self.add_node(e[1], pos + self.geo.g.edge_length(*e))

    def exposed_m(self, e: tuple, pos0: float) -> float:
        """Metres of edge e (laid from ``pos0``) the pack could see as the hare runs it."""
        if not self.cells:
            return 0.0
        w = self.timing.pack_window(pos0)
        if w is None:
            return 0.0
        lo, hi = w
        r2 = self.r * self.r
        samples = self.geo.__dict__.setdefault("_samples", {})
        key = (e, self.step)
        pts = samples.get(key)
        if pts is None:
            cum, x, y, _, _ = self.geo.edge_poly(e)
            n = max(2, int(cum[-1] // self.step) + 1)
            d = np.linspace(0.0, cum[-1], n)
            pts = samples[key] = list(zip(d.tolist(), np.interp(d, cum, x).tolist(), np.interp(d, cum, y).tolist()))
        hit = 0
        for _, x, y in pts:
            cx, cy = self._cell(x, y)
            found = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for px, py, ppos in self.cells.get((cx + dx, cy + dy), ()):
                        if lo <= ppos <= hi and (px - x) ** 2 + (py - y) ** 2 <= r2:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            hit += found
        return hit * self.step


def check_looks_at_hare(geo: Geometry, lines: SightLines, timing: PackTiming, node, ways: list,
                        true_xy: np.ndarray, true_s: np.ndarray, s: float,
                        probe_m: float = 150.0) -> bool:
    """Would hounds checking the wrong ways at a check ``s`` metres along have the hare in view?

    A hound runs ``probe_m`` out each way and looks on down the street from there.
    """
    lo, hi = timing.hare_window(s)
    sel = (true_s >= lo) & (true_s <= hi)
    if not sel.any():
        return False
    pts = true_xy[sel]
    reach = probe_m + SIGHT_M
    for nb in ways:
        cum, p, (x0, y0, x1, y1) = lines.ray(node, nb)
        inside = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        for x, y in pts[inside]:
            a, off = _project(cum, p, float(x), float(y))
            if off <= STREET_HALF_M and 0 < a <= reach:
                return True
    return False
