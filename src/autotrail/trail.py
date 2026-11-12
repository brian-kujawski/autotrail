"""The Trail format: the contract between generators, hares and the simulator.

A trail is a route on the street graph plus marks at exact positions along
its edges. It saves to GeoJSON, which any map editor (geojson.io, QGIS) can
open, and which carries the graph ids so it reloads exactly.

Two ways in:
  * ``Trail.from_geojson``: an Autotrail file (has edge ids). No graph needed.
  * ``snap_drawn_trail``: a hand-drawn GeoJSON (just lines and points). Lines
    are snapped onto the street graph, points with a ``kind`` become marks.
``load_trail`` picks the right one and falls back to snapping if the ids no
longer match the graph (for example after a fresh map download).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import gpxpy
import gpxpy.gpx

from .graph import StreetGraph

FORMAT = "autotrail.trail"
FORMAT_VERSION = 1
METRES_PER_MILE = 1609.344

NodeId = int


class MarkKind(str, Enum):
    BLOB = "BLOB"
    ARROW = "ARROW"
    CHECK = "CHECK"
    BACKCHECK = "BACKCHECK"
    FALSE = "FALSE"
    WHICHY_WAY = "WHICHY_WAY"
    SPLIT = "SPLIT"
    REJOIN = "REJOIN"
    BEER = "BEER"

    @classmethod
    def parse(cls, s: str) -> "MarkKind":
        key = str(s).strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {"BC": "BACKCHECK", "WHICHYWAY": "WHICHY_WAY", "BEER_CHECK": "BEER",
                   "BEER_NEAR": "BEER", "ON_ON": "BLOB", "TURKEY_EAGLE": "SPLIT"}
        return cls(aliases.get(key, key))


class BranchPurpose(str, Enum):
    FALSE = "FALSE"
    BACKCHECK_LURE = "BACKCHECK_LURE"
    TURKEY = "TURKEY"
    EAGLE = "EAGLE"


@dataclass(frozen=True)
class EdgeRef:
    u: NodeId
    v: NodeId
    key: int = 0

    def as_list(self) -> list:
        return [self.u, self.v, self.key]

    @classmethod
    def of(cls, seq) -> "EdgeRef":
        u, v, *rest = seq
        return cls(int(u), int(v), int(rest[0]) if rest else 0)


@dataclass
class EdgePos:
    """A position on the graph: an edge plus metres from its start, or a node."""
    edge: EdgeRef | None = None
    offset_m: float = 0.0
    node: NodeId | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {}
        if self.edge is not None:
            d["edge"] = self.edge.as_list()
            d["offset_m"] = round(self.offset_m, 2)
        if self.node is not None:
            d["node"] = self.node
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EdgePos":
        edge = EdgeRef.of(d["edge"]) if d.get("edge") is not None else None
        node = int(d["node"]) if d.get("node") is not None else None
        return cls(edge=edge, offset_m=float(d.get("offset_m", 0.0)), node=node)


@dataclass
class Mark:
    kind: MarkKind
    at: EdgePos
    params: dict = field(default_factory=dict)


@dataclass
class Branch:
    """A side path off the true trail: a false, a backcheck lure, or a split leg."""
    purpose: BranchPurpose
    edges: list[EdgeRef]
    label: str = ""

    @property
    def from_node(self) -> NodeId | None:
        return self.edges[0].u if self.edges else None


@dataclass
class Trail:
    start: NodeId
    end: NodeId                 # Beer Near: pre-staged cooler, ~5 mi out
    true_path: list[EdgeRef]    # the real trail Start -> Beer Near, in order
    branches: list[Branch] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # hare, date, notes, source
    return_path: list[EdgeRef] = field(default_factory=list)  # the shorter hared leg back to Start

    # ---------------------------------------------------------- measures
    @staticmethod
    def path_length_m(graph: StreetGraph, edges: Iterable[EdgeRef]) -> float:
        return sum(graph.edge_length(e.u, e.v, e.key) for e in edges)

    def true_trail_m(self, graph: StreetGraph) -> float:
        """Distance a perfect hound runs Start -> Beer Near."""
        return self.path_length_m(graph, self.true_path)

    def return_m(self, graph: StreetGraph) -> float:
        return self.path_length_m(graph, self.return_path)

    def all_edges(self) -> list[EdgeRef]:
        out = list(self.true_path) + list(self.return_path)
        for b in self.branches:
            out.extend(b.edges)
        return out

    # ---------------------------------------------------------- validation
    def validate(self, graph: StreetGraph) -> list[str]:
        """Return a list of problems; empty means the trail fits the graph."""
        errs: list[str] = []
        for e in self.all_edges():
            if not graph.has_edge(e.u, e.v, e.key):
                errs.append(f"edge {e.as_list()} is not in the graph")
        if errs:
            return errs
        errs += _check_chain("true_path", self.true_path, self.start, self.end)
        if self.return_path:
            errs += _check_chain("return_path", self.return_path, self.end, self.start)
        for i, b in enumerate(self.branches):
            if not b.edges:
                errs.append(f"branch {i} has no edges")
                continue
            errs += _check_chain(f"branch {i}", b.edges, b.edges[0].u, b.edges[-1].v)
        for i, m in enumerate(self.marks):
            if m.at.edge is None and m.at.node is None:
                errs.append(f"mark {i} ({m.kind.value}) has no position")
            if m.at.node is not None and m.at.node not in graph.G:
                errs.append(f"mark {i} ({m.kind.value}) node {m.at.node} not in graph")
            if m.at.edge is not None:
                e = m.at.edge
                if not graph.has_edge(e.u, e.v, e.key):
                    errs.append(f"mark {i} ({m.kind.value}) edge {e.as_list()} not in graph")
                elif not -0.5 <= m.at.offset_m <= graph.edge_length(e.u, e.v, e.key) + 0.5:
                    errs.append(f"mark {i} ({m.kind.value}) offset {m.at.offset_m:.1f} m is off its edge")
        return errs

    # ---------------------------------------------------------- GeoJSON
    def to_geojson(self, graph: StreetGraph) -> dict:
        feats: list[dict] = []

        def line_feature(role: str, edges: list[EdgeRef], **props) -> dict:
            return {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": _path_coords(graph, edges)},
                "properties": {"role": role, "edges": [e.as_list() for e in edges],
                               "length_m": round(self.path_length_m(graph, edges), 1), **props},
            }

        def point_feature(lonlat, props) -> dict:
            return {"type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [round(lonlat[0], 7), round(lonlat[1], 7)]},
                    "properties": props}

        feats.append(line_feature("true_path", self.true_path))
        if self.return_path:
            feats.append(line_feature("return_path", self.return_path))
        for b in self.branches:
            feats.append(line_feature("branch", b.edges, purpose=b.purpose.value, label=b.label))
        feats.append(point_feature(graph.node_lonlat(self.start), {"role": "start", "node": self.start}))
        feats.append(point_feature(graph.node_lonlat(self.end), {"role": "beer_near", "node": self.end}))
        for m in self.marks:
            props = {"role": "mark", "kind": m.kind.value, **m.at.to_dict()}
            if m.params:
                props["params"] = m.params
            feats.append(point_feature(_pos_lonlat(graph, m.at), props))
        return {
            "type": "FeatureCollection",
            "autotrail": {
                "format": FORMAT, "version": FORMAT_VERSION, "area": graph.area_key,
                "start": self.start, "end": self.end, "meta": self.meta,
                "true_trail_m": round(self.true_trail_m(graph), 1),
                "return_m": round(self.return_m(graph), 1),
            },
            "features": feats,
        }

    def save(self, path: str | Path, graph: StreetGraph) -> None:
        Path(path).write_text(json.dumps(self.to_geojson(graph), indent=1), encoding="utf-8")

    @classmethod
    def from_geojson(cls, data: dict) -> "Trail":
        head = data.get("autotrail")
        if not head or head.get("format") != FORMAT:
            raise ValueError("Not an Autotrail trail file (use snap_drawn_trail for drawn GeoJSON)")
        if int(head.get("version", 0)) > FORMAT_VERSION:
            raise ValueError(f"Trail file version {head['version']} is newer than this Autotrail")
        true_path: list[EdgeRef] = []
        return_path: list[EdgeRef] = []
        branches: list[Branch] = []
        marks: list[Mark] = []
        for f in data.get("features", []):
            p = f.get("properties") or {}
            role = p.get("role")
            if role in ("true_path", "return_path", "branch"):
                edges = [EdgeRef.of(e) for e in p.get("edges", [])]
                if role == "true_path":
                    true_path = edges
                elif role == "return_path":
                    return_path = edges
                else:
                    branches.append(Branch(BranchPurpose(p.get("purpose", "FALSE")), edges, p.get("label", "")))
            elif role == "mark":
                marks.append(Mark(MarkKind.parse(p["kind"]), EdgePos.from_dict(p), dict(p.get("params") or {})))
        return cls(start=int(head["start"]), end=int(head["end"]), true_path=true_path,
                   branches=branches, marks=marks, meta=dict(head.get("meta") or {}),
                   return_path=return_path)

    # ---------------------------------------------------------- GPX
    def to_gpx(self, graph: StreetGraph) -> str:
        """True path (and return leg) as GPX tracks, for phones and Strava-style tools."""
        gpx = gpxpy.gpx.GPX()
        gpx.name = self.meta.get("name", "Autotrail trail")
        for name, edges in (("Start to Beer Near", self.true_path), ("Back to Start", self.return_path)):
            if not edges:
                continue
            trk = gpxpy.gpx.GPXTrack(name=name)
            seg = gpxpy.gpx.GPXTrackSegment()
            for lon, lat in _path_coords(graph, edges):
                seg.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon))
            trk.segments.append(seg)
            gpx.tracks.append(trk)
        for m in self.marks:
            lon, lat = _pos_lonlat(graph, m.at)
            gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(lat, lon, name=m.kind.value))
        return gpx.to_xml()


# ---------------------------------------------------------------- helpers

def _check_chain(name: str, edges: list[EdgeRef], first: NodeId, last: NodeId) -> list[str]:
    if not edges:
        return [f"{name} is empty"]
    errs = []
    if edges[0].u != first:
        errs.append(f"{name} starts at {edges[0].u}, expected {first}")
    if edges[-1].v != last:
        errs.append(f"{name} ends at {edges[-1].v}, expected {last}")
    for a, b in zip(edges, edges[1:]):
        if a.v != b.u:
            errs.append(f"{name} breaks between {a.as_list()} and {b.as_list()}")
    return errs


def _path_coords(graph: StreetGraph, edges: list[EdgeRef]) -> list[list[float]]:
    coords: list[list[float]] = []
    for e in edges:
        pts = [[round(x, 7), round(y, 7)] for x, y in graph.edge_line(e.u, e.v, e.key).coords]
        coords.extend(pts if not coords else pts[1:])
    return coords


def _pos_lonlat(graph: StreetGraph, at: EdgePos) -> tuple[float, float]:
    if at.edge is not None:
        return graph.point_on_edge(at.edge.u, at.edge.v, at.edge.key, at.offset_m)
    return graph.node_lonlat(at.node)


# ---------------------------------------------------------------- drawn trails

def snap_line(graph: StreetGraph, coords: list[list[float]]) -> list[EdgeRef]:
    """Snap a drawn polyline onto the graph.

    Each drawn vertex goes to its nearest graph node, and consecutive nodes are
    joined by the shortest walking path, so a few clicks per street is enough.
    """
    nodes: list[NodeId] = []
    for lon, lat, *_ in coords:
        n = graph.nearest_node(lon, lat)
        if not nodes or nodes[-1] != n:
            nodes.append(n)
    edges: list[EdgeRef] = []
    for a, b in zip(nodes, nodes[1:]):
        edges.extend(EdgeRef(u, v, k) for u, v, k in graph.shortest_edges(a, b))
    return edges


def _line_parts(geom: dict) -> list[list[list[float]]]:
    if geom["type"] == "LineString":
        return [geom["coordinates"]]
    if geom["type"] == "MultiLineString":
        return list(geom["coordinates"])
    return []


def snap_drawn_trail(data: dict, graph: StreetGraph, *, max_mark_snap_m: float = 60.0) -> Trail:
    """Build a Trail from hand-drawn GeoJSON.

    Lines: the first line (or the one with ``role: true_path``) is the trail
    from Start to Beer Near. Optional lines with ``role: return_path`` and
    ``role: branch`` (plus ``purpose``: FALSE, BACKCHECK_LURE, TURKEY, EAGLE).
    Points: ``role: start`` / ``role: beer_near`` override the line's ends;
    any point with a ``kind`` (check, false, blob, backcheck, ...) is a mark,
    snapped to the nearest edge of the trail.
    """
    true_coords = None
    return_coords = None
    branch_specs: list[tuple[str, str, list]] = []
    start_pt = end_pt = None
    mark_specs: list[tuple[dict, list]] = []
    meta: dict = {}
    head = data.get("autotrail") or {}
    if isinstance(head.get("meta"), dict):
        meta.update(head["meta"])

    for f in data.get("features", []):
        geom = f.get("geometry") or {}
        p = f.get("properties") or {}
        role = str(p.get("role", "")).lower()
        if geom.get("type") in ("LineString", "MultiLineString"):
            coords = [c for part in _line_parts(geom) for c in part]
            if role == "return_path":
                return_coords = coords
            elif role == "branch":
                branch_specs.append((p.get("purpose", "FALSE"), p.get("label", ""), coords))
            elif role == "true_path" or true_coords is None:
                true_coords = coords
        elif geom.get("type") == "Point":
            if role == "start":
                start_pt = geom["coordinates"]
            elif role in ("beer_near", "end", "beer"):
                end_pt = geom["coordinates"]
            elif p.get("kind"):
                mark_specs.append((p, geom["coordinates"]))

    if true_coords is None:
        raise ValueError("Drawn trail has no line to follow")
    if start_pt is not None:
        true_coords = [start_pt] + list(true_coords)
    if end_pt is not None:
        true_coords = list(true_coords) + [end_pt]

    true_path = snap_line(graph, true_coords)
    if not true_path:
        raise ValueError("Drawn trail snapped to a single point; draw a longer line")
    start, end = true_path[0].u, true_path[-1].v

    return_path: list[EdgeRef] = []
    if return_coords is not None:
        return_path = snap_line(graph, [list(graph.node_lonlat(end))] + list(return_coords)
                                + [list(graph.node_lonlat(start))])

    branches = []
    for purpose, label, coords in branch_specs:
        edges = snap_line(graph, coords)
        if edges:
            branches.append(Branch(BranchPurpose(str(purpose).upper()), edges, label))

    trail_edges = [(e.u, e.v, e.key) for e in true_path + return_path]
    for b in branches:
        trail_edges += [(e.u, e.v, e.key) for e in b.edges]
    marks: list[Mark] = []
    for p, (lon, lat, *_) in mark_specs:
        e, offset, dist = graph.nearest_edge(lon, lat, among=trail_edges)
        if dist > max_mark_snap_m:
            e, offset, dist = graph.nearest_edge(lon, lat)
        params = {k: v for k, v in p.items() if k not in ("kind", "role")}
        marks.append(Mark(MarkKind.parse(p["kind"]), EdgePos(EdgeRef(*e), offset), params))

    meta.setdefault("source", "drawn")
    return Trail(start=start, end=end, true_path=true_path, branches=branches,
                 marks=marks, meta=meta, return_path=return_path)


def load_trail(path: str | Path, graph: StreetGraph | None = None) -> Trail:
    """Load an Autotrail file or a hand-drawn GeoJSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    head = data.get("autotrail") or {}
    if head.get("format") == FORMAT:
        trail = Trail.from_geojson(data)
        if graph is None or not trail.validate(graph):
            return trail
        # Ids no longer match this graph: rebuild from the saved geometry.
    if graph is None:
        raise ValueError("A street graph is needed to snap a drawn trail")
    trail = snap_drawn_trail(data, graph)
    trail.meta.setdefault("file", Path(path).name)
    return trail
