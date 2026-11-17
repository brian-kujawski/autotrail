"""The street graph: download, cache, gap stitching and spatial lookups.

The graph is an OSMnx ``MultiDiGraph`` in lat/lon (EPSG:4326). Walk graphs
are bidirectional, so every street appears as two directed edges.

Edge attributes Autotrail relies on:
    length / length_m  metres (``length`` is OSMnx's name; ``length_m`` is a copy)
    highway, surface, lit, access   OSM tags (may be lists after simplification)
    landmark_value     0.0 for now; filled in by later phases
    virtual            True for edges Autotrail added to stitch an OSM gap
Node attributes: x (lon), y (lat), degree (distinct neighbours).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pyproj
import shapely
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import substring

from .config import Area, data_dir

log = logging.getLogger(__name__)

EXTRA_WAY_TAGS = ["surface", "lit", "foot", "sidewalk", "footway", "service", "bicycle"]

# Attribute types for GraphML reload (GraphML stores everything as strings).
_EDGE_DTYPES = {
    "length_m": float,
    "landmark_value": float,
    "virtual": lambda s: str(s).lower() == "true",
}
_NODE_DTYPES = {"degree": int, "virtual": lambda s: str(s).lower() == "true"}


# ---------------------------------------------------------------- download

def configure_osmnx(cache_root: Path) -> None:
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_root / "http")
    ox.settings.log_console = False
    tags = list(ox.settings.useful_tags_way)
    for t in EXTRA_WAY_TAGS:
        if t not in tags:
            tags.append(t)
    ox.settings.useful_tags_way = tags


# OSMnx's "walk" filter, but keeping cycleways: hashes run on bike paths, and OSM often tags
# a shared path through a park as highway=cycleway. Ways marked foot=no still drop out.
HASH_WALK_FILTER = (
    '["highway"]["area"!~"yes"]["access"!~"private"]'
    '["highway"!~"abandoned|bus_guideway|construction|motor|no|planned|platform|proposed|raceway|razed|rest_area|services"]'
    '["foot"!~"no"]["service"!~"private"]'
    '["sidewalk"!~"separate"]["sidewalk:both"!~"separate"]["sidewalk:left"!~"separate"]["sidewalk:right"!~"separate"]'
)
GRAPH_FORMAT = 3  # 2: cycleways included; 3: every street two-way


def download_graph(area: Area) -> nx.MultiDiGraph:
    """Download the walk network (plus cycleways) for the union of the area's places."""
    log.info("Downloading %s network for %s", area.network_type, ", ".join(area.places))
    if area.network_type == "walk":
        # network_type="walk" still matters with a custom filter: it is what tells OSMnx to
        # make every street two-way (one-way signs are for cars, not hounds).
        return ox.graph_from_place(list(area.places), network_type="walk", custom_filter=HASH_WALK_FILTER,
                                   simplify=True, retain_all=False)
    return ox.graph_from_place(
        list(area.places), network_type=area.network_type, simplify=True, retain_all=False
    )


def cache_path(area: Area, root: Path | None = None) -> Path:
    return (root or data_dir()) / "cache" / f"{area.slug}.graphml"


def load_area_graph(
    area: Area,
    *,
    force: bool = False,
    root: Path | None = None,
    stitch: bool = True,
) -> "StreetGraph":
    """Load the area's graph from cache, downloading (and stitching) on a miss."""
    root = root or data_dir()
    path = cache_path(area, root)
    if path.exists() and not force:
        G = ox.load_graphml(path, node_dtypes=_NODE_DTYPES, edge_dtypes=_EDGE_DTYPES)
        log.info("Loaded cached graph %s", path)
        if int(G.graph.get("autotrail_format", 1)) < 2:
            log.warning("The cached map %s predates bike paths being included. Run "
                        "'autotrail download --force' to refresh it.", path)
        fixed = make_two_way(G)
        if fixed:
            log.warning("The cached map had %d one-way street(s); made them two-way in memory. "
                        "'autotrail download --force' fixes the file.", fixed)
        return StreetGraph(G, area_key=area.key)

    configure_osmnx(root)
    G = download_graph(area)
    annotate(G)
    make_two_way(G)
    added = stitch_gaps(G, area.stitch_tolerance_m) if stitch else []
    log.info("Stitched %d gaps", len(added))
    G.graph["autotrail_area"] = area.key
    G.graph["autotrail_format"] = GRAPH_FORMAT
    G.graph["autotrail_stitch_tolerance_m"] = area.stitch_tolerance_m if stitch else 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, path)
    return StreetGraph(G, area_key=area.key)


def make_two_way(G: nx.MultiDiGraph) -> int:
    """Give every edge a reverse twin, so every street can be run both ways. Returns how many were added.

    Hounds ignore one-way signs. OSMnx only builds both directions for its preset walk network,
    and a map downloaded with a custom filter alone came out with car one-ways, which broke
    anything that looks back along a street (KeyError in the hare sightlines).
    """
    missing = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True)
               if u != v and not _reverse_twins(G, u, v, d["length"])]
    for u, v, k, d in missing:
        r = dict(d)
        geom = d.get("geometry")
        if geom is not None:
            r["geometry"] = LineString(list(geom.coords)[::-1])
        r["reversed"] = not bool(d.get("reversed", False))
        r["oneway"] = False
        G.add_edge(v, u, **r)
    if missing:
        _refresh_degrees(G, {n for u, v, _, _ in missing for n in (u, v)})
    return len(missing)


def annotate(G: nx.MultiDiGraph) -> None:
    """Add the attributes Autotrail expects to every node and edge."""
    for _, _, data in G.edges(data=True):
        data["length_m"] = float(data["length"])
        data.setdefault("landmark_value", 0.0)
        data["virtual"] = bool(data.get("virtual", False))
    _refresh_degrees(G, G.nodes)


def _refresh_degrees(G: nx.MultiDiGraph, nodes) -> None:
    for n in nodes:
        nbrs = (set(G.successors(n)) | set(G.predecessors(n))) - {n}
        G.nodes[n]["degree"] = len(nbrs)
        G.nodes[n].setdefault("virtual", False)


# ---------------------------------------------------------------- geometry helpers

def transform(tf: pyproj.Transformer, geom):
    """Reproject a shapely geometry with a pyproj transformer's ``transform``."""
    def fn(a):
        x, y = tf(a[:, 0], a[:, 1])
        return np.column_stack([x, y])
    return shapely.transform(geom, fn)


def graph_centre(G: nx.MultiDiGraph) -> tuple[float, float]:
    """(lon, lat) the local metric projection is centred on: the mean of the nodes."""
    xs = [d["x"] for _, d in G.nodes(data=True)]
    ys = [d["y"] for _, d in G.nodes(data=True)]
    return float(np.mean(xs)), float(np.mean(ys))


def local_transformer(G: nx.MultiDiGraph, origin: tuple[float, float] | None = None
                      ) -> tuple[pyproj.Transformer, pyproj.Transformer]:
    """Forward/inverse transformers to a metric projection centred on ``origin`` (default: the graph)."""
    lon0, lat0 = origin or graph_centre(G)
    local = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m +ellps=WGS84")
    fwd = pyproj.Transformer.from_crs("EPSG:4326", local, always_xy=True)
    inv = pyproj.Transformer.from_crs(local, "EPSG:4326", always_xy=True)
    return fwd, inv


def edge_line(G: nx.MultiDiGraph, u, v, k) -> LineString:
    """Edge geometry in lon/lat, oriented u -> v."""
    data = G.edges[u, v, k]
    geom = data.get("geometry")
    if geom is None:
        return LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
    # Make sure the stored geometry starts at u.
    ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
    x0, y0 = geom.coords[0]
    x1, y1 = geom.coords[-1]
    if (x0 - ux) ** 2 + (y0 - uy) ** 2 > (x1 - ux) ** 2 + (y1 - uy) ** 2:
        geom = LineString(list(geom.coords)[::-1])
    return geom


# ---------------------------------------------------------------- gap stitching

@dataclass
class StitchedGap:
    dangling_node: int
    target_node: int
    gap_m: float
    split_edge: tuple | None  # the (u, v, key) that was split to make target_node, if any


def _reverse_twins(G, u, v, length) -> list:
    if not G.has_edge(v, u):
        return []
    return [k for k, d in G[v][u].items() if abs(d["length"] - length) <= max(0.5, 0.01 * length)]


def _split_edge(G, u, v, k, d_m, fwd, inv, new_id) -> int:
    """Split edge (u,v,k) and its reverse twin at d_m metres from u. Returns the new node."""
    data = dict(G.edges[u, v, k])
    line_ll = edge_line(G, u, v, k)
    line_m = transform(fwd.transform, line_ll)
    L = line_m.length
    frac = d_m / L if L else 0.5
    p_m = line_m.interpolate(d_m)
    lon, lat = inv.transform(p_m.x, p_m.y)
    G.add_node(new_id, x=lon, y=lat, street_count=2, virtual=True)

    g1 = transform(inv.transform, substring(line_m, 0, d_m))
    g2 = transform(inv.transform, substring(line_m, d_m, L))
    total = float(data["length"])
    twins = _reverse_twins(G, u, v, total)

    def piece(geom, share):
        d = dict(data)
        d["geometry"] = geom
        d["length"] = d["length_m"] = total * share
        return d

    G.remove_edge(u, v, k)
    G.add_edge(u, new_id, **piece(g1, frac))
    G.add_edge(new_id, v, **piece(g2, 1 - frac))
    for k2 in twins:
        rdata = dict(G.edges[v, u, k2])
        G.remove_edge(v, u, k2)
        for a, b, geom, share in ((v, new_id, g2, 1 - frac), (new_id, u, g1, frac)):
            d = dict(rdata)
            d["geometry"] = LineString(list(geom.coords)[::-1])
            d["length"] = d["length_m"] = total * share
            G.add_edge(a, b, **d)
    return new_id


def stitch_gaps(
    G: nx.MultiDiGraph,
    tolerance_m: float = 5.0,
    min_detour_m: float = 50.0,
) -> list[StitchedGap]:
    """Join dangling path ends to a nearby way that OSM left unconnected.

    A node with a single neighbour is a dangling end. If another edge passes
    within ``tolerance_m`` of it, and that edge isn't already reachable within
    ``min_detour_m`` of walking, the edge is split at the nearest point (unless
    that point is within 1 m of an end node) and a pair of virtual edges joins
    the two. Virtual edges carry ``virtual=True`` and ``highway="virtual"``.
    """
    if tolerance_m <= 0 or G.number_of_nodes() == 0:
        return []
    fwd, inv = local_transformer(G)
    xy = {n: fwd.transform(d["x"], d["y"]) for n, d in G.nodes(data=True)}

    dangling = [
        n for n in G.nodes
        if len((set(G.successors(n)) | set(G.predecessors(n))) - {n}) == 1
    ]
    if not dangling:
        return []

    # One canonical direction per street for the spatial index.
    canon: list[tuple] = []
    seen: set[tuple] = set()
    for u, v, k, d in G.edges(keys=True, data=True):
        if u == v:
            continue
        twins = _reverse_twins(G, u, v, d["length"])
        if any((v, u, k2) in seen for k2 in twins):
            continue
        seen.add((u, v, k))
        canon.append((u, v, k))
    lines_m = [transform(fwd.transform, edge_line(G, *e)) for e in canon]
    tree = STRtree(lines_m)
    # Each canonical street maps to its current forward pieces after splits.
    pieces: dict[int, list[tuple]] = {i: [e] for i, e in enumerate(canon)}

    next_id = min(0, min(G.nodes)) - 1
    added: list[StitchedGap] = []

    for n in dangling:
        p = Point(xy[n])
        best = None
        for i in tree.query(p.buffer(tolerance_m)):
            for (u, v, k) in pieces[int(i)]:
                if n in (u, v):
                    continue
                line = transform(fwd.transform, edge_line(G, u, v, k))
                dist = line.distance(p)
                if dist <= tolerance_m and (best is None or dist < best[0]):
                    best = (dist, int(i), (u, v, k), line)
        if best is None:
            continue
        gap, ci, (u, v, k), line = best
        along = line.project(p)

        # Already connected nearby? Then this isn't an OSM gap.
        reach = nx.single_source_dijkstra_path_length(G, n, cutoff=min_detour_m, weight="length")
        walk = min(reach.get(u, np.inf) + along, reach.get(v, np.inf) + line.length - along)
        if walk <= min_detour_m:
            continue

        split = None
        if along <= 1.0:
            target = u
        elif along >= line.length - 1.0:
            target = v
        else:
            target = _split_edge(G, u, v, k, along, fwd, inv, next_id)
            next_id -= 1
            xy[target] = fwd.transform(G.nodes[target]["x"], G.nodes[target]["y"])
            idx = pieces[ci].index((u, v, k))
            pieces[ci][idx:idx + 1] = [(u, target, _last_key(G, u, target)),
                                       (target, v, _last_key(G, target, v))]
            split = (u, v, k)

        gap_line = LineString([(G.nodes[n]["x"], G.nodes[n]["y"]),
                               (G.nodes[target]["x"], G.nodes[target]["y"])])
        gap_len = float(Point(xy[n]).distance(Point(xy[target])))
        for a, b, geom in ((n, target, gap_line), (target, n, LineString(gap_line.coords[::-1]))):
            G.add_edge(a, b, length=gap_len, length_m=gap_len, geometry=geom,
                       highway="virtual", virtual=True, oneway=False, reversed=False,
                       landmark_value=0.0, osmid=-1)
        added.append(StitchedGap(n, target, gap_len, split))

    touched = {g.dangling_node for g in added} | {g.target_node for g in added}
    _refresh_degrees(G, touched)
    return added


def _last_key(G, u, v):
    return max(G[u][v])


def without_virtual(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """A view of the graph with stitched (virtual) edges switched off."""
    return nx.subgraph_view(G, filter_edge=lambda u, v, k: not G.edges[u, v, k].get("virtual", False))


# ---------------------------------------------------------------- wrapper

class StreetGraph:
    """Thin wrapper around the OSMnx graph with lookups the rest of Autotrail needs."""

    def __init__(self, G: nx.MultiDiGraph, area_key: str | None = None,
                 origin: tuple[float, float] | None = None):
        self.G = G
        self.area_key = area_key or G.graph.get("autotrail_area")
        self.origin = origin or graph_centre(G)
        self._fwd, self._inv = local_transformer(G, self.origin)
        self._node_ids: np.ndarray | None = None
        self._node_xy: np.ndarray | None = None
        self._edge_tree: STRtree | None = None
        self._edge_keys: list[tuple] = []
        self._split_into: dict[tuple, list[tuple]] = {}  # edges split after the index was built
        self._index_lock = threading.Lock()

    def part(self, nodes) -> "StreetGraph":
        """A copy of just these nodes and the streets between them, in the same local metres
        (for worker processes: far smaller to send than the whole city)."""
        return StreetGraph(self.G.subgraph(nodes).copy(), self.area_key, origin=self.origin)

    def __getstate__(self):
        # The projection, the street index and the lock are rebuilt on the other side.
        return {"G": self.G, "area_key": self.area_key, "origin": self.origin}

    def __setstate__(self, state):
        self.__init__(state["G"], state["area_key"], origin=state["origin"])

    # -- basics
    def has_edge(self, u, v, k=0) -> bool:
        return self.G.has_edge(u, v, k)

    def edge_length(self, u, v, k=0) -> float:
        return float(self.G.edges[u, v, k]["length"])

    def edge_line(self, u, v, k=0) -> LineString:
        return edge_line(self.G, u, v, k)

    def node_lonlat(self, n) -> tuple[float, float]:
        d = self.G.nodes[n]
        return float(d["x"]), float(d["y"])

    def to_metres(self, lon: float, lat: float) -> tuple[float, float]:
        return self._fwd.transform(lon, lat)

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        return self._inv.transform(x, y)

    def best_key(self, u, v) -> int:
        """Shortest parallel edge between two adjacent nodes."""
        return min(self.G[u][v], key=lambda k: self.G[u][v][k]["length"])

    def virtual_edges(self) -> list[tuple]:
        return [(u, v, k) for u, v, k, d in self.G.edges(keys=True, data=True) if d.get("virtual")]

    # -- spatial lookups
    def nearest_node(self, lon: float, lat: float) -> int:
        if self._node_ids is None:
            ids = list(self.G.nodes)
            lons = np.array([self.G.nodes[n]["x"] for n in ids], dtype=float)
            lats = np.array([self.G.nodes[n]["y"] for n in ids], dtype=float)
            nx_, ny_ = self._fwd.transform(lons, lats)
            self._node_ids = np.array(ids, dtype=object)
            self._node_xy = np.column_stack([nx_, ny_])
        x, y = self.to_metres(lon, lat)
        d2 = (self._node_xy[:, 0] - x) ** 2 + (self._node_xy[:, 1] - y) ** 2
        return self._node_ids[int(np.argmin(d2))]

    def edge_line_m(self, u, v, k=0) -> LineString:
        return transform(self._fwd.transform, self.edge_line(u, v, k))

    def edge_index(self) -> tuple[STRtree, list[tuple]]:
        """A spatial index of every street (one direction each), built once and kept.

        Built in one vectorised pass: on the Detroit map (430k edges) going edge by
        edge took long enough that the first click on the GUI map seemed to do nothing.
        """
        with self._index_lock:
            if self._edge_tree is None:
                keys, seen, parts, counts = [], set(), [], []
                nodes = self.G.nodes
                for u, v, k, d in self.G.edges(keys=True, data=True):
                    twin = (v, u, round(float(d["length"]), 1))
                    if twin in seen:
                        continue
                    seen.add((u, v, round(float(d["length"]), 1)))
                    geom = d.get("geometry")
                    if geom is None:
                        c = np.array([(nodes[u]["x"], nodes[u]["y"]), (nodes[v]["x"], nodes[v]["y"])])
                    else:
                        c = np.asarray(geom.coords)[:, :2]
                    keys.append((u, v, k))
                    parts.append(c)
                    counts.append(len(c))
                xy = np.concatenate(parts)
                x, y = self._fwd.transform(xy[:, 0], xy[:, 1])
                lines = shapely.linestrings(np.column_stack([x, y]), indices=np.repeat(np.arange(len(keys)), counts))
                self._edge_keys = keys
                self._edge_tree = STRtree(lines)
            return self._edge_tree, self._edge_keys

    def nearest_edge(self, lon: float, lat: float, among: list[tuple] | None = None):
        """Return ((u, v, k), offset_m along the edge, distance_m) for the closest edge."""
        p = Point(self.to_metres(lon, lat))
        if among:
            best = min(((e, self.edge_line_m(*e)) for e in among), key=lambda t: t[1].distance(p))
            e, line = best
        else:
            tree, keys = self.edge_index()
            e = keys[int(tree.query_nearest(p)[0])]
            while e in self._split_into:  # split since the index was built: pick the nearer piece
                e = min(self._split_into[e], key=lambda x: self.edge_line_m(*x).distance(p))
            line = self.edge_line_m(*e)
        along = line.project(p)
        offset = along / line.length * self.edge_length(*e) if line.length else 0.0
        return e, offset, float(line.distance(p))

    def snap_to_street(self, lon: float, lat: float) -> dict:
        """The nearest point on any street or path to a click, without changing the graph."""
        (u, v, k), offset, dist = self.nearest_edge(lon, lat)
        plon, plat = self.point_on_edge(u, v, k, offset)
        d = self.G.edges[u, v, k]
        name = d.get("name")
        if isinstance(name, list):
            name = " / ".join(str(x) for x in name)
        return {"edge": (u, v, k), "offset_m": offset, "lon": plon, "lat": plat, "moved_m": dist,
                "name": name if isinstance(name, str) and name else None, "highway": str(d.get("highway", ""))}

    def node_at(self, lon: float, lat: float, end_tolerance_m: float = 3.0) -> int:
        """A node exactly where the user clicked: the nearest point on the nearest street.

        Within ``end_tolerance_m`` of an existing node, that node. Otherwise the street (and
        its reverse twin) is split there, in memory only, and the new node is returned.
        """
        (u, v, k), offset, _ = self.nearest_edge(lon, lat)
        length = self.edge_length(u, v, k)
        if offset <= end_tolerance_m:
            return u
        if offset >= length - end_tolerance_m:
            return v
        line_m = self.edge_line_m(u, v, k)
        d_m = offset / length * line_m.length if length else 0.0
        new_id = min(0, min(self.G.nodes)) - 1
        twins = _reverse_twins(self.G, u, v, length)
        _split_edge(self.G, u, v, k, d_m, self._fwd, self._inv, new_id)
        self.G.nodes[new_id]["virtual"] = False
        self.G.nodes[new_id]["picked"] = True
        _refresh_degrees(self.G, {new_id, u, v})
        self._node_ids = self._node_xy = None  # node lookup is cheap to rebuild
        # The edge index is not: remember where the old edges went instead.
        self._split_into[(u, v, k)] = [(u, new_id, _last_key(self.G, u, new_id)),
                                       (new_id, v, _last_key(self.G, new_id, v))]
        for k2 in twins:
            self._split_into[(v, u, k2)] = [(v, new_id, _last_key(self.G, v, new_id)),
                                            (new_id, u, _last_key(self.G, new_id, u))]
        return new_id

    def point_on_edge(self, u, v, k, offset_m: float) -> tuple[float, float]:
        length = self.edge_length(u, v, k)
        frac = 0.0 if length <= 0 else min(max(offset_m / length, 0.0), 1.0)
        p = self.edge_line(u, v, k).interpolate(frac, normalized=True)
        return p.x, p.y

    def shortest_edges(self, a, b) -> list[tuple]:
        """Edge list (u, v, k) of the shortest walking path from a to b."""
        nodes = nx.shortest_path(self.G, a, b, weight="length")
        return [(x, y, self.best_key(x, y)) for x, y in zip(nodes, nodes[1:])]

    def summary(self) -> dict:
        virt = self.virtual_edges()
        total_km = sum(d["length"] for _, _, d in self.G.edges(data=True)) / 2000.0
        return {
            "area": self.area_key,
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "street_km": round(total_km, 1),
            "virtual_edges": len(virt),
            "intersections_deg3plus": sum(1 for _, d in self.G.nodes(data=True) if d.get("degree", 0) >= 3),
        }
