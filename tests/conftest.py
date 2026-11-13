"""A small synthetic street grid in Detroit, in the shape OSMnx produces.

Grid: 5x5 nodes, 100 m apart, ids 1..25 (row-major from the south-west).
Extra footpaths (local metres, x east / y north from the grid's SW corner):
  * west path 101-102-103: dangling end 101 sits 3 m west of the grid edge
    between (0,100) and (0,200); its far end joins grid node (0,400).
  * east path 201-202-203: dangling end 201 sits 10 m east of the edge
    between (400,200) and (400,300); its far end joins grid node (400,0).
  * spur 301 off grid node (200,200), 20 m long, ending 3 m from the street north of it.
    It's a dead end already one short walk from that street: not a gap.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

LAT0, LON0 = 42.33, -83.05
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(LAT0))


def ll(x_m: float, y_m: float) -> tuple[float, float]:
    """Local metres -> (lon, lat)."""
    return LON0 + x_m / M_PER_DEG_LON, LAT0 + y_m / M_PER_DEG_LAT


def grid_id(col: int, row: int) -> int:
    return row * 5 + col + 1


def build_grid() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph(crs="epsg:4326")
    pos: dict[int, tuple[float, float]] = {}
    for row in range(5):
        for col in range(5):
            pos[grid_id(col, row)] = (col * 100.0, row * 100.0)
    pos.update({101: (-3, 150), 102: (-80, 150), 103: (-80, 400),
                201: (410, 250), 202: (480, 250), 203: (480, 0),
                301: (203, 220)})
    for n, (x, y) in pos.items():
        lon, lat = ll(x, y)
        G.add_node(n, x=lon, y=lat, street_count=0)

    def link(a, b, highway="residential"):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        length = math.hypot(xa - xb, ya - yb)
        for u, v in ((a, b), (b, a)):
            G.add_edge(u, v, length=length, highway=highway, oneway=False, reversed=u != a, osmid=1000 + a)

    for row in range(5):
        for col in range(5):
            if col < 4:
                link(grid_id(col, row), grid_id(col + 1, row))
            if row < 4:
                link(grid_id(col, row), grid_id(col, row + 1))
    link(101, 102, "footway")
    link(102, 103, "footway")
    link(103, grid_id(0, 4), "footway")
    link(201, 202, "footway")
    link(202, 203, "footway")
    link(203, grid_id(4, 0), "footway")
    link(grid_id(2, 2), 301, "footway")
    for n in G.nodes:
        G.nodes[n]["street_count"] = len(set(G.successors(n)))
    return G


@pytest.fixture
def grid() -> nx.MultiDiGraph:
    from autotrail.graph import annotate
    G = build_grid()
    annotate(G)
    return G


def build_city(n: int = 16, step: float = 110.0, seed: int = 1) -> nx.MultiDiGraph:
    """A bigger, slightly irregular grid (n x n, ~step m blocks, a few streets missing)."""
    import random

    rnd = random.Random(seed)
    G = nx.MultiDiGraph(crs="epsg:4326")
    pos = {}
    for r in range(n):
        for c in range(n):
            pos[r * n + c + 1] = (c * step + rnd.uniform(-8, 8), r * step + rnd.uniform(-8, 8))
    for k, (x, y) in pos.items():
        lon, lat = ll(x, y)
        G.add_node(k, x=lon, y=lat, street_count=0)

    def link(a, b):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        length = math.hypot(xa - xb, ya - yb)
        for u, v in ((a, b), (b, a)):
            G.add_edge(u, v, length=length, highway="residential", oneway=False, reversed=u != a, osmid=a)

    for r in range(n):
        for c in range(n):
            i = r * n + c + 1
            if c < n - 1 and rnd.random() > 0.08:
                link(i, i + 1)
            if r < n - 1 and rnd.random() > 0.08:
                link(i, i + n)
    G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
    return G


@pytest.fixture(scope="module")
def city():
    from autotrail.graph import StreetGraph, annotate
    G = build_city()
    annotate(G)
    return StreetGraph(G, area_key="city")
