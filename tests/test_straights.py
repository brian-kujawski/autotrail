"""Jogs are straight on; long straights and dull stretches; checks where the answer is straight on."""

import math

import networkx as nx
from conftest import ll
from test_sim import N, long_trail, street

from autotrail.generate import _cum, add_marks, corners
from autotrail.graph import StreetGraph, annotate
from autotrail.rules import Geometry, jog_corners
from autotrail.score import Variety, dull_stretches
from autotrail.sim import TrailModel, simulate_many
from autotrail.trail import EdgeRef, Trail


def jog_graph(offset_m: float) -> StreetGraph:
    """West-east street A-B, a step of ``offset_m`` south at B-C, then on east C-D; side streets at B and C."""
    pos = {1: (0, 0), 2: (100, 0), 3: (100, -offset_m), 4: (200, -offset_m),
           5: (100, 100), 6: (100, -offset_m - 100), 7: (-100, 0), 8: (300, -offset_m)}
    G = nx.MultiDiGraph(crs="epsg:4326")
    for n, (x, y) in pos.items():
        lon, lat = ll(x, y)
        G.add_node(n, x=lon, y=lat, street_count=0)
    for a, b in [(7, 1), (1, 2), (2, 3), (3, 4), (4, 8), (2, 5), (3, 6)]:
        (xa, ya), (xb, yb) = pos[a], pos[b]
        for u, v in ((a, b), (b, a)):
            G.add_edge(u, v, length=math.hypot(xa - xb, ya - yb), highway="residential", oneway=False,
                       reversed=u != a, osmid=a * 10 + b)
    annotate(G)
    return StreetGraph(G, area_key="jog")


def jog_trail(g):
    path = [(7, 1), (1, 2), (2, 3), (3, 4), (4, 8)]
    return Trail(start=7, end=8, true_path=[EdgeRef(u, v, 0) for u, v in path])


def test_a_short_jog_is_straight_on():
    g = jog_graph(8.0)
    geo = Geometry(g)
    t = jog_trail(g)
    tp = [(e.u, e.v, e.key) for e in t.true_path]
    cum = _cum(g, tp)
    assert jog_corners(geo, tp, cum) == {2, 3}
    assert not any(c.turn for c in corners(geo, tp, cum))
    t = add_marks(t, g)
    model = TrailModel(t, g)
    assert not model.unmarked_turns and not model.warnings
    for r in simulate_many(t, g, runs=3):
        assert len(r.finishers) == 12


def test_a_long_step_is_two_turns():
    g = jog_graph(40.0)
    geo = Geometry(g)
    t = jog_trail(g)
    tp = [(e.u, e.v, e.key) for e in t.true_path]
    cum = _cum(g, tp)
    assert jog_corners(geo, tp, cum) == set()
    assert sum(c.turn for c in corners(geo, tp, cum)) == 2


def test_dull_stretches_and_variety(city):
    a, b = 3 * N + 1, 3 * N + 16  # one straight street across the city, ~1.6 km
    t = add_marks(Trail(start=a, end=b, true_path=street(city, a, b)), city)
    m = TrailModel(t, city)
    assert dull_stretches(m, 402.0)
    results = simulate_many(t, city, runs=2)
    assert Variety().score(m.L, results).value < 0.6


def test_some_checks_are_straight_on(city):
    straight = turned = 0
    for seed in range(4):
        t = long_trail(city, seed=seed)
        m = TrailModel(t, city)
        assert not m.unmarked_turns
        straight += sum(c.fake for c in m.checks)
        turned += sum(not c.fake for c in m.checks)
    assert straight > 0 and turned > 0


def test_a_run_of_whichy_ways_ends_in_a_turn(city):
    found = False
    for seed in range(10):
        t = long_trail(city, seed=seed, whichy_run_share=1.0, fill_share=0.0)
        m = TrailModel(t, city)
        w = [c for c in m.checks if c.kind == "whichy"]
        for i in range(len(w) - 3):
            if all(c.fake for c in w[i:i + 3]) and not w[i + 3].fake:
                found = True
    assert found


def test_a_straight_avenue_with_a_parallel_sidewalk_still_counts_as_straight():
    """At a crossing where a sidewalk peels off at a shallow angle there's no single 'straight on',
    but the street is still a straightaway."""
    from autotrail.rules import straight_runs

    pos = {i: (100.0 * i, 0.0) for i in range(8)}
    pos.update({100 + i: (100.0 * i + 60, 12.0) for i in range(7)})  # stubs ~11 degrees off, at each corner
    G = nx.MultiDiGraph(crs="epsg:4326")
    for n, (x, y) in pos.items():
        lon, lat = ll(x, y)
        G.add_node(n, x=lon, y=lat, street_count=0)

    def link(a, b, hw="residential"):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        for u, v in ((a, b), (b, a)):
            G.add_edge(u, v, length=math.hypot(xa - xb, ya - yb), highway=hw, oneway=False,
                       reversed=u != a, osmid=a * 1000 + b)

    for i in range(7):
        link(i, i + 1)
        link(i, 100 + i, "footway")
    annotate(G)
    g = StreetGraph(G, area_key="avenue")
    geo = Geometry(g)
    edges = [(i, i + 1, 0) for i in range(7)]
    assert geo.straight_on(0, 1) is None  # ambiguous at every corner...
    assert straight_runs(geo, edges, _cum(g, edges)) == [(0.0, 700.0)]  # ...but one 700 m straightaway
