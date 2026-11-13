import json

import pytest

from autotrail.generate import GenerateOptions, generate
from autotrail.pack import HareProfile
from autotrail.rules import Footprint, Geometry
from autotrail.score import score_results
from autotrail.sim import TrailModel, simulate, simulate_many
from autotrail.trail import METRES_PER_MILE, MarkKind, Trail, load_trail


@pytest.fixture(scope="module")
def generated(city):
    opts = GenerateOptions(miles=2.0, return_miles=0.5, tries=60, candidates=2, placements=2, runs=3)
    return generate(city, 8 * 16 + 8, opts=opts, seed=4)


def test_generated_trail_is_valid_and_near_target(generated, city):
    t = generated.trail
    assert t.validate(city) == []
    assert t.true_trail_m(city) == pytest.approx(2.0 * METRES_PER_MILE, rel=0.10)
    assert t.return_path, "expected a hared leg back to Start"
    assert t.return_path[0].u == t.end and t.return_path[-1].v == t.start
    assert t.meta["source"] == "generated"


def test_generated_marks(generated, city):
    t = generated.trail
    model = TrailModel(t, city)
    assert len(model.checks) >= 2
    checks = [c for c in model.checks if c.kind == "check"]  # (a run of whichy-ways can be a block apart)
    spacing = [b.s - a.s for a, b in zip(checks, checks[1:])]
    assert all(s >= 150.0 - 1e-6 for s in spacing)  # straight-on checks keep 150 m from others
    check_nodes = {c.node for c in model.checks}
    assert all(b.from_node in check_nodes for b in t.branches)
    assert sum(m.kind == MarkKind.FALSE for m in t.marks) == len(t.branches)
    assert sum(b.label.startswith("false") for b in t.branches) <= 2
    assert any(m.params.get("leg") == "return" for m in t.marks)
    assert not model.unmarked_turns  # every turn has a check, arrow or whichy-way
    assert not model.warnings


def test_generated_trail_keeps_clear_of_itself(generated, city):
    """No crossing, no doubling back, no running back down the other side of a street."""
    t = generated.trail
    geo = Geometry(city)
    o = GenerateOptions()
    fp = Footprint(geo, o.keep_apart_m)
    sx, sy = geo.node_xy(t.start)
    ex, ey = geo.node_xy(t.end)
    fp.zones += [(sx, sy, 2 * o.keep_apart_m), (ex, ey, 2 * o.keep_apart_m)]
    assert not fp.path_clashes([(e.u, e.v, e.key) for e in t.true_path], "true")
    assert not fp.path_clashes([(e.u, e.v, e.key) for e in t.return_path], "return")
    nodes = [e.u for e in t.true_path] + [t.end]
    assert len(nodes) == len(set(nodes))


def test_generated_hare_stays_ahead(generated, city):
    t = generated.trail
    results = simulate_many(t, city, runs=5)
    card = score_results(results, t.true_trail_m(city), hare=HareProfile())
    assert card.term("hare_safety").value > 0.0


def test_generated_round_trip_sims_the_same(generated, city, tmp_path):
    t = generated.trail
    p = tmp_path / "gen.geojson"
    t.save(p, city)
    t2 = load_trail(p, city)
    assert t2 == Trail.from_geojson(json.loads(p.read_text()))
    a = simulate(t, city, seed=11)
    b = simulate(t2, city, seed=11)
    # Mark offsets are saved to the centimetre, so allow a hair of difference.
    assert [h.distance_m for h in a.hounds] == pytest.approx([h.distance_m for h in b.hounds], abs=0.5)


def _river_city(n=24, step=100.0, river=9, bridge=22):
    """An n x n grid with a river between rows ``river`` and ``river + 1``, one bridge far off."""
    import math

    import networkx as nx
    from conftest import ll

    from autotrail.graph import StreetGraph, annotate

    G = nx.MultiDiGraph(crs="epsg:4326")
    pos = {r * n + c + 1: (c * step, r * step) for r in range(n) for c in range(n)}
    for k, (x, y) in pos.items():
        lon, lat = ll(x, y)
        G.add_node(k, x=lon, y=lat, street_count=0)

    def link(a, b):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        for u, v in ((a, b), (b, a)):
            G.add_edge(u, v, length=math.hypot(xa - xb, ya - yb), highway="residential", oneway=False,
                       reversed=u != a, osmid=a)

    for r in range(n):
        for c in range(n):
            i = r * n + c + 1
            if c < n - 1:
                link(i, i + 1)
            if r < n - 1 and (r != river or c == bridge):
                link(i, i + n)
    annotate(G)
    return StreetGraph(G, area_key="river"), n, river


def test_waypoints_stay_on_the_walkable_side_of_a_river():
    import math

    from autotrail.generate import GenerateOptions, generate

    g, n, river = _river_city()
    start, end = (river + 2) * n + 10, (river + 4) * n + 13
    seen = []
    gen = generate(g, start, end, GenerateOptions(miles=2.5, tries=30, runs=2, placements=1, candidates=1,
                                                  return_miles=0.4), seed=0, on_route=seen.append)
    assert gen.trail.true_path
    # Nothing picked across the water unless it's a fair walk over the bridge: every waypoint's walk
    # from Start is within the detour limit, and most attempts stay north of the river.
    from autotrail.generate import WALK_DETOUR, WALK_SLACK_M, Network
    from autotrail.rules import Geometry
    geo = Geometry(g)
    walk = Network(g, start, 10_000).dijkstra(start)
    sx, sy = geo.node_xy(start)
    for r in seen:
        for w in r["waypoints"]:
            x, y = geo.node_xy(w)
            assert walk[w] <= WALK_DETOUR * math.hypot(x - sx, y - sy) + WALK_SLACK_M + 1e-6
    north = (river + 1) * n  # node ids above this are north of the river
    across = sum(any(w <= north for w in r["waypoints"]) for r in seen)
    assert across <= len(seen) // 5


def test_trail_may_use_the_way_back_near_the_beer_near(grid):
    """The corner next to the Beer Near is on the way back, but trail may still run in through it."""
    from conftest import grid_id

    from autotrail.generate import reserved_corners
    from autotrail.graph import StreetGraph

    g = StreetGraph(grid, area_key="grid")
    geo = Geometry(g)
    end, corner, far = grid_id(2, 2), grid_id(2, 1), grid_id(2, 0)
    back = [(end, corner, 0), (corner, far, 0), (far, grid_id(1, 0), 0)]
    start = grid_id(1, 0)
    assert reserved_corners(geo, back, start, end, 70.0) == {corner, far}
    # 100 m zone: the corner 100 m from the Beer Near is free to use too
    assert corner not in reserved_corners(geo, back, start, end, 101.0)


def test_loop_shapes_go_out_away_from_the_beer_near_and_vary():
    import math
    import random

    from autotrail.generate import loop_shape

    rng = random.Random(5)
    start, beer = (0.0, 0.0), (800.0, 0.0)  # Beer Near due east of Start
    ratios = []
    for _ in range(200):
        sh = loop_shape(rng, start, beer, 1500.0, 5)
        fx, fy = sh.far_point()
        assert fx < 400.0  # the far point lies west of the anchor: away from the Beer Near
        ratios.append(sh.width_m / sh.length_m)
        # Waypoints go round in order: out on Start's side, back on the Beer Near's.
        pts = [sh.point(i, rng) for i in range(sh.n)]
        ux, uy = math.cos(sh.heading), math.sin(sh.heading)
        across = [(-uy * (x - sh.ax) + ux * (y - sh.ay)) * sh.side for x, y, _ in pts]
        assert across[0] > 0 > across[-1] or sh.bend != 0  # (a bent loop can bow both ends one way)
    # Not all one shape: narrow out-and-backs through fat beans.
    assert min(ratios) < 0.25 and max(ratios) > 0.7


def test_seg_dist():
    from autotrail.generate import _seg_dist

    assert _seg_dist(5, 3, 0, 0, 10, 0) == 3
    assert _seg_dist(-4, 3, 0, 0, 10, 0) == 5
