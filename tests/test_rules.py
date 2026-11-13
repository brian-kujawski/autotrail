from autotrail.graph import StreetGraph
from autotrail.rules import Footprint, Geometry, is_sidewalk

from conftest import grid_id


def test_straight_on_at_a_crossing(grid):
    geo = Geometry(StreetGraph(grid, area_key="grid"))
    # Coming east along row 1 into a 4-way: straight on is the next corner east.
    assert geo.straight_on(grid_id(0, 1), grid_id(1, 1)) == grid_id(2, 1)
    assert geo.is_corner(grid_id(0, 1), grid_id(1, 1))


def test_no_straight_on_at_a_t(grid):
    geo = Geometry(StreetGraph(grid, area_key="grid"))
    # Coming north up column 1 into row 4 (the top edge): left or right, no straight on.
    assert geo.straight_on(grid_id(1, 3), grid_id(1, 4)) is None


def test_a_bend_with_one_way_on_is_straight(grid):
    geo = Geometry(StreetGraph(grid, area_key="grid"))
    # The SW corner of the grid only goes on north from the east.
    assert geo.straight_on(grid_id(1, 0), grid_id(0, 0)) == grid_id(0, 1)


def test_deviation(grid):
    geo = Geometry(StreetGraph(grid, area_key="grid"))
    assert geo.deviation(grid_id(0, 1), grid_id(1, 1), grid_id(2, 1)) < 1.0
    assert abs(geo.deviation(grid_id(0, 1), grid_id(1, 1), grid_id(1, 2)) - 90.0) < 1.0


def test_footprint_catches_the_other_side_of_the_street(grid):
    g = StreetGraph(grid, area_key="grid")
    geo = Geometry(g)
    fp = Footprint(geo, radius_m=25.0)
    a, b, c = grid_id(0, 1), grid_id(1, 1), grid_id(2, 1)
    fp.add_path([(a, b, 0), (b, c, 0)], "true")
    # Running back along the same street (or its other sidewalk) clashes...
    assert fp.clashes((c, b, 0), "true", 400.0)
    # ...the street one block north does not...
    assert not fp.clashes((grid_id(0, 2), grid_id(1, 2), 0), "true", 400.0)
    # ...and the stretch just run doesn't count against itself at a corner.
    assert not fp.clashes((c, grid_id(2, 2), 0), "true", 200.0)
    # Another leg can't touch it either.
    assert fp.clashes((grid_id(1, 0), b, 0), "return", 0.0)


def test_is_sidewalk():
    assert is_sidewalk({"highway": "footway", "footway": "sidewalk"})
    assert is_sidewalk({"highway": "footway", "footway": ["crossing", "sidewalk"]})
    assert not is_sidewalk({"highway": "footway"})
    assert not is_sidewalk({"highway": "residential"})


def test_turning_a_corner_is_not_coming_near_yourself(grid):
    # Round a square corner and the two streets are within 45 m of each other for ~32 m either side.
    geo = Geometry(StreetGraph(grid, area_key="grid"))
    path = [(grid_id(1, 1), grid_id(2, 1), 0), (grid_id(2, 1), grid_id(2, 2), 0)]
    assert not Footprint(geo, 45.0).path_clashes(path, "true")
    assert Footprint(geo, 45.0).path_clashes(path, "true", recent_m=50.0)  # the old, too-short exemption


def test_turns_per_mile(grid):
    import numpy as np
    from autotrail.rules import turns_per_mile

    g = StreetGraph(grid, area_key="grid")
    geo = Geometry(g)

    def path(nodes):
        edges = [(a, b, 0) for a, b in zip(nodes, nodes[1:])]
        return edges, np.concatenate([[0.0], np.cumsum([g.edge_length(*e) for e in edges])])

    straight = path([grid_id(c, 1) for c in range(5)])
    stairs = path([grid_id(0, 0), grid_id(1, 0), grid_id(1, 1), grid_id(2, 1), grid_id(2, 2), grid_id(3, 2)])
    assert turns_per_mile(geo, *straight) == 0.0
    # 4 turns in 500 m
    assert abs(turns_per_mile(geo, *stairs) - 4 / (500 / 1609.344)) < 0.1


def test_route_search_breaks_up_straights(grid):
    """With the straight-run penalty, A* zigzags instead of running a whole street."""
    from autotrail.generate import Network
    import numpy as np

    from autotrail.rules import straight_runs, turns_per_mile

    g = StreetGraph(grid, area_key="grid")
    geo = Geometry(g)
    net = Network(g, grid_id(0, 0), 2000.0)
    a, b = grid_id(0, 0), grid_id(4, 4)
    fp = Footprint(geo, 35.0)
    plain = net.clear_path(a, b, fp, "true", 0.0, set(), turn_cost_m=120.0)
    twisty = net.clear_path(a, b, fp, "true", 0.0, set(), turn_cost_m=8.0, straight_run_m=150.0, straight_cost=3.0)
    cum = lambda p: np.concatenate([[0.0], np.cumsum([g.edge_length(*e) for e in p])])  # noqa: E731
    assert turns_per_mile(geo, twisty, cum(twisty)) > turns_per_mile(geo, plain, cum(plain))
    assert max(b - a for a, b in straight_runs(geo, twisty, cum(twisty))) <= 200.0


def _boulevard():
    """A boulevard whose two sidewalks run east-west 40 m apart (y=0 and y=40), mapped separately,
    joined every 100 m by a crossing street; plus a residential street parallel 120 m north."""
    import math

    import networkx as nx

    from autotrail.graph import annotate
    from conftest import ll

    G = nx.MultiDiGraph(crs="epsg:4326")
    pos = {}
    for i in range(6):
        pos[("s", i)] = (i * 100.0, 0.0)
        pos[("n", i)] = (i * 100.0, 40.0)
        pos[("r", i)] = (i * 100.0, 160.0)
    ids = {k: n for n, k in enumerate(pos, start=1)}
    for k, (x, y) in pos.items():
        lon, lat = ll(x, y)
        G.add_node(ids[k], x=lon, y=lat, street_count=0)

    def link(a, b, **tags):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        for u, v in ((a, b), (b, a)):
            G.add_edge(ids[u], ids[v], length=math.hypot(xa - xb, ya - yb), oneway=False, reversed=u != a,
                       osmid=ids[a], **tags)

    for i in range(5):
        link(("s", i), ("s", i + 1), highway="footway", footway="sidewalk")
        link(("n", i), ("n", i + 1), highway="footway", footway="sidewalk")
        link(("r", i), ("r", i + 1), highway="residential", name="Elm")
    for i in range(6):
        link(("s", i), ("n", i), highway="residential", name=f"Cross {i}")
        link(("n", i), ("r", i), highway="residential", name=f"Cross {i}")
    annotate(G)
    return StreetGraph(G, area_key="blvd"), ids


def test_hopping_across_a_boulevard_is_a_jog_not_two_turns():
    import numpy as np

    from autotrail.rules import jog_corners, straight_runs, turns_per_mile

    g, ids = _boulevard()
    geo = Geometry(g)
    # North sidewalk east 200 m, hop across (40 m), carry on east along the south sidewalk.
    nodes = [ids[("n", 0)], ids[("n", 1)], ids[("n", 2)], ids[("s", 2)], ids[("s", 3)], ids[("s", 4)]]
    edges = [(a, b, 0) for a, b in zip(nodes, nodes[1:])]
    cum = np.concatenate([[0.0], np.cumsum([g.edge_length(*e) for e in edges])])
    assert jog_corners(geo, edges, cum) == {2, 3}
    assert turns_per_mile(geo, edges, cum) == 0.0
    assert len(straight_runs(geo, edges, cum)) == 1  # still one straightaway


def test_a_real_offset_over_20m_is_still_two_turns():
    import numpy as np

    from autotrail.rules import turns_per_mile

    g, ids = _boulevard()
    geo = Geometry(g)
    # Elm east, 120 m down Cross 2, then east on the north sidewalk: a real offset, two turns.
    nodes = [ids[("r", 0)], ids[("r", 1)], ids[("r", 2)], ids[("n", 2)], ids[("n", 3)]]
    edges = [(a, b, 0) for a, b in zip(nodes, nodes[1:])]
    cum = np.concatenate([[0.0], np.cumsum([g.edge_length(*e) for e in edges])])
    assert turns_per_mile(geo, edges, cum) > 0


def test_running_back_down_the_other_side_clashes():
    g, ids = _boulevard()
    geo = Geometry(g)
    fp = Footprint(geo, 35.0)
    out = [(ids[("n", i)], ids[("n", i + 1)], 0) for i in range(4)]  # north side, eastbound
    fp.add_path(out, "true")
    # Much later, westbound on the south side, 40 m away: the same street.
    assert fp.clashes((ids[("s", 2)], ids[("s", 1)], 0), "true", 2000.0)
    # Elm, 120 m north and parallel, is a different street.
    assert not fp.clashes((ids[("r", 2)], ids[("r", 1)], 0), "true", 2000.0)
