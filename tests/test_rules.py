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
