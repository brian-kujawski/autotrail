"""Corner facts: what a hound who takes each wrong way would find."""

from conftest import grid_id

from autotrail.facts import corner_facts, summary
from autotrail.graph import StreetGraph


def _path(g, nodes):
    return [(a, b, g.best_key(a, b)) for a, b in zip(nodes, nodes[1:])]


def _u_trail(grid):
    """East along row 1, up a block, back west along row 2: a U with 100 m between its arms."""
    g = StreetGraph(grid)
    nodes = [grid_id(c, 1) for c in range(5)] + [grid_id(c, 2) for c in range(4, -1, -1)]
    return g, _path(g, nodes)


def _way(facts, at, to):
    f = next(f for f in facts if f.node == at)
    return next(w for w in f.ways if w.to == to)


def test_a_way_that_lops_off_the_u_is_a_shortcut(grid):
    g, tp = _u_trail(grid)
    facts = corner_facts(g, tp)
    w = _way(facts, grid_id(1, 1), grid_id(1, 2))  # one block north meets the far arm, 700 m on
    assert w.kind == "shortcut"
    assert w.walk_m == 100.0 and w.skip_m == 700.0
    assert w.path[0] == grid_id(1, 1)
    # The same way off the far arm only meets trail already run: no shortcut.
    back = _way(facts, grid_id(2, 2), grid_id(2, 1))
    assert back.kind == "rejoins" and back.skip_m < 0


def test_dead_end_and_safe_ways(grid):
    g, tp = _u_trail(grid)
    facts = corner_facts(g, tp)
    assert _way(facts, grid_id(2, 2), 301).kind == "dead_end"  # the 20 m spur
    assert _way(facts, grid_id(2, 2), grid_id(2, 3)).kind == "open"


def test_trail_into_the_way_back_is_its_own_kind(grid):
    g, tp = _u_trail(grid)
    # A way back from the Beer Near (0,2) down column 0... is the way trail came out: use row 3 instead.
    ret = _path(g, [grid_id(0, 2), grid_id(0, 3), grid_id(1, 3), grid_id(2, 3), grid_id(2, 4)])
    facts = corner_facts(g, tp, ret)
    w = _way(facts, grid_id(2, 2), grid_id(2, 3))  # now lands on the way back
    assert w.kind == "way_back" and w.meets_s > 800.0


def test_summary_counts(grid):
    g, tp = _u_trail(grid)
    facts = corner_facts(g, tp)
    s = summary(facts)
    assert s["corners"] == len(facts) and s["shortcut_corners"] >= 1
    assert sum(s["ways"].values()) == sum(len(f.ways) for f in facts)
