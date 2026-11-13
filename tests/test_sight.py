"""Sightlines: a hound sees down its street (both ways) until it bends; the hare hides elsewhere."""

import pytest
from conftest import grid_id
from test_sim import long_trail

from autotrail.graph import StreetGraph
from autotrail.pack import HareProfile
from autotrail.rules import Geometry
from autotrail.score import score_results
from autotrail.sight import SightLines, hare_sightings, sighting_weight
from autotrail.sim import simulate_many


@pytest.fixture
def sight(grid):
    g = StreetGraph(grid, area_key="grid")
    return SightLines(Geometry(g)), Geometry(g)


def at(geo, col, row, dx=0.0, dy=0.0):
    x, y = geo.node_xy(grid_id(col, row))
    return (x + dx, y + dy)


def test_sees_down_the_street_both_ways(sight):
    s, geo = sight
    u, v = grid_id(1, 0), grid_id(2, 0)  # hound halfway along the south street
    dog = at(geo, 1, 0, 50)
    assert s.seen(u, v, 50, 100, dog, at(geo, 3, 0))      # 150 m ahead
    assert s.seen(u, v, 50, 100, dog, at(geo, 0, 0))      # 50 m behind
    assert not s.seen(u, v, 50, 100, dog, at(geo, 4, 0, 20))  # 270 m: too far


def test_buildings_block_the_next_street(sight):
    s, geo = sight
    u, v = grid_id(1, 0), grid_id(2, 0)
    dog = at(geo, 1, 0, 50)
    assert not s.seen(u, v, 50, 100, dog, at(geo, 1, 1, 50))  # parallel street, 100 m away
    assert not s.seen(u, v, 50, 100, dog, at(geo, 2, 0, 0, 40))  # 40 m up a cross street


def test_at_a_corner_a_hound_looks_down_every_street(sight):
    s, geo = sight
    u, v = grid_id(1, 1), grid_id(2, 1)
    dog = at(geo, 1, 1)
    assert s.seen(u, v, 0, 100, dog, at(geo, 1, 3))   # 200 m up the cross street
    assert s.seen(u, v, 0, 100, dog, at(geo, 1, 3, 5))  # crossing it, inside the street


def test_closer_sightings_count_more():
    assert sighting_weight(20) == 1.0
    assert sighting_weight(150) > sighting_weight(240) >= 0.3


def test_hare_seen_with_no_head_start_and_hidden_with_one(city):
    t = long_trail(city)
    results = simulate_many(t, city, runs=2)
    close = HareProfile(head_start_s=30.0)
    assert hare_sightings(results[0], close.pace_s_per_m, close.head_start_s)
    assert score_results(results, hare=close).term("hare_hidden").value < 0.3
    assert score_results(results, hare=HareProfile()).term("hare_hidden").value > 0.9


def test_pack_timing_windows():
    from autotrail.pack import default_pack
    from autotrail.sight import PackTiming
    tm = PackTiming.of(HareProfile(), default_pack())
    assert tm.pack_window(500.0) is None  # hare 500 m out after 15 min head start: pack still at Start
    lo, hi = tm.pack_window(6000.0)
    assert 0 < lo < hi < 6000.0  # the pack is somewhere behind the hare
    a, b = tm.hare_window(2000.0)
    assert 2000.0 < b and a < b


def test_route_search_pays_for_streets_in_view_of_the_pack(grid):
    from autotrail.sight import PackTiming, SightField
    g = StreetGraph(grid, area_key="grid")
    geo = Geometry(g)
    # Hare gets to the new street 2.5 km in; by then the pack is around 0-400 m along.
    tm = PackTiming(hare_pace=0.4, head_start_s=0.0, front_pace=0.4, back_pace=0.4, margin_m=200.0)
    sf = SightField(geo, tm)
    row0 = [(grid_id(c, 0), grid_id(c + 1, 0)) for c in range(4)]
    for i, (u, v) in enumerate(row0):
        sf.add_edge((u, v, 0), 100.0 * i)
    up_col0 = (grid_id(2, 1), grid_id(2, 2), 0)  # straight up a cross street from the trail
    across = (grid_id(1, 2), grid_id(2, 2), 0)     # crosses two streets the pack looks up
    up = sf.exposed_m(up_col0, 300.0)
    assert up > 90.0
    assert 0.0 < sf.exposed_m(across, 300.0) < 0.4 * up  # only the two crossings are in view
