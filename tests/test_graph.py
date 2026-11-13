import pytest

from autotrail import graph as graph_mod
from autotrail.config import Area, get_area
from autotrail.graph import StreetGraph, load_area_graph, stitch_gaps, without_virtual

from conftest import build_grid, grid_id, ll


def test_default_area_is_detroit_union():
    a = get_area()
    assert a.key == "detroit"
    assert len(a.places) == 3
    assert any("Hamtramck" in p for p in a.places)
    assert any("Highland Park" in p for p in a.places)


def test_stitch_joins_only_the_real_gap(grid):
    before_edges = grid.number_of_edges()
    gaps = stitch_gaps(grid, tolerance_m=5.0)
    assert [g.dangling_node for g in gaps] == [101]
    gap = gaps[0]
    assert gap.gap_m == pytest.approx(3.0, abs=0.2)
    assert gap.split_edge is not None  # joined mid-street, so the street was split

    new = gap.target_node
    assert grid.nodes[new]["virtual"] is True
    assert grid.nodes[new]["degree"] == 3
    # Both directions of the split street became two pieces each, summing to 100 m.
    a, b = grid_id(0, 1), grid_id(0, 2)
    assert not grid.has_edge(a, b) and not grid.has_edge(b, a)
    fwd = grid[a][new][0]["length"] + grid[new][b][0]["length"]
    rev = grid[b][new][0]["length"] + grid[new][a][0]["length"]
    assert fwd == pytest.approx(100.0) and rev == pytest.approx(100.0)
    # Virtual edges both ways.
    assert grid[101][new][0]["virtual"] and grid[new][101][0]["virtual"]
    # Net change: one street (2 directed edges) became 4, plus 2 virtual.
    assert grid.number_of_edges() == before_edges + 4


def test_wider_tolerance_catches_second_gap_but_not_the_spur(grid):
    gaps = stitch_gaps(grid, tolerance_m=15.0)
    assert sorted(g.dangling_node for g in gaps) == [101, 201]


def test_stitching_off(grid):
    assert stitch_gaps(grid, tolerance_m=0) == []


def test_without_virtual_view(grid):
    stitch_gaps(grid, tolerance_m=5.0)
    view = without_virtual(grid)
    assert not any(d.get("virtual") for _, _, d in view.edges(data=True))


def test_streetgraph_lookups(grid):
    g = StreetGraph(grid, area_key="test")
    lon, lat = ll(102, 98)
    assert g.nearest_node(lon, lat) == grid_id(1, 1)
    e, offset, dist = g.nearest_edge(*ll(150, 104))
    assert {e[0], e[1]} == {grid_id(1, 1), grid_id(2, 1)}
    assert dist == pytest.approx(4.0, abs=0.3)
    assert offset == pytest.approx(50.0, abs=1.0)
    path = g.shortest_edges(grid_id(0, 0), grid_id(2, 0))
    assert sum(g.edge_length(*e) for e in path) == pytest.approx(200.0)


def test_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_mod, "download_graph", lambda area: build_grid())
    area = Area(key="test-grid", name="Test grid", places=("Nowhere",), stitch_tolerance_m=5.0)
    g1 = load_area_graph(area, root=tmp_path)
    assert (tmp_path / "cache" / "test-grid.graphml").exists()
    assert len(g1.virtual_edges()) == 2

    monkeypatch.setattr(graph_mod, "download_graph", lambda area: pytest.fail("should use cache"))
    g2 = load_area_graph(area, root=tmp_path)
    assert len(g2.virtual_edges()) == 2
    u, v, k = g2.virtual_edges()[0]
    d = g2.G.edges[u, v, k]
    assert d["virtual"] is True
    assert isinstance(d["length_m"], float)
    assert isinstance(g2.G.nodes[u]["degree"], int)
    assert g2.summary()["nodes"] == g1.summary()["nodes"]
