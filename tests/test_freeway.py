"""Freeway cordon: edges near a freeway cost more, most when they run alongside it."""

import json

import pytest
from shapely.geometry import LineString

from autotrail import graph as graph_mod
from autotrail.config import Area
from autotrail.generate import Network
from autotrail.graph import StreetGraph, load_area_graph

from conftest import build_grid, grid_id, ll


def _freeway_along_row_2():
    # East-west freeway on the grid's middle row (y = 200 m), past both ends of the grid.
    return [LineString([ll(-150, 200), ll(650, 200)])]


def test_cordon_share_and_angle():
    g = StreetGraph(build_grid())
    n = g.mark_freeway_cordon(_freeway_along_row_2(), cordon_m=60.0)
    assert n > 0
    along = g.G.edges[grid_id(1, 2), grid_id(2, 2), 0]
    assert along["freeway_share"] == pytest.approx(1.0)
    assert along["freeway_parallel"] == pytest.approx(1.0, abs=0.01)
    cross = g.G.edges[grid_id(1, 1), grid_id(1, 2), 0]  # 100 m north-south, 60 m of it in the cordon
    assert cross["freeway_share"] == pytest.approx(0.6, abs=0.02)
    assert cross["freeway_parallel"] == pytest.approx(0.0, abs=0.01)
    far = g.G.edges[grid_id(1, 0), grid_id(2, 0), 0]
    assert far["freeway_share"] == 0.0


def test_network_cost_slides_with_angle():
    g = StreetGraph(build_grid())
    g.mark_freeway_cordon(_freeway_along_row_2(), cordon_m=60.0)
    net = Network(g, grid_id(2, 2), 2000.0, freeway_cost=(2.0, 4.0))

    def cost(u, v):
        return next(c for w, _, _, c in net.adj[u] if w == v)

    assert cost(grid_id(1, 2), grid_id(2, 2)) == pytest.approx(400.0, rel=0.01)   # alongside: 4x
    assert cost(grid_id(1, 1), grid_id(1, 2)) == pytest.approx(160.0, rel=0.02)   # 60 m at 2x + 40 m
    assert cost(grid_id(1, 0), grid_id(2, 0)) == pytest.approx(100.0)            # outside: 1x
    plain = Network(g, grid_id(2, 2), 2000.0)                                     # default: off
    assert next(c for w, _, _, c in plain.adj[grid_id(1, 2)] if w == grid_id(2, 2)) == pytest.approx(100.0)


def test_freeways_cached_and_offline_is_harmless(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_mod, "download_graph", lambda area: build_grid())
    monkeypatch.setattr(graph_mod, "download_freeways", lambda area: _freeway_along_row_2())
    area = Area(key="test-grid", name="Test grid", places=("Nowhere",))
    g = load_area_graph(area, root=tmp_path)
    assert (tmp_path / "cache" / "test-grid-freeways.geojson").exists()
    assert g.G.edges[grid_id(1, 2), grid_id(2, 2), 0]["freeway_share"] == pytest.approx(1.0)

    monkeypatch.setattr(graph_mod, "download_freeways", lambda area: pytest.fail("should use cache"))
    g2 = load_area_graph(area, root=tmp_path)
    assert g2.G.edges[grid_id(1, 2), grid_id(2, 2), 0]["freeway_share"] == pytest.approx(1.0)

    def offline(area):
        raise ConnectionError("no network")
    other = Area(key="offline", name="Offline", places=("Nowhere",))
    monkeypatch.setattr(graph_mod, "download_freeways", offline)
    g3 = load_area_graph(other, root=tmp_path)
    assert all(d["freeway_share"] == 0.0 for _, _, d in g3.G.edges(data=True))
