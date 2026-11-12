import json

import pytest

from autotrail.graph import StreetGraph, stitch_gaps
from autotrail.trail import (Branch, BranchPurpose, EdgePos, EdgeRef, Mark, MarkKind, Trail,
                             load_trail, snap_drawn_trail)
from autotrail.viz import save_trail_map

from conftest import grid_id, ll


@pytest.fixture
def g(grid):
    stitch_gaps(grid, 5.0)
    return StreetGraph(grid, area_key="test")


def make_trail(g: StreetGraph) -> Trail:
    start, end = grid_id(0, 0), grid_id(4, 4)
    path = [EdgeRef(*e) for e in g.shortest_edges(start, end)]
    back = [EdgeRef(*e) for e in g.shortest_edges(end, start)]
    false_leg = [EdgeRef(*e) for e in g.shortest_edges(grid_id(1, 0), grid_id(1, 1))]
    marks = [
        Mark(MarkKind.CHECK, EdgePos(node=grid_id(1, 0))),
        Mark(MarkKind.FALSE, EdgePos(false_leg[-1], 90.0)),
        Mark(MarkKind.BLOB, EdgePos(path[0], 50.0)),
        Mark(MarkKind.BACKCHECK, EdgePos(path[2], 10.0), {"n": 3}),
    ]
    return Trail(start=start, end=end, true_path=path, return_path=back,
                 branches=[Branch(BranchPurpose.FALSE, false_leg, "f1")], marks=marks,
                 meta={"hare": "Test Hare", "source": "test"})


def test_validate_ok(g):
    t = make_trail(g)
    assert t.validate(g) == []
    assert t.true_trail_m(g) == pytest.approx(800.0)
    assert t.return_m(g) == pytest.approx(800.0)


def test_validate_catches_broken_chain(g):
    t = make_trail(g)
    del t.true_path[1]
    assert any("breaks" in p for p in t.validate(g))


def test_geojson_round_trip(g, tmp_path):
    t = make_trail(g)
    p = tmp_path / "t.geojson"
    t.save(p, g)
    data = json.loads(p.read_text())
    assert data["type"] == "FeatureCollection"
    assert data["autotrail"]["true_trail_m"] == pytest.approx(800.0)
    t2 = Trail.from_geojson(data)
    assert t2 == t
    assert load_trail(p, g) == t


def test_stale_ids_fall_back_to_geometry(g, tmp_path):
    t = make_trail(g)
    data = t.to_geojson(g)
    for f in data["features"]:
        if f["properties"].get("role") == "true_path":
            f["properties"]["edges"][1] = [999, 998, 0]
    p = tmp_path / "stale.geojson"
    p.write_text(json.dumps(data))
    t2 = load_trail(p, g)
    assert t2.validate(g) == []
    assert t2.start == t.start and t2.end == t.end
    assert t2.true_trail_m(g) == pytest.approx(800.0)
    assert len(t2.marks) == len(t.marks)


def test_snap_hand_drawn(g):
    # A few clicks: SW corner, east along the bottom street, then north up the east side.
    drawn = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {},
             "geometry": {"type": "LineString",
                          "coordinates": [ll(2, -3), ll(395, 4), ll(403, 395)]}},
            {"type": "Feature", "properties": {"kind": "check"},
             "geometry": {"type": "Point", "coordinates": ll(250, 4)}},
            {"type": "Feature", "properties": {"kind": "BC", "n": 3},
             "geometry": {"type": "Point", "coordinates": ll(397, 150)}},
            {"type": "Feature", "properties": {"role": "branch", "purpose": "false"},
             "geometry": {"type": "LineString", "coordinates": [ll(200, 0), ll(200, 100)]}},
            {"type": "Feature", "properties": {"role": "return_path"},
             "geometry": {"type": "LineString", "coordinates": [ll(0, 400)]}},
        ],
    }
    t = snap_drawn_trail(drawn, g)
    assert t.start == grid_id(0, 0) and t.end == grid_id(4, 4)
    assert t.validate(g) == []
    assert t.true_trail_m(g) == pytest.approx(800.0)
    assert t.return_m(g) == pytest.approx(800.0)
    kinds = [m.kind for m in t.marks]
    assert kinds == [MarkKind.CHECK, MarkKind.BACKCHECK]
    assert t.marks[1].params == {"n": 3}
    assert t.branches[0].purpose == BranchPurpose.FALSE
    assert t.meta["source"] == "drawn"


def test_mark_kind_aliases():
    assert MarkKind.parse("whichy-way") is MarkKind.WHICHY_WAY
    assert MarkKind.parse("bc") is MarkKind.BACKCHECK
    assert MarkKind.parse("Beer Near") is MarkKind.BEER


def test_map_and_gpx(g, tmp_path):
    t = make_trail(g)
    out = save_trail_map(t, g, tmp_path / "map.html")
    html = out.read_text()
    assert "True trail" in html and "Beer Near" in html and "BC3" in html
    # OSM's tile servers block pages opened from a file, so the default basemap isn't theirs.
    assert "arcgisonline.com" in html and "tile.openstreetmap.org" not in html
    gpx = t.to_gpx(g)
    assert "<trk>" in gpx and "BACKCHECK" in gpx
