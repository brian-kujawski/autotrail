"""The web GUI's server and the shared pipeline, on the synthetic city (offline)."""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from autotrail.gui.server import GuiApp, serve
from autotrail.pipeline import RunSettings, check_ends, run


def _ll(g, n):
    lon, lat = g.node_lonlat(n)
    return [lat, lon]


START, BEER = 8 * 16 + 8, 8 * 16 + 11  # centre of the city, three blocks east


def test_check_ends(city):
    d, problem = check_ends(city, START, BEER, 2.0)
    assert 0.15 < d < 0.5 and problem is None
    _, problem = check_ends(city, START, START, 2.0)
    assert "same corner" in problem
    _, problem = check_ends(city, START, 16 * 16, 1.0)  # far corner, short trail
    assert "too far" in problem


def test_run_with_picked_beer_near(city, tmp_path):
    s = RunSettings(miles=2.0, seed=1, tries=40, runs=2)
    lon, lat = city.node_lonlat(START)
    elon, elat = city.node_lonlat(BEER)
    res = run(city, (lon, lat), (elon, elat), s, out=tmp_path / "t.geojson", gpx=tmp_path / "t.gpx")
    t = res.trail
    assert t.start == START and t.end == BEER
    assert t.validate(city) == []
    assert t.return_path and t.return_path[-1].v == START
    assert res.walk_back_mi == pytest.approx(check_ends(city, START, BEER, 2.0)[0])
    assert all(p.exists() for p in res.files.values())
    assert json.loads((tmp_path / "t.geojson").read_text())["autotrail"]["meta"]["beer_near"] == "picked"


@pytest.fixture(scope="module")
def server(city, tmp_path_factory):
    app = GuiApp(city, "Test city", tmp_path_factory.mktemp("runs"))
    srv = serve(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_page_and_info(server):
    with urllib.request.urlopen(server + "/") as r:
        assert b"Beer Near" in r.read()
    info = _get(server + "/api/info")
    assert info["area"] == "Test city" and info["defaults"]["miles"] == 5.0


def test_snap_and_check(server, city):
    lat, lon = _ll(city, START)
    p = _get(f"{server}/api/snap?lat={lat + 0.0001}&lon={lon}")
    assert int(p["node"]) == START and p["moved_m"] < 15
    e = _ll(city, BEER)
    c = _get(f"{server}/api/check?start={lat},{lon}&end={e[0]},{e[1]}&miles=2")
    assert c["problem"] is None and 0.15 < c["walk_mi"] < 0.5


def test_generate_job(server, city):
    body = {"start": _ll(city, START), "end": _ll(city, BEER),
            "settings": {"miles": 2, "tries": 40, "runs": 2, "seed": 3}}
    job = _post(server + "/api/generate", body)
    for _ in range(300):
        job = _get(f"{server}/api/jobs/{job['id']}")
        if job["state"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert job["state"] == "done", job
    r = job["result"]
    assert r["walk_back_mi"] and r["return_mi"] >= r["walk_back_mi"] - 0.01
    with urllib.request.urlopen(server + r["files"]["map"]) as resp:
        assert b"Pack replay" in resp.read()
    runs = _get(server + "/api/runs")
    assert runs[0]["run"] == r["run"]


def test_bad_requests(server, city):
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(server + "/api/generate", {"start": None})
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(server + "/api/generate", {"start": _ll(city, START), "end": None,
                                         "settings": {"return_leg": False}})
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(server + "/runs/..%2F..%2Fetc/passwd")
    assert e.value.code == 404
