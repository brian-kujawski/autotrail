"""Local web GUI: click Start and Beer Near on a map, generate, view the replay.

Run with ``autotrail gui``. The map graph is loaded once when the server
starts and kept in memory, so every trail after the first skips the load.
Standard library only (``http.server``); the page pulls Leaflet from a CDN.

Endpoints (all JSON unless noted):

* ``GET  /``                     the page
* ``GET  /api/info``             area, map bounds, default settings
* ``GET  /api/snap?lat=&lon=``   nearest intersection to a click
* ``GET  /api/check?start=lat,lon&end=lat,lon&miles=``  walk distance, problem if any
* ``POST /api/generate``         start a run; body {start: [lat, lon], end: [lat, lon] | null, settings}
* ``GET  /api/jobs/<id>``        progress log and, when done, the result
* ``GET  /api/runs``             earlier runs (newest first)
* ``GET  /runs/<run>/<file>``    a run's trail.geojson, trail.gpx or replay.html
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import traceback
import uuid
from dataclasses import asdict, fields
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..graph import StreetGraph
from ..pipeline import RunSettings, preview_walk_mi, run
from ..trail import METRES_PER_MILE

log = logging.getLogger(__name__)

RUN_FILES = {"trail.geojson": "application/geo+json", "trail.gpx": "application/gpx+xml",
             "replay.html": "text/html; charset=utf-8", "run.json": "application/json"}


def _settings_from(d: dict) -> RunSettings:
    known = {f.name: f.type for f in fields(RunSettings)}
    base = RunSettings()
    kw = {}
    for k, v in (d or {}).items():
        if k not in known or v is None or v == "":
            continue
        cur = getattr(base, k)
        if isinstance(cur, bool):
            kw[k] = bool(v)
        elif isinstance(cur, int):
            kw[k] = int(v)
        elif isinstance(cur, float):
            kw[k] = float(v)
        else:
            kw[k] = str(v)
    s = RunSettings(**kw)
    if not (0.5 <= s.miles <= 15):
        raise ValueError("Trail length should be between 0.5 and 15 miles.")
    if not (1 <= s.runners <= 60):
        raise ValueError("Pack size should be between 1 and 60.")
    if not (10 <= s.tries <= 2000):
        raise ValueError("Route attempts should be between 10 and 2000.")
    if not (1 <= s.runs <= 200):
        raise ValueError("Simulated runs should be between 1 and 200.")
    return s


class Job:
    def __init__(self, start, end, settings: RunSettings, folder: Path):
        self.id = uuid.uuid4().hex[:10]
        self.start, self.end, self.settings, self.folder = start, end, settings, folder
        self.state = "queued"
        self.log: list[str] = []
        self.result: dict | None = None
        self.error: str | None = None
        self.started = time.time()
        self.finished: float | None = None

    def say(self, msg: str) -> None:
        self.log.append(f"{time.time() - self.started:5.1f}s  {msg}")

    def view(self) -> dict:
        end = self.finished or time.time()
        return {"id": self.id, "state": self.state, "log": self.log, "result": self.result,
                "error": self.error, "elapsed_s": round(end - self.started, 1)}


class GuiApp:
    """The server's state: one loaded map, a folder of runs, one run at a time."""

    def __init__(self, g: StreetGraph, area_name: str, runs_dir: Path, tiles: str = "streets"):
        self.g = g
        self.area_name = area_name
        self.runs_dir = Path(runs_dir)
        self.tiles = tiles
        self.jobs: dict[str, Job] = {}
        self.busy = threading.Lock()
        g.edge_index()  # build the click-snapping index now, not on the first click
        lons = [d["x"] for _, d in g.G.nodes(data=True)]
        lats = [d["y"] for _, d in g.G.nodes(data=True)]
        self.bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]

    # -- api
    def info(self) -> dict:
        return {"area": self.area_name, "bounds": self.bounds, "defaults": asdict(RunSettings())}

    def snap(self, lat: float, lon: float) -> dict:
        """The nearest point on the nearest street or path: where the marker should sit."""
        p = self.g.snap_to_street(lon, lat)
        return {"lat": p["lat"], "lon": p["lon"], "moved_m": round(p["moved_m"], 1),
                "street": p["name"], "highway": p["highway"]}

    def check(self, start, end, miles: float) -> dict:
        d, problem = preview_walk_mi(self.g, (start[1], start[0]), (end[1], end[0]), miles)
        return {"walk_mi": None if math.isinf(d) else round(d, 2), "problem": problem}

    def start_job(self, body: dict) -> Job:
        start = body.get("start")
        end = body.get("end")
        if not start or len(start) != 2:
            raise ValueError("Pick a Start on the map first.")
        settings = _settings_from(body.get("settings") or {})
        if end is None and not settings.return_leg:
            raise ValueError("Without a leg back to Start, pick a Beer Near.")
        if not self.busy.acquire(blocking=False):
            raise RuntimeError("A trail is already being generated. Wait for it to finish.")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job = Job([float(start[0]), float(start[1])], None if end is None else [float(end[0]), float(end[1])],
                  settings, self.runs_dir / f"{stamp}-seed{settings.seed}")
        self.jobs[job.id] = job
        threading.Thread(target=self._work, args=(job,), daemon=True).start()
        return job

    def _work(self, job: Job) -> None:
        try:
            job.state = "running"
            job.say("Started")
            f = job.folder
            res = run(self.g, (job.start[1], job.start[0]),
                      None if job.end is None else (job.end[1], job.end[0]), job.settings,
                      out=f / "trail.geojson", map_out=f / "replay.html", gpx=f / "trail.gpx",
                      tiles=self.tiles, progress=job.say)
            t = res.trail
            s_lon, s_lat = self.g.node_lonlat(t.start)
            e_lon, e_lat = self.g.node_lonlat(t.end)
            summary = res.card.summary
            result = {
                "run": f.name,
                "score": round(res.card.total, 1),
                "report": res.report,
                "true_trail_mi": round(t.true_trail_m(self.g) / METRES_PER_MILE, 2),
                "return_mi": round(t.return_m(self.g) / METRES_PER_MILE, 2) if t.return_path else 0.0,
                "walk_back_mi": None if res.walk_back_mi is None else round(res.walk_back_mi, 2),
                "front_extra_mi": round(summary.get("front_extra_mi", 0.0), 2),
                "back_extra_mi": round(summary.get("back_extra_mi", 0.0), 2),
                "start": [s_lat, s_lon], "end": [e_lat, e_lon],
                "files": {k: f"/runs/{f.name}/{p.name}" for k, p in res.files.items()},
                "seconds": round(time.time() - job.started, 1),
            }
            job.result = result
            (f / "run.json").write_text(json.dumps({"settings": asdict(job.settings), "picked_start": job.start,
                                                    "picked_end": job.end, **result}, indent=2),
                                        encoding="utf-8")
            job.say(f"Done: score {result['score']:.0f}/100")
            job.state = "done"
        except Exception as e:  # report anything to the page rather than kill the thread silently
            log.warning("Run failed: %s", traceback.format_exc())
            job.error = str(e) or e.__class__.__name__
            job.say(f"Failed: {job.error}")
            job.state = "failed"
        finally:
            job.finished = time.time()
            self.busy.release()

    def list_runs(self) -> list[dict]:
        out = []
        if not self.runs_dir.exists():
            return out
        for d in sorted(self.runs_dir.iterdir(), reverse=True):
            meta = d / "run.json"
            if not meta.is_file():
                continue
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
            except ValueError:
                continue
            out.append({k: m.get(k) for k in ("run", "score", "true_trail_mi", "return_mi", "start", "end",
                                               "files", "settings", "seconds")})
        return out[:50]

    def run_file(self, run_name: str, file_name: str) -> Path | None:
        if file_name not in RUN_FILES or "/" in run_name or "\\" in run_name or run_name.startswith("."):
            return None
        p = self.runs_dir / run_name / file_name
        return p if p.is_file() else None


def _page() -> bytes:
    return resources.files("autotrail.gui").joinpath("index.html").read_bytes()


def make_handler(app: GuiApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Autotrail"

        def log_message(self, fmt, *args):  # keep the console quiet; uncomment for debugging
            log.debug(fmt, *args)

        def _send(self, status, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=HTTPStatus.OK) -> None:
            self._send(status, json.dumps(obj).encode("utf-8"), "application/json")

        def _err(self, msg: str, status=HTTPStatus.BAD_REQUEST) -> None:
            self._json({"error": msg}, status)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            parts = [p for p in u.path.split("/") if p]
            try:
                if not parts or parts == ["index.html"]:
                    return self._send(HTTPStatus.OK, _page(), "text/html; charset=utf-8")
                if parts == ["api", "info"]:
                    return self._json(app.info())
                if parts == ["api", "snap"]:
                    return self._json(app.snap(float(q["lat"]), float(q["lon"])))
                if parts == ["api", "check"]:
                    s = [float(x) for x in q["start"].split(",")]
                    e = [float(x) for x in q["end"].split(",")]
                    return self._json(app.check(s, e, float(q.get("miles", 5))))
                if parts == ["api", "runs"]:
                    return self._json(app.list_runs())
                if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                    job = app.jobs.get(parts[2])
                    return self._json(job.view()) if job else self._err("No such job", HTTPStatus.NOT_FOUND)
                if len(parts) == 3 and parts[0] == "runs":
                    p = app.run_file(parts[1], parts[2])
                    if p is None:
                        return self._err("Not found", HTTPStatus.NOT_FOUND)
                    return self._send(HTTPStatus.OK, p.read_bytes(), RUN_FILES[parts[2]])
                return self._err("Not found", HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError) as e:
                return self._err(f"Bad request: {e}")
            except Exception as e:  # show it on the page and in the console, don't just drop the request
                log.error("GET %s failed:\n%s", self.path, traceback.format_exc())
                return self._err(f"{e.__class__.__name__}: {e}", HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self):
            u = urlparse(self.path)
            if u.path.rstrip("/") != "/api/generate":
                return self._err("Not found", HTTPStatus.NOT_FOUND)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                job = app.start_job(body)
            except RuntimeError as e:
                return self._err(str(e), HTTPStatus.CONFLICT)
            except (ValueError, TypeError) as e:
                return self._err(str(e))
            return self._json(job.view(), HTTPStatus.ACCEPTED)

    return Handler


def serve(app: GuiApp, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Build (but don't start) the server. Call ``serve_forever()`` on the result."""
    return ThreadingHTTPServer((host, port), make_handler(app))
