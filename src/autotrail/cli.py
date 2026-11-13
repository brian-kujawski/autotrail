"""Command line: ``autotrail areas|download|show|generate|sim``.

Phase 2 adds ``autotrail ask``.
"""

from __future__ import annotations

import json
import logging
import webbrowser
from pathlib import Path
from typing import Optional

import typer

from .config import get_area, load_areas
from .graph import cache_path, load_area_graph
from .trail import METRES_PER_MILE, load_trail
from .viz import save_replay_map, save_trail_map

app = typer.Typer(add_completion=False, help="Hash trail suggestions on OpenStreetMap data.")

AreaOpt = typer.Option(None, "--area", "-a", help="Area key from areas.yaml (default: the file's default).")
AreasFileOpt = typer.Option(None, "--areas-file", help="Use your own areas YAML instead of the bundled one.")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")


@app.command()
def areas(areas_file: Optional[Path] = AreasFileOpt) -> None:
    """List the configured map areas."""
    found, default = load_areas(areas_file)
    for key, a in found.items():
        mark = " (default)" if key == default else ""
        typer.echo(f"{key}{mark}: {a.name}")
        for p in a.places:
            typer.echo(f"    {p}")


@app.command()
def download(
    area: Optional[str] = AreaOpt,
    areas_file: Optional[Path] = AreasFileOpt,
    force: bool = typer.Option(False, "--force", help="Download again even if cached."),
    no_stitch: bool = typer.Option(False, "--no-stitch", help="Skip joining dangling OSM path ends."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
) -> None:
    """Download and cache the walk network for an area."""
    _setup_logging(verbose)
    a = get_area(area, areas_file)
    typer.echo(f"Area: {a.name}")
    g = load_area_graph(a, force=force, stitch=not no_stitch)
    typer.echo(f"Cached at {cache_path(a)}")
    typer.echo(json.dumps(g.summary(), indent=2))


@app.command()
def show(
    trail_file: Path = typer.Argument(..., exists=True, help="Autotrail or hand-drawn GeoJSON."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="HTML map path (default: <trail>.html)."),
    save: Optional[Path] = typer.Option(None, "--save", help="Also save the snapped trail as Autotrail GeoJSON."),
    gpx: Optional[Path] = typer.Option(None, "--gpx", help="Also export the route as GPX."),
    area: Optional[str] = AreaOpt,
    areas_file: Optional[Path] = AreasFileOpt,
    tiles: str = typer.Option("streets", "--tiles", help="Basemap: streets, satellite, or osm (osm needs the map served over http)."),
    open_browser: bool = typer.Option(False, "--open", help="Open the map in a browser."),
    verbose: bool = typer.Option(False, "--verbose/--quiet"),
) -> None:
    """Load a trail, check it against the map, and draw it as an HTML map."""
    _setup_logging(verbose)
    g = load_area_graph(get_area(area, areas_file))
    trail = load_trail(trail_file, g)
    problems = trail.validate(g)
    if problems:
        typer.echo("Trail problems:")
        for p in problems:
            typer.echo(f"  - {p}")
        raise typer.Exit(1)

    typer.echo(f"True trail: {trail.true_trail_m(g) / METRES_PER_MILE:.2f} mi "
               f"({len(trail.true_path)} edges)")
    if trail.return_path:
        typer.echo(f"Return leg: {trail.return_m(g) / METRES_PER_MILE:.2f} mi")
    typer.echo(f"Marks: {len(trail.marks)}, branches: {len(trail.branches)}")
    virt = sum(1 for e in trail.all_edges() if g.G.edges[e.u, e.v, e.key].get("virtual"))
    if virt:
        typer.echo(f"Uses {virt} stitched gap edge(s); check them on the ground.")

    out = out or trail_file.with_suffix(".html")
    save_trail_map(trail, g, out, tiles=tiles)
    typer.echo(f"Map: {out}")
    if save:
        trail.save(save, g)
        typer.echo(f"Saved: {save}")
    if gpx:
        gpx.write_text(trail.to_gpx(g), encoding="utf-8")
        typer.echo(f"GPX: {gpx}")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())


def _latlon(text: str) -> tuple[float, float]:
    """'42.33,-83.07' -> (lon, lat)."""
    try:
        lat, lon = (float(x) for x in text.replace(" ", "").split(","))
    except ValueError:
        raise typer.BadParameter(f"expected 'lat,lon', got {text!r}")
    return lon, lat


def _mpm(t: str) -> float:
    m, _, s = t.partition(":")
    return int(m) + int(s or 0) / 60


def _pack(runners: int, fastest: str, slowest: str):
    from .pack import runners as make_runners
    return make_runners(runners, _mpm(fastest), _mpm(slowest))


def _hare(pace: str, head_start_min: float):
    from .pack import HareProfile, pace_from_min_per_mile
    return HareProfile(pace_from_min_per_mile(_mpm(pace)), head_start_min * 60.0)


HarePaceOpt = typer.Option("10:15", "--hare-pace", help="Hare's pace while laying trail, min:sec per mile.")
HeadStartOpt = typer.Option(15.0, "--head-start", help="Hare's head start in minutes.")
RunnersOpt = typer.Option(12, "--runners", help="Pack size (runners only in phase 1).")
FastOpt = typer.Option("7:30", "--fastest", help="Fastest pace, min:sec per mile.")
SlowOpt = typer.Option("9:30", "--slowest", help="Slowest pace, min:sec per mile.")


def _report(trail, g, results, card, out: Path, tiles: str, open_browser: bool, hare=None) -> None:
    from .score import format_report
    typer.echo(format_report(card, results))
    save_replay_map(trail, g, results[0], out, card=card, tiles=tiles, hare=hare)
    typer.echo(f"\nMap with pack replay: {out}")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())


@app.command()
def generate(
    start: str = typer.Option(..., "--start", help="Start as 'lat,lon' (snapped to the nearest intersection)."),
    end: Optional[str] = typer.Option(None, "--end", help="Beer Near as 'lat,lon'. Default: picked about --return-miles from Start."),
    miles: float = typer.Option(5.0, "--miles", help="True trail length, Start to Beer Near."),
    return_miles: float = typer.Option(1.0, "--return-miles", help="Hared leg back to Start. 0 for none."),
    out: Path = typer.Option(Path("trail.geojson"), "--out", "-o", help="Where to save the trail."),
    map_out: Optional[Path] = typer.Option(None, "--map", help="HTML map + replay (default: <out>.html)."),
    gpx: Optional[Path] = typer.Option(None, "--gpx", help="Also export the route as GPX."),
    seed: int = typer.Option(0, "--seed", help="Different seeds give different trails."),
    tries: int = typer.Option(300, "--tries", help="Route attempts."),
    runs: int = typer.Option(20, "--runs", help="Simulated runs for the final report."),
    runners: int = RunnersOpt, fastest: str = FastOpt, slowest: str = SlowOpt,
    hare_pace: str = HarePaceOpt, head_start: float = HeadStartOpt,
    area: Optional[str] = AreaOpt,
    areas_file: Optional[Path] = AreasFileOpt,
    tiles: str = typer.Option("streets", "--tiles", help="Basemap: streets, satellite, or osm."),
    open_browser: bool = typer.Option(False, "--open", help="Open the map in a browser."),
    verbose: bool = typer.Option(False, "--verbose/--quiet"),
) -> None:
    """One-click trail: route, marks (blobs, checks, falses), scored by simulating the pack."""
    from .generate import GenerateOptions, generate as gen
    from .score import score_results
    from .sim import simulate_many

    _setup_logging(verbose)
    g = load_area_graph(get_area(area, areas_file))
    s_node = g.nearest_node(*_latlon(start))
    e_node = g.nearest_node(*_latlon(end)) if end else None
    pack = _pack(runners, fastest, slowest)
    opts = GenerateOptions(miles=miles, return_miles=return_miles if return_miles > 0 else 0.0, tries=tries)
    if return_miles <= 0 and e_node is None:
        raise typer.BadParameter("with --return-miles 0, give --end")
    hare = _hare(hare_pace, head_start)
    res = gen(g, s_node, e_node, opts, pack=pack, hare=hare, seed=seed, progress=typer.echo)
    trail = res.trail
    if return_miles <= 0:
        trail.return_path = []
        trail.marks = [m for m in trail.marks if m.params.get("leg") != "return"]
    trail.meta.update({"name": f"Autotrail {miles:g} mi (seed {seed})"})
    results = simulate_many(trail, g, pack, runs=runs, seed=seed)
    card = score_results(results, miles * METRES_PER_MILE, hare=hare)
    trail.save(out, g)
    typer.echo(f"Trail: {out}\n")
    if gpx:
        gpx.write_text(trail.to_gpx(g), encoding="utf-8")
        typer.echo(f"GPX: {gpx}")
    _report(trail, g, results, card, map_out or out.with_suffix(".html"), tiles, open_browser, hare)


@app.command()
def sim(
    trail_file: Path = typer.Argument(..., exists=True, help="Autotrail or hand-drawn GeoJSON."),
    runs: int = typer.Option(20, "--runs", help="Simulated runs to average."),
    seed: int = typer.Option(0, "--seed"),
    target_miles: Optional[float] = typer.Option(None, "--target-miles",
                                                 help="Distance target for scoring (default: the trail's, else 5)."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="HTML map + replay (default: <trail>.sim.html)."),
    runners: int = RunnersOpt, fastest: str = FastOpt, slowest: str = SlowOpt,
    hare_pace: str = HarePaceOpt, head_start: float = HeadStartOpt,
    area: Optional[str] = AreaOpt,
    areas_file: Optional[Path] = AreasFileOpt,
    tiles: str = typer.Option("streets", "--tiles"),
    open_browser: bool = typer.Option(False, "--open", help="Open the map in a browser."),
    verbose: bool = typer.Option(False, "--verbose/--quiet"),
) -> None:
    """Run the pack on a trail and score it. Trails with no blobs get them filled in."""
    from .generate import add_marks, needs_marks
    from .score import score_results
    from .sim import TrailModel, simulate_many

    _setup_logging(verbose)
    g = load_area_graph(get_area(area, areas_file))
    trail = load_trail(trail_file, g)
    problems = trail.validate(g)
    if problems:
        typer.echo("Trail problems:")
        for p in problems:
            typer.echo(f"  - {p}")
        raise typer.Exit(1)
    if needs_marks(trail, g):
        trail = add_marks(trail, g, seed=seed)
        typer.echo("This trail was missing blobs or turn marks, so the sim filled them in: blobs every "
                   "~50 m (none for ~50 m after a check) and an arrow at every turn without a mark.")
    model = TrailModel(trail, g)
    for w in model.warnings:
        typer.echo(f"Note: {w}")
    target = target_miles or float(trail.meta.get("target_miles") or 5.0)
    results = simulate_many(trail, g, _pack(runners, fastest, slowest), runs=runs, seed=seed, model=model)
    hare = _hare(hare_pace, head_start)
    card = score_results(results, target * METRES_PER_MILE, hare=hare)
    _report(trail, g, results, card, out or trail_file.with_suffix(".sim.html"), tiles, open_browser, hare)


if __name__ == "__main__":
    app()
