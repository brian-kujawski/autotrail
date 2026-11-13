"""Command line: ``autotrail download``, ``autotrail show``, ``autotrail areas``.

Later phases add ``autotrail generate``, ``autotrail sim`` and ``autotrail ask``.
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
from .viz import save_trail_map

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


if __name__ == "__main__":
    app()
