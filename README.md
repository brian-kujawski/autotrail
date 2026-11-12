# Autotrail

Suggest hash trails from OpenStreetMap data and simulate a pack of hounds running them.

This is **Phase 0, the skeleton**: download and cache a walking map, a `Trail` file format
with GeoJSON round trip, and an HTML map view. Trail generation (`autotrail generate`),
the pack simulator (`autotrail sim`) and placement questions (`autotrail ask`) come in later
phases. See the design doc for the plan.

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -e ".[dev]"
```

## Use

```bash
autotrail areas                 # list configured map areas
autotrail download              # fetch + cache Detroit, Highland Park, Hamtramck (a few minutes the first time)
autotrail show examples/corktown_sample.geojson --open
```

`show` loads a trail, checks it against the map, prints the true trail distance and writes an
HTML map next to the file (`--out` to choose). Add `--save trail.geojson` to keep the snapped
Autotrail version and `--gpx trail.gpx` to export it for a phone.

The cached graph lives in `data/cache/<area>.graphml` (set `AUTOTRAIL_DATA` to move it).
Use `autotrail download --force` to refresh it.

### Drawing a trail by hand

Draw on [geojson.io](https://geojson.io) (or QGIS) and save the GeoJSON:

* **A line** from Start to the Beer Near. A few clicks per street is enough; each click is
  snapped to the nearest intersection and joined along the shortest walking route.
* Optional lines with property `role` set to `return_path` (the leg back to Start) or `branch`
  (add `purpose`: `FALSE`, `BACKCHECK_LURE`, `TURKEY` or `EAGLE`).
* Optional **points** with property `kind`: `check`, `false`, `blob`, `arrow`, `backcheck`
  (or `bc`, plus `n` for the count), `whichy-way`, `split`, `rejoin`, `beer`. Any other
  properties are kept as the mark's parameters.
* Optional points with `role` = `start` or `beer_near` to pin the ends.

## Areas

Areas live in [`src/autotrail/areas.yaml`](src/autotrail/areas.yaml). An area is a list of
place names (OSMnx downloads their union) plus a stitch tolerance. Add another city there, or
pass your own file with `--areas-file`, then `autotrail download --area <key>`.

## OSM gap stitching

OSM paths that visibly touch often don't share a node. On download, a dangling path end within
`stitch_tolerance_m` (default 5 m) of another way that isn't already a short walk away gets a
joining edge with `virtual=True` and `highway=virtual`. The map view has a layer (off by default)
showing stitched gaps near the trail, and `autotrail.graph.without_virtual(G)` switches them off.

## Trail file format

An Autotrail trail is a GeoJSON `FeatureCollection` with an `autotrail` header
(`format`, `version`, `area`, `start`, `end`, `meta`, distances) and features with a `role`:

| role | geometry | properties |
| --- | --- | --- |
| `true_path` | LineString | `edges`: `[[u, v, key], ...]` Start to Beer Near |
| `return_path` | LineString | `edges` Beer Near back to Start |
| `branch` | LineString | `edges`, `purpose`, `label` |
| `start`, `beer_near` | Point | `node` |
| `mark` | Point | `kind`, `edge` + `offset_m` (metres from the edge start) or `node`, `params` |

The edge ids tie the file to a map build. If they no longer match (for example after a fresh
download), `show` rebuilds the trail from the saved geometry instead.

## Code layout

| module | job |
| --- | --- |
| `autotrail.config` | areas and data folder |
| `autotrail.graph` | download, cache, gap stitching, `StreetGraph` lookups |
| `autotrail.trail` | `Trail`, `Mark`, `Branch`, `EdgeRef`, `EdgePos`; validation; GeoJSON and GPX; snapping drawn trails |
| `autotrail.viz` | Folium map |
| `autotrail.cli` | `autotrail` command |

## Tests

```bash
pytest
```

The tests use a small synthetic street grid, so they run offline.
