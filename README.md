# Autotrail

Suggest hash trails from OpenStreetMap data and simulate a pack of hounds running them.

This is **Phase 1, the MVP end to end**: download and cache a walking map, generate a
one-click trail with blobs, checks and falses (`autotrail generate`), run a simulated pack on
any trail and score it (`autotrail sim`), and watch the replay on a map. Backchecks,
whichy-ways, splits, walkers and placement questions (`autotrail ask`) are phase 2.
See the design doc for the plan.

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

### One-click trail

```bash
autotrail generate --start "42.3315,-83.0700" --open
autotrail generate --start "42.3315,-83.0700" --end "42.3350,-83.0600" --miles 4 --seed 2 --out tuesday.geojson
```

Start (and the Beer Near, if you give one) snap to the nearest intersection. Without `--end`
a Beer Near is picked about `--return-miles` (default 1) from Start, and a hared leg back is
added. It tries a few hundred routes, puts marks on the best few several ways, simulates the
pack on each and keeps the best. Takes 15 to 30 seconds on the Detroit map. `--seed` gives a
different trail. It writes the trail GeoJSON plus an HTML map with the pack replay (press
Play), and prints the score.

### Simulate a trail

```bash
autotrail sim tuesday.geojson --open
autotrail sim examples/corktown_generated.geojson --open                        # made by the generate example above
autotrail sim examples/corktown_sample.geojson --runs 50 --runners 15 --fastest 7:00 --slowest 10:00
```

Runs the pack (default 12 runners, 7:30 to 9:30 min/mile) 20 times and prints a scorecard,
each hound's finish time, distance run and extra over true trail, and what each check did.
A trail with no blobs (like a hand-drawn one) gets blobs filled in first.

### Show a trail

`show` loads a trail, checks it against the map, prints the true trail distance and writes an
HTML map next to the file (`--out` to choose). Add `--save trail.geojson` to keep the snapped
Autotrail version and `--gpx trail.gpx` to export it for a phone.

The basemap defaults to Esri's street map, with satellite as a switch in the layer menu
(`--tiles satellite` to start on it). OpenStreetMap's own tile servers refuse maps opened
straight from a file, under their [tile usage policy](https://operations.osmfoundation.org/policies/tiles/),
so `--tiles osm` only works if you serve the HTML over http (for example `python -m http.server`).

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

## How the simulated pack behaves

Each hound has a pace and follows a few rules (`autotrail.sim`; numbers in
`autotrail.pack.HoundProfile`):

* **Running**: spots each blob with probability 0.9. After 200 m with no mark it goes back
  to the last one and looks again.
* **Corners**: straight on is easy. If the trail turns, a hound spots the blob just past the
  corner (0.9), or runs straight on, finds nothing and comes back. A hound that saw someone
  take the corner the right way in the last 45 s just follows, which is why the back of the
  pack runs close to true trail.
* **Checks**: hounds spread over the directions nobody has tried (up to 150 m out and back,
  or to the F of a false). Two marks past the check and a hound calls On-On; everyone at that
  check within 150 m runs to it. Late arrivals at a solved check go straight on. A false with
  two or more blobs can fool a hound into calling On-On down it.
* **Done** at the Beer Near. The return leg isn't simulated yet.

## Score

`autotrail.score` gives each term 0 to 1 and a weighted total out of 100:

| term | weight | measures |
| --- | --- | --- |
| `distance_fit` | 1 | true trail vs target: full marks within 5%, zero at 25% off |
| `cohesion` | 2 | finish spread, mean and worst front-to-back gap along the trail (each 0.5 at 3 min), less time the back hound has nobody within 150 m, less any hound that doesn't finish |
| `check_effectiveness` | 1 | share of the front-to-back gap the checks close |

"Extra distance" (distance run minus true trail) is reported per hound and for the front and
back quarter of the pack.

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
| `autotrail.pack` | `HoundProfile`, pack presets |
| `autotrail.sim` | `TrailModel` (a trail compiled for the sim), event-driven tier 1 pack sim, `CheckPolicy` and `CallModel` hooks |
| `autotrail.score` | `ScoreTerm` plug-ins, `ScoreCard`, text report |
| `autotrail.generate` | route search, mark placement, `generate` (the one click) |
| `autotrail.viz` | Folium map and pack replay |
| `autotrail.cli` | `autotrail` command |

## Tests

```bash
pytest
```

The tests use a small synthetic street grid, so they run offline.
