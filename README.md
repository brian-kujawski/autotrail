# Autotrail

Suggest hash trails from OpenStreetMap data and simulate a pack of hounds running them.

This is **Phase 1, the MVP end to end**: download and cache a walking map, generate a
one-click trail with blobs, arrows, checks, whichy-ways and falses (`autotrail generate`), run
a simulated pack and hare on any trail and score it (`autotrail sim`), and watch the replay on
a map. Backchecks, splits, walkers and placement questions (`autotrail ask`) are phase 2.
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

### Web GUI (the easy way)

```bash
autotrail gui
```

Loads the map once (that part takes a while on Detroit), then opens `http://localhost:8765/` in
your browser. Click the map to set **Start**, then click where the **Beer Near** goes (the car
and cooler). Drag either marker to move it; each lands on the nearest point of the street or path you clicked (mid-block is fine), and the trail starts or ends exactly there. The page
tells you how far the Beer Near is on foot from Start and refuses a pair that can't make a
trail of the length you asked for. Set the trail length, seed and, under *Pack, hare and
search*, the pack, the hare and how many routes to try, then **Generate trail**. Progress shows
as it runs; when it's done the **Trail and replay** tab shows the map with the pack replay
(press Play), the score, and GeoJSON and GPX downloads. **Try another seed** reruns with the
next seed. Every run is saved under `data/gui_runs/<time>-seed<n>/` and listed under *Earlier
runs*, so you can flip back to one.

Tick *Let Autotrail pick the Beer Near* to have it choose one about a mile from Start, as the
command line does without `--end`.

`--port`, `--no-open` and `--runs-dir` change the defaults. The server only listens on this PC
(`127.0.0.1`). Leaflet and the map tiles come from the internet, so you need to be online.

### One-click trail

```bash
autotrail generate --start "42.3315,-83.0700" --open
autotrail generate --start "42.3315,-83.0700" --end "42.3350,-83.0600" --miles 4 --seed 2 --out tuesday.geojson
```

Start (and the Beer Near, if you give one) go on the nearest point of the nearest street or path. With `--end`
the leg back to Start is sized from the real walk between the two (`--return-miles 0` still
turns it off). Without `--end` a Beer Near is picked about `--return-miles` (default 1) from
Start, and a hared leg back is added. It tries about 60 loop-shaped routes that keep clear of themselves, puts marks on the
best few several ways, simulates the pack and the hare on each and keeps the best. Takes about
a minute on the Detroit map. `--seed` gives a different trail. It writes the trail GeoJSON plus
an HTML map with the pack replay (press Play), and prints the score.

`--hare-pace 10:15` (min/mile while marking) and `--head-start 15` (minutes) set the hare.

### Simulate a trail

```bash
autotrail sim tuesday.geojson --open
autotrail sim examples/corktown_generated.geojson --open                        # made by the generate example above
autotrail sim examples/corktown_sample.geojson --runs 50 --runners 15 --fastest 7:00 --slowest 10:00
```

Runs the pack (default 12 runners, 7:30 to 9:30 min/mile) 20 times and prints a scorecard,
each hound's finish time, distance run and extra over true trail, how close the pack got to the
hare, and what each check did. A trail with no blobs or with unmarked turns (like a
hand-drawn one) gets blobs and arrows filled in first.

### Show a trail

`show` loads a trail, checks it against the map, prints the true trail distance and writes an
HTML map next to the file (`--out` to choose). Add `--save trail.geojson` to keep the snapped
Autotrail version and `--gpx trail.gpx` to export it for a phone.

The basemap defaults to Esri's street map, with satellite as a switch in the layer menu
(`--tiles satellite` to start on it). OpenStreetMap's own tile servers refuse maps opened
straight from a file, under their [tile usage policy](https://operations.osmfoundation.org/policies/tiles/),
so `--tiles osm` only works if you serve the HTML over http (for example `python -m http.server`).

The download is OSMnx's walking network plus cycleways (OSM often tags a shared path through a
park as a cycleway, and hashes run on bike paths; ways marked `foot=no` are still left out). A map
cached before this change prints a warning: run `autotrail download --force` to refresh it.

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

## Hash rules

Both the generator and the simulated hounds follow these (`autotrail.rules`):

* **Trail never turns without a mark.** At a corner with no mark, trail goes straight on.
  "Straight" means one way on within 20 degrees of the way you came in, with every other way at
  least 20 degrees further off (a bend with only one way on counts too). Every other turn gets a
  check, a whichy-way or an arrow. A **jog** (two turns that cancel out within 20 m, like a
  step onto a sidewalk or across an offset crossing) is just straight on and gets no marks.
* **Trail never crosses or comes near itself**, the leg back to Start included: no point is
  within 35 m of trail laid earlier. That also rules out doubling back and running back down
  the other side of a street, where hounds could cut across or see the hare.
* **Twisty, not rectangles.** Turns slow the pack far more than the hare (a mark takes the hare
  about 3 s), so the route search charges a turn only that (~8 m), and charges 3x extra for
  every metre of a straight past 150 m (400 m on the shorter, simpler leg back to Start). The
  loop's waypoints alternate out on its rim and in toward its middle, so the route has
  distance to spare for twists, and routes are shortlisted (and scored, `twistiness`) on turns per mile as well as
  distance. On a 100 m test grid that's ~15 turns per mile with straights of a block or two.
* **One loop, out and back in.** Waypoints go round one loop in order: out from Start, round a
  far point roughly away from the Beer Near, and home to the Beer Near, so no leg has to cut
  across an earlier one. None sits near Start, the Beer Near or the streets between them (a
  quarter of the loop radius), leaving those for the first and last legs and the way back. The
  shape is drawn fresh each time: long and narrow (nearly an out-and-back) through fat beans,
  bent, lopsided, every point at its own distance. "Loop" is only topology. If the far end
  would be in a river or past a freeway, the loop swings wider.
* **Keep hounds honest.** Most checks are at turns, but some checks and whichy-ways have
  straight on as the answer: any quarter mile with no check or whichy-way may get one at a
  crossing, and now and then a run of short blocks gets a whichy-way at every crossing
  (straight, straight, straight... then the turn). No ratio is targeted; on test grids about a
  third of checks come out straight on.
  Three marks past a check solve it ("On one", "On two", "On-On"). A whichy-way (two arrows on
  the ground) needs one mark on the right way; the wrong way has no marks at all, so it costs
  the hare nothing. Most turns without a check get a whichy-way; arrows are used sparingly.
* **Know where every wrong way leads** (`autotrail.facts`). With the whole route known, each
  way off trail at each corner is walked out up to 250 m on foot (any path type but freeways
  and trunk roads). It's a **shortcut** if it meets trail still to come (not the way back: that's marked only
  after the pack is in) and
  skips at least 3x what it cost (and 100 m or more); otherwise it meets trail but saves
  little, is a dead end, or is safe. Each corner also notes whether checkers would see the
  hare. Step-through mode draws these on every kept route. Mark placement doesn't use them
  yet (that's the next step of the mark pass).
* **Falses are few and short**: none, one or two per trail, a couple of blocks off a check,
  marked often enough to call On-On, then an F. The hare runs every false out and back.
* **The pack never sees the hare.** A hound sees about 250 m down the street it's on (both
  ways) until the street bends, and down every street from a corner it stands at. A real
  sighting kills a trail (it turns into a footrace), so the score drops fast. A quick dash
  across a street two blocks down costs little; running away down a street the pack is
  looking along costs a lot. Two soft nudges in the generator: streets the pack could see when
  the hare runs them cost more in the route search, and a turn where hounds checking the wrong
  ways would look at the hare usually gets an arrow instead of a check (sometimes with a check
  one corner on). Hounds seeing *other hounds* on another part of trail is fine: it pulls the
  pack together.

## How the simulated pack behaves

Each hound has a pace and follows a few rules (`autotrail.sim`; numbers in
`autotrail.pack.HoundProfile`):

* **Running**: spots each blob with probability 0.9 (arrows and checks are never missed). It
  can't see marks down side streets. After 200 m with no mark it goes back to the last one and
  looks again.
* **Corners**: with no mark, straight on. If the trail turned there anyway, the hound runs out,
  finds nothing and comes back.
* **Checks**: the first hound picks a direction, the next takes another, and so on until every
  way is taken; later hounds wait for a call. Three marks out a hound calls On-On; everyone
  waiting at the check, or within 150 m, runs to it. The check is then marked, so late hounds
  go straight the right way. A false fools the hounds who call On-On down it until someone
  reaches the F and scratches the mark.
* **Done** at the Beer Near. The return leg isn't simulated yet.

The **hare** starts 15 minutes early at 10:15/mile and runs the trail plus every false out and
back. The replay shows the hare as a black dot ringed in yellow.

## Score

`autotrail.score` gives each term 0 to 1 and a weighted total out of 100:

| term | weight | measures |
| --- | --- | --- |
| `distance_fit` | 1 | true trail vs target: full marks within 5%, zero at 25% off |
| `cohesion` | 2 | finish spread, mean and worst front-to-back gap along the trail (each 0.5 at 3 min), less time the back hound has nobody within 150 m, less any hound that doesn't finish |
| `check_effectiveness` | 1 | share of the front-to-back gap the checks close |
| `hare_safety` | 2 | how close the front of the pack gets to the hare: full marks 2 min or more behind, zero if caught |
| `variety` | 1 | any quarter mile with nothing for the pack to do (no check or whichy-way) is a negative: each 0.5 mi of such dull trail past the first quarter mile halves it |
| `hare_hidden` | 3 | seconds the hare spends in a hound's sightline (weighted: full up to 50 m away, 0.3 at 250 m); 8 s halves it |

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
| `autotrail.pipeline` | one generate run (settings, Start/Beer Near checks, files), shared by the CLI and GUI |
| `autotrail.gui` | local web GUI: `server.py` (standard library HTTP server) and `index.html` (Leaflet page) |
| `autotrail.cli` | `autotrail` command |

## Tests

```bash
pytest
```

The tests use a small synthetic street grid, so they run offline.
