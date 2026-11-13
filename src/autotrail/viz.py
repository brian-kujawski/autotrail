"""Folium HTML map of a trail."""

from __future__ import annotations

from pathlib import Path

import folium
from shapely.geometry import box

from .graph import StreetGraph
from .trail import METRES_PER_MILE, MarkKind, Trail, _path_coords, _pos_lonlat

# (colour, short label) for each mark kind.
MARK_STYLE = {
    MarkKind.BLOB: ("#444444", "o"),
    MarkKind.ARROW: ("#444444", ">"),
    MarkKind.CHECK: ("#d62728", "X"),
    MarkKind.BACKCHECK: ("#9467bd", "BC"),
    MarkKind.FALSE: ("#000000", "F"),
    MarkKind.WHICHY_WAY: ("#ff7f0e", "?"),
    MarkKind.SPLIT: ("#2ca02c", "T/E"),
    MarkKind.REJOIN: ("#2ca02c", "RJ"),
    MarkKind.BEER: ("#bcbd22", "B"),
}
# Basemaps. OpenStreetMap's own tile servers refuse pages opened from a local
# file (no HTTP Referer, against their tile usage policy), so the default is
# Esri's street map, which needs no key. "osm" works only when the HTML is
# served over http(s), e.g. ``python -m http.server``.
BASEMAPS = {
    "streets": ("Esri.WorldStreetMap", "Streets (Esri)"),
    "satellite": ("Esri.WorldImagery", "Satellite (Esri)"),
    "osm": ("OpenStreetMap", "OpenStreetMap"),
}
BRANCH_COLOUR = {"FALSE": "#555555", "BACKCHECK_LURE": "#9467bd", "TURKEY": "#2ca02c", "EAGLE": "#17becf"}


def _latlon(coords):
    return [(lat, lon) for lon, lat in coords]


def _badge(text: str, colour: str) -> folium.DivIcon:
    return folium.DivIcon(
        icon_size=(26, 26), icon_anchor=(13, 13),
        html=(f'<div style="background:{colour};color:#fff;border:2px solid #fff;border-radius:13px;'
              f'width:26px;height:26px;line-height:22px;text-align:center;font:bold 10px sans-serif;'
              f'box-shadow:0 0 3px rgba(0,0,0,.6)">{text}</div>'),
    )


def _add_basemaps(m: folium.Map, default: str) -> None:
    if default not in BASEMAPS:
        raise ValueError(f"Unknown basemap {default!r}; choose from {', '.join(BASEMAPS)}")
    # The default goes first so it shows on load; satellite is offered as a switch.
    for key in [default] + [k for k in ("streets", "satellite") if k != default]:
        tiles, name = BASEMAPS[key]
        # Only the default is on at load; otherwise Leaflet shows the last one added.
        folium.TileLayer(tiles=tiles, name=name, max_zoom=19, show=(key == default)).add_to(m)


def trail_map(trail: Trail, graph: StreetGraph, *, show_virtual: bool = True,
              tiles: str = "streets") -> folium.Map:
    true_ll = _latlon(_path_coords(graph, trail.true_path))
    lats = [p[0] for p in true_ll]
    lons = [p[1] for p in true_ll]
    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=15,
                   tiles=None, control_scale=True)
    _add_basemaps(m, tiles)

    true_mi = trail.true_trail_m(graph) / METRES_PER_MILE
    route = folium.FeatureGroup(name=f"True trail ({true_mi:.2f} mi)")
    folium.PolyLine(true_ll, color="#d62728", weight=5, opacity=0.85,
                    tooltip=f"True trail: {true_mi:.2f} mi").add_to(route)
    route.add_to(m)

    if trail.return_path:
        ret_mi = trail.return_m(graph) / METRES_PER_MILE
        fg = folium.FeatureGroup(name=f"Return leg ({ret_mi:.2f} mi)")
        folium.PolyLine(_latlon(_path_coords(graph, trail.return_path)), color="#1f77b4", weight=4,
                        opacity=0.8, dash_array="8 6", tooltip=f"Return leg: {ret_mi:.2f} mi").add_to(fg)
        fg.add_to(m)

    if trail.branches:
        fg = folium.FeatureGroup(name="Falses and branches")
        for b in trail.branches:
            length = Trail.path_length_m(graph, b.edges)
            folium.PolyLine(_latlon(_path_coords(graph, b.edges)),
                            color=BRANCH_COLOUR.get(b.purpose.value, "#555"), weight=4, opacity=0.8,
                            dash_array="2 6",
                            tooltip=f"{b.purpose.value.title()} {b.label} ({length:.0f} m)").add_to(fg)
        fg.add_to(m)

    if trail.marks:
        fg = folium.FeatureGroup(name="Marks")
        blobs = folium.FeatureGroup(name="Blobs")
        for i, mk in enumerate(trail.marks):
            lon, lat = _pos_lonlat(graph, mk.at)
            colour, text = MARK_STYLE.get(mk.kind, ("#333", "?"))
            if mk.kind in (MarkKind.BLOB, MarkKind.ARROW):
                # Blobs are many and small: dots, not badges.
                folium.CircleMarker((lat, lon), radius=3, color="#222", weight=1, fill=True,
                                    fill_color="#fff", fill_opacity=1.0,
                                    tooltip="blob" + (" (return leg)" if mk.params.get("leg") == "return"
                                                      else "")).add_to(blobs)
                continue
            if mk.kind == MarkKind.BACKCHECK and mk.params.get("n"):
                text = f"BC{mk.params['n']}"
            note = ", ".join(f"{k}={v}" for k, v in mk.params.items())
            folium.Marker((lat, lon), icon=_badge(text, colour),
                          tooltip=f"#{i} {mk.kind.value}" + (f" ({note})" if note else "")).add_to(fg)
        blobs.add_to(m)
        fg.add_to(m)

    ends = folium.FeatureGroup(name="Start and Beer Near")
    slon, slat = graph.node_lonlat(trail.start)
    elon, elat = graph.node_lonlat(trail.end)
    folium.Marker((slat, slon), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(ends)
    folium.Marker((elat, elon), tooltip="Beer Near", icon=folium.Icon(color="orange", icon="glass")).add_to(ends)
    ends.add_to(m)

    if show_virtual:
        pad = 0.01  # about 1 km around the trail
        area = box(min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)
        fg = folium.FeatureGroup(name="Stitched OSM gaps (virtual edges)", show=False)
        seen = set()
        for u, v, k in graph.virtual_edges():
            if (v, u) in seen:
                continue
            seen.add((u, v))
            line = graph.edge_line(u, v, k)
            if area.intersects(line):
                folium.PolyLine(_latlon(line.coords), color="#e377c2", weight=6, opacity=0.9,
                                tooltip=f"Stitched gap ({graph.edge_length(u, v, k):.1f} m)").add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    return m


def save_trail_map(trail: Trail, graph: StreetGraph, out: str | Path, **kw) -> Path:
    out = Path(out)
    trail_map(trail, graph, **kw).save(str(out))
    return out


# ---------------------------------------------------------------- pack replay

def _pace_colour(frac: float) -> str:
    """0 = fastest (red) .. 1 = slowest (blue)."""
    a, b = (214, 39, 40), (31, 119, 180)
    r, g, bl = (round(x + (y - x) * frac) for x, y in zip(a, b))
    return f"#{r:02x}{g:02x}{bl:02x}"


_REPLAY_JS = """
window.addEventListener('load', function() {
  var map = %(map)s, D = %(data)s;
  var n = D.times.length, dt = D.dt, t = 0, playing = false, speed = 30, last = null;
  map.createPane('hounds');
  map.getPane('hounds').style.zIndex = 650;
  var dots = D.hounds.map(function(h) {
    return L.circleMarker(h.path[0], {pane: 'hounds', radius: 7, color: '#fff', weight: 2, fillColor: h.colour,
      fillOpacity: 0.95}).bindTooltip(h.name + ' (' + h.pace + ')').addTo(map);
  });
  function fmt(s) { s = Math.round(s); return Math.floor(s / 60) + ':' + ('0' + s %% 60).slice(-2); }
  var box = L.control({position: 'bottomleft'});
  box.onAdd = function() {
    var d = L.DomUtil.create('div');
    d.style.cssText = 'background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.4);' +
      'font:12px sans-serif;width:320px;max-width:80vw';
    d.innerHTML = '<div style="font-weight:bold;margin-bottom:4px">' + D.title + '</div>' +
      '<div style="margin-bottom:6px;color:#444">' + D.score + '</div>' +
      '<button id="rp-play">Play</button> <select id="rp-speed">' +
      '<option value="10">10x</option><option value="30" selected>30x</option><option value="60">60x</option>' +
      '<option value="120">120x</option></select> <span id="rp-clock" style="font-weight:bold">0:00</span>' +
      '<input id="rp-slider" type="range" min="0" max="' + (n - 1) * dt + '" step="1" value="0" style="width:100%%">' +
      '<div id="rp-log" style="height:84px;overflow:hidden;color:#333;line-height:14px"></div>' +
      '<div style="color:#777;margin-top:4px">Dots: red = fastest pace, blue = slowest.</div>';
    L.DomEvent.disableClickPropagation(d);
    L.DomEvent.disableScrollPropagation(d);
    return d;
  };
  box.addTo(map);
  var btn = document.getElementById('rp-play'), slider = document.getElementById('rp-slider'),
      clock = document.getElementById('rp-clock'), logEl = document.getElementById('rp-log');
  function pos(path, f) {
    var i = Math.min(Math.floor(f), path.length - 1), j = Math.min(i + 1, path.length - 1), w = f - i;
    return [path[i][0] + (path[j][0] - path[i][0]) * w, path[i][1] + (path[j][1] - path[i][1]) * w];
  }
  function draw() {
    var f = t / dt;
    D.hounds.forEach(function(h, k) { dots[k].setLatLng(pos(h.path, f)); });
    clock.textContent = fmt(t);
    slider.value = Math.round(t);
    var recent = D.events.filter(function(e) { return e[0] <= t; }).slice(-6).reverse();
    logEl.innerHTML = recent.map(function(e) { return fmt(e[0]) + ' ' + e[1]; }).join('<br>');
  }
  function frame(ts) {
    if (!playing) return;
    if (last !== null) t = Math.min((n - 1) * dt, t + (ts - last) / 1000 * speed);
    last = ts;
    draw();
    if (t >= (n - 1) * dt) { playing = false; btn.textContent = 'Play'; return; }
    requestAnimationFrame(frame);
  }
  btn.onclick = function() {
    playing = !playing; btn.textContent = playing ? 'Pause' : 'Play'; last = null;
    if (playing) { if (t >= (n - 1) * dt) t = 0; requestAnimationFrame(frame); }
  };
  document.getElementById('rp-speed').onchange = function(e) { speed = +e.target.value; };
  slider.oninput = function(e) { t = +e.target.value; draw(); };
  draw();
});
"""


def replay_map(trail: Trail, graph: StreetGraph, result, card=None, *, tiles: str = "streets",
               dt: float = 5.0) -> folium.Map:
    """The trail map plus an animated replay of one simulated run."""
    import json

    from .sim import sample_tracks

    m = trail_map(trail, graph, tiles=tiles)
    times, ll, _ = sample_tracks(result, dt)
    paces = [h.pace_s_per_m for h in result.hounds]
    lo, hi = min(paces), max(paces)
    hounds = []
    for i, h in enumerate(result.hounds):
        frac = 0.0 if hi == lo else (h.pace_s_per_m - lo) / (hi - lo)
        path = [[round(float(lat), 6), round(float(lon), 6)] for lon, lat in ll[i]]
        from .pack import fmt_pace
        hounds.append({"name": h.name, "pace": fmt_pace(h.pace_s_per_m), "colour": _pace_colour(frac),
                       "path": path})
    events = []
    for h in result.raw:
        for t, what in h.events:
            if what.startswith(("On-On", "beer", "false")):
                events.append([round(t, 1), f"{h.prof.name}: {what}"])
    for c in result.checks:
        if c.first_arrival_s is not None:
            events.append([round(c.first_arrival_s, 1), f"Front reaches check {c.idx + 1}"])
    events.sort()
    score = ""
    if card is not None:
        s = card.summary
        coh = card.term("cohesion").details
        score = (f"Score {card.total:.0f}/100. True trail {s['true_trail_mi']:.2f} mi; front runners ran "
                 f"+{s['front_extra_mi']:.2f} mi, back +{s['back_extra_mi']:.2f} mi. Finish spread "
                 f"{coh['finish_spread_min']:.1f} min.")
    data = {"dt": dt, "times": [float(x) for x in times], "hounds": hounds, "events": events,
            "title": f"Pack replay (run seed {result.seed})", "score": score}
    js = _REPLAY_JS % {"map": m.get_name(), "data": json.dumps(data, separators=(",", ":"))}
    m.get_root().script.add_child(folium.Element(js))
    return m


def save_replay_map(trail: Trail, graph: StreetGraph, result, out: str | Path, card=None, **kw) -> Path:
    out = Path(out)
    replay_map(trail, graph, result, card, **kw).save(str(out))
    return out
