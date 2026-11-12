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


def trail_map(trail: Trail, graph: StreetGraph, *, show_virtual: bool = True) -> folium.Map:
    true_ll = _latlon(_path_coords(graph, trail.true_path))
    lats = [p[0] for p in true_ll]
    lons = [p[1] for p in true_ll]
    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=15,
                   tiles="OpenStreetMap", control_scale=True)

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
        for i, mk in enumerate(trail.marks):
            lon, lat = _pos_lonlat(graph, mk.at)
            colour, text = MARK_STYLE.get(mk.kind, ("#333", "?"))
            if mk.kind == MarkKind.BACKCHECK and mk.params.get("n"):
                text = f"BC{mk.params['n']}"
            note = ", ".join(f"{k}={v}" for k, v in mk.params.items())
            folium.Marker((lat, lon), icon=_badge(text, colour),
                          tooltip=f"#{i} {mk.kind.value}" + (f" ({note})" if note else "")).add_to(fg)
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
