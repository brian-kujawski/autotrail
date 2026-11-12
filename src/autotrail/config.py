"""Area configuration: which places make up a map, and where caches live."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Area:
    key: str
    name: str
    places: tuple[str, ...]
    network_type: str = "walk"
    stitch_tolerance_m: float = 5.0

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.key.lower()).strip("-")


def _read_areas_file(path: Path | None) -> dict:
    if path is None:
        text = resources.files("autotrail").joinpath("areas.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def load_areas(path: Path | None = None) -> tuple[dict[str, Area], str | None]:
    """Return ({key: Area}, default_key) from the bundled or a given YAML file."""
    raw = _read_areas_file(path)
    areas: dict[str, Area] = {}
    for key, spec in (raw.get("areas") or {}).items():
        places = spec.get("places") or []
        if isinstance(places, str):
            places = [places]
        if not places:
            raise ValueError(f"Area {key!r} has no places")
        areas[key] = Area(
            key=key,
            name=spec.get("name", key),
            places=tuple(places),
            network_type=spec.get("network_type", "walk"),
            stitch_tolerance_m=float(spec.get("stitch_tolerance_m", 5.0)),
        )
    return areas, raw.get("default")


def get_area(key: str | None = None, path: Path | None = None) -> Area:
    areas, default = load_areas(path)
    key = key or default
    if key not in areas:
        known = ", ".join(sorted(areas)) or "none"
        raise KeyError(f"Unknown area {key!r}. Known areas: {known}")
    return areas[key]


def data_dir() -> Path:
    """Where downloaded graphs are cached. Override with AUTOTRAIL_DATA."""
    return Path(os.environ.get("AUTOTRAIL_DATA", "data")).resolve()
