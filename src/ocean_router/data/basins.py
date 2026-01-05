"""Basin classifier using polygons."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
from shapely.geometry import Point


@dataclass
class BasinClassifier:
    path: Path

    def __post_init__(self) -> None:
        self.gdf = gpd.read_file(self.path)
        if "id" not in self.gdf.columns:
            raise ValueError("basins.geojson must contain an 'id' property")

    def find_basin(self, lon: float, lat: float) -> Optional[str]:
        pt = Point(lon, lat)
        matches = self.gdf[self.gdf.contains(pt)]
        if matches.empty:
            return None
        return matches.iloc[0]["id"]
