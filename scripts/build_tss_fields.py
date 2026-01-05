"""Rasterize TSS lanes, separation zones, and direction fields from a single GeoJSON."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
from typing import Iterable, Tuple, List

import numpy as np
from shapely.geometry import LineString, Polygon, shape, mapping
from shapely.ops import nearest_points
from rasterio import features
import rasterio

from ocean_router.core.grid import GridSpec
from ocean_router.core.memmaps import save_memmap


# seamark:type values for different TSS features
TSS_LANE_TYPES = {"separation_lane"}
TSS_ZONE_TYPES = {"separation_zone"}
TSS_BOUNDARY_TYPES = {"separation_boundary", "separation_line"}


def load_geojson(path: Path) -> dict:
    """Load a GeoJSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_seamark_type(feature: dict) -> str | None:
    """Extract seamark:type from feature properties."""
    props = feature.get("properties", {})
    
    # Check parsed_other_tags first
    parsed = props.get("parsed_other_tags", {})
    if parsed and "seamark:type" in parsed:
        return parsed["seamark:type"]
    
    # Check other_tags string
    other_tags = props.get("other_tags", "")
    if "seamark:type" in other_tags:
        # Parse hstore-style string: "key"=>"value"
        import re
        match = re.search(r'"seamark:type"=>"([^"]+)"', other_tags)
        if match:
            return match.group(1)
    
    return None


def get_flow_bearing(feature: dict) -> float | None:
    """Extract flow bearing from feature properties."""
    props = feature.get("properties", {})
    bearing = props.get("tss_flow_bearing_deg")
    if bearing is not None:
        return float(bearing)
    return None


def rasterize_mask(shapes_iter: Iterable[Tuple[object, int]], grid: GridSpec, out_path: Path) -> np.ndarray:
    """Rasterize shapes to a binary mask."""
    transform = rasterio.transform.from_origin(grid.xmin, grid.ymax, grid.dx, grid.dy)
    shapes_list = list(shapes_iter)
    if not shapes_list:
        mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
    else:
        mask = features.rasterize(
            shapes_list, 
            out_shape=(grid.height, grid.width), 
            transform=transform, 
            fill=0, 
            dtype="uint8"
        )
    save_memmap(out_path, mask.astype(np.uint8), dtype=np.uint8)
    return mask


def build_direction_field_from_features(
    lane_features: List[dict], 
    grid: GridSpec, 
    out_path: Path, 
    influence_nm: float = 2.0
) -> None:
    """Build direction field from TSS lane features with pre-computed bearings.
    
    Uses rasterization approach - buffers each lane and fills with its bearing.
    Much faster than cell-by-cell iteration.
    """
    import rasterio
    from rasterio import features as rio_features
    
    field = np.full((grid.height, grid.width), -1, dtype=np.int16)
    influence_deg = influence_nm / 60
    
    # Pre-process lanes with bearings
    lanes_with_bearings = []
    for feat in lane_features:
        bearing = get_flow_bearing(feat)
        if bearing is not None:
            geom = shape(feat["geometry"])
            lanes_with_bearings.append((geom, int(bearing)))
    
    if not lanes_with_bearings:
        print("  Warning: No lanes with flow bearings found")
        save_memmap(out_path, field, dtype=np.int16)
        return
    
    print(f"  Processing {len(lanes_with_bearings)} lanes with flow bearings...")
    print(f"  Influence radius: {influence_nm} nm ({influence_deg:.4f}°)")
    
    transform = rasterio.transform.from_origin(grid.xmin, grid.ymax, grid.dx, grid.dy)
    
    # Process in batches to show progress
    batch_size = 100
    total_lanes = len(lanes_with_bearings)
    
    for i in range(0, total_lanes, batch_size):
        batch = lanes_with_bearings[i:i+batch_size]
        
        # Buffer and rasterize each lane
        shapes_with_bearings = []
        for geom, bearing in batch:
            buffered = geom.buffer(influence_deg)
            shapes_with_bearings.append((buffered, bearing))
        
        # Rasterize this batch (later lanes overwrite earlier ones where they overlap)
        if shapes_with_bearings:
            batch_field = rio_features.rasterize(
                shapes_with_bearings,
                out_shape=(grid.height, grid.width),
                transform=transform,
                fill=-1,
                dtype=np.int16
            )
            # Merge: new values overwrite -1, keep existing non-(-1) values
            mask = batch_field >= 0
            field[mask] = batch_field[mask]
        
        pct = min(100, ((i + batch_size) * 100) // total_lanes)
        print(f"    {pct}% complete... ({i + len(batch)}/{total_lanes} lanes)")
    
    save_memmap(out_path, field, dtype=np.int16)
    valid_cells = np.sum(field >= 0)
    print(f"  Direction field: {valid_cells:,} cells with TSS direction")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TSS fields from GeoJSON with flow directions")
    parser.add_argument("--grid", type=Path, default=Path("configs/grid_1nm.json"))
    parser.add_argument("--tss-geojson", type=Path, required=True, 
                        help="GeoJSON file with TSS features and tss_flow_bearing_deg")
    parser.add_argument("--outdir", type=Path, default=Path("data/processed/tss"))
    parser.add_argument("--influence-nm", type=float, default=2.0,
                        help="Influence radius in nautical miles for direction field")
    args = parser.parse_args()

    grid = GridSpec.from_file(args.grid)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TSS data from {args.tss_geojson}...")
    geojson = load_geojson(args.tss_geojson)
    features_list = geojson.get("features", [])
    print(f"  Loaded {len(features_list)} features")

    # Categorize features by seamark type
    lane_features = []
    zone_features = []
    boundary_features = []
    
    for feat in features_list:
        seamark_type = get_seamark_type(feat)
        if seamark_type in TSS_LANE_TYPES:
            lane_features.append(feat)
        elif seamark_type in TSS_ZONE_TYPES:
            zone_features.append(feat)
        elif seamark_type in TSS_BOUNDARY_TYPES:
            boundary_features.append(feat)
    
    print(f"  Separation lanes: {len(lane_features)}")
    print(f"  Separation zones: {len(zone_features)}")
    print(f"  Boundaries/lines: {len(boundary_features)}")

    # Rasterize lane mask (buffer lines to create lane areas)
    print("Building lane mask...")
    lane_shapes = []
    for feat in lane_features:
        geom = shape(feat["geometry"])
        # Buffer lines by ~0.5nm to create lane areas
        if geom.geom_type in ("LineString", "MultiLineString"):
            buffered = geom.buffer(0.5 / 60)  # 0.5nm in degrees
            lane_shapes.append((buffered, 1))
        else:
            lane_shapes.append((geom, 1))
    
    lane_mask = rasterize_mask(lane_shapes, grid, args.outdir / "tss_lane_mask_1nm.npy")
    print(f"  Lane mask: {np.sum(lane_mask > 0)} cells")

    # Rasterize separation zone mask
    print("Building separation zone mask...")
    zone_shapes = []
    for feat in zone_features:
        geom = shape(feat["geometry"])
        zone_shapes.append((geom, 1))
    
    sep_mask = rasterize_mask(zone_shapes, grid, args.outdir / "tss_sepzone_mask_1nm.npy")
    print(f"  Separation zone mask: {np.sum(sep_mask > 0)} cells")

    # Build direction field from lane features with bearings
    print("Building direction field...")
    build_direction_field_from_features(
        lane_features, grid, args.outdir / "tss_dir_field_1nm.npy", 
        influence_nm=args.influence_nm
    )

    # Remove separation zones from lane mask
    if np.any(sep_mask):
        lane_mask[sep_mask.astype(bool)] = 0
        save_memmap(args.outdir / "tss_lane_mask_1nm.npy", lane_mask, dtype=np.uint8)
        print("  Updated lane mask (removed separation zones)")

    print("Done!")


if __name__ == "__main__":
    main()
