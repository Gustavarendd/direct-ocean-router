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
from ocean_router.core.geodesy import angle_diff_deg, bearing_deg, rhumb_distance_nm
from ocean_router.core.memmaps import save_memmap


# seamark:type values for different TSS features
TSS_LANE_TYPES = {"separation_lane"}
TSS_ZONE_TYPES = {"separation_zone"}
TSS_BOUNDARY_TYPES = {"separation_boundary", "separation_line", "inshore_traffic_zone"}


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


def is_precautionary_area(feature: dict) -> bool:
    """Check if feature is a precautionary area (advisory, not mandatory)."""
    props = feature.get("properties", {})
    
    # Check parsed_other_tags first
    parsed = props.get("parsed_other_tags", {})
    if parsed and parsed.get("seamark:information") == "precautionary_area":
        return True
    
    # Check other_tags string
    other_tags = props.get("other_tags", "")
    if "precautionary_area" in other_tags:
        return True
    
    return False


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
    influence_nm: float = 0.0
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


def build_lane_graph(lane_features: List[dict], out_path: Path) -> None:
    """Build a lane-graph from lane centerlines with flow bearings."""
    nodes: list[tuple[float, float]] = []
    node_index: dict[tuple[float, float], int] = {}
    edges_u: list[int] = []
    edges_v: list[int] = []
    edges_weight: list[float] = []
    edges_flow_bearing: list[float] = []
    edges_length_nm: list[float] = []

    def _node_id(lon: float, lat: float) -> int:
        key = (lon, lat)
        idx = node_index.get(key)
        if idx is None:
            idx = len(nodes)
            node_index[key] = idx
            nodes.append(key)
        return idx

    def _add_edge(u: int, v: int, flow_bearing: float, length_nm: float, align_angle: float) -> None:
        edges_u.append(u)
        edges_v.append(v)
        edges_flow_bearing.append(flow_bearing)
        edges_length_nm.append(length_nm)
        edges_weight.append(length_nm * (1.0 + align_angle / 90.0))

    edges_added = 0
    for feat in lane_features:
        flow_bearing = get_flow_bearing(feat)
        if flow_bearing is None:
            continue
        geom = shape(feat["geometry"])
        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            for i in range(len(coords) - 1):
                lon0, lat0 = coords[i]
                lon1, lat1 = coords[i + 1]
                length_nm = rhumb_distance_nm(lon0, lat0, lon1, lat1)
                if length_nm <= 0:
                    continue
                seg_bearing = bearing_deg(lon0, lat0, lon1, lat1)
                align = angle_diff_deg(seg_bearing, flow_bearing)
                u = _node_id(lon0, lat0)
                v = _node_id(lon1, lat1)
                if align <= 60:
                    _add_edge(u, v, flow_bearing, length_nm, align)
                    edges_added += 1
                else:
                    rev_bearing = (seg_bearing + 180.0) % 360.0
                    rev_align = angle_diff_deg(rev_bearing, flow_bearing)
                    if rev_align <= 90:
                        _add_edge(v, u, flow_bearing, length_nm, rev_align)
                        edges_added += 1

    if not nodes or not edges_u:
        print("  Warning: No lane graph edges built")
        return

    nodes_arr = np.array(nodes, dtype=np.float64)
    np.savez(
        out_path,
        nodes_lon=nodes_arr[:, 0],
        nodes_lat=nodes_arr[:, 1],
        edges_u=np.array(edges_u, dtype=np.int32),
        edges_v=np.array(edges_v, dtype=np.int32),
        edges_weight=np.array(edges_weight, dtype=np.float32),
        edges_flow_bearing=np.array(edges_flow_bearing, dtype=np.float32),
        edges_length_nm=np.array(edges_length_nm, dtype=np.float32),
    )
    print(f"  Lane graph: {len(nodes)} nodes, {edges_added} directed edges")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TSS fields from GeoJSON with flow directions")
    parser.add_argument("--grid", type=Path, default=Path("configs/grid_1nm.json"))
    parser.add_argument("--resolution", type=str, default="1nm", 
                        help="Resolution suffix for output files (1nm or 0.5nm)")
    parser.add_argument("--tss-geojson", type=Path, required=True, 
                        help="GeoJSON file with TSS features and tss_flow_bearing_deg")
    parser.add_argument("--outdir", type=Path, default=Path("data/processed/tss"))
    parser.add_argument("--influence-nm", type=float, default=0.5,
                        help="Influence radius in nautical miles for direction field")
    args = parser.parse_args()
    
    # Derive file suffix from resolution (e.g., "0.5nm" -> "05nm")
    suffix = args.resolution.replace(".", "")

    grid = GridSpec.from_file(args.grid)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"[RESOLUTION] Building TSS fields at {args.resolution} ({grid.width}x{grid.height})")
    print(f"Loading TSS data from {args.tss_geojson}...")
    geojson = load_geojson(args.tss_geojson)
    features_list = geojson.get("features", [])
    print(f"  Loaded {len(features_list)} features")

    # Categorize features by seamark type
    lane_features = []
    zone_features = []
    boundary_features = []
    precautionary_count = 0
    
    for feat in features_list:
        seamark_type = get_seamark_type(feat)
        if seamark_type in TSS_LANE_TYPES:
            lane_features.append(feat)
        elif seamark_type in TSS_ZONE_TYPES:
            zone_features.append(feat)
        elif seamark_type in TSS_BOUNDARY_TYPES:
            # Skip precautionary areas - they are advisory, not mandatory
            if is_precautionary_area(feat):
                precautionary_count += 1
                continue
            boundary_features.append(feat)
    
    print(f"  Separation lanes: {len(lane_features)}")
    print(f"  Separation zones: {len(zone_features)}")
    print(f"  Boundaries/lines: {len(boundary_features)} (excluded {precautionary_count} precautionary areas)")

    # Rasterize lane mask (buffer lines to create lane areas)
    print("Building lane mask...")
    lane_shapes = []
    for feat in lane_features:
        geom = shape(feat["geometry"])
        # Buffer lines to create continuous lane areas
        # Use same physical buffer (1.5nm) regardless of resolution
        # This ensures lanes are wide enough for simplification to work
        buffer_nm = 0.50
        if geom.geom_type in ("LineString", "MultiLineString"):
            buffered = geom.buffer(buffer_nm / 60)  # Convert nm to degrees
            lane_shapes.append((buffered, 1))
        else:
            lane_shapes.append((geom, 1))
    
    lane_mask = rasterize_mask(lane_shapes, grid, args.outdir / f"tss_lane_mask_{suffix}.npy")
    print(f"  Lane mask: {np.sum(lane_mask > 0)} cells")

    # Rasterize separation zone mask
    print("Building separation zone mask...")
    zone_shapes = []
    for feat in zone_features:
        geom = shape(feat["geometry"])
        buffer_nm = 0.1
        if geom.geom_type in ("LineString", "MultiLineString"):
            # if start/end are same, make polygon
            if isinstance(geom, LineString) and geom.is_ring:
                geom = Polygon(geom)
            buffered = geom.buffer(buffer_nm / 60)  # Convert nm to degrees
            zone_shapes.append((buffered, 1))
        else:
            zone_shapes.append((geom, 1))
    
    sep_mask = rasterize_mask(zone_shapes, grid, args.outdir / f"tss_sepzone_mask_{suffix}.npy")
    print(f"  Separation zone mask: {np.sum(sep_mask > 0)} cells")

    # Rasterize separation boundary mask (lines that shouldn't be crossed)
    print("Building separation boundary mask...")
    boundary_shapes = []
    for feat in boundary_features:
        geom = shape(feat["geometry"])
        # Buffer boundary lines by 1nm to ensure continuous coverage
        # This is important for routing to detect crossings properly
        buffer_nm = 0.1
        if geom.geom_type in ("LineString", "MultiLineString"):
            buffered = geom.buffer(buffer_nm / 60)  # Convert nm to degrees
            boundary_shapes.append((buffered, 1))
        else:
            boundary_shapes.append((geom, 1))
    
    boundary_mask = rasterize_mask(boundary_shapes, grid, args.outdir / f"tss_sepboundary_mask_{suffix}.npy")
    print(f"  Separation boundary mask: {np.sum(boundary_mask > 0)} cells")

    # Build direction field from lane features with bearings
    print("Building direction field...")
    build_direction_field_from_features(
        lane_features, grid, args.outdir / f"tss_dir_field_{suffix}.npy", 
        influence_nm=args.influence_nm
    )

    print("Building lane graph...")
    build_lane_graph(
        lane_features,
        args.outdir / f"tss_lane_graph_{suffix}.npz",
    )

    # Clean up overlaps - proper layering:
    # 1. Remove separation zones from lane mask (zones are in the middle, not lanes)
    # 2. Remove lanes from boundary mask (boundaries are edges, not inside lanes)
    # 3. Remove separation zones from boundary mask (zones have their own mask)
    print("Cleaning up layer overlaps...")
    
    # if np.any(sep_mask):
    #     lane_mask[sep_mask.astype(bool)] = 0
    #     print(f"  Removed separation zones from lanes: {np.sum(lane_mask > 0)} lane cells remaining")
    
    # if np.any(lane_mask):
    #     boundary_mask[lane_mask.astype(bool)] = 0
    #     sep_mask[lane_mask.astype(bool)] = 0
    #     print(f"  Removed lanes from boundaries: {np.sum(boundary_mask > 0)} boundary cells remaining")
    
    # if np.any(sep_mask):
    #     boundary_mask[sep_mask.astype(bool)] = 0
    #     print(f"  Removed sep zones from boundaries: {np.sum(boundary_mask > 0)} boundary cells remaining")

    lane_bool = lane_mask.astype(bool)

    # Lanes are sacred — nothing touches them
    sep_mask[lane_bool] = 0
    boundary_mask[lane_bool] = 0

    # Optional: separation zones override boundaries
    sep_bool = sep_mask.astype(bool)
    boundary_mask[sep_bool] = 0
    
    # Save cleaned masks
    save_memmap(args.outdir / f"tss_lane_mask_{suffix}.npy", lane_mask, dtype=np.uint8)
    save_memmap(args.outdir / f"tss_sepzone_mask_{suffix}.npy", sep_mask, dtype=np.uint8)
    save_memmap(args.outdir / f"tss_sepboundary_mask_{suffix}.npy", boundary_mask, dtype=np.uint8)


    print("Done!")


if __name__ == "__main__":
    main()
