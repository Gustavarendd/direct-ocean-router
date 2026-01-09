#!/usr/bin/env python3
"""Create an interactive Leaflet map visualization of routes with TSS and land."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import numpy as np
from PIL import Image
import base64
import io
import requests

from ocean_router.core.grid import GridSpec


def load_mask(path: Path) -> np.ndarray:
    """Load a memmap file with its metadata."""
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return np.memmap(path, mode='r', dtype=meta['dtype'], shape=tuple(meta['shape']))
    else:
        return np.load(path, mmap_mode='r')


def create_tile_image(
    grid: GridSpec,
    land_mask: np.ndarray,
    lane_mask: np.ndarray | None,
    sepzone_mask: np.ndarray | None,
    sepboundary_mask: np.ndarray | None,
    bbox: tuple,
    depth: np.ndarray | None = None,
    depth_nodata: int = -32768,
    depth_max_m: int = 1000,
    size: int = 512,
) -> tuple[str, tuple]:
    """Create a PNG image for the bbox and return as base64 with actual bounds.
    
    Returns:
        Tuple of (base64_image, actual_bounds) where actual_bounds is 
        (lon_min, lon_max, lat_min, lat_max) matching the extracted pixels.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    
    # Convert bbox to grid indices
    x_min = max(0, int((lon_min - grid.xmin) / grid.dx))
    x_max = min(grid.width, int((lon_max - grid.xmin) / grid.dx))
    y_min = max(0, int((grid.ymax - lat_max) / grid.dy))
    y_max = min(grid.height, int((grid.ymax - lat_min) / grid.dy))
    
    # Calculate actual geographic bounds for these pixel indices
    # This ensures the overlay aligns exactly with the extracted pixels
    actual_lon_min = grid.xmin + x_min * grid.dx
    actual_lon_max = grid.xmin + x_max * grid.dx
    actual_lat_max = grid.ymax - y_min * grid.dy  # top edge
    actual_lat_min = grid.ymax - y_max * grid.dy  # bottom edge
    actual_bounds = (actual_lon_min, actual_lon_max, actual_lat_min, actual_lat_max)
    
    # Extract subsets
    land_sub = land_mask[y_min:y_max, x_min:x_max]
    depth_sub = None
    if depth is not None:
        depth_sub = depth[y_min:y_max, x_min:x_max]
    
    # Create RGBA image
    h, w = land_sub.shape
    img_data = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Ocean - transparent
    # Land - tan. Also treat cells with non-negative bathymetry as land
    if depth_sub is not None:
        land_pixels = (land_sub > 0) | (depth_sub >= 0)
    else:
        land_pixels = land_sub > 0
    img_data[land_pixels] = [212, 196, 168, 255]
    
    # TSS lanes - light green
    if lane_mask is not None:
        lane_sub = lane_mask[y_min:y_max, x_min:x_max]
        lane_pixels = (lane_sub > 0) & (~land_pixels)
        img_data[lane_pixels] = [144, 238, 144, 200]
    
    # Separation zones - gold
    if sepzone_mask is not None:
        sepzone_sub = sepzone_mask[y_min:y_max, x_min:x_max]
        sepzone_pixels = (sepzone_sub > 0) & (~land_pixels)
        img_data[sepzone_pixels] = [255, 215, 0, 200]
    
    # Separation boundaries - red
    if sepboundary_mask is not None:
        sepboundary_sub = sepboundary_mask[y_min:y_max, x_min:x_max]
        boundary_pixels = (sepboundary_sub > 0) & (~land_pixels)
        img_data[boundary_pixels] = [255, 107, 107, 200]

    # Bathymetry shading (discrete bands) and boundary lines
    if depth_sub is not None:
        # valid water cells: not nodata and negative depth values
        valid_water = (depth_sub != depth_nodata) & (depth_sub < 0) & (~land_pixels)
        if valid_water.any():
            # Convert to positive depth in meters
            depth_m = -depth_sub.astype(np.float32)


            # Define discrete bands (meters): exact 0, (0,5], (5,10], (10,20], >20
            band_ids = np.full(depth_m.shape, -1, dtype=np.int8)
            band_ids[valid_water & (depth_m == 0.0)] = 0
            band_ids[valid_water & (depth_m > 0.0) & (depth_m <= 5.0)] = 1
            band_ids[valid_water & (depth_m > 5.0) & (depth_m <= 10.0)] = 2
            band_ids[valid_water & (depth_m > 10.0) & (depth_m <= 20.0)] = 3
            band_ids[valid_water & (depth_m > 20.0)] = 4

            # Colors for bands
            colors = {
                0: (255, 255, 255, 220),   # 0 m -> white
                1: (173, 216, 230, 220),   # 0-5 m -> light blue
                2: (34, 139, 34, 220),     # 5-10 m -> green (ForestGreen)
                3: (0, 30, 100, 220),      # 10-20 m -> dark blue
                4: (220, 20, 60, 220),     # >20 m -> red (Crimson)
            }

            for bid, col in colors.items():
                mask = band_ids == bid
                if mask.any():
                    img_data[mask, 0] = col[0]
                    img_data[mask, 1] = col[1]
                    img_data[mask, 2] = col[2]
                    img_data[mask, 3] = col[3]

            # (Edge drawing removed to avoid heavy black outlines)
    
    # Create PIL image - DO NOT resize to square, keep original aspect ratio
    # The geographic bounds will handle proper placement on the map
    img = Image.fromarray(img_data, mode='RGBA')
    # No resize - keep 1:1 pixel to grid cell correspondence
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode(), actual_bounds


def fetch_route(start: tuple, end: tuple, api_url: str = "http://127.0.0.1:8000") -> dict:
    """Fetch a route from the API. Start/end are (lat, lon), API expects [lon, lat]."""
    response = requests.post(
        f"{api_url}/route",
        json={
            "start": [start[1], start[0]],  # Convert (lat, lon) to [lon, lat]
            "end": [end[1], end[0]],
            "min_draft": 10
        }
    )
    response.raise_for_status()
    return response.json()


def create_html_map(
    route_coords: list,
    grid: GridSpec,
    land_mask: np.ndarray,
    lane_mask: np.ndarray | None,
    sepzone_mask: np.ndarray | None,
    sepboundary_mask: np.ndarray | None,
    output_path: Path,
    depth: np.ndarray | None = None,
):
    """Create an interactive HTML map with Leaflet."""
    
    lons = [p[0] for p in route_coords]
    lats = [p[1] for p in route_coords]
    
    # Calculate center and bounds
    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2
    
    margin = 1.0
    bounds = [
        [min(lats) - margin, min(lons) - margin],
        [max(lats) + margin, max(lons) + margin]
    ]
    
    # Create overlay image
    bbox = (
        min(lons) - margin,
        max(lons) + margin,
        min(lats) - margin,
        max(lats) + margin,
    )
    
    overlay_base64, actual_bbox = create_tile_image(
        grid, land_mask, lane_mask, sepzone_mask, sepboundary_mask,
        bbox,
        depth=depth,
    )
    # Use actual_bbox for overlay bounds to ensure pixel-perfect alignment
    bbox = actual_bbox
    
    # Generate waypoint markers
    waypoint_markers = []
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        color = 'green' if i == 0 else ('red' if i == len(lons) - 1 else 'blue')
        label = 'Start' if i == 0 else ('End' if i == len(lons) - 1 else f'WP{i}')
        waypoint_markers.append(f'''
            L.circleMarker([{lat}, {lon}], {{
                radius: {12 if i in [0, len(lons)-1] else 8},
                fillColor: '{color}',
                color: 'white',
                weight: 2,
                fillOpacity: 0.9
            }}).addTo(map).bindPopup('<b>{label}</b><br>Lat: {lat:.4f}<br>Lon: {lon:.4f}');
        ''')
    
    # Route line coordinates
    route_line = [[lat, lon] for lon, lat in zip(lons, lats)]
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Ocean Route Visualization</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ width: 100vw; height: 100vh; }}
        .info-box {{
            padding: 10px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        }}
        .legend {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            line-height: 1.8;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 20px;
            height: 12px;
            border: 1px solid #999;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 8);
        
        // Base tiles - OpenStreetMap
        var osmLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '© OpenStreetMap'
        }});
        
        // ESRI Ocean basemap
        var esriOcean = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            maxZoom: 13,
            attribution: 'Tiles © Esri'
        }}).addTo(map);
        
        // TSS Overlay
        var overlayBounds = [[{bbox[2]}, {bbox[0]}], [{bbox[3]}, {bbox[1]}]];
        var tssOverlay = L.imageOverlay('data:image/png;base64,{overlay_base64}', overlayBounds, {{
            opacity: 1.0
        }}).addTo(map);
        
        // Route line
        var routeLine = L.polyline({route_line}, {{
            color: 'blue',
            weight: 4,
            opacity: 0.8
        }}).addTo(map);
        
        // White outline for route
        L.polyline({route_line}, {{
            color: 'white',
            weight: 6,
            opacity: 0.5
        }}).addTo(map).bringToBack();
        
        // Waypoint markers
        {"".join(waypoint_markers)}
        
        // Fit bounds
        map.fitBounds(routeLine.getBounds().pad(0.1));
        
        // Layer control
        var baseMaps = {{
            "ESRI Ocean": esriOcean,
            "OpenStreetMap": osmLayer
        }};
        var overlays = {{
            "TSS Overlay": tssOverlay,
            "Route": routeLine
        }};
        L.control.layers(baseMaps, overlays).addTo(map);
        
        // Info box
        var info = L.control({{position: 'topright'}});
        info.onAdd = function(map) {{
            var div = L.DomUtil.create('div', 'info-box');
            div.innerHTML = '<b>Route Info</b><br>' +
                'Waypoints: {len(route_coords)}<br>' +
                'Start: ({lats[0]:.4f}, {lons[0]:.4f})<br>' +
                'End: ({lats[-1]:.4f}, {lons[-1]:.4f})';
            return div;
        }};
        info.addTo(map);
        
        // Legend
        var legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function(map) {{
            var div = L.DomUtil.create('div', 'legend');
            div.innerHTML = '<b>Legend</b><br>' +
                '<div class="legend-item"><div class="legend-color" style="background:#d4c4a8"></div>Land</div>' +
                '<div class="legend-item"><div class="legend-color" style="background:#ffffff;border:1px solid #999"></div>0 m (dry)</div>' +
                '<div class="legend-item"><div class="legend-color" style="background:#ADD8E6;border:1px solid #999"></div>0–5 m</div>' +
                '<div class="legend-item"><div class="legend-color" style="background:#228B22;border:1px solid #999"></div>5–10 m</div>' +
                '<div class="legend-item"><div class="legend-color" style="background:#001E64;border:1px solid #999"></div>10–20 m</div>' +
                '<div class="legend-item"><div class="legend-color" style="background:#DC143C;border:1px solid #999"></div>>20 m</div>' +
                '<div class="legend-item"><div class="legend-color" style="background:#90EE90"></div>TSS Lane</div>' +
                '<div class="legend-item"><div class="legend-color" style="background:#FFD700"></div>Separation Zone</div>' +
                '<div class="legend-item"><div class="legend-color" style="background:#FF6B6B"></div>Separation Boundary</div>' +
                '<div class="legend-item"><div class="legend-color" style="background:blue"></div>Route</div>';
            return div;
        }};
        legend.addTo(map);
    </script>
</body>
</html>
'''
    
    output_path.write_text(html)
    print(f"Saved interactive map to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create interactive HTML map of ocean routes")
    parser.add_argument("--start", type=str, help="Start coords as 'lat,lon'")
    parser.add_argument("--end", type=str, help="End coords as 'lat,lon'")
    parser.add_argument("--route-json", type=Path, help="JSON file with route response")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="API URL")
    parser.add_argument("--output", "-o", type=Path, default=Path("route_map.html"), help="Output HTML path")
    parser.add_argument("--grid", type=Path, default=Path("configs/grid_1nm.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--show-depth", action="store_true", help="Include bathymetry overlay (depth_1nm.npy)")
    args = parser.parse_args()
    
    # Load grid
    grid = GridSpec.from_file(args.grid)
    
    # Load masks (prefer buffered land mask if available to include small islands)
    land_base = args.data_dir / "land/land_mask_1nm.npy"
    land_buf = args.data_dir / "land/land_mask_1nm_buffered.npy"
    if land_buf.exists():
        land_mask = load_mask(land_buf)
    elif land_base.exists():
        land_mask = load_mask(land_base)
    else:
        raise FileNotFoundError("No land mask found in data directory")
    
    lane_mask = sepzone_mask = sepboundary_mask = None
    tss_dir = args.data_dir / "tss"
    if (tss_dir / "tss_lane_mask_1nm.npy").exists():
        lane_mask = load_mask(tss_dir / "tss_lane_mask_1nm.npy")
    if (tss_dir / "tss_sepzone_mask_1nm.npy").exists():
        sepzone_mask = load_mask(tss_dir / "tss_sepzone_mask_1nm.npy")
    if (tss_dir / "tss_sepboundary_mask_1nm.npy").exists():
        sepboundary_mask = load_mask(tss_dir / "tss_sepboundary_mask_1nm.npy")
    depth = None
    if args.show_depth:
        depth_path = args.data_dir / "bathy" / "depth_1nm.npy"
        if depth_path.exists():
            depth = load_mask(depth_path)
        else:
            print(f"[WARNING] Depth file not found: {depth_path}; continuing without depth overlay")
    
    # Get route
    if args.route_json:
        with open(args.route_json) as f:
            route_data = json.load(f)
    elif args.start and args.end:
        start = tuple(map(float, args.start.split(',')))
        end = tuple(map(float, args.end.split(',')))
        print(f"Fetching route from {start} to {end}...")
        route_data = fetch_route(start, end, args.api_url)
    else:
        print("Using default Gibraltar route...")
        route_data = fetch_route((36.05, -5.35), (35.95, -5.85), args.api_url)
    
    print(f"Route has {len(route_data['path'])} waypoints, {route_data['distance_nm']:.1f} nm")
    if route_data.get('warnings'):
        print(f"Warnings: {route_data['warnings']}")
    
    create_html_map(
        route_coords=route_data['path'],
        grid=grid,
        land_mask=land_mask,
        lane_mask=lane_mask,
        sepzone_mask=sepzone_mask,
        sepboundary_mask=sepboundary_mask,
        output_path=args.output,
        depth=depth,
    )


if __name__ == "__main__":
    main()
