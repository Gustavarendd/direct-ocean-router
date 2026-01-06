#!/usr/bin/env python3
"""Visualize routes on a map with TSS and land layers."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
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
        # Fallback to regular load
        return np.load(path, mmap_mode='r')


def visualize_route(
    route_coords: list,
    grid: GridSpec,
    land_mask: np.ndarray,
    lane_mask: np.ndarray | None = None,
    sepzone_mask: np.ndarray | None = None,
    sepboundary_mask: np.ndarray | None = None,
    title: str = "Ocean Route",
    output_path: Path | None = None,
    bbox: tuple | None = None,  # (lon_min, lon_max, lat_min, lat_max)
):
    """Visualize a route on a map with TSS and land layers."""
    
    # Extract route coordinates
    lons = [p[0] for p in route_coords]
    lats = [p[1] for p in route_coords]
    
    # Determine bounding box
    if bbox is None:
        margin = 2.0  # degrees
        lon_min = min(lons) - margin
        lon_max = max(lons) + margin
        lat_min = min(lats) - margin
        lat_max = max(lats) + margin
    else:
        lon_min, lon_max, lat_min, lat_max = bbox
    
    # Convert bbox to grid indices
    x_min = max(0, int((lon_min - grid.xmin) / grid.dx))
    x_max = min(grid.width, int((lon_max - grid.xmin) / grid.dx))
    y_min = max(0, int((grid.ymax - lat_max) / grid.dy))
    y_max = min(grid.height, int((grid.ymax - lat_min) / grid.dy))
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Create extent for imshow
    extent = [lon_min, lon_max, lat_min, lat_max]
    
    # Extract subsets of masks for the bbox
    land_sub = land_mask[y_min:y_max, x_min:x_max]
    
    # Create composite image
    # 0 = ocean (blue), 1 = land (tan), 2 = TSS lane (light green), 
    # 3 = sep zone (yellow), 4 = sep boundary (red)
    composite = np.zeros_like(land_sub, dtype=np.uint8)
    
    # Ocean is default (0)
    # Land
    composite[land_sub > 0] = 1
    
    # TSS layers (only where not land)
    if lane_mask is not None:
        lane_sub = lane_mask[y_min:y_max, x_min:x_max]
        composite[(lane_sub > 0) & (land_sub == 0)] = 2
    
    if sepzone_mask is not None:
        sepzone_sub = sepzone_mask[y_min:y_max, x_min:x_max]
        composite[(sepzone_sub > 0) & (land_sub == 0)] = 3
    
    if sepboundary_mask is not None:
        sepboundary_sub = sepboundary_mask[y_min:y_max, x_min:x_max]
        composite[(sepboundary_sub > 0) & (land_sub == 0)] = 4
    
    # Define colormap
    colors = [
        '#1a3a5c',  # 0: Ocean (dark blue)
        '#d4c4a8',  # 1: Land (tan)
        '#90EE90',  # 2: TSS Lane (light green)
        '#FFD700',  # 3: Separation Zone (gold)
        '#FF6B6B',  # 4: Separation Boundary (red)
    ]
    cmap = ListedColormap(colors)
    
    # Plot composite
    ax.imshow(composite, extent=extent, origin='upper', cmap=cmap, 
              aspect='auto', interpolation='nearest')
    
    # Plot route
    ax.plot(lons, lats, 'w-', linewidth=3, label='Route', zorder=10)
    ax.plot(lons, lats, 'b-', linewidth=1.5, zorder=11)
    
    # Plot waypoints
    ax.scatter(lons, lats, c='white', s=80, edgecolors='blue', linewidths=2, zorder=12)
    
    # Add waypoint labels
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        ax.annotate(f'WP{i}', (lon, lat), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8,
                   color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='blue', alpha=0.7))
    
    # Mark start and end
    ax.scatter([lons[0]], [lats[0]], c='lime', s=150, marker='o', 
              edgecolors='darkgreen', linewidths=2, zorder=13, label='Start')
    ax.scatter([lons[-1]], [lats[-1]], c='red', s=150, marker='s', 
              edgecolors='darkred', linewidths=2, zorder=13, label='End')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=colors[0], label='Ocean'),
        mpatches.Patch(facecolor=colors[1], label='Land'),
        mpatches.Patch(facecolor=colors[2], label='TSS Lane'),
        mpatches.Patch(facecolor=colors[3], label='Separation Zone'),
        mpatches.Patch(facecolor=colors[4], label='Sep. Boundary'),
        plt.Line2D([0], [0], color='blue', linewidth=2, label='Route'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'{title}\n{len(route_coords)} waypoints')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


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


def main():
    parser = argparse.ArgumentParser(description="Visualize ocean routes with TSS and land")
    parser.add_argument("--start", type=str, help="Start coords as 'lat,lon'")
    parser.add_argument("--end", type=str, help="End coords as 'lat,lon'")
    parser.add_argument("--route-json", type=Path, help="JSON file with route response")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="API URL")
    parser.add_argument("--output", "-o", type=Path, help="Output image path")
    parser.add_argument("--bbox", type=str, help="Bounding box as 'lon_min,lon_max,lat_min,lat_max'")
    parser.add_argument("--grid", type=Path, default=Path("configs/grid_1nm.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    
    # Load grid
    grid = GridSpec.from_file(args.grid)
    
    # Load masks
    land_mask = load_mask(args.data_dir / "land/land_mask_1nm.npy")
    
    lane_mask = None
    sepzone_mask = None
    sepboundary_mask = None
    
    tss_dir = args.data_dir / "tss"
    if (tss_dir / "tss_lane_mask_1nm.npy").exists():
        lane_mask = load_mask(tss_dir / "tss_lane_mask_1nm.npy")
    if (tss_dir / "tss_sepzone_mask_1nm.npy").exists():
        sepzone_mask = load_mask(tss_dir / "tss_sepzone_mask_1nm.npy")
    if (tss_dir / "tss_sepboundary_mask_1nm.npy").exists():
        sepboundary_mask = load_mask(tss_dir / "tss_sepboundary_mask_1nm.npy")
    
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
        # Default: Gibraltar strait route
        print("Using default Gibraltar route...")
        route_data = fetch_route((36.1, -5.4), (36.0, -5.9), args.api_url)
    
    print(f"Route has {len(route_data['path'])} waypoints, {route_data['distance_nm']:.1f} nm")
    if route_data.get('warnings'):
        print(f"Warnings: {route_data['warnings']}")
    
    # Parse bbox if provided
    bbox = None
    if args.bbox:
        bbox = tuple(map(float, args.bbox.split(',')))
    
    # Create title
    path = route_data['path']
    title = f"Route: ({path[0][1]:.2f}°N, {path[0][0]:.2f}°W) → ({path[-1][1]:.2f}°N, {path[-1][0]:.2f}°W)"
    
    # Visualize
    visualize_route(
        route_coords=route_data['path'],
        grid=grid,
        land_mask=land_mask,
        lane_mask=lane_mask,
        sepzone_mask=sepzone_mask,
        sepboundary_mask=sepboundary_mask,
        title=title,
        output_path=args.output,
        bbox=bbox
    )


if __name__ == "__main__":
    main()
