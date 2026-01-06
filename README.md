# Direct Ocean Router

Direct Ocean Router is a corridor-limited voyage planning pipeline that guarantees no land crossings and prefers safe, regulated water. The project is organized to separate offline preprocessing from fast runtime routing.

## Key guarantees

- **No land crossings:** Land is rasterized to a global mask and buffered to prevent coastal threading. Every A\* expansion checks the mask.
- **Depth safety:** Depth lookups enforce `depth >= draft + safety_margin` with optional near-threshold penalties.
- **Directional lanes:** Traffic Separation Scheme (TSS) lanes are encouraged via direction-aware penalties instead of hard locks.
- **Macro guidance:** A curated passage graph prevents global searches and keeps routing inside known corridors.

## Repository layout

- `configs/` – grid specification, vessel profiles, routing defaults, passage graph, and basin polygons.
- `data/` – raw source data and processed memory-mappable grids (land, bathy, TSS, density, basin index, cached corridors).
- `scripts/` – preprocessing utilities to rasterize and normalize source data into aligned grids.
- `src/ocean_router/` – Python package for grid math, data loaders, routing logic, API, and CLI.

## Quick Start

```bash
# Install in development mode
pip install -e .

# Check current configuration
python -m ocean_router.cli.main info

# Start API server
uvicorn ocean_router.api.main:app --reload
```

## Preprocessing Scripts

Build the routing grids from raw data. All scripts support `--resolution` flag for multi-resolution builds.

### Build All Data (Recommended)

```bash
# Build both 1nm and 0.5nm resolution data
./scripts/build_all_resolutions.sh

# Build only 1nm (default)
./scripts/build_all_resolutions.sh 1nm

# Build only 0.5nm (higher resolution, ~4x larger files)
./scripts/build_all_resolutions.sh 0.5nm
```

### Individual Scripts

```bash
# Land mask (requires land_polygons.shp from OpenStreetMap)
python scripts/build_land_mask.py \
    --grid configs/grid_1nm.json \
    --resolution 1nm \
    --land data/raw/land_polygons.shp

# Bathymetry (requires GEBCO GeoTIFF tiles)
python scripts/build_bathy_grid.py \
    --grid configs/grid_1nm.json \
    --resolution 1nm \
    --gebco-dir data/raw/gebco_2025_sub_ice_topo_geotiff

# TSS fields (requires separation_lanes_with_direction.geojson)
python scripts/build_tss_fields.py \
    --grid configs/grid_1nm.json \
    --resolution 1nm \
    --tss-geojson data/raw/separation_lanes_with_direction.geojson

# For 0.5nm resolution, use grid_0.5nm.json (note the dot!)
python scripts/build_tss_fields.py \
    --grid configs/grid_0.5nm.json \
    --resolution 0.5nm \
    --tss-geojson data/raw/separation_lanes_with_direction.geojson
```

## Resolution Switching

The router supports multiple grid resolutions. Switch via environment variable:

```bash
# Use 1nm resolution (default, 21600×10800 cells)
export OCEAN_ROUTER_RESOLUTION=1nm

# Use 0.5nm resolution (43200×21600 cells, ~4x more detail)
export OCEAN_ROUTER_RESOLUTION=0.5nm

# Start server with specific resolution
OCEAN_ROUTER_RESOLUTION=0.5nm uvicorn ocean_router.api.main:app --reload
```

**Note:** You must build the data for a resolution before using it. If requested resolution data doesn't exist, the system falls back to 1nm.

## Data Sizes

| Resolution | Grid Size | Land Mask | Bathymetry | TSS Fields | Total |
|------------|-----------|-----------|------------|------------|-------|
| 1nm | 21600×10800 | ~220 MB | ~445 MB | ~890 MB | ~2.7 GB |
| 0.5nm | 43200×21600 | ~890 MB | ~1.8 GB | ~3.6 GB | ~10.8 GB |

## Pipeline overview

1. **Preprocess** (offline):
   - Rasterize land polygons to a 1 nm grid, buffer, and store as `.npy`.
   - Warp bathymetry to the same grid and store as `depth_1nm.npy` (int16 meters with nodata).
   - Rasterize TSS lanes and separation zones; build a direction field from centerlines.
   - Normalize ship-density rasters (optional) to `.npy`.
   - Build basin classifier and passage graph caches.
2. **Runtime** (fast):
   - Load memmaps for land, depth, TSS, density, and basins once.
   - Classify start/end basins and compute a macro path over `passages.yaml`.
   - Build a buffered corridor mask around macro waypoints.
   - Run corridor-limited A\* with depth checks, TSS penalties, density bias, and turn smoothing.
   - Validate that the resulting polyline never intersects land; simplify and return GeoJSON.

## CLI Commands

```bash
# Show configuration and available data
python -m ocean_router.cli.main info

# Show/set resolution
python -m ocean_router.cli.main resolution         # Show current
python -m ocean_router.cli.main resolution 0.5nm   # Set to 0.5nm

# Compute a route
python -m ocean_router.cli.main route --start "lat,lon" --end "lat,lon" --draft 9.5 --margin 2.0
```

## Data expectations

Place raw inputs under `data/raw/`:
- `land_polygons.shp` – OSM land polygons (from osmdata.openstreetmap.de)
- `gebco_2025_sub_ice_topo_geotiff/` – GEBCO bathymetry tiles
- `separation_lanes_with_direction.geojson` – TSS lanes with flow bearings

Processed outputs live in `data/processed/` under subfolders for land, bathy, tss, density, basins, and cached corridors.

## API Server

```bash
# Start with default 1nm resolution
pip install -e . && uvicorn ocean_router.api.main:app --reload

# Start with 0.5nm resolution
OCEAN_ROUTER_RESOLUTION=0.5nm uvicorn ocean_router.api.main:app --reload
```
