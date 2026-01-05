# Direct Ocean Router

Direct Ocean Router is a corridor-limited voyage planning pipeline that guarantees no land crossings and prefers safe, regulated water. The project is organized to separate offline preprocessing from fast runtime routing.

## Key guarantees
- **No land crossings:** Land is rasterized to a global mask and buffered to prevent coastal threading. Every A* expansion checks the mask.
- **Depth safety:** Depth lookups enforce `depth >= draft + safety_margin` with optional near-threshold penalties.
- **Directional lanes:** Traffic Separation Scheme (TSS) lanes are encouraged via direction-aware penalties instead of hard locks.
- **Macro guidance:** A curated passage graph prevents global searches and keeps routing inside known corridors.

## Repository layout
- `configs/` – grid specification, vessel profiles, routing defaults, passage graph, and basin polygons.
- `data/` – raw source data and processed memory-mappable grids (land, bathy, TSS, density, basin index, cached corridors).
- `scripts/` – preprocessing utilities to rasterize and normalize source data into aligned grids.
- `src/ocean_router/` – Python package for grid math, data loaders, routing logic, API, and CLI.

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
   - Run corridor-limited A* with depth checks, TSS penalties, density bias, and turn smoothing.
   - Validate that the resulting polyline never intersects land; simplify and return GeoJSON.

## Commands
- `ocean-router preprocess all` – run all preprocessing scripts.
- `ocean-router route --start "lat,lon" --end "lat,lon" --draft 9.5 --margin 2.0` – compute a route using the cached grids.

## Data expectations
Place raw inputs under `data/raw/` (land polygons, ETOPO bathymetry, TSS vectors, density rasters). Processed outputs live in `data/processed/` under subfolders for land, bathy, TSS, density, basins, and cached corridors.
