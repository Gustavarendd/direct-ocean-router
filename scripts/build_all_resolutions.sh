#!/bin/bash
# Build preprocessed data at both 1nm and 0.5nm resolutions
# Usage: ./scripts/build_all_resolutions.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Paths to source data
LAND_SHP="data/raw/land_polygons.shp"
GEBCO_DIR="data/raw/gebco_2025_sub_ice_topo_geotiff"
TSS_GEOJSON="data/raw/separation_lanes_with_direction.geojson"

echo "========================================"
echo "Building data at BOTH resolutions"
echo "========================================"

build_resolution() {
    local RES=$1
    local GRID_FILE="configs/grid_${RES}.json"
    
    echo ""
    echo "========================================"
    echo "Building ${RES} resolution data..."
    echo "========================================"
    
    if [ ! -f "$GRID_FILE" ]; then
        echo "ERROR: Grid config $GRID_FILE not found!"
        return 1
    fi
    
    # Build land mask
    if [ -f "$LAND_SHP" ]; then
        echo ""
        echo "--- Building land mask at ${RES} ---"
        python scripts/build_land_mask.py \
            --grid "$GRID_FILE" \
            --resolution "$RES" \
            --land "$LAND_SHP"
    else
        echo "SKIP: Land shapefile not found at $LAND_SHP"
    fi
    
    # Build bathymetry
    if [ -d "$GEBCO_DIR" ]; then
        echo ""
        echo "--- Building bathymetry at ${RES} ---"
        python scripts/build_bathy_grid.py \
            --grid "$GRID_FILE" \
            --resolution "$RES" \
            --gebco-dir "$GEBCO_DIR"
    else
        echo "SKIP: GEBCO directory not found at $GEBCO_DIR"
    fi
    
    # Build TSS fields
    if [ -f "$TSS_GEOJSON" ]; then
        echo ""
        echo "--- Building TSS fields at ${RES} ---"
        python scripts/build_tss_fields.py \
            --grid "$GRID_FILE" \
            --resolution "$RES" \
            --tss-geojson "$TSS_GEOJSON"
    else
        echo "SKIP: TSS GeoJSON not found at $TSS_GEOJSON"
    fi
    
    echo ""
    echo "Done building ${RES} data!"
}

# Parse arguments
if [ $# -eq 0 ]; then
    # No arguments - build 1nm + 0.5nm
    build_resolution "1nm"
    build_resolution "0.5nm"
elif [ "$1" == "1nm" ] || [ "$1" == "0.5nm" ] || [ "$1" == "0.25nm" ]; then
    # Build specific resolution
    build_resolution "$1"
else
    echo "Usage: $0 [1nm|0.5nm|0.25nm]"
    echo "  No argument: Build 1nm + 0.5nm"
    echo "  1nm:         Build only 1nm resolution"
    echo "  0.5nm:       Build only 0.5nm resolution"
    echo "  0.25nm:      Build only 0.25nm resolution"
    exit 1
fi

echo ""
echo "========================================"
echo "All done!"
echo ""
echo "To switch resolutions at runtime, use:"
echo "  export OCEAN_ROUTER_RESOLUTION=0.5nm"
echo "  # or"
echo "  export OCEAN_ROUTER_RESOLUTION=1nm"
echo "  # or"
echo "  export OCEAN_ROUTER_RESOLUTION=0.25nm"
echo "========================================"
