import time
from ocean_router.api.endpoints import compute_route
from ocean_router.api.dependencies import get_grid_spec, get_bathy, get_land_mask, get_tss, get_cost_weights, get_canals

# Get dependencies
grid = get_grid_spec()
bathy = get_bathy()
land = get_land_mask()
tss = get_tss()
weights = get_cost_weights()
canals = get_canals()

# Test the user's problematic route
start = (-14.853515625, 36.914764288955936)  # (lon, lat)
end = (-54.0966796875, 37.68382032669382)    # (lon, lat)

print('Testing the user route that might cross land...')
print(f'From: {start} (lon, lat)')
print(f'To: {end} (lon, lat)')

# First check what the current fast path logic would do
from ocean_router.core.geodesy import haversine_nm
dist = haversine_nm(start[0], start[1], end[0], end[1])
print(f'Distance: {dist:.1f} nm')

if dist > 1500.0:
    print('Current logic: SKIP land checking (unsafe!)')
elif dist > 1000.0:
    print('Current logic: DO land checking')
else:
    print('Current logic: Use full A*')

# Now run the route
t0 = time.time()

# Check if start/end points are blocked
from ocean_router.routing.costs import CostContext
context = CostContext(bathy, tss, None, grid.dx, grid.dy, land=land)
start_x, start_y = grid.lonlat_to_xy(*start)
end_x, end_y = grid.lonlat_to_xy(*end)

print(f'Start point ({start[0]:.2f}, {start[1]:.2f}) -> grid ({start_y}, {start_x})')
print(f'End point ({end[0]:.2f}, {end[1]:.2f}) -> grid ({end_y}, {end_x})')

# Check round-trip conversion
start_lon2, start_lat2 = grid.xy_to_lonlat(start_x, start_y)
end_lon2, end_lat2 = grid.xy_to_lonlat(end_x, end_y)
print(f'Round-trip start: ({start_lat2:.2f}, {start_lon2:.2f})')
print(f'Round-trip end: ({end_lat2:.2f}, {end_lon2:.2f})')

# Check bathy depth at start and end
start_depth = bathy.depth[start_y, start_x]
end_depth = bathy.depth[end_y, end_x]
print(f'Start depth: {start_depth:.1f}m, End depth: {end_depth:.1f}m')

# Check land mask at start and end
start_land = land.base[start_y, start_x] if land else None
end_land = land.base[end_y, end_x] if land else None
print(f'Start land mask: {start_land}, End land mask: {end_land}')

# Check land distance at start and end
start_dist = land.distance_from_land[start_y, start_x] if land else None
end_dist = land.distance_from_land[end_y, end_x] if land else None
print(f'Start land distance: {start_dist:.2f}nm, End land distance: {end_dist:.2f}nm')

# Check if blocked (using default min_draft=10m)
min_draft = 10.0
start_blocked = context.blocked(start_y, start_x, min_draft)
end_blocked = context.blocked(end_y, end_x, min_draft)
print(f'Start blocked: {start_blocked}, End blocked: {end_blocked}')

path, distance, warnings = compute_route(start, end, grid, bathy, land, tss, canals, weights)
t1 = time.time()

print(f'Route computed in {t1-t0:.3f} seconds')
print(f'Path length: {len(path)} points')
print(f'Distance: {distance:.1f} nm')
if warnings:
    print(f'Warnings: {warnings}')
else:
    print('No warnings')
