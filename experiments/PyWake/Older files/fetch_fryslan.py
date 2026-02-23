"""
Fetch Windpark Fryslån Turbine Coordinates from OpenStreetMap
==============================================================

Uses Overpass API to get the exact coordinates of all 89 turbines.

According to verified specs:
- 89 turbines in hexagonal layout
- 660 m spacing
- Farm center: 52.9969° N, 5.2620° E
- Siemens Gamesa SWT-DD-130 (4.3 MW)

Author: Thodoris
Date: 2026-02-09
"""

import requests
import pandas as pd
import numpy as np
import time

print("=" * 70)
print("FETCHING WINDPARK FRYSLÅN COORDINATES FROM OPENSTREETMAP")
print("=" * 70)

# =============================================================================
# OVERPASS API QUERY
# =============================================================================

overpass_url = "https://overpass-api.de/api/interpreter"

# Query for wind turbines in IJsselmeer region
# Bounding box covers Windpark Fryslån area
overpass_query = """
[out:csv(::lat, ::lon, name, "power", "generator:source", "generator:output:electricity")];
node
  ["power"="generator"]
  ["generator:source"="wind"]
  (52.9, 5.1, 53.1, 5.4);
out;
"""

print("\n1. Querying Overpass API...")
print(f"   URL: {overpass_url}")
print(f"   Bounding box: IJsselmeer (52.9-53.1°N, 5.1-5.4°E)")

# Make request
try:
    response = requests.post(
        overpass_url,
        data={'data': overpass_query},
        timeout=60
    )
    
    if response.status_code == 200:
        print(f"   ✓ Request successful!")
    else:
        print(f"   ✗ Request failed: HTTP {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        exit(1)
        
except Exception as e:
    print(f"   ✗ Request failed: {e}")
    print("\n   Alternative: Use Overpass Turbo web interface:")
    print("   https://overpass-turbo.eu/")
    exit(1)

# =============================================================================
# PARSE RESPONSE
# =============================================================================

print("\n2. Parsing response...")

# Split into lines
lines = response.text.strip().split('\n')

if len(lines) < 2:
    print("   ✗ No data returned")
    print(f"   Response: {response.text}")
    exit(1)

# First line is header
header = lines[0].split('\t')
print(f"   ✓ Columns: {header}")

# Parse data
data = []
for line in lines[1:]:
    parts = line.split('\t')
    if len(parts) >= 2:
        try:
            lat = float(parts[0].replace('@lat', ''))
            lon = float(parts[1].replace('@lon', ''))
            
            # Try to get name and power info if available
            name = parts[2] if len(parts) > 2 else ''
            power = parts[3] if len(parts) > 3 else ''
            
            data.append({
                'latitude': lat,
                'longitude': lon,
                'name': name,
                'power': power
            })
        except (ValueError, IndexError) as e:
            continue

print(f"   ✓ Found {len(data)} wind turbines in region")

# =============================================================================
# FILTER FOR WINDPARK FRYSLÅN
# =============================================================================

print("\n3. Filtering for Windpark Fryslån...")

# Farm center (from spec)
FARM_CENTER_LAT = 52.9969
FARM_CENTER_LON = 5.2620

# Calculate distance from farm center
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

# Filter turbines within reasonable radius of farm center
# Hexagon with 660m spacing, ~10 rows → radius ~5 km
MAX_RADIUS_KM = 5.0

fryslan_turbines = []
for turbine in data:
    dist_km = haversine_distance(
        FARM_CENTER_LAT, FARM_CENTER_LON,
        turbine['latitude'], turbine['longitude']
    )
    
    if dist_km <= MAX_RADIUS_KM:
        turbine['distance_from_center_km'] = dist_km
        fryslan_turbines.append(turbine)

print(f"   ✓ Filtered to {len(fryslan_turbines)} turbines within {MAX_RADIUS_KM} km of farm center")

# Expected: 89 turbines
if len(fryslan_turbines) != 89:
    print(f"   ⚠️  Expected 89 turbines, found {len(fryslan_turbines)}")
    print(f"   This may be OK if OSM data is incomplete or includes nearby farms")

# =============================================================================
# CONVERT TO LOCAL COORDINATES (METERS)
# =============================================================================

print("\n4. Converting to local Cartesian coordinates...")

# Convert lat/lon to approximate local meters
# 1 degree latitude ≈ 111.32 km
# 1 degree longitude at 53°N ≈ 67.6 km

LAT_TO_M = 111320  # meters per degree
LON_TO_M = 67600   # meters per degree at 53°N

# Use farm center as origin
for turbine in fryslan_turbines:
    turbine['x_m'] = (turbine['longitude'] - FARM_CENTER_LON) * LON_TO_M
    turbine['y_m'] = (turbine['latitude'] - FARM_CENTER_LAT) * LAT_TO_M

print(f"   ✓ Converted to local coordinates (origin at farm center)")

# =============================================================================
# CREATE DATAFRAME AND SAVE
# =============================================================================

print("\n5. Saving coordinates...")

df = pd.DataFrame(fryslan_turbines)

# Sort by distance from center (so turbine 0 is closest to center)
df = df.sort_values('distance_from_center_km').reset_index(drop=True)

# Add turbine IDs
df['turbine_id'] = range(len(df))

# Reorder columns
cols = ['turbine_id', 'latitude', 'longitude', 'x_m', 'y_m', 
        'distance_from_center_km', 'name', 'power']
df = df[cols]

# Save CSV
csv_file = 'fryslan_turbine_coordinates.csv'
df.to_csv(csv_file, index=False)
print(f"   ✓ Saved: {csv_file}")

# Save just x, y for PyWake
coords_file = 'fryslan_coordinates_xy.csv'
df[['x_m', 'y_m']].to_csv(coords_file, index=False)
print(f"   ✓ Saved: {coords_file} (for PyWake)")

# =============================================================================
# STATISTICS
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nTurbines found: {len(df)}")
print(f"Expected: 89")

if len(df) == 89:
    print("✓ PERFECT MATCH!")
elif abs(len(df) - 89) <= 5:
    print("⚠️  Close to expected (may include nearby turbines)")
else:
    print("⚠️  Significant difference - verify bounding box")

print(f"\nCoordinate range:")
print(f"  X: {df['x_m'].min():.0f} to {df['x_m'].max():.0f} m")
print(f"  Y: {df['y_m'].min():.0f} to {df['y_m'].max():.0f} m")
print(f"  Farm extent: {df['x_m'].max() - df['x_m'].min():.0f} × {df['y_m'].max() - df['y_m'].min():.0f} m")

# Check spacing
if len(df) > 1:
    # Calculate nearest neighbor distances
    from scipy.spatial import distance_matrix
    coords = df[['x_m', 'y_m']].values
    dist_matrix = distance_matrix(coords, coords)
    
    # Set diagonal to inf so we don't get 0 distance to self
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Find minimum distance for each turbine
    min_distances = dist_matrix.min(axis=1)
    
    print(f"\nTurbine spacing:")
    print(f"  Mean nearest neighbor: {min_distances.mean():.0f} m")
    print(f"  Expected: 660 m (hexagonal)")
    print(f"  Std: {min_distances.std():.0f} m")
    
    if 600 <= min_distances.mean() <= 720:
        print("  ✓ Matches expected hexagonal spacing!")

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)
print(f"\n1. {csv_file}")
print(f"   Complete data with lat/lon, local coords, IDs")
print(f"\n2. {coords_file}")
print(f"   Just x, y in meters for PyWake")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("""
1. Review coordinates in CSV files
2. Update fryslan_windfarm.yaml with these coordinates
3. Generate power curve for SWT-DD-130 (4.3 MW)
4. Run PyWake analysis with real data!
""")
print("=" * 70)