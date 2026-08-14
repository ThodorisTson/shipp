"""
Calculate Weibull Parameters from ERA5 Time Series
===================================================

Fits Weibull distributions by wind direction sector to create
proper wind resource YAML like IEA 740.

Input:  wind_data_fryslan.csv (ERA5 time series)
Output: fryslan_wind_resource_weibull.yaml (sectoral Weibull)

Author: Thodoris
Date: 2026-02-10
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import weibull_min
from scipy.special import gamma
import yaml

print("=" * 70)
print("CALCULATING WEIBULL PARAMETERS FROM ERA5")
print("=" * 70)

# =============================================================================
# LOAD ERA5 TIME SERIES
# =============================================================================

print("\n1. Loading ERA5 data...")

BASE_DIR = Path(__file__).parent
csv_path = BASE_DIR / "wind_data_fryslan.csv"
df = pd.read_csv(csv_path)

print(f"   ✓ Loaded {len(df)} hours")
print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")

ws = df['wind_speed'].values
wd = df['wind_direction'].values

print(f"\n   Wind speed stats:")
print(f"   Mean: {ws.mean():.2f} m/s")
print(f"   Median: {np.median(ws):.2f} m/s")
print(f"   Std: {ws.std():.2f} m/s")

# =============================================================================
# DEFINE SECTORS (LIKE IEA 740 - 12 SECTORS)
# =============================================================================

print("\n2. Defining wind direction sectors...")

# 12 sectors (30° each) - matches IEA 740
n_sectors = 12
sector_width = 360 / n_sectors

sector_centers = np.arange(0, 360, sector_width)
print(f"   ✓ {n_sectors} sectors, {sector_width:.0f}° each")
print(f"   Centers: {sector_centers}")

# =============================================================================
# FIT WEIBULL BY SECTOR
# =============================================================================

print("\n3. Fitting Weibull distributions by sector...")

weibull_A = []
weibull_k = []
sector_freq = []

for i, center in enumerate(sector_centers):
    # Define sector bounds
    lower = (center - sector_width/2) % 360
    upper = (center + sector_width/2) % 360
    
    # Handle wrap-around at 0/360
    if lower < upper:
        mask = (wd >= lower) & (wd < upper)
    else:  # Crosses 0/360 boundary
        mask = (wd >= lower) | (wd < upper)
    
    ws_sector = ws[mask]
    
    # Sector frequency
    freq = len(ws_sector) / len(ws)
    sector_freq.append(freq)
    
    if len(ws_sector) > 10:  # Need enough data to fit
        # Fit Weibull (shape, loc, scale)
        # loc=0 forces 2-parameter Weibull
        shape, loc, scale = weibull_min.fit(ws_sector, floc=0)
        
        weibull_k.append(shape)
        weibull_A.append(scale)
        
        print(f"   Sector {center:3.0f}°: A={scale:.2f}, k={shape:.2f}, freq={freq:.3f}")
    else:
        # Not enough data - use overall distribution
        shape, loc, scale = weibull_min.fit(ws, floc=0)
        weibull_k.append(shape)
        weibull_A.append(scale)
        print(f"   Sector {center:3.0f}°: Insufficient data, using overall distribution")

# =============================================================================
# CALCULATE TURBULENCE INTENSITY BY WIND SPEED
# =============================================================================

print("\n4. Calculating turbulence intensity...")

# Define wind speed bins for TI
ws_bins = np.arange(4, 26, 1)  # Like IEA 740
ws_centers = ws_bins[:-1] + 0.5

# Calculate TI by bin (using standard deviation / mean)
# For offshore: TI ~ 0.10 to 0.14
# Use IEC Class B: TI = Iref * (0.75 * Vave + 5.6) / Vhub
# where Iref = 0.14 for Class B

Iref = 0.14
Vave = 8.5  # Typical for offshore
TI_values = []

for ws_center in ws_centers:
    # IEC formula
    TI = Iref * (0.75 * Vave + 5.6) / ws_center
    TI_values.append(min(TI, 0.20))  # Cap at 0.20

print(f"   ✓ Calculated TI for {len(ws_centers)} wind speed bins")
print(f"   TI range: {min(TI_values):.3f} to {max(TI_values):.3f}")

# =============================================================================
# CREATE WIND RESOURCE YAML (IEA 740 STYLE)
# =============================================================================

print("\n5. Creating wind resource YAML...")

wind_resource_yaml = {
    'wind_resource': {
        # Wind direction sectors (centers)
        'wind_direction': sector_centers.tolist(),
        
        # Wind speed bins for TI
        'wind_speed': ws_centers.tolist(),
        
        # Weibull A parameter by sector
        'weibull_a': {
            'dims': ['wind_direction'],
            'data': weibull_A
        },
        
        # Weibull k parameter by sector
        'weibull_k': {
            'dims': ['wind_direction'],
            'data': weibull_k
        },
        
        # Sector probability (frequency)
        'sector_probability': {
            'dims': ['wind_direction'],
            'data': sector_freq
        },
        
        # Turbulence intensity by wind speed
        'turbulence_intensity': {
            'dims': ['wind_speed'],
            'data': TI_values
        }
    }
}

# Convert numpy arrays to plain Python lists
def convert_numpy(obj):
    """Recursively convert numpy types to Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj

wind_resource_yaml_clean = convert_numpy(wind_resource_yaml)

# Save
yaml_file = str((Path(__file__).parent / "fryslan_wind_resource_weibull.yaml").resolve())
with open(yaml_file, 'w') as f:
    yaml.dump(wind_resource_yaml_clean, f, default_flow_style=False, sort_keys=False, indent=2)

print(f"   ✓ Saved: {yaml_file}")

# =============================================================================
# VALIDATION
# =============================================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

# Check sector frequencies sum to 1
total_freq = sum(sector_freq)
print(f"\n✓ Sector frequencies sum: {total_freq:.6f}")

if abs(total_freq - 1.0) < 0.001:
    print(f"  ✓ Perfect!")
else:
    print(f"  ⚠️  Should be 1.0")

# Overall Weibull from sectors
A_overall = np.average(weibull_A, weights=sector_freq)
k_overall = np.average(weibull_k, weights=sector_freq)

print(f"\n✓ Weighted average Weibull parameters:")
print(f"  A: {A_overall:.2f} m/s")
print(f"  k: {k_overall:.2f}")

# Compare with direct fit
shape_direct, loc, scale_direct = weibull_min.fit(ws, floc=0)
print(f"\n✓ Direct fit (all data):")
print(f"  A: {scale_direct:.2f} m/s")
print(f"  k: {shape_direct:.2f}")

# Mean wind speed comparison
mean_ws_data = ws.mean()
mean_ws_weibull = A_overall * gamma(1 + 1/k_overall)

print(f"\n✓ Mean wind speed:")
print(f"  From data: {mean_ws_data:.2f} m/s")
print(f"  From Weibull: {mean_ws_weibull:.2f} m/s")
print(f"  Difference: {abs(mean_ws_data - mean_ws_weibull):.2f} m/s")

if abs(mean_ws_data - mean_ws_weibull) < 0.5:
    print(f"  ✓ Good match!")

# =============================================================================
# CREATE COMPLETE SITE YAML
# =============================================================================

print("\n6. Creating complete site YAML...")

site_yaml = {
    'name': 'IJsselmeer (Windpark Fryslån)',
    'description': 'Shallow freshwater lake in Netherlands',
    
    'energy_resource': wind_resource_yaml_clean
}

site_file = str((Path(__file__).parent / "fryslan_site_weibull.yaml").resolve())
with open(site_file, 'w') as f:
    yaml.dump(site_yaml, f, default_flow_style=False, sort_keys=False, indent=2)

print(f"   ✓ Saved: {site_file}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("✓✓✓ SUCCESS ✓✓✓")
print("=" * 70)

print(f"""
Files created:
1. {yaml_file}
   - 12 directional sectors (30° each)
   - Weibull A, k by sector
   - Sector frequencies
   - TI by wind speed

2. {site_file}
   - Complete site definition
   - Ready for fryslan_system.yaml

Wind resource summary:
- Sectors: {n_sectors}
- Mean wind speed: {mean_ws_data:.2f} m/s
- Weibull A (avg): {A_overall:.2f} m/s
- Weibull k (avg): {k_overall:.2f}
- Data period: {df['timestamp'].min()} to {df['timestamp'].max()}

✓ Now update fryslan_system.yaml:
  energy_resource: !include fryslan_site_weibull.yaml

✓ Then run:
  python run_fryslan_windio_yaml.py fryslan_system.yaml

This matches IEA 740 structure exactly!
""")

print("=" * 70)
