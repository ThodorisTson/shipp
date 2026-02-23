"""
Assemble Complete Fryslån System YAML (IEA 740 Style)
======================================================

Combines all components into a single WindIO-compliant system file:
1. Wind resource (Weibull from ERA5)
2. Turbine specification (SWT-DD-130)
3. Wind farm layout (OSM coordinates)

Output: fryslan_system_iea740.yaml

Author: Thodoris
Date: 2026-02-10
"""
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("ASSEMBLING FRYSLÅN SYSTEM YAML (IEA 740 STYLE)")
print("=" * 70)

BASE_DIR = Path(__file__).parent

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files
WIND_RESOURCE_YAML = BASE_DIR / "fryslan_wind_resource_weibull.yaml"
TURBINE_YAML = BASE_DIR / "siemens_swt_dd_130_turbine.yaml"
COORDINATES_CSV = BASE_DIR / "fryslan_coordinates_xy.csv"

# Output file
OUTPUT_YAML = BASE_DIR / "fryslan_system_iea740.yaml"

# =============================================================================
# STEP 1: LOAD WIND RESOURCE
# =============================================================================

print("\n1. Loading wind resource...")


# Check if file exists
wind_resource_path = Path(WIND_RESOURCE_YAML)
if not wind_resource_path.exists():
    print(f"   ⚠️  {WIND_RESOURCE_YAML} not found!")
    print("   Run: python calculate_weibull_from_era5.py")
    exit(1)

with open(wind_resource_path, 'r') as f:
    wind_resource = yaml.safe_load(f)

print(f"   ✓ Loaded: {WIND_RESOURCE_YAML}")
print(f"   Sectors: {len(wind_resource['wind_resource']['wind_direction'])}")
print(f"   Wind speeds: {len(wind_resource['wind_resource']['wind_speed'])}")

# =============================================================================
# STEP 2: LOAD TURBINE SPECIFICATION
# =============================================================================

print("\n2. Loading turbine specification...")

turbine_paths = [TURBINE_YAML]

turbine_data = None
turbine_path_used = None

for turbine_path in turbine_paths:
    if turbine_path.exists():
        with open(turbine_path, 'r') as f:
            turbine_data = yaml.safe_load(f)
        turbine_path_used = turbine_path
        break

if turbine_data is None:
    print(f"   ⚠️  {TURBINE_YAML} not found in any location!")
    print("   Searched:")
    for p in turbine_paths:
        print(f"     - {p}")
    exit(1)

print(f"   ✓ Loaded: {turbine_path_used}")
print(f"   Name: {turbine_data['name']}")
print(f"   Rated power: {turbine_data['performance']['rated_power']/1e6:.1f} MW")
print(f"   Hub height: {turbine_data['hub_height']} m")
print(f"   Rotor diameter: {turbine_data['rotor_diameter']} m")

# =============================================================================
# STEP 3: LOAD COORDINATES
# =============================================================================

print("\n3. Loading wind farm layout...")

coords_paths = [COORDINATES_CSV]

coords_df = None
coords_path_used = None

for coords_path in coords_paths:
    if coords_path.exists():
        coords_df = pd.read_csv(coords_path)
        coords_path_used = coords_path
        break

if coords_df is None:
    print(f"   ⚠️  {COORDINATES_CSV} not found in any location!")
    print("   Searched:")
    for p in coords_paths:
        print(f"     - {p}")
    exit(1)

print(f"   ✓ Loaded: {coords_path_used}")
print(f"   Turbines: {len(coords_df)}")
print(f"   X range: [{coords_df['x_m'].min():.0f}, {coords_df['x_m'].max():.0f}] m")
print(f"   Y range: [{coords_df['y_m'].min():.0f}, {coords_df['y_m'].max():.0f}] m")

# =============================================================================
# STEP 4: ASSEMBLE SYSTEM YAML
# =============================================================================

print("\n4. Assembling complete system YAML...")

# Calculate total capacity
n_turbines = len(coords_df)
turbine_rating_mw = turbine_data['performance']['rated_power'] / 1e6
total_capacity_mw = n_turbines * turbine_rating_mw

# Create complete system structure (IEA 740 style)
system = {
    'name': 'Windpark Fryslån System',
    'description': 'Complete wind farm system following IEA Wind Task 37 WindIO structure',
    
    'metadata': {
        'author': 'Thodoris',
        'supervisor': 'Jenna',
        'institution': 'TU Delft / DTU',
        'project': 'MSc Thesis - Hybrid Wind-Battery Systems',
        'created': '2026-02-10',
        'version': '1.0 (IEA 740 compliant)',
        'purpose': 'PyWake AEP calculation with validated Weibull wind resource',
        'notes': 'Wind resource from ERA5 2024, turbine curves verified, coordinates from OSM',
    },
    
    # Site with Weibull wind resource
    'site': {
        'name': 'IJsselmeer (Windpark Fryslån)',
        'description': 'Shallow freshwater lake in Netherlands, nearshore environment',
        'location': {
            'latitude': 52.9969,  # Farm center
            'longitude': 5.2620,
            'elevation': 0,  # Sea level
        },
        'energy_resource': wind_resource,
    },
    
    # Wind farm
    'wind_farm': {
        'name': 'Windpark Fryslån',
        'description': 'Operational offshore wind farm in IJsselmeer',
        
        'capacity': total_capacity_mw,
        'number_of_turbines': n_turbines,
        
        # Turbine specification
        'turbines': turbine_data,
        
        # Layout
        'layouts': {
            'initial_layout': {
                'description': 'As-built turbine positions from OpenStreetMap',
                'coordinate_system': 'Local Cartesian (meters, origin at farm center)',
                'reference_point': {
                    'latitude': 52.9969,
                    'longitude': 5.2620,
                },
                'coordinates': {
                    'x': coords_df['x_m'].tolist(),
                    'y': coords_df['y_m'].tolist(),
                    'units': 'm',
                }
            }
        }
    },
    
    # Expected results (to be calculated)
    'attributes': {
        'net_AEP': None,  # GWh
        'capacity_factor': None,  # %
        
        'analyses': {
            'wake_model': {
                'name': 'NOJ',
                'k': 0.05,
                'turbulence_model': None,
                'rotor_avg_model': 'RotorCenter',
                'notes': 'Following IEA Wind Task 37 Case Study 10 protocol',
            }
        }
    }
}

# =============================================================================
# STEP 5: SAVE YAML
# =============================================================================

print("\n5. Saving system YAML...")

# Use safe_dump with custom options for readability
class literal_str(str):
    pass

def literal_str_representer(dumper, data):
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(literal_str, literal_str_representer)

with open(OUTPUT_YAML, 'w') as f:
    yaml.dump(system, f, 
              default_flow_style=False, 
              sort_keys=False,
              allow_unicode=True,
              width=1000)  # Prevent line wrapping

print(f"   ✓ Saved: {OUTPUT_YAML}")

# =============================================================================
# VALIDATION
# =============================================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

# Reload to verify
with open(OUTPUT_YAML, 'r') as f:
    reloaded = yaml.safe_load(f)

# Check structure
checks = {
    'Site defined': 'site' in reloaded,
    'Wind resource present': 'energy_resource' in reloaded.get('site', {}),
    'Wind farm defined': 'wind_farm' in reloaded,
    'Turbines present': 'turbines' in reloaded.get('wind_farm', {}),
    'Layout present': 'layouts' in reloaded.get('wind_farm', {}),
    'Coordinates match': len(reloaded['wind_farm']['layouts']['initial_layout']['coordinates']['x']) == n_turbines,
}

print("\nStructure checks:")
for check, passed in checks.items():
    status = "✓" if passed else "✗"
    print(f"  {status} {check}")

all_passed = all(checks.values())

if all_passed:
    print("\n✅ All checks passed!")
else:
    print("\n⚠️  Some checks failed!")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
System: {reloaded['name']}
Wind Farm: {reloaded['wind_farm']['name']}

Wind Resource:
  Sectors: {len(wind_resource['wind_resource']['wind_direction'])}
  Wind speeds: {len(wind_resource['wind_resource']['wind_speed'])}
  Type: Weibull (sectoral)

Wind Farm:
  Turbines: {n_turbines}
  Model: {turbine_data['name']}
  Rating: {turbine_rating_mw:.1f} MW each
  Total capacity: {total_capacity_mw:.1f} MW
  
  Layout:
    X range: [{coords_df['x_m'].min():.0f}, {coords_df['x_m'].max():.0f}] m
    Y range: [{coords_df['y_m'].min():.0f}, {coords_df['y_m'].max():.0f}] m
    Extent: ~{(coords_df['x_m'].max() - coords_df['x_m'].min())/1000:.1f} × {(coords_df['y_m'].max() - coords_df['y_m'].min())/1000:.1f} km

Output File: {OUTPUT_YAML}
File size: {Path(OUTPUT_YAML).stat().st_size / 1024:.1f} KB

✅ Ready for PyWake IEA 740 analysis!

Next step:
  python run_fryslan_iea_style.py {OUTPUT_YAML}
""")

print("=" * 70)
