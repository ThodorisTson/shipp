"""
WP2 Denmark Offshore AEP Calculation (IEA 740 Style)
=====================================================

Follows EXACT structure of IEA 740 PyWake example:
- Loads system YAML (layout + turbine + Weibull wind resource)
- Builds XRSite + WindTurbine (PowerCtTabular)
- Runs NOJ wake model with same parameters as IEA 740
- Prints AEP + capacity factor

Wind farm specifications:
- WP2: 65 × 5.0 MW Generic turbines
- Turbine: D=161.5m, hub=100m, sp=244 W/m²
- Location: Denmark North Sea (56.23°N, 8.59°E)
- Weibull parameters from ERA5 2022 with shear correction (86m→100m)
- Coordinates from HyDesign optimization

Author: Thodoris  
Date: 2026-02-11
Based on: IEA Wind Task 37 Case Study 10 (example_pywake.py)
"""

import numpy as np
import sys
import xarray as xr
from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake import NOJ
from py_wake.rotor_avg_models import RotorCenter
from pathlib import Path


# Try windIO, fall back to yaml (with !include support)
try:
    from windIO.utils.yml_utils import load_yaml  # Best case
except ImportError:
    import yaml
    from pathlib import Path

    def _yaml_loader_with_include(base_path: Path):
        class Loader(yaml.SafeLoader):
            pass

        def include(loader, node):
            rel_path = loader.construct_scalar(node)
            inc_path = (base_path.parent / rel_path).resolve()
            try:
                with open(inc_path, "r", encoding="utf-8-sig") as f:
                    return yaml.load(f, Loader=_yaml_loader_with_include(inc_path))
            except UnicodeDecodeError:
                with open(inc_path, "r", encoding="cp1252") as f:
                    return yaml.load(f, Loader=_yaml_loader_with_include(inc_path))

        Loader.add_constructor("!include", include)
        return Loader

    def load_yaml(path):
        path = Path(path).resolve()
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                return yaml.load(f, Loader=_yaml_loader_with_include(path))
        except UnicodeDecodeError:
            with open(path, "r", encoding="cp1252") as f:
                return yaml.load(f, Loader=_yaml_loader_with_include(path))


print("=" * 70)
print("WP2 DENMARK OFFSHORE - AEP CALCULATION (IEA 740 STYLE)")
print("=" * 70)

# =============================================================================
# INPUT
# =============================================================================

ws_sw = 1   # Wind speed step [m/s] (same as IEA 740)
wd_sw = 1   # Wind direction step [deg] (same as IEA 740)

# Load YAML file
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    # Default to wp2_site_weibull.yaml which contains both site and wind resource
    file_path = Path(__file__).parent / 'wp2_site_weibull.yaml'

print(f"\nLoading: {file_path}")

# =============================================================================
# LOAD DATA FROM YAML (WP2 STRUCTURE - SIMPLIFIED)
# =============================================================================

data = load_yaml(file_path)

# WP2 has a simpler structure than IEA 740
# Extract site data
site_dat = data['site']
wr = data['wind_resource']

# Extract Weibull parameters by sector
sectors = wr['sectors']
wd = np.array([s['direction_deg'] for s in sectors])
A = np.array([s['weibull_a'] for s in sectors])
k = np.array([s['weibull_k'] for s in sectors])
freq = np.array([s['probability'] for s in sectors])
TI_const = np.array([s['turbulence_intensity'] for s in sectors])

# Create wind speed bins for TI interpolation
ws = np.arange(4.5, 25.0, 1.0)  # Standard bins
TI_data = np.full(len(ws), TI_const.mean())  # Constant TI per wind speed

print(f"\n✓ Site loaded: {site_dat['name']}")
print(f"  Location: {site_dat['latitude']:.2f}°N, {site_dat['longitude']:.2f}°E")
print(f"  Altitude: {site_dat['altitude']} m")

print(f"\n✓ Wind resource loaded:")
print(f"  Sectors: {len(wd)} directions")
print(f"  Weibull A: {A.mean():.2f} m/s (avg)")
print(f"  Weibull k: {k.mean():.2f} (avg)")
print(f"  Turbulence intensity: {TI_const.mean()*100:.1f}%")
print(f"  Reference height: {wr['reference_height_m']} m")

# Load turbine data from separate file
turbine_file = Path(__file__).parent / 'wp2_turbine.yaml'
print(f"\nLoading turbine: {turbine_file}")
turbine_dat = load_yaml(turbine_file)

hh = float(turbine_dat['hub_height'])
rd = float(turbine_dat['rotor_diameter'])

# Extract power and Ct curves
perf = turbine_dat['performance']
p_ws = np.array(perf['power_curve']['power_wind_speeds'])
p = np.array(perf['power_curve']['power_values'])
ct_ws = np.array(perf['Ct_curve']['Ct_wind_speeds'])
ct = np.array(perf['Ct_curve']['Ct_values'])

rated_power = perf['rated_power']
cut_in = p_ws[np.where(p > 0)[0][0]]  # First non-zero power
cut_out = p_ws[-1]  # Last wind speed
rated_ws_idx = np.where(p >= rated_power * 0.99)[0][0]  # First time at 99% of rated
rated = p_ws[rated_ws_idx]

print(f"\n✓ Turbine loaded: {turbine_dat['name']}")
print(f"  Hub height: {hh} m")
print(f"  Rotor diameter: {rd:.2f} m")
print(f"  Rated power: {rated_power/1e6:.1f} MW")
print(f"  Cut-in: {cut_in:.1f} m/s")
print(f"  Rated: {rated:.1f} m/s")
print(f"  Cut-out: {cut_out:.1f} m/s")
print(f"  Power curve: {len(p_ws)} points")
print(f"  Ct curve: {len(ct_ws)} points")

# Load layout
layout_file = Path(__file__).parent / 'wp2_turbine_coordinates.csv'
print(f"\nLoading layout: {layout_file}")

import pandas as pd
layout = pd.read_csv(layout_file)
x = layout['X_m'].values
y = layout['Y_m'].values

print(f"\n✓ Layout loaded:")
print(f"  Turbines: {len(x)}")
print(f"  X range: [{x.min():.0f}, {x.max():.0f}] m")
print(f"  Y range: [{y.min():.0f}, {y.max():.0f}] m")

# =============================================================================
# INTERPOLATE POWER/CT CURVES (IEA 740 STYLE: 10,000 POINTS!)
# =============================================================================

print(f"\n✓ Interpolating curves to 10,000 points (IEA 740 style)...")

int_speeds = np.linspace(
    min(p_ws.min(), ct_ws.min()), 
    max(p_ws.max(), ct_ws.max()), 
    10000  # Same as IEA 740!
)

ps_int = np.interp(int_speeds, p_ws, p)
cts_int = np.interp(int_speeds, ct_ws, ct)

print(f"  ✓ Created smooth 10,000-point curves")

# =============================================================================
# CREATE PYWAKE OBJECTS (IEA 740 STYLE)
# =============================================================================

print(f"\n✓ Creating PyWake objects...")

# Wind turbine
windTurbines = WindTurbine(
    name=turbine_dat['name'],
    diameter=rd,
    hub_height=hh,
    powerCtFunction=PowerCtTabular(int_speeds, ps_int, power_unit='W', ct=cts_int)
)

# Site (XRSite with Weibull) - make wind direction cyclic for 360° wraparound
# Add duplicate points at 0° (= 360°) and 360° to enable full circle interpolation
wd_cyclic = np.concatenate([wd - 360, wd, wd + 360])
freq_cyclic = np.concatenate([freq, freq, freq])
A_cyclic = np.concatenate([A, A, A])
k_cyclic = np.concatenate([k, k, k])
TI_const_cyclic = np.concatenate([TI_const, TI_const, TI_const])

site = XRSite(
    ds=xr.Dataset(
        data_vars={
            'Sector_frequency': ('wd', freq_cyclic),
            'Weibull_A': ('wd', A_cyclic),
            'Weibull_k': ('wd', k_cyclic),
            'TI': ('ws', TI_data),
        },
        coords={'wd': wd_cyclic, 'ws': ws}
    )
)
site.interp_method = 'linear'

print(f"  ✓ Created XRSite with Weibull parameters")
print(f"  ✓ Created WindTurbine with PowerCtTabular")

# =============================================================================
# WIND ROSE DISCRETIZATION (IEA 740 STYLE)
# =============================================================================

print(f"\n✓ Creating wind rose discretization...")

ws_min = max(cut_in, ws.min())
ws_max = min(cut_out, ws.max())

ws_py = np.arange(ws_min, ws_max + ws_sw, ws_sw)
wd_py = np.arange(0, 360, wd_sw)
TI = np.interp(ws_py, ws, TI_data)

print(f"  Wind speeds: {len(ws_py)} bins ({ws_py.min():.0f}-{ws_py.max():.0f} m/s)")
print(f"  Wind directions: {len(wd_py)} bins (0-360°)")
print(f"  Total cases: {len(ws_py) * len(wd_py):,}")

# =============================================================================
# RUN PYWAKE (IEA 740 STYLE)
# =============================================================================

print(f"\n" + "=" * 70)
print("RUNNING PYWAKE NOJ MODEL")
print("=" * 70)

# NOJ model with EXACT IEA 740 parameters
noj = NOJ(
    site, 
    windTurbines, 
    turbulenceModel=None,      
    k=0.05,                    
    rotorAvgModel=RotorCenter()
)

print(f"\nNOJ parameters:")
print(f"  turbulenceModel: None")
print(f"  k: 0.05")
print(f"  rotorAvgModel: RotorCenter()")

print(f"\nRunning simulation...")

sim_res = noj(
    x, y, 
    time=False,  # Weibull-based, not time series
    ws=ws_py, 
    wd=wd_py, 
    TI=TI
)

# Calculate AEP (IEA 740 style: normalize_probabilities=False!)
aep = sim_res.aep(normalize_probabilities=False).sum()
aep_gwh = float(xr.DataArray.to_numpy(aep))

# Capacity factor
cap_factor = aep_gwh / (len(x) * rated_power * 8760 / 1e9)


# =============================================================================
# RESULTS
# =============================================================================

print(f"\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\n{'='*70}")
print(f"  AEP: {aep_gwh:.2f} GWh")
print(f"  Capacity factor: {cap_factor*100:.2f}%")
print(f"{'='*70}")

print(f"\nFarm details:")
print(f"  Turbines: {len(x)}")
print(f"  Rated power: {rated_power/1e6:.1f} MW each")
print(f"  Total capacity: {len(x) * rated_power/1e6:.1f} MW")

print(f"\nAnnual production:")
print(f"  Total: {aep_gwh:.2f} GWh/year")
print(f"  Per turbine: {aep_gwh/len(x):.2f} GWh/year")
print(f"  Per MW: {aep_gwh/(len(x) * rated_power/1e6):.2f} GWh/MW/year")

# Sanity checks
expected_cf_min = 0.35  # 35% is reasonable for North Sea offshore
expected_cf_max = 0.55  # 55% is excellent for offshore

if expected_cf_min <= cap_factor <= expected_cf_max:
    print(f"\n✓ Capacity factor looks reasonable for Denmark North Sea offshore!")
else:
    print(f"\n⚠️  Capacity factor outside expected range ({expected_cf_min*100:.0f}-{expected_cf_max*100:.0f}%)")
    print(f"   Check wind resource and turbine curves")

print(f"\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
