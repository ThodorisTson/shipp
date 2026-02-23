"""
Windpark Fryslån AEP Calculation (IEA 740 Style)
=================================================

Follows EXACT structure of IEA 740 PyWake example:
- Loads system YAML (layout + turbine + Weibull wind resource)
- Builds XRSite + WindTurbine (PowerCtTabular)
- Runs NOJ wake model with same parameters as IEA 740
- Prints AEP + capacity factor

Differences from IEA 740:
- Fryslån: 89 × 4.3 MW SWT-DD-130 turbines
- Weibull parameters estimated from ERA5 time series
- Coordinates from OpenStreetMap

Author: Thodoris  
Date: 2026-02-10
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
print("WINDPARK FRYSLÅN - AEP CALCULATION (IEA 740 STYLE)")
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
    file_path = Path(__file__).parent / 'fryslan_system_iea740.yaml'

print(f"\nLoading: {file_path}")

# =============================================================================
# LOAD DATA FROM YAML (IEA 740 STRUCTURE)
# =============================================================================

system_dat = load_yaml(file_path)
farm_dat = system_dat['wind_farm']
resource_dat = system_dat['site']['energy_resource']

# Extract site data (Weibull by sector)
wr = resource_dat['wind_resource']
A = np.asarray(wr['weibull_a']['data'], dtype=float)
k = np.asarray(wr['weibull_k']['data'], dtype=float)
freq = np.asarray(wr['sector_probability']['data'], dtype=float)
wd = np.asarray(wr['wind_direction'], dtype=float)
ws = np.asarray(wr['wind_speed'], dtype=float)
TI_data = np.asarray(wr['turbulence_intensity']['data'], dtype=float)

print(f"\n✓ Wind resource loaded:")
print(f"  Sectors: {len(wd)} directions")
print(f"  Weibull A: {A.mean():.2f} m/s (avg)")
print(f"  Weibull k: {k.mean():.2f} (avg)")
print(f"  TI points: {len(TI_data)}")

# Extract layout
x = np.asarray(farm_dat['layouts']['initial_layout']['coordinates']['x'], dtype=float)
y = np.asarray(farm_dat['layouts']['initial_layout']['coordinates']['y'], dtype=float)

print(f"\n✓ Layout loaded:")
print(f"  Turbines: {len(x)}")
print(f"  X range: [{x.min():.0f}, {x.max():.0f}] m")
print(f"  Y range: [{y.min():.0f}, {y.max():.0f}] m")

# Extract turbine data
t = farm_dat['turbines']
hh = float(t['hub_height'])
rd = float(t['rotor_diameter'])

p_ws = np.asarray(t['performance']['power_curve']['power_wind_speeds'], dtype=float)
p = np.asarray(t['performance']['power_curve']['power_values'], dtype=float)
ct_ws = np.asarray(t['performance']['Ct_curve']['Ct_wind_speeds'], dtype=float)
ct = np.asarray(t['performance']['Ct_curve']['Ct_values'], dtype=float)

cut_in = float(t['performance']['cutin_wind_speed'])
cut_out = float(t['performance']['cutout_wind_speed'])
rated = float(t['performance']['rated_wind_speed'])

print(f"\n✓ Turbine loaded: {t['name']}")
print(f"  Hub height: {hh} m")
print(f"  Rotor diameter: {rd} m")
print(f"  Rated power: {t['performance']['rated_power']/1e6:.1f} MW")
print(f"  Cut-in: {cut_in} m/s")
print(f"  Rated: {rated} m/s")
print(f"  Cut-out: {cut_out} m/s")
print(f"  Power curve: {len(p_ws)} points")
print(f"  Ct curve: {len(ct_ws)} points")

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
    name=t['name'],
    diameter=rd,
    hub_height=hh,
    powerCtFunction=PowerCtTabular(int_speeds, ps_int, power_unit='W', ct=cts_int)
)

# Site (XRSite with Weibull)
site = XRSite(
    ds=xr.Dataset(
        data_vars={
            'Sector_frequency': ('wd', freq),
            'Weibull_A': ('wd', A),
            'Weibull_k': ('wd', k),
            'TI': ('ws', TI_data),
        },
        coords={'wd': wd, 'ws': ws}
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
rated_power_w = float(t['performance']['rated_power'])

cap_factor = aep_gwh / (len(x) * rated_power_w * 8760 / 1e9)


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
print(f"  Rated power: {t['performance']['rated_power']/1e6:.1f} MW each")
print(f"  Total capacity: {len(x) * t['performance']['rated_power']/1e6:.1f} MW")

print(f"\nAnnual production:")
print(f"  Total: {aep_gwh:.2f} GWh/year")
print(f"  Per turbine: {aep_gwh/len(x):.2f} GWh/year")
print(f"  Per MW: {aep_gwh/(len(x) * t['performance']['rated_power']/1e6):.2f} GWh/MW/year")

# Sanity checks
expected_cf_min = 0.30  # 30% is reasonable for nearshore
expected_cf_max = 0.55  # 55% is high but possible

if expected_cf_min <= cap_factor <= expected_cf_max:
    print(f"\n✓ Capacity factor looks reasonable for IJsselmeer nearshore!")
else:
    print(f"\n⚠️  Capacity factor outside expected range ({expected_cf_min*100:.0f}-{expected_cf_max*100:.0f}%)")
    print(f"   Check wind resource and turbine curves")

print(f"\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
