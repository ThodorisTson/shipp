"""
WP2 Wake Model Comparison
==========================

Compares two wake modeling approaches for WP2 offshore wind farm:

1. **NOJ (Jensen)** - IEA 740 baseline
   - Simple linear wake expansion
   - No turbulence modeling
   - Rotor center point only
   
2. **Bastankhah-Gaussian + Crespo-Hernandez** - Advanced
   - Gaussian wake shape
   - Wake-induced turbulence
   - 3×3 rotor grid averaging

Purpose: Quantify impact of advanced wake modeling on AEP prediction

Wind farm: WP2 Denmark (65 × 5MW, D=161.5m, hub=100m)
Author: Thodoris
Date: 2026-02-11
"""

import numpy as np
import xarray as xr
import pandas as pd
from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake import NOJ
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.turbulence_models.crespo import CrespoHernandez
from py_wake.rotor_avg_models import RotorCenter, EqGridRotorAvg
from pathlib import Path
import time


# Try windIO, fall back to yaml
try:
    from windIO.utils.yml_utils import load_yaml
except ImportError:
    import yaml
    
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


print("=" * 80)
print("WP2 WAKE MODEL COMPARISON")
print("=" * 80)

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n📂 Loading WP2 data...")

# Site + wind resource
site_file = Path(__file__).parent / 'wp2_site_weibull.yaml'
data = load_yaml(site_file)

site_dat = data['site']
wr = data['wind_resource']

sectors = wr['sectors']
wd = np.array([s['direction_deg'] for s in sectors])
A = np.array([s['weibull_a'] for s in sectors])
k = np.array([s['weibull_k'] for s in sectors])
freq = np.array([s['probability'] for s in sectors])
TI_const = np.array([s['turbulence_intensity'] for s in sectors])

ws = np.arange(4.5, 25.0, 1.0)
TI_data = np.full(len(ws), TI_const.mean())

# Turbine
turbine_file = Path(__file__).parent / 'wp2_turbine.yaml'
turbine_dat = load_yaml(turbine_file)

hh = float(turbine_dat['hub_height'])
rd = float(turbine_dat['rotor_diameter'])

perf = turbine_dat['performance']
p_ws = np.array(perf['power_curve']['power_wind_speeds'])
p = np.array(perf['power_curve']['power_values'])
ct_ws = np.array(perf['Ct_curve']['Ct_wind_speeds'])
ct = np.array(perf['Ct_curve']['Ct_values'])
rated_power = perf['rated_power']

cut_in = p_ws[np.where(p > 0)[0][0]]
cut_out = p_ws[-1]

# Layout
layout_file = Path(__file__).parent / 'wp2_turbine_coordinates.csv'
layout = pd.read_csv(layout_file)
x = layout['X_m'].values
y = layout['Y_m'].values

print(f"  ✓ Site: {site_dat['name']}")
print(f"  ✓ Turbines: {len(x)} × {rated_power/1e6:.1f} MW")
print(f"  ✓ Total capacity: {len(x) * rated_power/1e6:.1f} MW")
print(f"  ✓ Mean wind speed: {A.mean():.2f} m/s at {hh}m")

# =============================================================================
# CREATE PYWAKE OBJECTS
# =============================================================================

print("\n🔧 Creating PyWake objects...")

# Interpolate curves
int_speeds = np.linspace(min(p_ws.min(), ct_ws.min()), max(p_ws.max(), ct_ws.max()), 10000)
ps_int = np.interp(int_speeds, p_ws, p)
cts_int = np.interp(int_speeds, ct_ws, ct)

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

# Wind rose discretization
ws_sw = 1
wd_sw = 1
ws_py = np.arange(max(cut_in, ws.min()), min(cut_out, ws.max()) + ws_sw, ws_sw)
wd_py = np.arange(0, 360, wd_sw)
TI = np.interp(ws_py, ws, TI_data)

print(f"  ✓ Wind cases: {len(ws_py)} speeds × {len(wd_py)} directions = {len(ws_py)*len(wd_py):,} total")

# =============================================================================
# MODEL 1: NOJ (IEA 740 BASELINE)
# =============================================================================

print("\n" + "=" * 80)
print("MODEL 1: NOJ (Jensen) - IEA 740 Baseline")
print("=" * 80)

print("\nParameters:")
print("  - Linear wake expansion")
print("  - k = 0.05")
print("  - No turbulence model")
print("  - Rotor center point")

noj = NOJ(site, windTurbines, turbulenceModel=None, k=0.05, rotorAvgModel=RotorCenter())

print("\n⏱️  Running NOJ simulation...")
t0 = time.time()
sim_noj = noj(x, y, time=False, ws=ws_py, wd=wd_py, TI=TI)
t_noj = time.time() - t0

aep_noj = sim_noj.aep(normalize_probabilities=False).sum()
aep_noj_gwh = float(xr.DataArray.to_numpy(aep_noj))
cf_noj = aep_noj_gwh / (len(x) * rated_power * 8760 / 1e9)

print(f"  ✓ Completed in {t_noj:.2f} seconds")

# =============================================================================
# MODEL 2: BASTANKHAH-GAUSSIAN + CRESPO-HERNANDEZ
# =============================================================================

print("\n" + "=" * 80)
print("MODEL 2: Bastankhah-Gaussian + Crespo-Hernandez")
print("=" * 80)

print("\nParameters:")
print("  - Gaussian wake shape")
print("  - k = 0.04 (offshore)")
print("  - CrespoHernandez turbulence model")
print("  - 3×3 rotor grid averaging")

bastankhah = Bastankhah_PorteAgel_2014(
    site, 
    windTurbines,
    k=0.04,
    turbulenceModel=CrespoHernandez(),
    rotorAvgModel=EqGridRotorAvg(3)
)

print("\n⏱️  Running Bastankhah simulation...")
t0 = time.time()
sim_bast = bastankhah(x, y, time=False, ws=ws_py, wd=wd_py, TI=TI)
t_bast = time.time() - t0

aep_bast = sim_bast.aep(normalize_probabilities=False).sum()
aep_bast_gwh = float(xr.DataArray.to_numpy(aep_bast))
cf_bast = aep_bast_gwh / (len(x) * rated_power * 8760 / 1e9)

print(f"  ✓ Completed in {t_bast:.2f} seconds")

# =============================================================================
# COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

diff_gwh = aep_bast_gwh - aep_noj_gwh
diff_pct = (aep_bast_gwh / aep_noj_gwh - 1) * 100
diff_cf = cf_bast - cf_noj

print(f"\n{'Model':<40} {'AEP [GWh]':>15} {'Cap. Factor':>15} {'Time [s]':>12}")
print("─" * 80)
print(f"{'NOJ (IEA 740 baseline)':<40} {aep_noj_gwh:>15.2f} {cf_noj*100:>14.2f}% {t_noj:>12.2f}")
print(f"{'Bastankhah + Crespo':<40} {aep_bast_gwh:>15.2f} {cf_bast*100:>14.2f}% {t_bast:>12.2f}")
print("─" * 80)
print(f"{'Difference (Bast - NOJ)':<40} {diff_gwh:>+15.2f} {diff_cf*100:>+14.2f}% {t_bast-t_noj:>+12.2f}")
print(f"{'Relative difference':<40} {diff_pct:>+14.2f}%")

print(f"\n📊 Key findings:")
print(f"  • Advanced wake model predicts {diff_pct:+.2f}% {'higher' if diff_pct > 0 else 'lower'} AEP")
print(f"  • Difference: {abs(diff_gwh):.2f} GWh/year")
print(f"  • Revenue impact: €{abs(diff_gwh) * 50:.0f}k/year @ €50/MWh")

if abs(diff_pct) < 2:
    print(f"\n✓ Models agree well (< 2% difference)")
    print(f"  → Wake modeling choice has minor impact for this layout")
elif diff_pct > 0:
    print(f"\n⚠️  Bastankhah predicts {diff_pct:.1f}% higher AEP")
    print(f"  → Possible reasons:")
    print(f"    • Gaussian wake model has faster recovery")
    print(f"    • Rotor averaging captures beneficial wind shear")
    print(f"    • Wake-induced turbulence improves mixing")
else:
    print(f"\n⚠️  Bastankhah predicts {abs(diff_pct):.1f}% lower AEP")
    print(f"  → Possible reasons:")
    print(f"    • More realistic wake deficit in near wake")
    print(f"    • Turbulence model increases losses")

print(f"\n⏱️  Computational cost:")
print(f"  • NOJ: {t_noj:.2f}s (baseline)")
print(f"  • Bastankhah: {t_bast:.2f}s ({t_bast/t_noj:.1f}× slower)")

print(f"\n💡 Recommendation:")
if abs(diff_pct) < 3:
    print(f"  For optimization: Use NOJ (faster, similar results)")
    print(f"  For final validation: Run both models")
else:
    print(f"  Model choice significantly impacts results")
    print(f"  Use Bastankhah for more accurate predictions")
    print(f"  Consider validation with CFD or field data")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
