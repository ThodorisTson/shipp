"""
compare_wake_yield_onerun.py  -- INTERNAL diagnostic (not for the thesis)
-------------------------------------------------------------------------
ONE run produces the full OLD-vs-NEW x OLD-k-vs-NEW-k wake-yield grid.

Loads BOTH wind-resource config files (each carries its own h_ref / alpha / TI / included series), builds the PyWake site+turbine exactly as production, and runs
the 65-turbine array. Sweeps k inline because get_wake_model() hard-codes 0.0572.

CORRECTNESS NOTE (important):
  wp2_common scales ONLY the Weibull A from h_ref to hub height; the hourly series fed to the wake model stays at h_ref. The OLD config is at 86 m, the
  NEW at 90 m. To compare like-for-like at 90 m hub height, this script applies the power-law shear to the HOURLY series itself, per each config's h_ref/alpha,
  before the wake run. Set APPLY_HOURLY_SHEAR=False to reproduce the raw (uncorrected) behaviour instead.

Run from the folder with wp2_common.py and both YAMLs:
    python compare_wake_yield_onerun.py
Deps: pywake, xarray, numpy, yaml (whatever wp2_common needs).
"""
from pathlib import Path
import numpy as np
import yaml
import wp2_common as wp2

HERE = Path(__file__).parent
HUB_HEIGHT  = 90.0
GRID_CAP_MW = 300.0
K_OLD, K_NEW = 0.0324809, 0.0572
APPLY_HOURLY_SHEAR = False     # True = both cases compared at true 90 m hub height

# Each case points at a CONFIG file (not the raw series). The config carries
# h_ref / alpha / TI / the !included hourly file, so the loader does the rest.
WIND_CONFIGS = {
    "NEW(90m,TI0.14)": HERE / "WP2_Wind_Resource.yaml",
    "OLD(86m,TI0.10)": HERE / "WP2_Wind_Resource - old.yaml",
}

def build_inline_bastankhah(site, wt, k):
    from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
    from py_wake.turbulence_models.crespo import CrespoHernandez
    from py_wake.superposition_models import LinearSum
    from py_wake.rotor_avg_models import EqGridRotorAvg
    return Bastankhah_PorteAgel_2014(
        site, wt, k=k,
        turbulenceModel=CrespoHernandez(),
        superpositionModel=LinearSum(),
        rotorAvgModel=EqGridRotorAvg(3),
    )

def load_hpp_with_wind(hpp_yaml, wind_cfg_path):
    """Load the master HPP, then splice in the chosen wind-resource config."""
    hpp = wp2.load_yaml(hpp_yaml)
    wind_cfg = wp2.load_yaml(wind_cfg_path)   # resolves its own !include
    hpp['site']['energy_resource'] = wind_cfg \
        if 'h_ref' in wind_cfg else hpp['site']['energy_resource']
    # wp2.load_yaml on the wind-resource file returns the dict with h_ref, shear,
    # weibull_fit, time_series already resolved. Place it where the loader expects.
    hpp['site']['energy_resource'] = wind_cfg
    return hpp

def build_setup(hpp):
    wr  = wp2.load_wind_resource(hpp, verbose=False)
    tb  = wp2.load_turbine(hpp)
    site, wt, ws_bins, wd_bins = wp2.build_pywake_objects(wr, tb)
    x, y, n = wp2.load_layout(hpp, verbose=False)
    return wr, tb, site, wt, x, y, n

def run_case(cfg_label, cfg_path):
    hpp = load_hpp_with_wind(HERE / "WP2_HPP.yaml", cfg_path)
    wr, tb, site, wt, x, y, n = build_setup(hpp)

    ts = hpp['site']['energy_resource']['time_series']['wind_resource']
    ws = np.asarray(ts['wind_speed'], float)
    wd = np.asarray(ts['wind_direction'], float)
    ti = ts.get('turbulence_intensity')
    ti = np.asarray(ti['data'], float) if isinstance(ti, dict) else None

    h_ref = float(hpp['site']['energy_resource']['h_ref'])
    alpha = float(hpp['site']['energy_resource']['shear']['alpha'])
    if APPLY_HOURLY_SHEAR and abs(h_ref - HUB_HEIGHT) > 1e-9:
        factor = (HUB_HEIGHT / h_ref) ** alpha
        ws = ws * factor
        shear_note = f"hourly x{factor:.4f} ({h_ref:.0f}->{HUB_HEIGHT:.0f}m)"
    else:
        shear_note = f"hourly as-is @ {h_ref:.0f}m"

    rated_farm = n * tb['rated_power'] / 1e6
    out = {}
    for klabel, k in [("oldk", K_OLD), ("newk", K_NEW)]:
        model = build_inline_bastankhah(site, wt, k)
        kw = dict(x=x, y=y, wd=wd, ws=ws, time=True)
        if ti is not None: kw['TI'] = ti
        sim = model(**kw)
        farm_MW = sim.Power.values.sum(axis=0) / 1e6
        out[klabel] = dict(
            gen=farm_MW.sum()/1e3, cf=farm_MW.mean()/rated_farm,
            over=int((farm_MW > GRID_CAP_MW).sum()),
            surplus=np.maximum(farm_MW-GRID_CAP_MW,0).sum()/1e3,
            ws=ws.mean())
    return out, shear_note

# ---------------------------------------------------------------- run all
print(f"APPLY_HOURLY_SHEAR = {APPLY_HOURLY_SHEAR}\n")
print(f"{'case':28s} {'ws':>5} {'gen GWh':>9} {'CF %':>6} {'hrs>cap':>8} {'surplus':>9}")
R = {}
for label, path in WIND_CONFIGS.items():
    res, note = run_case(label, path)
    R[label] = res
    for klabel in ("oldk", "newk"):
        r = res[klabel]
        print(f"{label+' '+klabel:28s} {r['ws']:5.2f} {r['gen']:9.1f} {r['cf']*100:6.1f} "
              f"{r['over']:8d} {r['surplus']:9.1f}")
    print(f"   ({note})")

# ---------------------------------------------------------------- decomposition
try:
    base = R["OLD(86m,TI0.10)"]["oldk"]["gen"]
    full = R["NEW(90m,TI0.14)"]["newk"]["gen"]
    wind_only = R["NEW(90m,TI0.14)"]["oldk"]["gen"]   # new wind, old k
    k_only    = R["OLD(86m,TI0.10)"]["newk"]["gen"]   # old wind, new k
    print("\n--- delivered-energy decomposition (GWh) ---")
    print(f"  total  new setup - old setup : {full-base:+.1f}")
    print(f"   wind+TI change alone        : {wind_only-base:+.1f}")
    print(f"   k change alone              : {k_only-base:+.1f}")
    print(f"   interaction (remainder)     : {(full-base)-(wind_only-base)-(k_only-base):+.1f}")
except KeyError:
    pass