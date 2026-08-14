"""
diagnose_wind_provenance.py
---------------------------
Answer the only question that matters: WHICH wind series did a run consume,
and what generation did it imply -- without re-running the 20-year LP.

It does two independent things:

  PART A  Read the run's OUTPUT csvs and print every wind-sensitive proxy that
          is actually stored (revenue, NPV, cycles/yr, mean DoD, EFC). These do
          not prove the wind input, but a wind change must move them.

  PART B  Load a wind-resource YAML the SAME way the pipeline does and compute
          the farm-level annual generation, capacity factor and curtailment
          headroom from the turbine power curve. THIS is the definitive
          fingerprint: run it once per YAML (new and old) and compare.

Part B needs only the YAMLs + the turbine power curve; no PyWake, no LP.
It approximates farm power as 65 x single-turbine power (no wake losses), which
is fine for a RELATIVE old-vs-new comparison since wakes apply equally to both.

Reproducible in VS Code on Windows. Deps: numpy, pyyaml.
    python diagnose_wind_provenance.py
"""
from pathlib import Path
import csv, glob
import numpy as np
import yaml

HERE = Path(__file__).parent if "__file__" in globals() else Path(".")

# ----------------------------------------------------------------------
# EDIT THESE: which run csv-prefix, and which wind YAML to fingerprint
# ----------------------------------------------------------------------
RUN_GLOB   = "*195133*dk2022*"                     # today's run outputs
WIND_YAML  = HERE / "wind_resource_2022_era5_90m.yaml"   # the series to measure
FARM_YAML  = HERE / "WP2_Wind_Farm.yaml"
N_TURBINES = 65
GRID_CAP_MW = 300.0          # plant grid connection (for curtailment-headroom proxy)

# ====================== PART A: run-output proxies ======================
def find(prefix_glob, kind):
    hits = glob.glob(str(HERE/ f"{kind}_{prefix_glob}.csv")) or glob.glob(str(HERE/f"*{kind}*{prefix_glob.strip('*')}*.csv"))
    return hits[0] if hits else None

def row0(path):
    with open(path) as f: return next(csv.DictReader(f))

print("="*64)
print("PART A -- wind-sensitive proxies stored in the run outputs")
print("="*64)
opt = find(RUN_GLOB, "battery_optimization_results")
if opt:
    for r in csv.DictReader(open(opt)):
        if r["optimization_type"] == "dispatch_fixed":
            print(f"  revenue_kEUR     : {float(r['revenue_kEUR']):.1f}")
            print(f"  npv_MEUR         : {float(r['npv_MEUR']):.1f}")
            print(f"  cycles_per_year  : {float(r['cycles_per_year']):.1f}")
            print(f"  revenue_incr_pct : {float(r['revenue_increase_pct']):.3f}  (battery vs no-battery)")
print("  NOTE: no wind generation / curtailment column exists in these outputs.")
print("        Wind provenance CANNOT be confirmed from the run csvs alone.")

# ====================== PART B: measure the YAML ========================
print("\n" + "="*64)
print(f"PART B -- generation fingerprint of: {WIND_YAML.name}")
print("="*64)

# build single-turbine power curve from the farm YAML (same formula as wp2_common)
wf = yaml.safe_load(open(FARM_YAML))["turbines"]
perf = wf["performance"]
D = float(wf["rotor_diameter"]); rated = float(perf["rated_power"])
cp_u = np.array(perf["Cp_curve"]["Cp_wind_speeds"], float)
cp_v = np.array(perf["Cp_curve"]["Cp_values"], float)
rho = 1.225; area = np.pi/4*D**2
cut_in, cut_out = cp_u.min(), cp_u.max()
def turbine_power(ws):
    cp = np.interp(ws, cp_u, cp_v, left=0, right=0)
    p = cp * 0.5 * rho * area * ws**3
    p = np.minimum(p, rated)
    p[(ws < cut_in) | (ws > cut_out)] = 0.0
    return p   # [W]

wr = yaml.safe_load(open(WIND_YAML))["wind_resource"]
ws = np.asarray(wr["wind_speed"], float)
ti = np.asarray(wr["turbulence_intensity"]["data"], float)
p1 = turbine_power(ws)                     # W per turbine
farm_MW = N_TURBINES * p1 / 1e6            # MW, no wakes
gen_GWh = farm_MW.sum() / 1e3             # MWh->GWh (hourly steps)
cf = farm_MW.mean() / (N_TURBINES * rated/1e6)
hrs_over_cap = int((farm_MW > GRID_CAP_MW).sum())
surplus_GWh = np.maximum(farm_MW - GRID_CAP_MW, 0).sum()/1e3

print(f"  mean wind speed   : {ws.mean():.3f} m/s")
print(f"  TI (unique)       : {np.unique(ti)}")
print(f"  annual generation : {gen_GWh:,.1f} GWh  (no-wake upper bound)")
print(f"  gross capacity factor : {cf*100:.1f}%")
print(f"  hours over {GRID_CAP_MW:.0f} MW grid cap : {hrs_over_cap}  ({100*hrs_over_cap/len(ws):.1f}% of year)")
print(f"  energy above grid cap : {surplus_GWh:,.1f} GWh  (arbitrage/curtailment pool)")
print("\nRun this again with WIND_YAML = the OLD 86 m file to get the same")
print("five numbers; the differences are the true effect of the wind swap.")
