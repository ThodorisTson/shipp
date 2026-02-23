"""
WP2 Battery Optimization & Degradation Testing
==============================================

Following Jenna's Example 2 pattern EXACTLY.
Step-by-step battery implementation - no nesting yet.

Structure matches example_2.py:
1. Load power and price time series
2. Create SHIPP components (Storage, Production, TimeSeries)
3. Run sizing optimization
4. Run dispatch optimization (fixed capacity)
5. Compare results
6. Plot

Next step: Add degradation modeling to this working baseline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SHIPP imports (exactly like Example 2)
from shipp.kernel import solve_lp_sparse
from shipp.kernel_pyomo import solve_lp_pyomo  # For large problems
from shipp.components import Storage, Production, TimeSeries

# WP2 imports
from wp2_common import load_yaml, quick_setup, get_wake_model
from degradation import count_equivalent_full_cycles


# PyWake for time series
import xarray as xr
from py_wake.site import XRSite

# =============================================================================
# CONFIGURATION (matching Example 2 variable names)
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
HPP_YAML = SCRIPT_DIR / "WP2_HPP.yaml"
PRICE_CSV = SCRIPT_DIR / "dk1_prices_2022.csv"

eur_to_usd = 1.18        # EUR → USD conversion (Example 2 uses USD)
discount_rate = 0.03     # Discount rate (Example 2: 0.03)
n_year = 20              # Project duration [years] (Example 2: 20)
p_min = 0                # Minimum power [MW] (Example 2: 10, but we have no baseload)

# =============================================================================
# SOLVER CONFIGURATION (exactly like Example 2)
# =============================================================================
# For full year: use scipy (auto-limits to 8,752 hours which is fine)
# For 30 days: can use either
pyo_solver = 'none'      # 'none' = scipy sparse (most stable on Windows)
                         # 'appsi_highs' = HiGHS (faster but can have issues on large problems)
                         # 'glpk' = GLPK (needs separate install)

# =============================================================================
# QUICK TOGGLE: Testing vs Full Year
# =============================================================================
RUN_FULL_YEAR = False     # True = full year (8760h, ~4min), False = quick test (180d, ~2min)

# Simulation period
if RUN_FULL_YEAR:
    n = None             # Full year (all 8760 hours)
    print("Running FULL YEAR simulation (8760 hours, ~4 minutes)")
else:
    n = 30 * 24          # Quick testing (30 days)
    print("Running QUICK TEST (30 days, ~1 minute)")

dt = 1                   # Time step duration [hour]

# PyWake settings
WAKE_MODEL = 'Bastankhah'

print("=" * 80)
print("WP2 BATTERY OPTIMIZATION - Example 2 Pattern")
print("=" * 80)

# =============================================================================
# 1. LOAD POWER TIME SERIES (replacing renewable.ninja in Example 2)
# =============================================================================

print("\n[1/4] Loading wind farm configuration and running PyWake...")

# Load WP2 configuration
setup = quick_setup(HPP_YAML, config={'interp_n': 2000}, verbose=False)
hpp = setup['hpp']

# Extract ERA5 hourly time series
ts = hpp['site']['energy_resource']['time_series']['wind_resource']
ws = np.array(ts['wind_speed'], dtype=float)
wd = np.array(ts['wind_direction'], dtype=float)

# Check for hourly TI
ti = None
if 'turbulence_intensity' in ts:
    ti_data = ts['turbulence_intensity']
    if isinstance(ti_data, dict) and 'data' in ti_data:
        ti = np.array(ti_data['data'], dtype=float)

# Set n to data length if None (full year)
if n is None:
    n = len(ws)
    print(f"  Using full dataset: {n:,} hours ({n/24:.0f} days)")
else:
    print(f"  Using subset: {n:,} hours ({n/24:.0f} days)")

# =============================================================================
# 2. LOAD PRICE TIME SERIES (load early to check length)
# =============================================================================

print("\n[2/4] Loading electricity prices...")

df_price = pd.read_csv(PRICE_CSV)

# Find price column (DK1 prices use 'price_eur_mwh')
price_col = None
for col in ['price_eur_mwh', 'price_eur_per_mwh', 'Price', 'price']:
    if col in df_price.columns:
        price_col = col
        break

if price_col is None:
    raise KeyError(f"Cannot find price column in {PRICE_CSV}. Available columns: {df_price.columns.tolist()}")

# Extract price data - matching Example 2 'data_price'
data_price = df_price[price_col].values  # EUR/MWh, as array

# Handle length mismatch between wind and price data
n_price = len(data_price)
n_wind = n

if n_wind != n_price:
    print(f"  ⚠️  Data length mismatch:")
    print(f"      Wind data: {n_wind:,} hours")
    print(f"      Price data: {n_price:,} hours")
    
    # Use the shorter length (don't pad - causes numerical issues)
    n = min(n_wind, n_price)
    print(f"  → Using {n:,} hours (8 hours short of full year, but clean data)")
    print(f"     {n/24:.0f} days = {n/24/365*100:.1f}% of year")
    
    if n_wind > n_price:
        # Price data is shorter - we'll trim wind data later
        pass
    else:
        # Wind data is shorter - trim price data now
        data_price = data_price[:n_wind]
else:
    print(f"  ✓ Data length match: {n:,} hours")

data_price = data_price[:n]  # Final trim to n
print(f"  ✓ Mean price: {np.mean(data_price):.2f} EUR/MWh ({np.mean(data_price)*eur_to_usd:.2f} USD/MWh)")

# =============================================================================
# 3. FINALIZE DATA LENGTHS AND RUN PYWAKE
# =============================================================================

# Trim wind data to final n
ws = ws[:n]
wd = wd[:n]
if ti is not None:
    ti = ti[:n]

print(f"\n  Final simulation length: {n:,} hours ({n/24:.1f} days)")
print(f"  Running PyWake {WAKE_MODEL} model...")

# Run PyWake time series (minimal site for time series mode)
site = XRSite(ds=xr.Dataset(data_vars=dict(P=1)))
wf_model = get_wake_model(WAKE_MODEL, site, setup['windturbine'])

time_days = np.arange(n) / 24.0
sim_kwargs = {'x': setup['x'], 'y': setup['y'], 'wd': wd, 'ws': ws, 'time': time_days}
if ti is not None:
    sim_kwargs['TI'] = ti

sim_res = wf_model(**sim_kwargs)

# Extract power [W → MW, as list] - matching Example 2 'data_power'
power_W = sim_res.Power.sum(['wt']).values
data_power = (power_W / 1e6).tolist()  # MW, as list (Example 2 format)

print(f"  ✓ Mean wind power: {np.mean(data_power):.1f} MW")
print(f"  ✓ Peak wind power: {np.max(data_power):.1f} MW")

# =============================================================================
# 3. SETUP STORAGE AND PRODUCTION (exactly like Example 2)
# =============================================================================

print("\n[3/4] Building SHIPP components...")

# Grid limit [W → MW]
p_max = float(hpp['grid_connection_capacity']) / 1e6  # Example 2 uses 'p_max'

# Battery parameters [Wh/W → MWh/MW]
bat = setup['battery']
e_cap = bat['energy_capacity_Wh'] / 1e6   # MWh (Example 2 name)
p_cap = bat['power_capacity_W'] / 1e6     # MW  (Example 2 name)

# Efficiency (matching Example 2 exactly)
rte_dc = bat['rte_nominal']
pcu_eff = bat['pcu_efficiency']
rte_ac = rte_dc * (pcu_eff ** 2)
eta = rte_ac  # Example 2 uses eta as the round-trip efficiency (NOT sqrt!)

# Cost parameters (used for sizing optimization)
# Note: Power and energy costs scale differently - this is simplified

e_cost = bat['capex_EUR_per_kWh'] * 1000 * eur_to_usd  # USD/MWh
p_cost = bat['capex_EUR_per_kW']  * 1000 * eur_to_usd   # USD/MW

print(f"  Battery: {e_cap:.0f} MWh / {p_cap:.0f} MW")
print(f"  Round-trip efficiency: {rte_ac*100:.1f}%")
print(f"  eta parameter (eff_out): {eta:.4f}")
print(f"  Grid limit: {p_max:.0f} MW")

# Build SHIPP components (EXACTLY like Example 2)
stor = Storage(e_cap=e_cap, p_cap=p_cap, eff_in=1, eff_out=eta, 
               e_cost=e_cost, p_cost=p_cost)
stor_null = Storage(e_cap=0, p_cap=0, eff_in=1, eff_out=1, 
                    e_cost=0, p_cost=0)

# Verify all data arrays have exactly n elements (should pass after padding)
assert len(data_power) == n, f"Power data mismatch: {len(data_power)} != {n}"
assert len(data_price) == n, f"Price data mismatch: {len(data_price)} != {n}"

price_dam = TimeSeries((data_price * eur_to_usd).tolist(), dt)  # Convert to USD

power_ts = TimeSeries(data_power, dt)
prod = Production(power_ts, p_cost=0)
prod_null = Production(TimeSeries([0 for _ in range(n)], dt), 0)

# =============================================================================
# 4. SOLVE (exactly like Example 2)
# =============================================================================

print("\n[4/4] Running SHIPP optimization...")

# Check solver and timestep limits (exactly like Example 2)
if pyo_solver == 'none':
    # Using scipy sparse solver - works well up to ~6 months
    max_hours_sparse = 180 * 24  # 6 months = 4,320 hours
    if n > max_hours_sparse:
        print(f"  ⚠️  {n:,} hours exceeds scipy sparse solver limit")
        print(f"  → Limiting to {max_hours_sparse:,} hours ({max_hours_sparse/24:.0f} days / 6 months)")
        print(f"  → This captures half-year seasonality (e.g., winter + summer)")
        print(f"  → For full year: get Gurobi academic license, set pyo_solver='gurobi'")
        n = max_hours_sparse
        # Re-trim all data
        data_power = data_power[:n]
        data_price = data_price[:n]
        # Rebuild TimeSeries with trimmed data
        price_dam = TimeSeries((data_price * eur_to_usd).tolist(), dt)
        power_ts = TimeSeries(data_power, dt)
        prod = Production(power_ts, p_cost=0)
        prod_null = Production(TimeSeries([0 for _ in range(n)], dt), 0)
    print(f"  ℹ️  Using scipy sparse solver")
else:
    # Using Pyomo solver
    pass

print(f"  Solver: {pyo_solver if pyo_solver != 'none' else 'scipy linprog (limited)'}")
print(f"  Timesteps: {n:,} hours ({n/24:.0f} days)")
print("  (This may take 30-60 seconds...)")

# Sizing optimization (Example 2: 'os')
if pyo_solver == 'none':
    os = solve_lp_sparse(price_dam, prod, prod_null, stor, stor_null, 
                         discount_rate, n_year, p_min, p_max, n)
else:
    os = solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null, 
                        discount_rate, n_year, p_min, p_max, n, pyo_solver)

# Dispatch only with fixed capacity (Example 2: 'os_fixed')
if pyo_solver == 'none':
    os_fixed = solve_lp_sparse(price_dam, prod, prod_null, stor, stor_null, 
                               discount_rate, n_year, p_min, p_max, n, 
                               fixed_cap=True)
else:
    os_fixed = solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null, 
                              discount_rate, n_year, p_min, p_max, n, pyo_solver,
                              fixed_cap=True)

# Calculate yearly revenues for renewable power only (Example 2: 'revenues_res_only')
revenues_res_only = 365 * 24 / n * np.dot(data_price, np.minimum(data_power, p_max)) * dt

# Get NPV (Example 2 does this)
os.get_added_npv(discount_rate, n_year)
os_fixed.get_added_npv(discount_rate, n_year)

print("  ✓ Optimization complete!")

# =============================================================================
# CYCLES (use same method as degradation.py)
# =============================================================================

# Equivalent full cycles over the simulated horizon
cycles_opt = count_equivalent_full_cycles(
    storage_p=os.storage_p[0].data,
    storage_e=os.storage_e[0].data,
    e_cap=os.storage_list[0].e_cap
)

cycles_fixed = count_equivalent_full_cycles(
    storage_p=os_fixed.storage_p[0].data,
    storage_e=os_fixed.storage_e[0].data,
    e_cap=os_fixed.storage_list[0].e_cap
)

period_days = n * dt / 24.0
cycles_opt_per_year = cycles_opt / period_days * 365.0
cycles_fixed_per_year = cycles_fixed / period_days * 365.0

# =============================================================================
# 5. RESULTS (exactly like Example 2 table format)
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

# Example 2 table format
print('\n\t\tP_min [MW]\tRevenue [kUSD]\tRev. increase\tp_cap/e_cap\t\t'
      'Cost [M.USD]\tTot NPV [M.USD]')
print('-' * 100)

print('Sizing Opt.\t{:.1f}\t\t{:.1f}\t\t{:.2f}%\t\t{:.2f}/{:.2f}\t\t'
      '{:.2f}\t\t{:.1f}'.format(
    p_min,
    os.revenue * 1e-3,
    100 * (os.revenue / revenues_res_only - 1),
    os.storage_list[0].p_cap,
    os.storage_list[0].e_cap,
    -os.a_npv,
    os.npv
))

print('Dispatch only\t{:.1f}\t\t{:.1f}\t\t{:.2f}%\t\t{:.2f}/{:.2f}\t\t'
      '{:.2f}\t\t{:.1f}'.format(
    p_min,
    os_fixed.revenue * 1e-3,
    100 * (os_fixed.revenue / revenues_res_only - 1),
    os_fixed.storage_list[0].p_cap,
    os_fixed.storage_list[0].e_cap,
    -os_fixed.a_npv,
    os_fixed.npv
))

print('-' * 100)

print(f"\nCycles (EFC) over {period_days:.1f} days:")
print(f"  Sizing opt:   {cycles_opt:.2f}  (≈ {cycles_opt_per_year:.0f}/year)")
print(f"  Fixed cap:    {cycles_fixed:.2f}  (≈ {cycles_fixed_per_year:.0f}/year)")

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================

results_file = SCRIPT_DIR / "battery_optimization_results.csv"

# Check if file exists to determine if we need headers
file_exists = results_file.exists()

import csv
from datetime import datetime

with open(results_file, 'a', newline='') as f:
    writer = csv.writer(f)
    
    # Write header if file is new
    if not file_exists:
        writer.writerow([
            'timestamp', 'hours_simulated', 'days_simulated', 
            'optimization_type', 'p_min_MW', 'revenue_kUSD', 'revenue_increase_pct',
            'p_cap_MW', 'e_cap_MWh', 'cost_MUSD', 'npv_MUSD',
            'wind_mean_MW', 'wind_peak_MW', 'rte_pct', 'solver'
        ])
    
    # Write sizing optimization result
    writer.writerow([
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        n, n/24,
        'sizing', p_min, os.revenue * 1e-3, 
        100 * (os.revenue / revenues_res_only - 1),
        os.storage_list[0].p_cap, os.storage_list[0].e_cap,
        -os.a_npv, os.npv,
        np.mean(data_power), np.max(data_power),
        rte_ac*100, pyo_solver if pyo_solver != 'none' else 'scipy'
    ])
    
    # Write fixed capacity result
    writer.writerow([
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        n, n/24,
        'dispatch_fixed', p_min, os_fixed.revenue * 1e-3,
        100 * (os_fixed.revenue / revenues_res_only - 1),
        os_fixed.storage_list[0].p_cap, os_fixed.storage_list[0].e_cap,
        -os_fixed.a_npv, os_fixed.npv,
        np.mean(data_power), np.max(data_power),
        rte_ac*100, pyo_solver if pyo_solver != 'none' else 'scipy'
    ])

print(f"\n✓ Results saved to: {results_file}")

# Also save detailed text report
report_file = SCRIPT_DIR / f"battery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(report_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("WP2 BATTERY OPTIMIZATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("SIMULATION PARAMETERS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Hours simulated: {n:,} ({n/24:.1f} days)\n")
    f.write(f"Solver: {pyo_solver if pyo_solver != 'none' else 'scipy sparse'}\n")
    f.write(f"Discount rate: {discount_rate:.1%}\n")
    f.write(f"Project duration: {n_year} years\n")
    f.write(f"Minimum power: {p_min} MW\n")
    f.write(f"Grid limit: {p_max:.0f} MW\n\n")
    
    f.write("WIND FARM\n")
    f.write("-" * 80 + "\n")
    f.write(f"Mean power: {np.mean(data_power):.1f} MW\n")
    f.write(f"Peak power: {np.max(data_power):.1f} MW\n")
    f.write(f"Mean export as % of grid limit: {np.mean(data_power)/p_max*100:.1f}%\n\n")
    
    f.write("BATTERY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Energy capacity: {e_cap:.0f} MWh\n")
    f.write(f"Power capacity: {p_cap:.0f} MW\n")
    f.write(f"Round-trip efficiency: {rte_ac*100:.1f}%\n")
    f.write(f"E/P ratio: {e_cap/p_cap:.2f} hours\n\n")
    
    f.write("ELECTRICITY PRICES\n")
    f.write("-" * 80 + "\n")
    f.write(f"Mean price: {np.mean(data_price):.2f} EUR/MWh\n")
    f.write(f"Min price: {np.min(data_price):.2f} EUR/MWh\n")
    f.write(f"Max price: {np.max(data_price):.2f} EUR/MWh\n")
    f.write(f"Price spread: {np.max(data_price) - np.min(data_price):.2f} EUR/MWh\n\n")
    
    f.write("RESULTS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("Wind-only baseline:\n")
    f.write(f"  Revenue: {revenues_res_only * 1e-3:.1f} kUSD\n\n")
    
    f.write("Sizing Optimization:\n")
    f.write(f"  Revenue: {os.revenue * 1e-3:.1f} kUSD\n")
    f.write(f"  Revenue increase: {100*(os.revenue/revenues_res_only-1):.2f}%\n")
    f.write(f"  Optimal battery: {os.storage_list[0].p_cap:.1f} MW / {os.storage_list[0].e_cap:.1f} MWh\n")
    f.write(f"  Cost: ${-os.a_npv:.2f} M\n")
    f.write(f"  NPV: ${os.npv:.2f} M\n\n")
    
    f.write(f"Fixed Capacity ({p_cap:.0f} MW / {e_cap:.0f} MWh):\n")
    f.write(f"  Revenue: {os_fixed.revenue * 1e-3:.1f} kUSD\n")
    f.write(f"  Revenue increase: {100*(os_fixed.revenue/revenues_res_only-1):.2f}%\n")
    f.write(f"  Cost: ${-os_fixed.a_npv:.2f} M\n")
    f.write(f"  NPV: ${os_fixed.npv:.2f} M\n\n")
    
    if n < 8000:  # If less than full year
        f.write("ANNUAL EXTRAPOLATION\n")
        f.write("-" * 80 + "\n")
        scale_factor = 8760 / n
        f.write(f"Simulated: {n:,} hours ({n/24:.1f} days)\n")
        f.write(f"Scale factor to annual: {scale_factor:.3f}\n\n")
        f.write(f"Estimated annual revenue (fixed cap): {os_fixed.revenue * 1e-3 * scale_factor:.1f} kUSD\n")
        f.write(f"Estimated annual revenue increase: {100*(os_fixed.revenue/revenues_res_only-1):.2f}%\n")
        f.write(f"Equivalent full cycles (fixed cap) in period: {cycles_fixed:.2f}\n")
        f.write(f"Estimated cycles/year (fixed cap): {cycles_fixed_per_year:.0f}\n")
        f.write(f"Equivalent full cycles (sizing opt) in period: {cycles_opt:.2f}\n")
        f.write(f"Estimated cycles/year (sizing opt): {cycles_opt_per_year:.0f}\n")

print(f"✓ Detailed report saved to: {report_file}")

# =============================================================================
# 6. PLOT (exactly like Example 2)
# =============================================================================

print("\nGenerating plots...")

# Time vector (Example 2 uses days)
time_vec = np.arange(n) * dt / 24

# Create figure (Example 2: 1 row, 2 columns)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Left plot: Power flows (Example 2 left plot)
ax[0].plot(time_vec, np.array(data_power) + os.storage_p[0].data, 
           label='Power + opt. storage')
ax[0].plot(time_vec, np.array(data_power) + os_fixed.storage_p[0].data, 
           label='Power + fixed storage')
ax[0].plot(time_vec, data_power, label='Wind power')
ax[0].legend()
ax[0].set_xlabel('Time [days]')
ax[0].set_ylabel('Power [MW]')
ax[0].grid(True, alpha=0.3)

# Right plot: State of charge (Example 2 right plot)
ax[1].plot(time_vec, os.storage_e[0].data, label='Opt. storage')
ax[1].plot(time_vec, os_fixed.storage_e[0].data, label='Fixed storage')
ax[1].legend()
ax[1].set_xlabel('Time [days]')
ax[1].set_ylabel('State of charge [MWh]')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'battery_optimization_results.png', dpi=200)
print(f"✓ Saved: battery_optimization_results.png")

plt.show()

# =============================================================================
# READY FOR DEGRADATION MODELING
# =============================================================================

print("\n" + "=" * 80)
print("✓ BASELINE COMPLETE - READY FOR DEGRADATION")
print("=" * 80)

print("""
This script follows Example 2 pattern exactly.

Battery dispatch results available in:
  • os.storage_p[0].data  - battery power [MW] for sizing opt
  • os.storage_e[0].data  - battery energy [MWh] for sizing opt
  • os_fixed.storage_p[0].data  - battery power [MW] for fixed cap
  • os_fixed.storage_e[0].data  - battery energy [MWh] for fixed cap

Next step: Add degradation modeling
  1. Count cycles from storage_p and storage_e
  2. Calculate capacity fade
  3. Update battery limits over time
  4. Re-run optimization with degraded capacity
""")