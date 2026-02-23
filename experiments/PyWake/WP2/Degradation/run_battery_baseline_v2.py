"""
WP2 Battery Optimization Baseline (Minimal, readable)

Follows Example 2 pattern:
1) Load wind + price time series
2) Build SHIPP components
3) Solve sizing optimization (os)
4) Solve dispatch-only at fixed capacity (os_fixed)
5) Compare results, compute real cycles, save CSV/report, optional plot

Notes:
- Uses the SAME cycle counter as degradation.py (consistent results).
- For scipy sparse solver, caps horizon to 6 months for stability.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SHIPP
from shipp.kernel import solve_lp_sparse
from shipp.kernel_pyomo import solve_lp_pyomo
from shipp.components import Storage, Production, TimeSeries

# WP2
from wp2_common import quick_setup, get_wake_model
from degradation_v_2 import count_equivalent_full_cycles

# PyWake minimal site for time-series runs
import xarray as xr
from py_wake.site import XRSite

# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
HPP_YAML = SCRIPT_DIR / "WP2_HPP.yaml"
PRICE_CSV = SCRIPT_DIR / "dk1_prices_2019.csv"

# Output folders
RESULTS_DIR = SCRIPT_DIR / "Results"
PLOTS_DIR   = SCRIPT_DIR / "Baseline Plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Example 2 style parameters
eur_to_usd = 1.18
discount_rate = 0.03
n_year = 20
p_min = 15.0
dt = 1.0  # hours

# Solver:
#   'none' => scipy sparse (recommended on Windows for long horizons)
#   'appsi_highs' => Pyomo HiGHS (good for shorter horizons; may fail for full year depending on feasibility checks)
pyo_solver = "none"

# Run length
RUN_FULL_YEAR = False
N_DAYS_TEST = 120  # used if RUN_FULL_YEAR=False

# Wake model
WAKE_MODEL = "Bastankhah"

# Outputs
SAVE_CSV = True
SAVE_REPORT = True
MAKE_PLOT = True

run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# =============================================================================
# Helpers
# =============================================================================

def find_price_column(df: pd.DataFrame) -> str:
    candidates = ["price_eur_mwh", "price_eur_per_mwh", "Price", "price"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Cannot find price column. Available columns: {df.columns.tolist()}")

def determine_horizon_hours(
    n_wind_available: int,
    n_price_available: int,
    run_full_year: bool,
    n_days_test: int,
    solver_name: str,
) -> int:
    # requested horizon
    n_req = n_wind_available if run_full_year else int(n_days_test * 24)

    # match to available data
    n = min(n_req, n_wind_available, n_price_available)

    # scipy sparse stability cap
    if solver_name == "none":
        max_hours_sparse = 180 * 24  # 6 months
        n = min(n, max_hours_sparse)

    return int(n)

def run_pywake_timeseries_power_MW(setup: dict, wd: np.ndarray, ws: np.ndarray, ti: np.ndarray | None, wake_model: str = "Bastankhah") -> np.ndarray:
    n = len(ws)
    site = XRSite(ds=xr.Dataset(data_vars=dict(P=1)))
    wf_model = get_wake_model(wake_model, site, setup["windturbine"])

    time_days = np.arange(n) / 24.0

    kwargs = {
        "x": setup["x"],
        "y": setup["y"],
        "wd": wd,
        "ws": ws,
        "time": time_days,
    }
    if ti is not None:
        kwargs["TI"] = ti

    sim_res = wf_model(**kwargs)
    power_W = sim_res.Power.sum(["wt"]).values
    return power_W / 1e6  # MW

def append_results_csv(results_path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    header = not results_path.exists()
    df.to_csv(results_path, mode="a", header=header, index=False)

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("WP2 BATTERY OPTIMIZATION - Minimal Example 2 Pattern")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1) Load WP2 config and time series from YAML
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading WP2 configuration...")
    setup = quick_setup(HPP_YAML, config={"interp_n": 2000}, verbose=False)
    hpp = setup["hpp"]

    ts = hpp["site"]["energy_resource"]["time_series"]["wind_resource"]
    ws_all = np.array(ts["wind_speed"], dtype=float)
    wd_all = np.array(ts["wind_direction"], dtype=float)

    ti_all = None
    ti_dat = ts.get("turbulence_intensity", None)
    if isinstance(ti_dat, dict) and "data" in ti_dat:
        ti_all = np.array(ti_dat["data"], dtype=float)

    n_wind_avail = len(ws_all)
    if len(wd_all) != n_wind_avail:
        raise ValueError("Wind speed and wind direction lengths differ in YAML time series.")

    # -------------------------------------------------------------------------
    # 2) Load prices
    # -------------------------------------------------------------------------
    print("\n[2/4] Loading electricity prices...")
    df_price = pd.read_csv(PRICE_CSV)
    price_col = find_price_column(df_price)
    price_eur_all = df_price[price_col].astype(float).to_numpy()
    n_price_avail = len(price_eur_all)

    # -------------------------------------------------------------------------
    # 3) Decide horizon ONCE (prevents rebuilds later)
    # -------------------------------------------------------------------------
    n = determine_horizon_hours(
        n_wind_available=n_wind_avail,
        n_price_available=n_price_avail,
        run_full_year=RUN_FULL_YEAR,
        n_days_test=N_DAYS_TEST,
        solver_name=pyo_solver,
    )

    print(f"  Wind available:  {n_wind_avail:,} h")
    print(f"  Price available: {n_price_avail:,} h")
    print(f"  Using horizon:   {n:,} h ({n/24:.1f} days)")
    if pyo_solver == "none" and n < (n_wind_avail if RUN_FULL_YEAR else N_DAYS_TEST * 24):
        print("  Note: capped horizon for scipy sparse stability (6 months max).")

    # Trim all inputs to the same horizon
    ws = ws_all[:n]
    wd = wd_all[:n]
    ti = ti_all[:n] if ti_all is not None else None
    price_eur = price_eur_all[:n]

    print(f"  Mean price: {np.mean(price_eur):.2f} EUR/MWh ({np.mean(price_eur)*eur_to_usd:.2f} USD/MWh)")

    # -------------------------------------------------------------------------
    # 4) Run PyWake for wind power
    # -------------------------------------------------------------------------
    print("\n[3/4] Running PyWake time-series simulation...")
    power_wind_MW = run_pywake_timeseries_power_MW(setup, wd, ws, ti, wake_model=WAKE_MODEL)
    print(f"  Wind mean: {np.mean(power_wind_MW):.1f} MW")
    print(f"  Wind peak: {np.max(power_wind_MW):.1f} MW")

    # -------------------------------------------------------------------------
    # 5) Build SHIPP components
    # -------------------------------------------------------------------------
    print("\n[4/4] Building SHIPP components + solving...")

    # Grid limit (W -> MW)
    p_max = float(hpp["grid_connection_capacity"]) / 1e6

    # Battery (Wh/W -> MWh/MW)
    bat = setup["battery"]
    e_cap = bat["energy_capacity_Wh"] / 1e6
    p_cap = bat["power_capacity_W"] / 1e6

    # Efficiency (Example 2 style: eff_in=1, eff_out=eta where eta acts like round-trip)
    rte_dc = float(bat["rte_nominal"])
    pcu_eff = float(bat["pcu_efficiency"])
    rte_ac = rte_dc * (pcu_eff ** 2)
    eta = rte_ac

    # Costs (now both energy and power are priced)
    e_cost = float(bat["capex_EUR_per_kWh"]) * 1000.0 * eur_to_usd  # USD/MWh
    p_cost = float(bat["capex_EUR_per_kW"]) * 1000.0 * eur_to_usd   # USD/MW

    print(f"  Battery: {p_cap:.0f} MW / {e_cap:.0f} MWh (E/P={e_cap/p_cap:.2f} h)")
    print(f"  Grid limit: {p_max:.0f} MW")
    print(f"  RTE(ac): {rte_ac*100:.1f}%")
    print(f"  Solver: {('scipy sparse' if pyo_solver == 'none' else pyo_solver)}")

    stor = Storage(e_cap=e_cap, p_cap=p_cap, eff_in=1.0, eff_out=eta, e_cost=e_cost, p_cost=p_cost)
    stor_null = Storage(e_cap=0.0, p_cap=0.0, eff_in=1.0, eff_out=1.0, e_cost=0.0, p_cost=0.0)

    price_usd = (price_eur * eur_to_usd).tolist()
    power_list = power_wind_MW.tolist()

    price_dam = TimeSeries(price_usd, dt)
    prod = Production(TimeSeries(power_list, dt), p_cost=0.0)
    prod_null = Production(TimeSeries([0.0] * n, dt), p_cost=0.0)

    # Solve sizing and fixed
    if pyo_solver == "none":
        os = solve_lp_sparse(price_dam, prod, prod_null, stor, stor_null, discount_rate, n_year, p_min, p_max, n)
        os_fixed = solve_lp_sparse(price_dam, prod, prod_null, stor, stor_null, discount_rate, n_year, p_min, p_max, n, fixed_cap=True)
    else:
        os = solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null, discount_rate, n_year, p_min, p_max, n, pyo_solver)
        os_fixed = solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null, discount_rate, n_year, p_min, p_max, n, pyo_solver, fixed_cap=True)

    # Wind-only baseline revenue (same as Example 2 idea)
    revenues_res_only = 365.0 * 24.0 / n * np.dot(price_eur, np.minimum(power_wind_MW, p_max)) * dt

    os.get_added_npv(discount_rate, n_year)
    os_fixed.get_added_npv(discount_rate, n_year)

    # Cycles (consistent with degradation.py)
    cycles_opt = count_equivalent_full_cycles(os.storage_p[0].data, os.storage_e[0].data, os.storage_list[0].e_cap)
    cycles_fixed = count_equivalent_full_cycles(os_fixed.storage_p[0].data, os_fixed.storage_e[0].data, os_fixed.storage_list[0].e_cap)

    period_days = n * dt / 24.0
    cycles_opt_per_year = cycles_opt / period_days * 365.0
    cycles_fixed_per_year = cycles_fixed / period_days * 365.0

    # -------------------------------------------------------------------------
    # Print results table (Example 2 style)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    def rev_inc_pct(rev: float) -> float:
        return 100.0 * (rev / revenues_res_only - 1.0)

    print("                P_min [MW]      Revenue [kUSD]  Rev. increase   p_cap/e_cap             Cost [M.USD]    Tot NPV [M.USD]")
    print("-" * 100)
    print(f"Sizing Opt.     {p_min:<14.1f}{os.revenue*1e-3:<15.1f}{rev_inc_pct(os.revenue):<14.2f}%"
          f"   {os.storage_list[0].p_cap:>8.2f}/{os.storage_list[0].e_cap:<8.2f}"
          f"        {-os.a_npv:>10.2f}        {os.npv:>10.1f}")
    print(f"Dispatch only   {p_min:<14.1f}{os_fixed.revenue*1e-3:<15.1f}{rev_inc_pct(os_fixed.revenue):<14.2f}%"
          f"   {os_fixed.storage_list[0].p_cap:>8.2f}/{os_fixed.storage_list[0].e_cap:<8.2f}"
          f"        {-os_fixed.a_npv:>10.2f}        {os_fixed.npv:>10.1f}")
    print("-" * 100)

    print(f"\nCycles (EFC) over {period_days:.1f} days:")
    print(f"  Sizing opt: {cycles_opt:.2f} (≈ {cycles_opt_per_year:.0f}/year)")
    print(f"  Fixed cap:  {cycles_fixed:.2f} (≈ {cycles_fixed_per_year:.0f}/year)")

    # -------------------------------------------------------------------------
    # Save CSV + report
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if SAVE_CSV:
        results_file = RESULTS_DIR / "battery_optimization_results.csv"
        rows = [
            {
                "timestamp": timestamp,
                "hours_simulated": n,
                "days_simulated": n / 24.0,
                "optimization_type": "sizing",
                "p_min_MW": p_min,
                "revenue_kUSD": os.revenue * 1e-3,
                "revenue_increase_pct": rev_inc_pct(os.revenue),
                "p_cap_MW": os.storage_list[0].p_cap,
                "e_cap_MWh": os.storage_list[0].e_cap,
                "cost_MUSD": -os.a_npv,
                "npv_MUSD": os.npv,
                "cycles_efc": cycles_opt,
                "cycles_per_year": cycles_opt_per_year,
                "wind_mean_MW": float(np.mean(power_wind_MW)),
                "wind_peak_MW": float(np.max(power_wind_MW)),
                "rte_pct": rte_ac * 100.0,
                "solver": ("scipy" if pyo_solver == "none" else pyo_solver),
            },
            {
                "timestamp": timestamp,
                "hours_simulated": n,
                "days_simulated": n / 24.0,
                "optimization_type": "dispatch_fixed",
                "p_min_MW": p_min,
                "revenue_kUSD": os_fixed.revenue * 1e-3,
                "revenue_increase_pct": rev_inc_pct(os_fixed.revenue),
                "p_cap_MW": os_fixed.storage_list[0].p_cap,
                "e_cap_MWh": os_fixed.storage_list[0].e_cap,
                "cost_MUSD": -os_fixed.a_npv,
                "npv_MUSD": os_fixed.npv,
                "cycles_efc": cycles_fixed,
                "cycles_per_year": cycles_fixed_per_year,
                "wind_mean_MW": float(np.mean(power_wind_MW)),
                "wind_peak_MW": float(np.max(power_wind_MW)),
                "rte_pct": rte_ac * 100.0,
                "solver": ("scipy" if pyo_solver == "none" else pyo_solver),
            },
        ]
        append_results_csv(results_file, rows)
        print(f"\n✓ Results saved to: {results_file}")

    if SAVE_REPORT:
        report_file = RESULTS_DIR / f"battery_report_{run_ts}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("WP2 BATTERY OPTIMIZATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {timestamp}\n\n")

            f.write("SIMULATION PARAMETERS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Hours simulated: {n:,} ({n/24:.1f} days)\n")
            f.write(f"Solver: {('scipy sparse' if pyo_solver == 'none' else pyo_solver)}\n")
            f.write(f"Discount rate: {discount_rate:.1%}\n")
            f.write(f"Project duration: {n_year} years\n")
            f.write(f"Minimum power: {p_min:.1f} MW\n")
            f.write(f"Grid limit: {p_max:.0f} MW\n\n")

            f.write("WIND FARM\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean power: {np.mean(power_wind_MW):.1f} MW\n")
            f.write(f"Peak power: {np.max(power_wind_MW):.1f} MW\n")
            f.write(f"Grid utilization (mean/limit): {np.mean(power_wind_MW)/p_max*100:.1f}%\n\n")

            f.write("BATTERY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Energy capacity (nominal): {e_cap:.0f} MWh\n")
            f.write(f"Power capacity (nominal): {p_cap:.0f} MW\n")
            f.write(f"Round-trip efficiency (AC): {rte_ac*100:.1f}%\n")
            f.write(f"E/P ratio: {e_cap/p_cap:.2f} hours\n")
            f.write(f"CAPEX energy: {e_cost:.0f} USD/MWh\n")
            f.write(f"CAPEX power:  {p_cost:.0f} USD/MW\n\n")

            f.write("RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Wind-only baseline revenue: {revenues_res_only*1e-3:.1f} kUSD\n\n")

            f.write("Sizing optimization:\n")
            f.write(f"  Revenue: {os.revenue*1e-3:.1f} kUSD\n")
            f.write(f"  Revenue increase: {rev_inc_pct(os.revenue):.2f}%\n")
            f.write(f"  Optimal battery: {os.storage_list[0].p_cap:.1f} MW / {os.storage_list[0].e_cap:.1f} MWh\n")
            f.write(f"  Cost: {-os.a_npv:.2f} MUSD\n")
            f.write(f"  NPV: {os.npv:.2f} MUSD\n")
            f.write(f"  EFC in period: {cycles_opt:.2f} (≈ {cycles_opt_per_year:.0f}/year)\n\n")

            f.write("Fixed capacity dispatch:\n")
            f.write(f"  Fixed battery: {p_cap:.1f} MW / {e_cap:.1f} MWh\n")
            f.write(f"  Revenue: {os_fixed.revenue*1e-3:.1f} kUSD\n")
            f.write(f"  Revenue increase: {rev_inc_pct(os_fixed.revenue):.2f}%\n")
            f.write(f"  Cost: {-os_fixed.a_npv:.2f} MUSD\n")
            f.write(f"  NPV: {os_fixed.npv:.2f} MUSD\n")
            f.write(f"  EFC in period: {cycles_fixed:.2f} (≈ {cycles_fixed_per_year:.0f}/year)\n\n")

        print(f"✓ Detailed report saved to: {report_file}")

    # -------------------------------------------------------------------------
    # Plot (optional)
    # -------------------------------------------------------------------------
    if MAKE_PLOT:
        print("\nGenerating plots...")
        time_days = np.arange(n) * dt / 24.0

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].plot(time_days, power_wind_MW + os.storage_p[0].data, label="Power + opt. storage")
        ax[0].plot(time_days, power_wind_MW + os_fixed.storage_p[0].data, label="Power + fixed storage")
        ax[0].plot(time_days, power_wind_MW, label="Wind power")
        ax[0].set_xlabel("Time [days]")
        ax[0].set_ylabel("Power [MW]")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()

        ax[1].plot(time_days, os.storage_e[0].data, label="Opt. storage")
        ax[1].plot(time_days, os_fixed.storage_e[0].data, label="Fixed storage")
        ax[1].set_xlabel("Time [days]")
        ax[1].set_ylabel("State of charge [MWh]")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend()

        plt.tight_layout()
        out_png = PLOTS_DIR / f"battery_optimization_results_{run_ts}.png"
        plt.savefig(out_png, dpi=200)
        print(f"✓ Saved: {out_png.name}")
        plt.show()

    print("\n" + "=" * 80)
    print("✓ BASELINE COMPLETE - READY FOR DEGRADATION")
    print("=" * 80)


if __name__ == "__main__":
    main()
