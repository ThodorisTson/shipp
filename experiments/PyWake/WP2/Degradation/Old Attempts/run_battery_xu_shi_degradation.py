"""WP2 battery optimization + degradation (clean, baseline-style).

Goal
- Reuse the same "baseline_v2" flow (load -> horizon -> PyWake -> SHIPP).
- Then run degradation as a post-processing step for BOTH:
    1) sizing optimization (os)
    2) fixed-capacity dispatch (os_fixed)

Notes
- This file assumes you have:
    - WP2_HPP.yaml and dk1_prices_2022.csv/dk1_prices_2019.csv next to it
    - wp2_common.py providing quick_setup() and get_wake_model()
    - degradation_v_2.py providing analyze_degradation() + plotting/report helpers
- SHIPP convention: storage_p > 0 discharge, storage_p < 0 charge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shipp.kernel import solve_lp_sparse
from shipp.kernel_pyomo import solve_lp_pyomo
from shipp.components import Storage, Production, TimeSeries

from wp2_common import quick_setup, get_wake_model
from degradation_xu import (
    analyze_degradation,
    count_equivalent_full_cycles,
)

from degradation_xu import rainflow_cycle_counting
from degradation_subgradient import compute_subgradient, fit_shi_polynomial

from degradation_plots import (
    plot_degradation_analysis,
    print_degradation_report,
)

import xarray as xr
from py_wake.site import XRSite

import numpy_financial as npf

# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
HPP_YAML = SCRIPT_DIR / "WP2_HPP.yaml"
PRICE_CSV = SCRIPT_DIR / "dk1_prices_2022.csv"

# Output folders
RESULTS_DIR = SCRIPT_DIR / "Results"
PLOTS_DIR   = SCRIPT_DIR / "Degradation Plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

eur_to_usd = 1.18
discount_rate = 0.03
n_year = 20
dt = 1.0           # hours

# Solver:
#   'none'    => scipy sparse (fast but: no dod constraint, max ~6 months)
#   'gurobi'  => recommended for full-year runs with dod constraint
pyo_solver = "gurobi"

# Horizon control
RUN_FULL_YEAR = True
N_DAYS_TEST = 120
MAX_HOURS_SPARSE = 180 * 24  # 6-month cap for scipy

# Problem settings
p_min = 0.0        # 0 = no minimum export; required for feasibility with dod < 1
WAKE_MODEL = "Bastankhah"

# End-of-Life thresholds for degradation analysis and plots.
# 0.80 = IEC/EV industry convention (Xu et al. 2016 case study)
# 0.60 = typical grid-storage operational limit (stationary storage practice)
# Add or remove values to test other scenarios, e.g. [0.80, 0.70, 0.60]
eol_thresholds = [0.80, 0.70, 0.60]

# Output toggles
print_baseline_table = True
print_degr_reports   = True
SAVE_CSV    = True
SAVE_REPORT = True
MAKE_PLOT   = True
show_plots  = False

run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

def _build_run_label(ts: str, price_csv: Path, p_cap: float, e_cap: float) -> str:
    """Build a descriptive label for output filenames: {ts}_{dataset}_{pcap}mw_{ecap}mwh"""
    stem = price_csv.stem.lower()
    year = ''.join(filter(str.isdigit, stem))[-4:]
    dataset = f"dk{year}"
    bat = f"{int(round(p_cap))}mw_{int(round(e_cap))}mwh"
    return f"{ts}_{dataset}_{bat}"
# =============================================================================
# Small structs
# =============================================================================

@dataclass
class HorizonData:
    n: int
    ws: np.ndarray
    wd: np.ndarray
    ti: Optional[np.ndarray]
    price_eur: np.ndarray


# =============================================================================
# Helpers
# =============================================================================

def _find_price_column(df: pd.DataFrame) -> str:
    for c in ("price_eur_mwh", "price_eur_per_mwh", "Price", "price"):
        if c in df.columns:
            return c
    raise KeyError(f"No recognized price column. Columns: {df.columns.tolist()}")


def _choose_horizon(n_wind: int, n_price: int) -> int:
    requested = n_wind if RUN_FULL_YEAR else int(N_DAYS_TEST * 24)
    n = min(requested, n_wind, n_price)
    if pyo_solver == "none":
        n = min(n, MAX_HOURS_SPARSE)
    return int(n)


def _load_inputs(hpp: dict) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    ts = hpp["site"]["energy_resource"]["time_series"]["wind_resource"]
    ws = np.asarray(ts["wind_speed"], dtype=float)
    wd = np.asarray(ts["wind_direction"], dtype=float)

    ti = None
    ti_dat = ts.get("turbulence_intensity")
    if isinstance(ti_dat, dict) and "data" in ti_dat:
        ti = np.asarray(ti_dat["data"], dtype=float)

    if ws.shape != wd.shape:
        raise ValueError("Wind speed and wind direction arrays have different lengths.")
    if ti is not None and ti.shape != ws.shape:
        # Some datasets omit TI or provide a constant; we only support hourly TI arrays here
        ti = None

    return ws, wd, ti


def _load_prices() -> np.ndarray:
    df = pd.read_csv(PRICE_CSV)
    col = _find_price_column(df)
    return df[col].astype(float).to_numpy()


def _run_pywake_power_MW(setup: dict, wd: np.ndarray, ws: np.ndarray, ti: Optional[np.ndarray]) -> np.ndarray:
    n = len(ws)
    site = XRSite(ds=xr.Dataset(data_vars=dict(P=1)))
    wf_model = get_wake_model(WAKE_MODEL, site, setup["windturbine"])

    time_days = np.arange(n) / 24.0
    kwargs = {"x": setup["x"], "y": setup["y"], "wd": wd, "ws": ws, "time": time_days}
    if ti is not None:
        kwargs["TI"] = ti

    sim_res = wf_model(**kwargs)
    power_W = sim_res.Power.sum(["wt"]).values
    return power_W / 1e6


def _build_shipp_components(setup: dict, p_max_MW: float) -> Tuple[Storage, Storage, float, float, float]:
    bat = setup["battery"]

    e_cap_MWh = float(bat["energy_capacity_Wh"]) / 1e6
    p_cap_MW = float(bat["power_capacity_W"]) / 1e6

    rte_dc = float(bat["rte_nominal"])
    pcu_eff = float(bat["pcu_efficiency"])
    rte_ac = rte_dc * (pcu_eff ** 2)

    # Costs
    e_cost_USD_per_MWh = float(bat["capex_EUR_per_kWh"]) * 1000.0 * eur_to_usd
    p_cost_USD_per_MW = float(bat["capex_EUR_per_kW"]) * 1000.0 * eur_to_usd

    # solve_lp_sparse asserts dod==1.0; only enforce real dod with Pyomo solvers
    soc_min  = float(bat.get('soc_min', 0.10))
    soc_max  = float(bat.get('soc_max', 0.90))
    dod_yaml = 1.0 - soc_min    # floor: how far down from 100% the battery can go
    if pyo_solver == "none":
        dod_eff = 1.0
        print(f"  \u26a0 DoD ({dod_yaml:.0%}) ignored — scipy sparse requires dod=1.0. Use gurobi to enforce.")
    else:
        dod_eff = dod_yaml
    print(f"  SoC window: {soc_min*100:.0f}% – {soc_max*100:.0f}%  (DoD floor={dod_eff:.0%}, ceiling={soc_max:.0%}, enforced by {pyo_solver})")
    stor = Storage(
        e_cap=e_cap_MWh,
        p_cap=p_cap_MW,
        eff_in=1.0,
        eff_out=rte_ac,
        e_cost=e_cost_USD_per_MWh,
        p_cost=p_cost_USD_per_MW,
        dod=dod_eff,
    )
    stor_null = Storage(e_cap=0.0, p_cap=0.0, eff_in=1.0, eff_out=1.0, e_cost=0.0, p_cost=0.0)

    return stor, stor_null, e_cap_MWh, p_cap_MW, rte_ac, e_cost_USD_per_MWh, soc_min, soc_max


def _solve_shipp(
    price_eur: np.ndarray,
    power_wind_MW: np.ndarray,
    stor: Storage,
    stor_null: Storage,
    p_max_MW: float,
    n: int,
    soc_max: float = 1.0,
):
    price_dam = TimeSeries((price_eur * eur_to_usd).tolist(), dt)
    prod = Production(TimeSeries(power_wind_MW.tolist(), dt), p_cost=0.0)
    prod_null = Production(TimeSeries([0.0] * n, dt), p_cost=0.0)

    if pyo_solver == "none":
        os = solve_lp_sparse(price_dam, prod, prod_null, stor, stor_null, discount_rate, n_year, p_min, p_max_MW, n)
        os_fixed = solve_lp_sparse(price_dam, prod, prod_null, stor, stor_null, discount_rate, n_year, p_min, p_max_MW, n, fixed_cap=True)
    else:
        os = solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null, discount_rate, n_year, p_min, p_max_MW, n, pyo_solver, soc_max1=soc_max)
        os_fixed = solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null, discount_rate, n_year, p_min, p_max_MW, n, pyo_solver, fixed_cap=True, return_duals=True, soc_max1=soc_max)

    return os, os_fixed


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("WP2 BATTERY OPTIMIZATION + DEGRADATION (clean, baseline-style)")
    print("=" * 80)

    # 1) Load config + raw series
    print("\n[1/5] Loading WP2 configuration...")
    setup = quick_setup(HPP_YAML, config={"interp_n": 2000}, verbose=False)
    hpp = setup["hpp"]

    ws_all, wd_all, ti_all = _load_inputs(hpp)
    price_all = _load_prices()

    # 2) Choose horizon once and trim
    print("\n[2/5] Loading electricity prices...")
    n = _choose_horizon(len(ws_all), len(price_all))

    ws = ws_all[:n]
    wd = wd_all[:n]
    ti = ti_all[:n] if ti_all is not None else None
    price_eur = price_all[:n]

    print(f"  Horizon: {n:,} h ({n/24:.1f} days)")
    print(f"  Mean price: {float(np.mean(price_eur)):.2f} EUR/MWh")

    # 3) PyWake time series
    print("\n[3/5] Running PyWake time-series simulation...")
    power_wind_MW = _run_pywake_power_MW(setup, wd, ws, ti)
    print(f"  Wind mean: {float(np.mean(power_wind_MW)):.1f} MW | peak: {float(np.max(power_wind_MW)):.1f} MW")

    # 4) SHIPP
    print("\n[4/5] Running SHIPP optimization...")
    p_max_MW = float(hpp["grid_connection_capacity"]) / 1e6

    stor, stor_null, e_cap_MWh, p_cap_MW, rte_ac, e_cost_USD_per_MWh, soc_min, soc_max = _build_shipp_components(setup, p_max_MW)

    # Fit Shi polynomial over the actual YAML SoC window — used for gradient computation only.
    # Using the physical DoD ceiling (soc_max - soc_min) gives better R² than fitting over [0,1].
    shi_fit = fit_shi_polynomial(soc_min, soc_max, verbose=False)

    print(
        f"  Battery bounds: {p_cap_MW:.0f} MW / {e_cap_MWh:.0f} MWh | "
        f"Grid limit: {p_max_MW:.0f} MW | RTE(ac): {rte_ac*100:.1f}%"
    )
    print(f"  Solver: {('scipy sparse' if pyo_solver == 'none' else pyo_solver)}")

    run_label = _build_run_label(run_ts, PRICE_CSV, p_cap_MW, e_cap_MWh)

    os, os_fixed = _solve_shipp(price_eur, power_wind_MW, stor, stor_null, p_max_MW, n, soc_max)

    # Save SoC profiles for degradation_subgradient.py self-test
    np.save(RESULTS_DIR / "storage_e_fixed.npy", np.array(os_fixed.storage_e[0].data))
    np.save(RESULTS_DIR / "e_cap_fixed.npy",     np.array([os_fixed.storage_list[0].e_cap]))

    # Baseline metrics
    revenues_res_only = 365.0 * 24.0 / n * np.dot(price_eur, np.minimum(power_wind_MW, p_max_MW)) * dt
    os.get_added_npv(discount_rate, n_year)
    os_fixed.get_added_npv(discount_rate, n_year)

    period_days = n * dt / 24.0

    def _rev_inc_pct(rev: float) -> float:
        return 100.0 * (rev / revenues_res_only - 1.0)

    cycles_opt = count_equivalent_full_cycles(os.storage_p[0].data, os.storage_e[0].data, os.storage_list[0].e_cap, dt_hours=dt)
    cycles_fixed = count_equivalent_full_cycles(os_fixed.storage_p[0].data, os_fixed.storage_e[0].data, os_fixed.storage_list[0].e_cap, dt_hours=dt)

    if print_baseline_table:
        print("\n" + "=" * 80)
        print("BASELINE RESULTS")
        print("=" * 80)
        print("                Revenue [kUSD]  Rev. increase    p_cap/e_cap         NPV [M.USD]   Cycles/year")
        print("-" * 90)
        print(
            f"Sizing Opt.     {os.annual_revenue*1e-3:>10.1f}      {_rev_inc_pct(os.annual_revenue):>8.2f}%   "
            f"{os.storage_list[0].p_cap:>7.2f}/{os.storage_list[0].e_cap:<7.2f}   "
            f"{os.npv:>10.1f}      {cycles_opt/period_days*365:>8.0f}"
        )
        print(
            f"Dispatch only   {os_fixed.annual_revenue*1e-3:>10.1f}      {_rev_inc_pct(os_fixed.annual_revenue):>8.2f}%   "
            f"{os_fixed.storage_list[0].p_cap:>7.2f}/{os_fixed.storage_list[0].e_cap:<7.2f}   "
            f"{os_fixed.npv:>10.1f}      {cycles_fixed/period_days*365:>8.0f}"
        )
        print("-" * 90)

    # -----------------------------------------------------------------------
    # 5) Degradation analysis
    # -----------------------------------------------------------------------
    print("\n[5/5] Degradation analysis...")
    bat_params = setup["battery"]

    # Dispatch-fixed: always has a battery
    degr_fixed = analyze_degradation(
        storage_p=os_fixed.storage_p[0].data,
        storage_e=os_fixed.storage_e[0].data,
        e_cap_nominal=float(os_fixed.storage_list[0].e_cap),
        battery_params=bat_params,
        dt_hours=dt,
        enable_rainflow=True,
        T_cell_C=25.0,
        eol_thresholds=eol_thresholds,
    )

    # Sizing opt: guard — 2019 prices often return no battery as optimal
    opt_e_cap = float(os.storage_list[0].e_cap)
    no_battery = opt_e_cap < 1.0
    if no_battery:
        print(
            f"\n  \u26a0  Sizing optimizer chose no battery  "
            f"(e_cap = {opt_e_cap:.3f} MWh < 1 MWh).\n"
            f"     Battery is not economically viable at this price level.\n"
            f"     Degradation analysis for sizing-opt case skipped."
        )
        degr_opt = None
    else:
        degr_opt = analyze_degradation(
            storage_p=os.storage_p[0].data,
            storage_e=os.storage_e[0].data,
            e_cap_nominal=opt_e_cap,
            battery_params=bat_params,
            dt_hours=dt,
            enable_rainflow=True,
            T_cell_C=25.0,
            eol_thresholds=eol_thresholds,
        )

    # -----------------------------------------------------------------------
    # Gap D verification — dual prices + full gradient chain rule
    # -----------------------------------------------------------------------


    if os_fixed.dual_prices is not None:
        print("\n" + "=" * 60)
        print("GAP D VERIFICATION — dual prices + gradient signal")
        print("=" * 60)

        dual  = os_fixed.dual_prices["dual_e_min1"]     # shape (8760,)
        e_cap = os_fixed.dual_prices["e_cap1"]          # 300.0 MWh

        print(f"\n  dual_e_min1  : min={dual.min():.4e}  max={dual.max():.4e}"
              f"  mean={dual.mean():.4e}")
        print(f"  n_nonzero    : {np.sum(dual != 0)} / {len(dual)} timesteps")
        print(f"  e_cap1       : {e_cap:.1f} MWh")

        # Recompute subgradient (already done inside degradation but we need it here)
        cycles = rainflow_cycle_counting(
            os_fixed.storage_e[0].data, e_cap
        )
        sg = compute_subgradient(
            storage_e=os_fixed.storage_e[0].data,
            cycles=cycles,
            dt_hours=dt,
            battery_replacement_cost_per_MWh=e_cost_USD_per_MWh,
            eff_in=1.0,
            eff_out=rte_ac,
            shi_fit=shi_fit,
        )

        # Chain rule: dDeg/dDoD = -e_cap * dot(subgrad_combined, dual_e_min1)
        factor = npf.npv(discount_rate, np.ones(n_year)) - 1  # same factor as in kernel_pyomo
        dual_per_year = dual / factor
        grad_deg_dod = -e_cap * float(np.dot(sg["subgrad_combined"], dual_per_year))

        print(f"\n  subgrad_combined range : [{sg['subgrad_combined'].min():.4e},"
              f"  {sg['subgrad_combined'].max():.4e}]")
        print(f"\n  dDeg/dDoD (chain rule) : {grad_deg_dod:.6e}")
        print(f"  Interpretation: a 1% DoD increase changes degradation cost"
              f" by {grad_deg_dod * 0.01:.4e} USD")
        print("=" * 60)
    else:
        print("\n  WARNING: dual_prices is None — dual extraction failed.")

    # -----------------------------------------------------------------------
    # Console output
    # -----------------------------------------------------------------------
    if print_degr_reports:
        def _print_degr_block(degr: dict, label: str) -> None:
            W = 72
            print("\n" + "\u2554" + "\u2550" * W + "\u2557")
            print(f"\u2551  DEGRADATION \u2014 {label:<{W-2}}\u2551")
            print("\u255a" + "\u2550" * W + "\u255d")
            print_degradation_report(degr, period_days=period_days, enabled=True)

            # Hourly discharge level distribution (bar chart in terminal)
            dod_bins, dod_counts = degr["dod_distribution"]
            total_h = max(int(np.sum(dod_counts)), 1)
            print("\n  Hourly discharge level distribution")
            print("  (how many hours the battery sat at each discharge level):")
            for b, c in zip(dod_bins, dod_counts):
                if c > 0:
                    bar = "\u2588" * max(1, int(c / total_h * 35))
                    print(f"    {b*100:5.1f}% discharged : {int(c):>5d} h  "
                          f"({int(c)/total_h*100:5.1f}%)  {bar}")

        _print_degr_block(degr_fixed, "DISPATCH ONLY \u2014 fixed 150 MW / 300 MWh")
        if degr_opt is not None:
            _print_degr_block(degr_opt, "SIZING OPTIMIZATION \u2014 optimal capacity")
        else:
            print("\n  [Sizing-opt degradation skipped \u2014 no battery installed]")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if SAVE_CSV:
        base_csv = RESULTS_DIR / f"battery_optimization_results_{run_label}.csv"
        degr_csv = RESULTS_DIR / f"battery_degradation_results_{run_label}.csv"

        base_rows = [
            {
                "timestamp":            timestamp,
                "hours_simulated":      n,
                "days_simulated":       period_days,
                "optimization_type":    "sizing",
                "revenue_kUSD":         os.annual_revenue * 1e-3,
                "revenue_increase_pct": _rev_inc_pct(os.annual_revenue),
                "p_cap_MW":             os.storage_list[0].p_cap,
                "e_cap_MWh":            os.storage_list[0].e_cap,
                "npv_MUSD":             os.npv,
                "cycles_per_year":      cycles_opt / period_days * 365.0,
                "solver":               ("scipy" if pyo_solver == "none" else pyo_solver),
                "no_battery":           no_battery,
            },
            {
                "timestamp":            timestamp,
                "hours_simulated":      n,
                "days_simulated":       period_days,
                "optimization_type":    "dispatch_fixed",
                "revenue_kUSD":         os_fixed.annual_revenue * 1e-3,
                "revenue_increase_pct": _rev_inc_pct(os_fixed.annual_revenue),
                "p_cap_MW":             os_fixed.storage_list[0].p_cap,
                "e_cap_MWh":            os_fixed.storage_list[0].e_cap,
                "npv_MUSD":             os_fixed.npv,
                "cycles_per_year":      cycles_fixed / period_days * 365.0,
                "solver":               ("scipy" if pyo_solver == "none" else pyo_solver),
                "no_battery":           False,
            },
        ]
        pd.DataFrame(base_rows).to_csv(base_csv, mode="a", header=not base_csv.exists(), index=False)

        def _degr_row(d: dict, case: str) -> dict:
            stats = d.get("xu_cycle_stats", {})
            return {
                "timestamp":          timestamp,
                "hours_simulated":    n,
                "days_simulated":     period_days,
                "case":               case,
                "total_cycles_efc":   d["total_cycles"],
                "cycles_per_year":    d["total_cycles"] / period_days * 365.0,
                "fd_total":           d["fd"],
                "fd_cycle":           d["fd_cycle"],
                "fd_calendar":        d["fd_calendar"],
                "soh_pct":            d["soh"],
                "capacity_fade_pct":  d["capacity_fade_percent"],
                "e_cap_degraded_MWh": d["e_cap_degraded"],
                "p_cap_degraded_MW":  d["p_cap_degraded"],
                "mean_dod_pct":       stats.get("mean_dod", 0) * 100,
                "mean_soc_pct":       stats.get("mean_soc", 0) * 100,
                "n_rainflow_cycles":  stats.get("n_rainflow_cycles", 0),
                "model":              d["meta"].get("model", "Xu2016"),
            }

        degr_rows = [_degr_row(degr_fixed, "dispatch_fixed")]
        if degr_opt is not None:
            degr_rows.append(_degr_row(degr_opt, "sizing_opt"))
        pd.DataFrame(degr_rows).to_csv(degr_csv, mode="a", header=not degr_csv.exists(), index=False)
        print(f"\n  \u2713 CSV: {base_csv.name}")
        print(f"  \u2713 CSV: {degr_csv.name}")

    # -----------------------------------------------------------------------
    # Save text report
    # -----------------------------------------------------------------------
    if SAVE_REPORT:
        report_path = RESULTS_DIR / f"degradation_report_{run_label}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("WP2 BATTERY OPTIMIZATION + DEGRADATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated : {timestamp}\n")
            f.write(f"Dataset   : {PRICE_CSV.name}\n")
            f.write(f"Horizon   : {n:,} h  ({period_days:.1f} days)\n")
            f.write(f"Solver    : {('scipy sparse' if pyo_solver == 'none' else pyo_solver)}\n")
            f.write(f"Grid limit: {p_max_MW:.0f} MW\n\n")
            f.write(f"Mean price: {float(np.mean(price_eur)):.2f} EUR/MWh\n")
            f.write(f"Wind mean : {float(np.mean(power_wind_MW)):.1f} MW  |  "
                    f"peak: {float(np.max(power_wind_MW)):.1f} MW\n\n")

            f.write("OPTIMIZATION RESULTS\n" + "-" * 80 + "\n")
            f.write(f"Wind-only revenue : {revenues_res_only*1e-3:.1f} kUSD/yr\n")
            f.write(f"Sizing opt revenue: {os.annual_revenue*1e-3:.1f} kUSD/yr  |  "
                    f"NPV: {os.npv:.1f} MUSD\n")
            f.write(f"  Battery chosen  : {os.storage_list[0].p_cap:.1f} MW / "
                    f"{os.storage_list[0].e_cap:.1f} MWh")
            if no_battery:
                f.write("  <- optimizer chose NO battery (not economically viable at this price level)\n")
            else:
                f.write("\n")
            f.write(f"Dispatch revenue  : {os_fixed.annual_revenue*1e-3:.1f} kUSD/yr  |  "
                    f"NPV: {os_fixed.npv:.1f} MUSD\n\n")

            pairs = [("dispatch_fixed", degr_fixed)]
            if degr_opt is not None:
                pairs.append(("sizing_opt", degr_opt))
            for name, d in pairs:
                stats = d.get("xu_cycle_stats", {})
                f.write(f"DEGRADATION ({name}) \u2014 Xu (2016) LMO model\n" + "-" * 80 + "\n")
                f.write(f"EFC           : {d['total_cycles']:.2f}  "
                        f"(\u2248 {d['total_cycles']/period_days*365:.0f}/yr)\n")
                f.write(f"Rainflow cyc  : {stats.get('n_rainflow_cycles', 0):.1f}\n")
                f.write(f"Mean cycle DoD: {stats.get('mean_dod', 0)*100:.1f}%\n")
                f.write(f"Mean cycle SoC: {stats.get('mean_soc', 0)*100:.1f}%\n")
                f.write(f"fd_total      : {d['fd']:.6f}\n")
                f.write(f"  fd_cycle    : {d['fd_cycle']:.6f}  "
                        f"({100*d['fd_cycle']/max(d['fd'],1e-30):.0f}%)\n")
                f.write(f"  fd_calendar : {d['fd_calendar']:.6f}  "
                        f"({100*d['fd_calendar']/max(d['fd'],1e-30):.0f}%)\n")
                f.write(f"SoH           : {d['soh']:.3f}%  |  Fade: {d['capacity_fade_percent']:.4f}%\n")
                f.write(f"Degraded E    : {d['e_cap_degraded']:.3f} MWh  |  "
                        f"P: {d['p_cap_degraded']:.3f} MW\n\n")

        print(f"  \u2713 Report: {report_path.name}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    if MAKE_PLOT:
        time_vec = np.arange(n) * dt / 24.0

        # Power export + SoC overview
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(time_vec, power_wind_MW + os_fixed.storage_p[0].data,
                   linewidth=0.7, label="Wind + battery export")
        ax[0].plot(time_vec, power_wind_MW,
                   linewidth=0.7, alpha=0.6, label="Wind only")
        ax[0].axhline(p_max_MW, linestyle="--", alpha=0.5,
                      label=f"Grid limit ({p_max_MW:.0f} MW)")
        ax[0].set_xlabel("Time [days]")
        ax[0].set_ylabel("Power [MW]")
        ax[0].set_title("Dispatch-Fixed: Power Export Profile")
        ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)

        ax[1].plot(time_vec, os_fixed.storage_e[0].data, linewidth=0.7)
        ax[1].set_xlabel("Time [days]")
        ax[1].set_ylabel("SoC [MWh]")
        ax[1].set_title("Dispatch-Fixed: Battery State of Charge")
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"battery_baseline_results_{run_label}.png", dpi=200)

        # Degradation detail — dispatch fixed (always)
        plot_degradation_analysis(
            degr_fixed,
            storage_e=os_fixed.storage_e[0].data,
            time_vec=time_vec,
            save_path=str(PLOTS_DIR / f"battery_degradation_analysis_fixed_{run_label}.png"),
            show=False,
            verbose=True,
            eol_thresholds=eol_thresholds,
        )

        # Degradation detail — sizing opt (only if a battery was installed)
        if degr_opt is not None:
            plot_degradation_analysis(
                degr_opt,
                storage_e=os.storage_e[0].data,
                time_vec=time_vec,
                save_path=str(PLOTS_DIR / f"battery_degradation_analysis_sizing_{run_label}.png"),
                show=False,
                verbose=True,
                eol_thresholds=eol_thresholds,
            )
        else:
            print("  [Sizing-opt plot skipped \u2014 no battery installed]")

        if show_plots:
            plt.show()
        else:
            plt.close("all")

    print("\n" + "=" * 80)
    print("\u2713 COMPLETE \u2014 BASELINE + DEGRADATION")
    print("=" * 80)


if __name__ == "__main__":
    main()