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
from degradation_v_2 import (
    analyze_degradation,
    plot_degradation_analysis,
    print_degradation_report,
    count_equivalent_full_cycles,
)

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
PLOTS_DIR   = SCRIPT_DIR / "Degradation Plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

EUR_TO_USD = 1.18
DISCOUNT_RATE = 0.03
N_YEAR = 20
DT_H = 1.0

# Solver:
#   "none" => scipy sparse (fast to set up, but limited horizon)
#   else   => pyomo solver name, e.g. "appsi_highs" or "gurobi"
PYO_SOLVER = "none"

# Horizon control
RUN_FULL_YEAR = False
N_DAYS_TEST = 120
MAX_HOURS_SPARSE = 180 * 24  # 6 months

# Problem settings
P_MIN = 15.0
WAKE_MODEL = "Bastankhah"

# Output toggles
PRINT_BASELINE_TABLE = True
PRINT_DEGR_REPORTS = True
SAVE_CSV = True
SAVE_REPORT_TXT = True
MAKE_PLOTS = True
SHOW_PLOTS = True

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
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
    if PYO_SOLVER == "none":
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
    e_cost_USD_per_MWh = float(bat["capex_EUR_per_kWh"]) * 1000.0 * EUR_TO_USD
    p_cost_USD_per_MW = float(bat["capex_EUR_per_kW"]) * 1000.0 * EUR_TO_USD

    stor = Storage(
        e_cap=e_cap_MWh,
        p_cap=p_cap_MW,
        eff_in=1.0,
        eff_out=rte_ac,
        e_cost=e_cost_USD_per_MWh,
        p_cost=p_cost_USD_per_MW,
    )
    stor_null = Storage(e_cap=0.0, p_cap=0.0, eff_in=1.0, eff_out=1.0, e_cost=0.0, p_cost=0.0)

    return stor, stor_null, e_cap_MWh, p_cap_MW, rte_ac


def _solve_shipp(
    price_eur: np.ndarray,
    power_wind_MW: np.ndarray,
    stor: Storage,
    stor_null: Storage,
    p_max_MW: float,
    n: int,
):
    price_dam = TimeSeries((price_eur * EUR_TO_USD).tolist(), DT_H)
    prod = Production(TimeSeries(power_wind_MW.tolist(), DT_H), p_cost=0.0)
    prod_null = Production(TimeSeries([0.0] * n, DT_H), p_cost=0.0)

    if PYO_SOLVER == "none":
        os = solve_lp_sparse(price_dam, prod, prod_null, stor, stor_null, DISCOUNT_RATE, N_YEAR, P_MIN, p_max_MW, n)
        os_fixed = solve_lp_sparse(price_dam, prod, prod_null, stor, stor_null, DISCOUNT_RATE, N_YEAR, P_MIN, p_max_MW, n, fixed_cap=True)
    else:
        os = solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null, DISCOUNT_RATE, N_YEAR, P_MIN, p_max_MW, n, PYO_SOLVER)
        os_fixed = solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null, DISCOUNT_RATE, N_YEAR, P_MIN, p_max_MW, n, PYO_SOLVER, fixed_cap=True)

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

    stor, stor_null, e_cap_MWh, p_cap_MW, rte_ac = _build_shipp_components(setup, p_max_MW)

    print(
        f"  Battery bounds: {p_cap_MW:.0f} MW / {e_cap_MWh:.0f} MWh | "
        f"Grid limit: {p_max_MW:.0f} MW | RTE(ac): {rte_ac*100:.1f}%"
    )
    print(f"  Solver: {('scipy sparse' if PYO_SOLVER == 'none' else PYO_SOLVER)}")

    os, os_fixed = _solve_shipp(price_eur, power_wind_MW, stor, stor_null, p_max_MW, n)

    # Baseline metrics
    revenues_res_only = 365.0 * 24.0 / n * np.dot(price_eur, np.minimum(power_wind_MW, p_max_MW)) * DT_H
    os.get_added_npv(DISCOUNT_RATE, N_YEAR)
    os_fixed.get_added_npv(DISCOUNT_RATE, N_YEAR)

    period_days = n * DT_H / 24.0

    def _rev_inc_pct(rev: float) -> float:
        return 100.0 * (rev / revenues_res_only - 1.0)

    cycles_opt = count_equivalent_full_cycles(os.storage_p[0].data, os.storage_e[0].data, os.storage_list[0].e_cap, dt_hours=DT_H)
    cycles_fixed = count_equivalent_full_cycles(os_fixed.storage_p[0].data, os_fixed.storage_e[0].data, os_fixed.storage_list[0].e_cap, dt_hours=DT_H)

    if PRINT_BASELINE_TABLE:
        print("\n" + "=" * 80)
        print("BASELINE RESULTS")
        print("=" * 80)
        print("                Revenue [kUSD]  Rev. increase    p_cap/e_cap         NPV [M.USD]   Cycles/year")
        print("-" * 90)
        print(
            f"Sizing Opt.     {os.revenue*1e-3:>10.1f}      {_rev_inc_pct(os.revenue):>8.2f}%   "
            f"{os.storage_list[0].p_cap:>7.2f}/{os.storage_list[0].e_cap:<7.2f}   "
            f"{os.npv:>10.1f}      {cycles_opt/period_days*365:>8.0f}"
        )
        print(
            f"Dispatch only   {os_fixed.revenue*1e-3:>10.1f}      {_rev_inc_pct(os_fixed.revenue):>8.2f}%   "
            f"{os_fixed.storage_list[0].p_cap:>7.2f}/{os_fixed.storage_list[0].e_cap:<7.2f}   "
            f"{os_fixed.npv:>10.1f}      {cycles_fixed/period_days*365:>8.0f}"
        )
        print("-" * 90)

    # 5) Degradation
    print("\n[5/5] Degradation analysis...")
    bat_params = setup["battery"]

    degr_fixed = analyze_degradation(
        storage_p=os_fixed.storage_p[0].data,
        storage_e=os_fixed.storage_e[0].data,
        e_cap_nominal=float(os_fixed.storage_list[0].e_cap),
        battery_params=bat_params,
        dt_hours=DT_H,
        enable_rainflow=True,
    )
    # DEBUG — remove after checking
    print("DEBUG degr keys:", list(degr_fixed.keys()))

    degr_opt = analyze_degradation(
        storage_p=os.storage_p[0].data,
        storage_e=os.storage_e[0].data,
        e_cap_nominal=float(os.storage_list[0].e_cap),
        battery_params=bat_params,
        dt_hours=DT_H,
        enable_rainflow=True,
    )

    if PRINT_DEGR_REPORTS:
        def _print_dod_detail(degr: dict, label: str) -> None:
            print("\n" + "=" * 72)
            print(f"DEGRADATION REPORT — {label}")
            print("=" * 72)
            print_degradation_report(degr, period_days=period_days, enabled=True)
            # dod_distribution is stored as a tuple (bin_centers, counts)
            dod_bins, dod_counts = degr["dod_distribution"]
            total_h = max(int(np.sum(dod_counts)), 1)
            print("  DoD distribution (bin center → hourly frequency):")
            for b, c in zip(dod_bins, dod_counts):
                if c > 0:
                    print(f"    DoD {b*100:5.1f}%: {int(c):>5d} h  ({int(c) / total_h * 100:.1f}%)")

        _print_dod_detail(degr_fixed, "DISPATCH ONLY (fixed capacity)")
        _print_dod_detail(degr_opt,   "SIZING OPTIMIZATION (optimal capacity)")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if SAVE_CSV:
        base_csv = RESULTS_DIR / "battery_optimization_results.csv"
        degr_csv = RESULTS_DIR / "battery_degradation_results.csv"

        base_rows = [
            {
                "timestamp": timestamp,
                "hours_simulated": n,
                "days_simulated": period_days,
                "optimization_type": "sizing",
                "revenue_kUSD": os.revenue * 1e-3,
                "revenue_increase_pct": _rev_inc_pct(os.revenue),
                "p_cap_MW": os.storage_list[0].p_cap,
                "e_cap_MWh": os.storage_list[0].e_cap,
                "npv_MUSD": os.npv,
                "cycles_per_year": cycles_opt / period_days * 365.0,
                "solver": ("scipy" if PYO_SOLVER == "none" else PYO_SOLVER),
            },
            {
                "timestamp": timestamp,
                "hours_simulated": n,
                "days_simulated": period_days,
                "optimization_type": "dispatch_fixed",
                "revenue_kUSD": os_fixed.revenue * 1e-3,
                "revenue_increase_pct": _rev_inc_pct(os_fixed.revenue),
                "p_cap_MW": os_fixed.storage_list[0].p_cap,
                "e_cap_MWh": os_fixed.storage_list[0].e_cap,
                "npv_MUSD": os_fixed.npv,
                "cycles_per_year": cycles_fixed / period_days * 365.0,
                "solver": ("scipy" if PYO_SOLVER == "none" else PYO_SOLVER),
            },
        ]
        pd.DataFrame(base_rows).to_csv(base_csv, mode="a", header=not base_csv.exists(), index=False)

        degr_rows = [
            {
                "timestamp": timestamp,
                "hours_simulated": n,
                "days_simulated": period_days,
                "case": "dispatch_fixed",
                "total_cycles": degr_fixed["total_cycles"],
                "cycles_per_year": degr_fixed["total_cycles"] / period_days * 365.0,
                "soh_pct": degr_fixed["soh"],
                "capacity_fade_pct": degr_fixed["capacity_fade_percent"],
                "e_cap_degraded_MWh": degr_fixed["e_cap_degraded"],
                "p_cap_degraded_MW": degr_fixed["p_cap_degraded"],
            },
            {
                "timestamp": timestamp,
                "hours_simulated": n,
                "days_simulated": period_days,
                "case": "sizing_opt",
                "total_cycles": degr_opt["total_cycles"],
                "cycles_per_year": degr_opt["total_cycles"] / period_days * 365.0,
                "soh_pct": degr_opt["soh"],
                "capacity_fade_pct": degr_opt["capacity_fade_percent"],
                "e_cap_degraded_MWh": degr_opt["e_cap_degraded"],
                "p_cap_degraded_MW": degr_opt["p_cap_degraded"],
            },
        ]
        pd.DataFrame(degr_rows).to_csv(degr_csv, mode="a", header=not degr_csv.exists(), index=False)

        print(f"  ✓ Saved CSV: {base_csv.name} and {degr_csv.name}")

    if SAVE_REPORT_TXT:
        report_path = RESULTS_DIR / f"degradation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("WP2 BASELINE + DEGRADATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {timestamp}\n\n")
            f.write(f"Hours: {n:,} | Days: {period_days:.1f}\n")
            f.write(f"Solver: {('scipy sparse' if PYO_SOLVER == 'none' else PYO_SOLVER)}\n")
            f.write(f"Grid limit: {p_max_MW:.0f} MW\n\n")
            f.write(f"Mean price: {float(np.mean(price_eur)):.2f} EUR/MWh\n")
            f.write(f"Wind mean: {float(np.mean(power_wind_MW)):.1f} MW | peak: {float(np.max(power_wind_MW)):.1f} MW\n\n")

            f.write("BASELINE\n" + "-" * 80 + "\n")
            f.write(f"Wind-only revenue: {revenues_res_only*1e-3:.1f} kUSD\n")
            f.write(f"Sizing rev: {os.revenue*1e-3:.1f} kUSD | NPV: {os.npv:.1f} MUSD\n")
            f.write(f"Fixed  rev: {os_fixed.revenue*1e-3:.1f} kUSD | NPV: {os_fixed.npv:.1f} MUSD\n\n")

            for name, d in (("dispatch_fixed", degr_fixed), ("sizing_opt", degr_opt)):
                f.write(f"DEGRADATION ({name})\n" + "-" * 80 + "\n")
                f.write(f"Cycles: {d['total_cycles']:.2f} (≈ {d['total_cycles']/period_days*365:.0f}/yr)\n")
                f.write(f"SoH: {d['soh']:.2f}% | Fade: {d['capacity_fade_percent']:.3f}%\n")
                f.write(f"Degraded E: {d['e_cap_degraded']:.2f} MWh | Degraded P: {d['p_cap_degraded']:.2f} MW\n\n")

        print(f"  ✓ Saved report: {report_path.name}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if MAKE_PLOTS:
        time_vec = np.arange(n) * DT_H / 24.0

        # Baseline plot (fixed)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(time_vec, power_wind_MW + os_fixed.storage_p[0].data, label="Wind + battery export")
        ax[0].plot(time_vec, power_wind_MW, label="Wind", alpha=0.6)
        ax[0].axhline(p_max_MW, linestyle="--", alpha=0.5, label="Grid limit")
        ax[0].set_xlabel("Time [days]")
        ax[0].set_ylabel("Power [MW]")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(time_vec, os_fixed.storage_e[0].data, label="SOC (fixed)")
        ax[1].set_xlabel("Time [days]")
        ax[1].set_ylabel("SOC [MWh]")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"battery_baseline_results_{ts}.png", dpi=200)

        # Degradation plots
        plot_degradation_analysis(
            degr_fixed,
            storage_e=os_fixed.storage_e[0].data,
            time_vec=time_vec,
            save_path=str(PLOTS_DIR / f"battery_degradation_analysis_fixed_{ts}.png"),
            show=False,
        )
        plot_degradation_analysis(
            degr_opt,
            storage_e=os.storage_e[0].data,
            time_vec=time_vec,
            save_path=str(PLOTS_DIR / f"battery_degradation_analysis_sizing_{ts}.png"),
            show=False,
        )

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close("all")

    print("\n" + "=" * 80)
    print("✓ COMPLETE - BASELINE + DEGRADATION")
    print("=" * 80)


if __name__ == "__main__":
    main()