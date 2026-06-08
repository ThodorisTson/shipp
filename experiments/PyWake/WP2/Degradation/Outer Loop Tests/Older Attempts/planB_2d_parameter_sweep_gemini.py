"""planB_2d_parameter_sweep_gemini.py

This script executes a 2D parameter sweep over both battery energy capacity (E_cap) 
and power inverter capacity (P_cap) to find the optimal co-located asset sizing 
under multi-model degradation regimes (No-Degradation baseline, Xu, and Shi).

What the Code Does:
-------------------
1. Loads historical wind generation profiles and spot market electricity prices.
2. Loops through a 2D grid of E_cap and P_cap values, using an execution guard to bypass 
   unphysical or uneconomic durations (P > E or duration > 8 hours) with a fast 
   neighbor-cloning fallback to maintain heatmap continuity without slowing down.
3. Formulates and solves a Pyomo linear program (LP) via Gurobi at each active sizing coordinate 
   to maximize annual arbitrage and market dispatch revenues.
4. Post-processes the resulting hourly battery dispatch profiles through non-linear degradation 
   equations to track cumulative capacity loss (State of Health fade).
5. Supports both 'annual' mode (fast 1-year extrapolation) and 'lifetime' mode (sequential 
   20-year multi-loop tracking with uneven battery replacement schedules).

The 4 Portfolio Outputs:
------------------------
All data and images are timestamped and saved into the 'Plan B Results' directory:
1. planB_2d_npv_comparison_[ts].png : Three side-by-side absolute NPV heatmaps showing 
   where the optimal asset investment coordinates land (marked by colored stars).
2. planB_2d_margins_[ts].png        : Two heatmaps displaying Financial Deltas (the total wealth 
   destroyed by degradation, and the model divergence risk gap between Xu and Shi).
3. planB_2d_sweep_[ts].png          : Degradation-focused maps tracking Fractional Degradation (fd) 
   limits and annual Equivalent Full Cycles (EFC).
4. planB_2d_slices_[ts].png         : Linear 1D cross-section slices cut through the peak 
   coordinates to highlight how steeply NPV drops off away from the optimum.

Usage:
    python planB_2d_parameter_sweep_gemini.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ─── Path setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "Plan B Results"
OUTPUT_DIR.mkdir(exist_ok=True)

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import numpy as np
import numpy_financial as npf
import pyomo.environ as pyo

from shipp.kernel_pyomo import solve_lp_pyomo
from shipp.components import Storage, Production, TimeSeries

from degradation_xu import (
    rainflow_cycle_counting,
    analyze_degradation,
    ft_calendar,
    sei_capacity_loss,
    count_equivalent_full_cycles,
)
from degradation_shi import analyze_degradation_shi
from degradation_subgradient import fit_shi_polynomial

HPP_YAML  = PARENT_DIR / "WP2_HPP.yaml"
PRICE_CSV = PARENT_DIR / "dk1_prices_2022.csv"


# ════════════════════════════════════════════════════════════════════════════
# 0.  QUICK TOGGLES
# ════════════════════════════════════════════════════════════════════════════

# Expanded, high-resolution grid parameters

# E_CAP_GRID = [150, 300, 450, 600, 700, 800, 900, 1000, 1100, 1300, 1500] # 11 points
# P_CAP_GRID = [75, 125, 175, 200, 225, 250, 275, 325]                   # 8 points

E_CAP_GRID = [450, 800, 1100, 1500]   # diagnostic: 4 points across the curtailment range
P_CAP_GRID = [225]                      # one P; pairs with each E

# ── Re-plot from a previous CSV run (skip the LP sweep entirely) ─────────
# Set REPLOT_FROM_LAST = True to automatically find and load the most recent
# CSV in the output folder — no need to type a filename.
# Set REPLOT_CSV to an explicit path only if you want a specific older run.
# If both are set, REPLOT_CSV takes priority.
# Leave both at their defaults to run the full sweep normally.
REPLOT_FROM_LAST: bool = False
REPLOT_CSV: str | None = None

# Include calendar aging in the degradation evaluation?
INCLUDE_CALENDAR = True

# ── Sweep mode ────────────────────────────────────────────────────────────
# "annual"   : one LP per (E, P) point, single-year dispatch, degradation
#              post-processed, 20-year NPV via annuity factor (fast, ~5 min)
# "lifetime" : 20 LPs per (E, P) point, capacity degrades year-on-year,
#              replacement triggered when SoH < REPL_THRESHOLD, three parallel
#              loops (no-deg / Xu / Shi), ~20× slower than annual
SWEEP_MODE: str = "annual"   # change to "lifetime" to run the 20-year version

# Battery replacement SoH threshold for lifetime mode
REPL_THRESHOLD: float = 0.70   # replace when capacity retention drops below 70%

# ── Lifetime smart filtering ─────────────────────────────────────────────
# Runs the fast annual sweep first, then skips lifetime evaluation for
# (E, P) points that are clearly non-competitive based on their 1-year NPV.
# A point is kept if its annual NPV (Xu OR Shi) exceeds this fraction of
# the best annual NPV for that model.  0.65 is safe — the v5.2 lifetime
# optimum at 800 MWh had ~90% of the annual optimum's NPV at that point.
LIFETIME_NPV_FLOOR: float = 0.65

# Duration guard for lifetime mode (same logic as annual smart filter)
LIFETIME_MIN_DURATION_H: float = 1.0
LIFETIME_MAX_DURATION_H: float = 8.0

# Horizons
RUN_HOURS  = 8760         # full year
SKIP_PLOT  = False

# Economics
EUR_TO_USD    = 1.18
DISCOUNT_RATE = 0.03
N_YEARS       = 20
DT            = 1.0
P_MIN_MW      = 0.0


# ════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_data(n_hours: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load wind power and price data. Returns (wind_MW, price_eur, params)."""
    import pandas as pd
    import xarray as xr
    from py_wake.site import XRSite
    from wp2_common import quick_setup, get_wake_model

    print("  Loading WP2 configuration...")
    setup = quick_setup(HPP_YAML, config={"interp_n": 2000}, verbose=False)
    hpp = setup["hpp"]

    ts = hpp["site"]["energy_resource"]["time_series"]["wind_resource"]
    ws = np.asarray(ts["wind_speed"], dtype=float)
    wd = np.asarray(ts["wind_direction"], dtype=float)
    ti_dat = ts.get("turbulence_intensity")
    ti = None
    if isinstance(ti_dat, dict) and "data" in ti_dat:
        ti = np.asarray(ti_dat["data"], dtype=float)
        if ti.shape != ws.shape:
            ti = None

    n = min(n_hours, len(ws))
    ws, wd = ws[:n], wd[:n]
    if ti is not None:
        ti = ti[:n]

    print("  Running PyWake simulation...")
    site = XRSite(ds=xr.Dataset(data_vars=dict(P=1)))
    wf_model = get_wake_model("Bastankhah", site, setup["windturbine"])
    kwargs = {"x": setup["x"], "y": setup["y"],
              "wd": wd, "ws": ws, "time": np.arange(n) / 24.0}
    if ti is not None:
        kwargs["TI"] = ti
    sim_res = wf_model(**kwargs)
    power_wind = sim_res.Power.sum(["wt"]).values / 1e6

    df = pd.read_csv(PRICE_CSV)
    for col in ("price_eur_mwh", "price_eur_per_mwh", "Price", "price"):
        if col in df.columns:
            break
    price_eur = df[col].astype(float).to_numpy()[:n]

    bat = setup["battery"]
    rte_dc  = float(bat["rte_nominal"])
    pcu_eff = float(bat["pcu_efficiency"])

    params = {
        "e_cap":   float(bat["energy_capacity_Wh"]) / 1e6,
        "p_cap":   float(bat["power_capacity_W"]) / 1e6,
        "eta_in":  1.0,
        "eta_out": rte_dc * pcu_eff**2,
        "soc_min": float(bat.get("soc_min", 0.10)),
        "soc_max": float(bat.get("soc_max", 0.90)),
        "e_cost_eur_per_kwh": float(bat["capex_EUR_per_kWh"]),
        "p_cost_eur_per_kw":  float(bat["capex_EUR_per_kW"]),
        "e_cost_usd_per_mwh": float(bat["capex_EUR_per_kWh"]) * 1000.0 * EUR_TO_USD,
        "p_cost_usd_per_mw":  float(bat["capex_EUR_per_kW"]) * 1000.0 * EUR_TO_USD,
        "p_max": float(hpp["grid_connection_capacity"]) / 1e6,
        "bat_params": {"power_capacity_W": float(bat["power_capacity_W"])},
        "T_cell_C": 25.0,
    }

    shi_fit = fit_shi_polynomial(params["soc_min"], params["soc_max"], verbose=True)
    params["shi_fit"] = shi_fit
    params["k3"] = shi_fit.k3
    params["k4"] = shi_fit.k4

    # E/P ratio from YAML (for scaled mode)
    params["ep_ratio"] = params["e_cap"] / params["p_cap"]

    print(f"  Battery (YAML): {params['p_cap']:.0f} MW / {params['e_cap']:.0f} MWh")
    print(f"  Grid: {params['p_max']:.0f} MW")
    print(f"  RTE(ac): {params['eta_out']*100:.1f}%")
    print(f"  SoC: {params['soc_min']*100:.0f}%–{params['soc_max']*100:.0f}%")

    return power_wind[:n], price_eur[:n], params


# ════════════════════════════════════════════════════════════════════════════
# 2.  SINGLE-POINT EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_single_ecap(
    e_cap:      float,
    p_cap:      float,
    power_wind: np.ndarray,
    price_eur:  np.ndarray,
    params:     dict,
) -> dict:
    """Solve LP and evaluate degradation for one (E_cap, P_cap) point."""
    n       = len(price_eur)
    eta_out = params["eta_out"]
    soc_min = params["soc_min"]
    soc_max = params["soc_max"]
    p_max   = params["p_max"]
    e_cost  = params["e_cost_usd_per_mwh"]
    p_cost  = params["p_cost_usd_per_mw"]

    dod = 1.0 - soc_min

    stor = Storage(e_cap=e_cap, p_cap=p_cap, eff_in=1.0, eff_out=eta_out,
                   e_cost=e_cost, p_cost=p_cost, dod=dod)
    stor_null = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
                        e_cost=0, p_cost=0)

    price_dam = TimeSeries((price_eur * EUR_TO_USD).tolist(), DT)
    prod      = Production(TimeSeries(power_wind.tolist(), DT), p_cost=0.0)
    prod_null = Production(TimeSeries([0.0] * n, DT), p_cost=0.0)

    # ── Solve LP ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    os_res = solve_lp_pyomo(
        price_dam, prod, prod_null, stor, stor_null,
        DISCOUNT_RATE, N_YEARS, P_MIN_MW, p_max, n,
        "gurobi", fixed_cap=True, soc_max1=soc_max,
    )
    solve_s = time.perf_counter() - t0

    e_vec = np.array(os_res.storage_e[0].data, dtype=float)
    p_vec = np.array(os_res.storage_p[0].data, dtype=float)

    factor = npf.npv(DISCOUNT_RATE, np.ones(N_YEARS + 1)) - 1

    # Revenue (annual, NPV-discounted)
    revenue_usd = 365.0 * 24.0 / n * factor * float(
        np.dot(price_eur * EUR_TO_USD, p_vec)) * DT

    # Capex
    capex_usd = p_cost * p_cap + e_cost * e_cap

    # ── DIAGNOSTIC: what does SHIPP a_npv actually count? (remove after one run) ──
    os_res.get_added_npv(DISCOUNT_RATE, N_YEARS)            # populates os_res.a_npv
    price_used  = price_eur * EUR_TO_USD                    # same prices fed to the solver
    p_prod      = np.array(os_res.production_p[0].data, dtype=float)
    wind_no_bat = np.minimum(power_wind, p_max)
    ann         = 365.0 * 24.0 / n
    rev_arb  = ann * float(np.dot(price_used, p_vec)) * DT                                        # arbitrage (current Plan B)
    rev_marg = ann * float(np.dot(price_used, p_prod + p_vec) - np.dot(price_used, wind_no_bat)) * DT  # marginal (v5.2)
    npv_arb  = rev_arb  * factor - capex_usd
    npv_marg = rev_marg * factor - capex_usd
    print(f"[a_npv chk] E={e_cap:.0f} P={p_cap:.0f} | a_npv={os_res.a_npv:+.4f}M "
          f"| arb={npv_arb*1e-6:+.4f}M (d={os_res.a_npv - npv_arb*1e-6:+.2e}) "
          f"| marg={npv_marg*1e-6:+.4f}M (d={os_res.a_npv - npv_marg*1e-6:+.2e})")

    # ── Degradation evaluation ───────────────────────────────────────────
    # Xu model (full: cycle + calendar internally via compute_fd)
    xu_result = analyze_degradation(
        storage_p=p_vec.tolist(),
        storage_e=e_vec.tolist(),
        e_cap_nominal=e_cap,
        battery_params=params["bat_params"],
        dt_hours=DT,
        T_cell_C=params["T_cell_C"],
    )
    # BUG FIX vs 1D version: xu_result["fd"] is the TOTAL (cycle + calendar).
    # Using it as fd_xu_cycle and then adding fd_calendar again double-counted.
    # Correct: read the separate components directly from the result dict.
    fd_xu_cycle  = xu_result["fd_cycle"]     # cycle-only  (Xu rainflow)
    fd_calendar  = xu_result["fd_calendar"]  # calendar-only (Xu ft_calendar)

    # Shi model (cycle-only by design — no calendar term in Shi framework)
    shi_result = analyze_degradation_shi(
        storage_p=p_vec.tolist(),
        storage_e=e_vec.tolist(),
        e_cap_nominal=e_cap,
        battery_params=params["bat_params"],
        shi_fit=params["shi_fit"],
        T_cell_C=params["T_cell_C"],
        dt_hours=DT,
    )
    fd_shi_cycle = shi_result["fd_shi"]

    # Total fd: add calendar to both branches if enabled
    # For Xu:  calendar already computed inside analyze_degradation, read directly
    # For Shi: calendar is structurally absent; add Xu calendar term to reporting
    fd_xu_total  = xu_result["fd"] if INCLUDE_CALENDAR else fd_xu_cycle
    fd_shi_total = (fd_shi_cycle + fd_calendar) if INCLUDE_CALENDAR else fd_shi_cycle

    # EFC
    efc = count_equivalent_full_cycles(p_vec.tolist(), e_vec.tolist(), e_cap, DT)

    # Degradation cost (annualised over project life)
    deg_scale = 365.0 * 24.0 / n * factor
    deg_cost_xu  = fd_xu_total  * e_cost * e_cap * deg_scale
    deg_cost_shi = fd_shi_total * e_cost * e_cap * deg_scale

    # NPV variants
    npv_no_deg     = revenue_usd - capex_usd
    npv_with_xu    = revenue_usd - capex_usd - deg_cost_xu
    npv_with_shi   = revenue_usd - capex_usd - deg_cost_shi

    # SoH at end of year 1
    soh_xu  = (1.0 - sei_capacity_loss(fd_xu_total)) * 100.0
    soh_shi = (1.0 - sei_capacity_loss(fd_shi_total)) * 100.0

    # Rainflow cycle count
    cycles = rainflow_cycle_counting(e_vec, e_cap)
    n_cycles = len(cycles)

    return {
        "e_cap":          e_cap,
        "p_cap":          p_cap,
        "revenue_usd":    revenue_usd,
        "capex_usd":      capex_usd,
        "npv_no_deg":     npv_no_deg,
        "npv_with_xu":    npv_with_xu,
        "npv_with_shi":   npv_with_shi,
        "fd_xu_cycle":    fd_xu_cycle,
        "fd_shi_cycle":   fd_shi_cycle,
        "fd_calendar":    fd_calendar,
        "fd_xu_total":    fd_xu_total,
        "fd_shi_total":   fd_shi_total,
        "deg_cost_xu":    deg_cost_xu,
        "deg_cost_shi":   deg_cost_shi,
        "efc":            efc,
        "n_cycles":       n_cycles,
        "soh_xu_pct":     soh_xu,
        "soh_shi_pct":    soh_shi,
        "solve_s":        solve_s,
        "throughput_mwh": float(np.sum(np.abs(p_vec))) * DT,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3.  SWEEP + OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def find_latest_csv() -> Path:
    """Return the most recent planB sweep CSV in OUTPUT_DIR.

    Searches for annual (planB_2d_sweep_*.csv) or lifetime
    (planB_lifetime_sweep_*.csv) files depending on SWEEP_MODE.
    Sorting is done on the timestamp embedded in the filename
    (YYYYMMDD_HHMMSS), so it is independent of filesystem modification times.
    Raises FileNotFoundError if no matching CSV exists yet.
    """
    if SWEEP_MODE == "lifetime":
        pattern = "planB_lifetime_sweep_*.csv"
    else:
        pattern = "planB_2d_sweep_*.csv"

    candidates = sorted(OUTPUT_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No {pattern} files found in {OUTPUT_DIR}.\n"
            "Run the full sweep first (set REPLOT_FROM_LAST = False)."
        )
    latest = candidates[-1]
    print(f"  Auto-selected most recent CSV: {latest.name}")
    return latest


def load_and_verify(csv_path: str) -> tuple[List[dict], dict, int]:
    """Load sweep results from a previous CSV run and verify config match.

    Parameters
    ----------
    csv_path : str or Path
        Path to a previously saved planB_2d_sweep_*.csv file.

    Returns
    -------
    results : List[dict]   — list of row dicts, same format as run_sweep_2d
    params  : dict         — minimal params dict needed for plotting
    n       : int          — number of time steps from the saved run

    Config verification
    -------------------
    Looks for a .json sidecar next to the CSV (same stem).  If found, compares
    E_CAP_GRID, P_CAP_GRID, INCLUDE_CALENDAR, and RUN_HOURS against the current
    module-level toggles.  Prints a warning for each mismatch so you know
    whether the saved data is still valid for the current configuration.
    If no sidecar exists, prints a reminder and continues.

    Raises
    ------
    FileNotFoundError if the CSV does not exist.
    """
    import json
    import pandas as pd

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"REPLOT_CSV not found: {csv_path}")

    print(f"  Loading results from: {csv_path.name}")
    df = pd.read_csv(csv_path)
    results = df.to_dict(orient="records")
    print(f"  Loaded {len(results)} rows.")

    # ── Config verification ────────────────────────────────────────────────
    json_path = csv_path.with_suffix(".json")
    if json_path.exists():
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        mismatches = []

        if saved.get("E_CAP_GRID") != E_CAP_GRID:
            mismatches.append(
                f"  E_CAP_GRID: saved={saved['E_CAP_GRID']}  current={E_CAP_GRID}"
            )
        if saved.get("P_CAP_GRID") != P_CAP_GRID:
            mismatches.append(
                f"  P_CAP_GRID: saved={saved['P_CAP_GRID']}  current={P_CAP_GRID}"
            )
        if saved.get("INCLUDE_CALENDAR") != INCLUDE_CALENDAR:
            mismatches.append(
                f"  INCLUDE_CALENDAR: saved={saved['INCLUDE_CALENDAR']}  "
                f"current={INCLUDE_CALENDAR}"
            )
        if saved.get("RUN_HOURS") != RUN_HOURS:
            mismatches.append(
                f"  RUN_HOURS: saved={saved['RUN_HOURS']}  current={RUN_HOURS}"
            )

        if mismatches:
            print("\n  WARNING: config mismatch between saved run and current toggles:")
            for m in mismatches:
                print(m)
            print("  Plots will use saved data but reflect the OLD configuration.")
            print("  Set REPLOT_CSV = None and re-run if you changed the grid or settings.\n")
        else:
            print("  Config verified, saved run matches current toggles. ✓")

        # Recover n from the sidecar
        n = int(saved.get("RUN_HOURS", RUN_HOURS))

        # Reconstruct minimal params dict for plotting (costs, reference design)
        params = {
            "e_cap":             300.0,    # WP2 reference — used only for plot markers
            "p_cap":             150.0,
            "soc_min":           saved.get("soc_min", 0.10),
            "soc_max":           saved.get("soc_max", 0.90),
            "eta_out":           saved.get("eta_out", 0.877),
            "e_cost_eur_per_kwh":saved.get("e_cost_eur_kwh", 150.0),
            "k3":                saved.get("k3", 3.24e-5),
            "k4":                saved.get("k4", 1.179),
        }
    else:
        print(
            "  NOTE: no JSON sidecar found next to CSV.  Cannot verify config.\n"
            "  Proceeding with current toggle values for plot parameters."
        )
        n = RUN_HOURS
        params = {
            "e_cap": 300.0, "p_cap": 150.0,
            "soc_min": 0.10, "soc_max": 0.90,
            "eta_out": 0.877, "e_cost_eur_per_kwh": 150.0,
            "k3": 3.24e-5, "k4": 1.179,
        }

    return results, params, n


# ════════════════════════════════════════════════════════════════════════════
# 3b.  LIFETIME SINGLE-POINT EVALUATION  (three parallel 20-year loops)
# ════════════════════════════════════════════════════════════════════════════

def evaluate_20year(
    e_cap:      float,
    p_cap:      float,
    power_wind: np.ndarray,
    price_eur:  np.ndarray,
    params:     dict,
) -> dict:
    """Run three parallel 20-year loops for one (E_cap, P_cap) design point.

    The three loops share identical price/wind data and LP setup.
    They differ only in how SoH evolves and whether replacement is triggered:

      Loop A — no degradation:
        SoH = 1.0 every year.  e_cap_eff = e_cap_nom always.
        No replacement cost.  Upper bound on NPV.

      Loop B — Xu degradation:
        fd accumulated via Xu rainflow + calendar each year.
        SoH = 1 - sei_capacity_loss(fd_cum_xu).
        Replace when SoH < REPL_THRESHOLD.  Reset fd_cum and SoH on replacement.

      Loop C — Shi degradation:
        Same structure as Loop B but fd accumulated via Shi polynomial.
        Calendar term added from Xu (Shi has no calendar model of its own).

    Price/wind data is repeated identically for all 20 years.
    This is standard practice for design sweeps — the year's dispatch pattern
    is assumed stationary.  Year-to-year price variation is a future extension.

    NPV assembly (all three loops):
        NPV = -capex_at_t0
              + sum_k [ annual_rev_k * (1+r)^(-k) ]
              - sum_repl [ repl_cost * (1+r)^(-k_repl) ]

    Capex includes both energy and power components.
    Replacement cost covers energy capex only (power electronics not replaced).
    """
    n         = len(price_eur)
    eta_out   = params["eta_out"]
    soc_min   = params["soc_min"]
    soc_max   = params["soc_max"]
    p_max_g   = params["p_max"]
    e_cost_usd = params["e_cost_usd_per_mwh"]   # USD/MWh
    p_cost_usd = params["p_cost_usd_per_mw"]    # USD/MW

    # Initial capex — paid at t=0, not discounted
    capex_usd = e_cost_usd * e_cap + p_cost_usd * p_cap

    # Replacement cost — energy side only (power electronics stay)
    repl_cost_usd = e_cost_usd * e_cap

    # ── Shared state for all three loops ──────────────────────────────────
    # Each loop tracks its own fd_cum and SoH independently.
    # Revenue is identical every year for loop A (SoH=1 always).
    # Loops B and C re-solve the LP with reduced e_cap each year.

    # Initialise per-loop accumulators
    npv_nd = -capex_usd
    npv_xu = -capex_usd
    npv_shi = -capex_usd

    soh_xu  = 1.0;  fd_cum_xu  = 0.0
    soh_shi = 1.0;  fd_cum_shi = 0.0

    n_repl_xu  = 0;  eol_xu  = None
    n_repl_shi = 0;  eol_shi = None

    # Tracking for output
    soh_traj_xu  = []
    soh_traj_shi = []
    fd_annual_xu = []
    fd_annual_shi = []

    for k in range(1, N_YEARS + 1):
        discount_f = (1.0 + DISCOUNT_RATE) ** (-k)

        # ── Loop A: no degradation — always full capacity ──────────────────
        # The LP is identical every year so we only solve it once at k=1
        # and reuse the revenue for all 20 years (SoH=1, capacity never drops).
        if k == 1:
            stor_nd = Storage(
                e_cap=e_cap, p_cap=p_cap,
                eff_in=1.0, eff_out=eta_out,
                e_cost=e_cost_usd, p_cost=p_cost_usd,
                dod=1.0 - soc_min,
            )
            stor_null_nd = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
                                   e_cost=0, p_cost=0)
            price_ts  = TimeSeries((price_eur * EUR_TO_USD).tolist(), DT)
            prod      = Production(TimeSeries(power_wind.tolist(), DT), p_cost=0.0)
            prod_null = Production(TimeSeries([0.0]*n, DT), p_cost=0.0)

            # Solve with n_year=N_YEARS to match evaluate_single_ecap exactly.
            # This means storage_p[0].data and os.revenue are computed on the
            # same basis as the annual sweep — consistent sign convention and
            # annuity scaling.
            os_nd = solve_lp_pyomo(
                price_ts, prod, prod_null, stor_nd, stor_null_nd,
                DISCOUNT_RATE, N_YEARS, P_MIN_MW, p_max_g, n,
                "gurobi", fixed_cap=True, soc_max1=soc_max,
            )

            # Extract raw annual revenue exactly as evaluate_single_ecap does:
            # revenue_usd (from annual sweep) = factor * dot(price, p_vec) * DT
            # → raw_annual = revenue_usd / factor
            p_vec_nd  = np.array(os_nd.storage_p[0].data, dtype=float)
            factor_20 = npf.npv(DISCOUNT_RATE, np.ones(N_YEARS)) - 1
            # Total 20-year discounted revenue from the annual formula
            total_rev_nd = (365.0 * 24.0 / n * factor_20
                            * float(np.dot(price_eur * EUR_TO_USD, p_vec_nd)) * DT)
            # Back out the single-year undiscounted revenue for manual discounting
            raw_rev_nd = total_rev_nd / factor_20   # = dot(price, p_vec) * DT * (365*24/n)

        npv_nd += raw_rev_nd * discount_f

        # ── Loops B & C: degradation — solve LP with degraded capacity ─────
        # B — Xu effective capacity
        e_cap_eff_xu  = e_cap * soh_xu
        # C — Shi effective capacity
        e_cap_eff_shi = e_cap * soh_shi

        for loop_id, e_eff in [("xu", e_cap_eff_xu), ("shi", e_cap_eff_shi)]:
            # Rebuild Storage with this year's effective capacity.
            # Use actual e_cost/p_cost so the LP objective is on the same
            # basis as evaluate_single_ecap (matching sign convention in p_vec).
            stor = Storage(
                e_cap=e_eff, p_cap=p_cap,
                eff_in=1.0, eff_out=eta_out,
                e_cost=e_cost_usd, p_cost=p_cost_usd,
                dod=1.0 - soc_min,
            )
            stor_null = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
                                e_cost=0, p_cost=0)

            try:
                os_deg = solve_lp_pyomo(
                    price_ts, prod, prod_null, stor, stor_null,
                    DISCOUNT_RATE, N_YEARS, P_MIN_MW, p_max_g, n,
                    "gurobi", fixed_cap=True, soc_max1=soc_max,
                )
            except RuntimeError:
                # Battery too small to operate — stop this loop early
                break

            p_vec = np.array(os_deg.storage_p[0].data, dtype=float)
            e_vec = np.array(os_deg.storage_e[0].data, dtype=float)

            # Same revenue formula as evaluate_single_ecap, divided by factor
            # to get the raw single-year undiscounted revenue for this year's
            # degraded capacity.  Discount applied manually below.
            factor_20 = npf.npv(DISCOUNT_RATE, np.ones(N_YEARS)) - 1
            raw_rev = ((365.0 * 24.0 / n * factor_20
                        * float(np.dot(price_eur * EUR_TO_USD, p_vec)) * DT)
                       / factor_20)   # = 365*24/n * dot(price, p_vec) * DT

            if loop_id == "xu":
                npv_xu += raw_rev * discount_f
            else:
                npv_shi += raw_rev * discount_f

            # ── Degradation this year ──────────────────────────────────────
            xu_res = analyze_degradation(
                storage_p=p_vec.tolist(),
                storage_e=e_vec.tolist(),
                e_cap_nominal=e_eff,
                battery_params=params["bat_params"],
                dt_hours=DT,
                T_cell_C=params["T_cell_C"],
            )
            fd_xu_yr  = xu_res["fd"] if INCLUDE_CALENDAR else xu_res["fd_cycle"]
            fd_cal_yr = xu_res["fd_calendar"]   # used by Shi branch below

            if loop_id == "xu":
                fd_cum_xu += fd_xu_yr
                soh_xu = 1.0 - sei_capacity_loss(fd_cum_xu)
                soh_traj_xu.append(soh_xu)
                fd_annual_xu.append(fd_xu_yr)

                # Replacement check
                if soh_xu < REPL_THRESHOLD:
                    if eol_xu is None:
                        eol_xu = k
                    npv_xu -= repl_cost_usd * discount_f
                    n_repl_xu += 1
                    fd_cum_xu = 0.0
                    soh_xu    = 1.0

            else:  # Shi
                shi_res = analyze_degradation_shi(
                    storage_p=p_vec.tolist(),
                    storage_e=e_vec.tolist(),
                    e_cap_nominal=e_eff,
                    battery_params=params["bat_params"],
                    shi_fit=params["shi_fit"],
                    T_cell_C=params["T_cell_C"],
                    dt_hours=DT,
                )
                fd_shi_yr = shi_res["fd_shi"]
                # Add calendar term from Xu (Shi has no calendar model)
                fd_shi_total_yr = (fd_shi_yr + fd_cal_yr) if INCLUDE_CALENDAR \
                                  else fd_shi_yr

                fd_cum_shi += fd_shi_total_yr
                soh_shi = 1.0 - sei_capacity_loss(fd_cum_shi)
                soh_traj_shi.append(soh_shi)
                fd_annual_shi.append(fd_shi_total_yr)

                if soh_shi < REPL_THRESHOLD:
                    if eol_shi is None:
                        eol_shi = k
                    npv_shi -= repl_cost_usd * discount_f
                    n_repl_shi += 1
                    fd_cum_shi = 0.0
                    soh_shi    = 1.0

    return {
        # Design point
        "e_cap":          e_cap,
        "p_cap":          p_cap,
        # Lifetime NPVs (USD)
        "npv_no_deg":     npv_nd,
        "npv_with_xu":    npv_xu,
        "npv_with_shi":   npv_shi,
        # Capex (for reference)
        "capex_usd":      capex_usd,
        # Replacement info
        "n_repl_xu":      n_repl_xu,
        "n_repl_shi":     n_repl_shi,
        "eol_year_xu":    eol_xu  if eol_xu  is not None else float("nan"),
        "eol_year_shi":   eol_shi if eol_shi is not None else float("nan"),
        # Final SoH at year 20 (or end of last complete cycle)
        "soh_final_xu":   soh_traj_xu[-1]  if soh_traj_xu  else float("nan"),
        "soh_final_shi":  soh_traj_shi[-1] if soh_traj_shi else float("nan"),
        # Mean annual fd across all years (useful for heatmaps)
        "fd_mean_xu":     float(np.mean(fd_annual_xu))  if fd_annual_xu  else float("nan"),
        "fd_mean_shi":    float(np.mean(fd_annual_shi)) if fd_annual_shi else float("nan"),
    }


def _build_viable_set(
    annual_results: List[dict],
    p_max: float,
) -> Tuple[set, dict]:
    """Identify (E, P) pairs worth evaluating in lifetime mode.

    Applies three filters to the annual sweep results:
      1. Duration guard: skip E/P < LIFETIME_MIN_DURATION_H or > LIFETIME_MAX_DURATION_H
      2. NPV floor: keep if annual Xu NPV ≥ floor × max(Xu) OR annual Shi NPV ≥ floor × max(Shi)
         (We don't filter on no-deg because it's linear and not the optimisation target.)
      3. Always keep the annual optima themselves (safety net)

    Returns
    -------
    viable : set of (e_cap, p_cap) tuples
    stats  : dict with filtering diagnostics for logging
    """
    # Find annual maxima for the two degradation models
    max_xu  = max(r.get("npv_with_xu",  -np.inf) for r in annual_results)
    max_shi = max(r.get("npv_with_shi", -np.inf) for r in annual_results)
    floor_xu  = LIFETIME_NPV_FLOOR * max_xu
    floor_shi = LIFETIME_NPV_FLOOR * max_shi

    # Always-include: the annual optima for each model
    best_xu  = max(annual_results, key=lambda r: r.get("npv_with_xu",  -np.inf))
    best_shi = max(annual_results, key=lambda r: r.get("npv_with_shi", -np.inf))
    best_nd  = max(annual_results, key=lambda r: r.get("npv_no_deg",   -np.inf))
    always_keep = {
        (best_xu["e_cap"],  best_xu["p_cap"]),
        (best_shi["e_cap"], best_shi["p_cap"]),
        (best_nd["e_cap"],  best_nd["p_cap"]),
    }

    viable = set()
    n_duration_skip = 0
    n_npv_skip = 0

    for r in annual_results:
        e = r["e_cap"]
        p = min(r["p_cap"], p_max)
        key = (e, p)

        # Filter 1: duration guard
        dur = e / p if p > 0 else 0.0
        if dur < LIFETIME_MIN_DURATION_H or dur > LIFETIME_MAX_DURATION_H:
            if key not in always_keep:
                n_duration_skip += 1
                continue

        # Filter 2: NPV floor (must pass for at least one degradation model)
        npv_xu  = r.get("npv_with_xu",  -np.inf)
        npv_shi = r.get("npv_with_shi", -np.inf)
        if npv_xu < floor_xu and npv_shi < floor_shi:
            if key not in always_keep:
                n_npv_skip += 1
                continue

        viable.add(key)

    # Safety: ensure always_keep points are included
    viable |= always_keep

    stats = {
        "total_annual": len(annual_results),
        "n_duration_skip": n_duration_skip,
        "n_npv_skip": n_npv_skip,
        "n_viable": len(viable),
        "floor_xu_musd": floor_xu / 1e6,
        "floor_shi_musd": floor_shi / 1e6,
    }
    return viable, stats


def _nan_result_20year(e_cap: float, p_cap: float) -> dict:
    """Return a placeholder result dict with NaN values for skipped points."""
    return {
        "e_cap": e_cap, "p_cap": p_cap,
        "npv_no_deg": float("nan"), "npv_with_xu": float("nan"),
        "npv_with_shi": float("nan"), "capex_usd": float("nan"),
        "n_repl_xu": 0, "n_repl_shi": 0,
        "eol_year_xu": float("nan"), "eol_year_shi": float("nan"),
        "soh_final_xu": float("nan"), "soh_final_shi": float("nan"),
        "fd_mean_xu": float("nan"), "fd_mean_shi": float("nan"),
    }


def run_sweep_20year(
    power_wind: np.ndarray,
    price_eur:  np.ndarray,
    params:     dict,
    annual_results: List[dict] | None = None,
) -> List[dict]:
    """Nested loop calling evaluate_20year for every viable (E_cap, P_cap) point.

    If annual_results is provided, applies smart filtering (duration + NPV floor)
    to skip non-competitive points.  Otherwise runs the full grid.

    Each evaluated point runs 2×20 + 1 = 41 LP solves.  With filtering this
    typically reduces the grid from 88 to ~30–35 points → ~1.5–2 hours.
    """
    # ── Build viable set if annual results available ──────────────────────
    if annual_results is not None:
        viable, stats = _build_viable_set(annual_results, params["p_max"])
        print(f"\n  ── Smart filtering ──")
        print(f"  Annual points:     {stats['total_annual']}")
        print(f"  Duration skipped:  {stats['n_duration_skip']}")
        print(f"  NPV floor skipped: {stats['n_npv_skip']}  "
              f"(Xu floor={stats['floor_xu_musd']:.1f} MUSD, "
              f"Shi floor={stats['floor_shi_musd']:.1f} MUSD)")
        print(f"  Viable points:     {stats['n_viable']}")
        print(f"  LP solves:         ~{stats['n_viable'] * 41}  "
              f"(est. {stats['n_viable'] * 41 * 5 / 3600:.1f} hours)")
        print()
    else:
        viable = None  # run everything

    total = len(E_CAP_GRID) * len(P_CAP_GRID)
    results = []
    done = 0
    n_evaluated = 0
    t_start = time.perf_counter()

    for e_cap in E_CAP_GRID:
        for p_cap in P_CAP_GRID:
            done += 1
            p_cap_eff = min(p_cap, params["p_max"])

            # ── Check if this point should be evaluated ───────────────────
            if viable is not None and (e_cap, p_cap_eff) not in viable:
                print(f"  [{done:3d}/{total}]  E={e_cap:5.0f} MWh  P={p_cap_eff:5.0f} MW  "
                      f"-> SKIPPED (filtered)")
                results.append(_nan_result_20year(e_cap, p_cap_eff))
                continue

            n_evaluated += 1
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / n_evaluated) * (len(viable or E_CAP_GRID) * (1 if viable else len(P_CAP_GRID)) - n_evaluated) if n_evaluated > 1 else 0.0
            print(f"  [{done:3d}/{total}]  E={e_cap:5.0f} MWh  P={p_cap_eff:5.0f} MW  "
                  f"[{n_evaluated}/{len(viable) if viable else total}]  "
                  f"ETA={eta/60:.1f}min ...", end="", flush=True)

            r = evaluate_20year(e_cap, p_cap_eff, power_wind, price_eur, params)
            results.append(r)

            print(f"  NPV_nd={r['npv_no_deg']/1e6:.2f}M  "
                  f"NPV_xu={r['npv_with_xu']/1e6:.2f}M  "
                  f"NPV_shi={r['npv_with_shi']/1e6:.2f}M  "
                  f"repl_xu={r['n_repl_xu']}  repl_shi={r['n_repl_shi']}")

    elapsed_total = time.perf_counter() - t_start
    print(f"\n  Evaluated {n_evaluated}/{total} points in {elapsed_total/60:.1f} min "
          f"(skipped {total - n_evaluated})")
    return results


def save_results_20year(results: List[dict], params: dict, n: int):
    """Save lifetime sweep results to CSV, JSON sidecar, and text report."""
    import json
    import pandas as pd
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── CSV ──────────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / f"planB_lifetime_sweep_{n}h_{N_YEARS}yr_{ts}.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path.name}")

    # ── JSON sidecar ──────────────────────────────────────────────────────
    config_snap = {
        "E_CAP_GRID":        E_CAP_GRID,
        "P_CAP_GRID":        P_CAP_GRID,
        "INCLUDE_CALENDAR":  INCLUDE_CALENDAR,
        "RUN_HOURS":         n,
        "N_YEARS":           N_YEARS,
        "REPL_THRESHOLD":    REPL_THRESHOLD,
        "DISCOUNT_RATE":     DISCOUNT_RATE,
        "LIFETIME_NPV_FLOOR": LIFETIME_NPV_FLOOR,
        "LIFETIME_MIN_DURATION_H": LIFETIME_MIN_DURATION_H,
        "LIFETIME_MAX_DURATION_H": LIFETIME_MAX_DURATION_H,
        "soc_min":           params["soc_min"],
        "soc_max":           params["soc_max"],
        "eta_out":           params["eta_out"],
        "e_cost_eur_kwh":    params["e_cost_eur_per_kwh"],
        "k3":                params["k3"],
        "k4":                params["k4"],
    }
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(config_snap, indent=2), encoding="utf-8")
    print(f"  JSON: {json_path.name}")

    # ── Text report ──────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / f"planB_lifetime_report_{n}h_{N_YEARS}yr_{ts}.txt"
    lines = []
    w = lines.append

    w("=" * 110)
    w(f"PLAN B: Lifetime sweep, {N_YEARS}-year NPV vs (E_cap, P_cap), three degradation scenarios")
    w("=" * 110)
    w("")
    w("CONFIGURATION")
    w(f"  Horizon per year:  {n} hours ({n/24:.1f} days), repeated {N_YEARS} years")
    w(f"  E_cap grid:        {E_CAP_GRID} MWh")
    w(f"  P_cap grid:        {P_CAP_GRID} MW")
    w(f"  Grid points:       {len(results)}")
    w(f"  Calendar aging:    {INCLUDE_CALENDAR}")
    w(f"  Replacement SoH:   {REPL_THRESHOLD*100:.0f}%")
    w(f"  Discount rate:     {DISCOUNT_RATE*100:.1f}%")
    w(f"  SoC window:        {params['soc_min']*100:.0f}%–{params['soc_max']*100:.0f}%")
    w(f"  RTE(ac):           {params['eta_out']*100:.1f}%")
    w("")
    w("=" * 110)
    w("RESULTS")
    w("=" * 110)
    w("")
    header = (f"  {'E':>5}  {'P':>5}  {'E/P':>4}  "
              f"{'NPV_noDeg':>10}  {'NPV_Xu':>10}  {'NPV_Shi':>10}  "
              f"{'Repl_Xu':>7}  {'Repl_Shi':>8}  "
              f"{'EoL_Xu':>6}  {'EoL_Shi':>7}  "
              f"{'SoH_Xu':>6}  {'SoH_Shi':>7}")
    units  = (f"  {'MWh':>5}  {'MW':>5}  {'h':>4}  "
              f"{'MUSD':>10}  {'MUSD':>10}  {'MUSD':>10}  "
              f"{'n':>7}  {'n':>8}  "
              f"{'yr':>6}  {'yr':>7}  "
              f"{'%':>6}  {'%':>7}")
    w(header)
    w(units)
    w("  " + "-" * 107)

    for r in results:
        ep = r['e_cap'] / r['p_cap']
        eol_xu  = f"{r['eol_year_xu']:.0f}"  if not np.isnan(r['eol_year_xu'])  else "none"
        eol_shi = f"{r['eol_year_shi']:.0f}" if not np.isnan(r['eol_year_shi']) else "none"
        w(f"  {r['e_cap']:5.0f}  {r['p_cap']:5.0f}  {ep:4.1f}  "
          f"{r['npv_no_deg']/1e6:10.2f}  "
          f"{r['npv_with_xu']/1e6:10.2f}  "
          f"{r['npv_with_shi']/1e6:10.2f}  "
          f"{r['n_repl_xu']:7.0f}  {r['n_repl_shi']:8.0f}  "
          f"{eol_xu:>6}  {eol_shi:>7}  "
          f"{r['soh_final_xu']*100:6.1f}  {r['soh_final_shi']*100:7.1f}")

    w("")
    w("OPTIMAL DESIGN POINT")
    w("")
    for label, key in [("No degradation", "npv_no_deg"),
                       ("Xu degradation", "npv_with_xu"),
                       ("Shi degradation", "npv_with_shi")]:
        best = max(results, key=lambda r, k=key: r[k] if np.isfinite(r[k]) else -np.inf)
        ep = best['e_cap'] / best['p_cap']
        w(f"  {label:<18s}: E={best['e_cap']:.0f} MWh  P={best['p_cap']:.0f} MW  "
          f"E/P={ep:.1f}h  NPV={best[key]/1e6:.2f} MUSD")

    w("")
    w("=" * 110)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {report_path.name}")

    return csv_path, report_path, ts


def run_sweep_2d(
    power_wind: np.ndarray,
    price_eur:  np.ndarray,
    params:     dict,
) -> List[dict]:
    """Run the 2D parameter sweep over all (E_cap, P_cap) combinations.
    
    Uses intelligent fallback approximation for extreme durations to keep 
    contour grids unbroken without executing slow LP solves.
    """
    total = len(E_CAP_GRID) * len(P_CAP_GRID)
    results = []
    done = 0
    t_sweep_start = time.perf_counter()

    # We can keep a quick lookup dictionary of computed valid results
    # keyed by (e_cap, p_cap_eff)
    valid_lookup = {}

    for e_cap in E_CAP_GRID:
        for p_cap in P_CAP_GRID:
            done += 1
            p_cap_eff = min(p_cap, params["p_max"])
            duration_h = e_cap / p_cap_eff if p_cap_eff > 0 else 0.0

            elapsed = time.perf_counter() - t_sweep_start
            eta = (elapsed / done) * (total - done) if done > 1 else 0.0

            # ── Check Pruning Boundaries ──
            is_pruned = (duration_h < 1.0) or (duration_h > 8.0)

            if is_pruned:
                # Find a realistic valid companion neighbor to clone market dynamics from
                # If duration too short, revenue is bottlenecked by E_cap (clone P = E)
                # If duration too long, revenue is bottlenecked by P_cap (clone E = 8 * P)
                target_p = p_cap_eff if duration_h <= 8.0 else min(P_CAP_GRID, key=lambda p: abs(e_cap/p - 8.0 if p > 0 else 999))
                target_e = e_cap if duration_h >= 1.0 else min(E_CAP_GRID, key=lambda e: abs(e/p_cap_eff - 1.0))
                
                neighbor_key = (target_e, target_p)
                
                if neighbor_key in valid_lookup:
                    n_r = valid_lookup[neighbor_key]
                    print(f"  [{done:3d}/{total}]  E={e_cap:5.0f} MWh  P={p_cap_eff:5.0f} MW  "
                          f"Duration={duration_h:.1f}h -> SKIPPED (Cloned from neighbor)")
                    
                    # Recalculate true CAPEX and NPV for this point using the cloned revenue
                    p_cost = params["p_cost_usd_per_mw"]
                    e_cost = params["e_cost_usd_per_mwh"]
                    true_capex = (p_cost * p_cap_eff) + (e_cost * e_cap)
                    
                    r_clone = n_r.copy()
                    r_clone["e_cap"] = e_cap
                    r_clone["p_cap"] = p_cap_eff
                    r_clone["capex_usd"] = true_capex
                    r_clone["npv_no_deg"] = n_r["revenue_usd"] - true_capex
                    r_clone["npv_with_xu"] = n_r["revenue_usd"] - true_capex - n_r.get("deg_cost_xu", 0)
                    r_clone["npv_with_shi"] = n_r["revenue_usd"] - true_capex - n_r.get("deg_cost_shi", 0)
                    r_clone["solve_s"] = 0.0
                    
                    results.append(r_clone)
                    continue

            # ── Normal Execution Block ──
            print(f"  [{done:3d}/{total}]  E={e_cap:5.0f} MWh  P={p_cap_eff:5.0f} MW  "
                  f"ETA={eta:.0f}s ...", end="", flush=True)

            r = evaluate_single_ecap(e_cap, p_cap_eff, power_wind, price_eur, params)
            results.append(r)
            
            # Save to lookup matrix so skipped points can read from it
            valid_lookup[(e_cap, p_cap_eff)] = r

            print(f"  rev={r['revenue_usd']/1e3:7.0f}k  "
                  f"fd_xu={r['fd_xu_total']:.5f}  "
                  f"EFC={r['efc']:.0f}  t={r['solve_s']:.1f}s")

    return results


def save_results_2d(results: List[dict], params: dict, n: int) -> None:
    """Save 2D sweep results to CSV, JSON sidecar, and text report."""
    import json
    import pandas as pd
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── CSV ──────────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / f"planB_2d_sweep_{n}h_{ts}.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path.name}")

    # ── JSON sidecar — config snapshot for replot verification ───────────
    # Saved next to the CSV with the same stem.  load_and_verify() reads
    # this to check whether the current toggles match the saved run.
    config_snap = {
        "E_CAP_GRID":       E_CAP_GRID,
        "P_CAP_GRID":       P_CAP_GRID,
        "INCLUDE_CALENDAR": INCLUDE_CALENDAR,
        "RUN_HOURS":        n,
        "soc_min":          params["soc_min"],
        "soc_max":          params["soc_max"],
        "eta_out":          params["eta_out"],
        "e_cost_eur_kwh":   params["e_cost_eur_per_kwh"],
        "k3":               params["k3"],
        "k4":               params["k4"],
    }
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(config_snap, indent=2), encoding="utf-8")
    print(f"  JSON: {json_path.name}")

    # ── Text report ──────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / f"planB_2d_report_{n}h_{ts}.txt"

    lines = []
    w = lines.append

    w("=" * 100)
    w("PLAN B 2D: Parameter sweep, NPV vs (E_cap, P_cap) with degradation")
    w("=" * 100)
    w("")
    w("CONFIGURATION")
    w(f"  Horizon:         {n} hours ({n/24:.1f} days)")
    w(f"  E_cap grid:      {E_CAP_GRID} MWh")
    w(f"  P_cap grid:      {P_CAP_GRID} MW")
    w(f"  Grid points:     {len(E_CAP_GRID) * len(P_CAP_GRID)}")
    w(f"  Calendar aging:  {INCLUDE_CALENDAR}")
    w(f"  SoC window:      {params['soc_min']*100:.0f}%–{params['soc_max']*100:.0f}%")
    w(f"  RTE(ac):         {params['eta_out']*100:.1f}%")
    w(f"  Shi k3:          {params['k3']:.4e}")
    w(f"  Shi k4:          {params['k4']:.4f}")
    w(f"  Cost:            {params['e_cost_eur_per_kwh']:.0f} EUR/kWh")

    w("")
    w("=" * 100)
    w("RESULTS")
    w("=" * 100)
    w("")
    w(f"  {'E_cap':>6s}  {'P_cap':>6s}  {'E/P_h':>5s}  {'Revenue':>10s}  "
      f"{'NPV_noDeg':>10s}  {'NPV_Xu':>10s}  {'NPV_Shi':>10s}  "
      f"{'fd_Xu':>8s}  {'fd_Shi':>8s}  {'EFC':>5s}")
    w(f"  {'MWh':>6s}  {'MW':>6s}  {'h':>5s}  {'kUSD':>10s}  "
      f"{'kUSD':>10s}  {'kUSD':>10s}  {'kUSD':>10s}  "
      f"{'':>8s}  {'':>8s}  {'':>5s}")
    w(f"  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*10}  "
      f"{'─'*10}  {'─'*8}  {'─'*8}  {'─'*5}")

    for r in results:
        ep_h = r['e_cap'] / r['p_cap'] if r['p_cap'] > 0 else float('inf')
        w(f"  {r['e_cap']:6.0f}  {r['p_cap']:6.0f}  {ep_h:5.1f}  "
          f"{r['revenue_usd']/1e3:10.0f}  "
          f"{r['npv_no_deg']/1e3:10.0f}  "
          f"{r['npv_with_xu']/1e3:10.0f}  "
          f"{r['npv_with_shi']/1e3:10.0f}  "
          f"{r['fd_xu_total']:8.5f}  "
          f"{r['fd_shi_total']:8.5f}  "
          f"{r['efc']:5.0f}")

    # Best design point for each NPV metric
    w("")
    w("OPTIMAL DESIGN POINT")
    w("")
    for label, key in [("Without degradation", "npv_no_deg"),
                       ("With Xu degradation", "npv_with_xu"),
                       ("With Shi degradation", "npv_with_shi")]:
        best = max(results, key=lambda r: r[key])
        ep_h = best['e_cap'] / best['p_cap']
        w(f"  {label:<25s}:  E={best['e_cap']:.0f} MWh  P={best['p_cap']:.0f} MW  "
          f"E/P={ep_h:.1f}h  NPV={best[key]/1e6:.2f} MUSD")

    w("")
    w("=" * 100)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {report_path.name}")

    return csv_path, report_path, ts


def _to_grid(results: List[dict], key: str) -> np.ndarray:
    """Reshape flat results list into a 2D array (n_e × n_p) for heatmaps."""
    n_e = len(E_CAP_GRID)
    n_p = len(P_CAP_GRID)
    grid = np.full((n_e, n_p), np.nan)
    for r in results:
        i = E_CAP_GRID.index(r["e_cap"]) if r["e_cap"] in E_CAP_GRID else None
        # p_cap may have been clamped to p_max — find nearest grid value
        p_diffs = [abs(r["p_cap"] - p) for p in P_CAP_GRID]
        j = int(np.argmin(p_diffs))
        if i is not None:
            grid[i, j] = r[key]
    return grid


# ════════════════════════════════════════════════════════════════════════════
# 3c.  QUADRATIC SURFACE FITTING
# ════════════════════════════════════════════════════════════════════════════

def fit_quadratic_surface(
    results: List[dict],
    npv_key: str,
    label: str = "",
) -> dict | None:
    """Fit a 2D quadratic to NPV(E, P) and find the continuous optimum.

    Model: NPV = a0 + a1*E + a2*P + a3*E² + a4*P² + a5*E*P

    Optimum: solve  dNPV/dE = 0,  dNPV/dP = 0  →  2×2 linear system.
    Valid only if the Hessian is negative definite (both eigenvalues < 0),
    meaning the surface is concave (has a maximum, not a saddle or minimum).

    Returns dict with fitted optimum, R², and diagnostics, or None if the
    fit fails (too few points, non-concave surface, etc).
    """
    # Filter out NaN results (from pruned/skipped points)
    valid = [(r["e_cap"], r["p_cap"], r[npv_key])
             for r in results if np.isfinite(r.get(npv_key, float("nan")))]
    if len(valid) < 6:
        print(f"  [{label}] Too few valid points ({len(valid)}) for quadratic fit.")
        return None

    E = np.array([v[0] for v in valid])
    P = np.array([v[1] for v in valid])
    Z = np.array([v[2] for v in valid])

    # Normalise for numerical stability
    E_mean, E_std = E.mean(), max(E.std(), 1.0)
    P_mean, P_std = P.mean(), max(P.std(), 1.0)
    En = (E - E_mean) / E_std
    Pn = (P - P_mean) / P_std

    # Design matrix: [1, En, Pn, En², Pn², En·Pn]
    A = np.column_stack([np.ones(len(En)), En, Pn, En**2, Pn**2, En * Pn])

    coeffs, residuals, rank, sv = np.linalg.lstsq(A, Z, rcond=None)
    a0, a1, a2, a3, a4, a5 = coeffs

    # R²
    Z_pred = A @ coeffs
    ss_res = np.sum((Z - Z_pred) ** 2)
    ss_tot = np.sum((Z - Z.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Hessian in normalised coordinates
    H = np.array([[2 * a3, a5],
                   [a5, 2 * a4]])
    eigvals = np.linalg.eigvalsh(H)
    is_concave = bool(np.all(eigvals < 0))

    # Grid optimum (always available as fallback)
    best_grid = max(results, key=lambda r: r.get(npv_key, -np.inf) if np.isfinite(r.get(npv_key, -np.inf)) else -np.inf)

    if not is_concave:
        print(f"  [{label}] Hessian NOT negative definite "
              f"(eigenvalues: {eigvals[0]:.2e}, {eigvals[1]:.2e}).")
        print(f"    Surface is not concave — grid optimum is more reliable.")
        return {
            "method": "grid_only",
            "e_opt": best_grid["e_cap"],
            "p_opt": best_grid["p_cap"],
            "npv_opt": best_grid[npv_key],
            "r_squared": r_squared,
            "eigvals": eigvals.tolist(),
            "is_concave": False,
            "in_range": True,
            "label": label,
        }

    # Solve for optimum: H @ [En*, Pn*] = -[a1, a2]
    opt_norm = np.linalg.solve(H, -np.array([a1, a2]))
    E_opt = opt_norm[0] * E_std + E_mean
    P_opt = opt_norm[1] * P_std + P_mean
    NPV_opt = float(np.array([1, opt_norm[0], opt_norm[1],
                                opt_norm[0]**2, opt_norm[1]**2,
                                opt_norm[0] * opt_norm[1]]) @ coeffs)

    # Sanity: is the optimum within a reasonable range of the grid?
    e_range = (min(E), max(E))
    p_range = (min(P), max(P))
    in_range = (e_range[0] * 0.8 <= E_opt <= e_range[1] * 1.2 and
                p_range[0] * 0.8 <= P_opt <= p_range[1] * 1.2)

    print(f"  [{label}] Quadratic fit: R²={r_squared:.4f}")
    print(f"    Grid optimum:       E={best_grid['e_cap']:.0f} MWh, "
          f"P={best_grid['p_cap']:.0f} MW, "
          f"NPV={best_grid[npv_key]/1e6:.2f} MUSD")
    print(f"    Continuous optimum: E*={E_opt:.1f} MWh, P*={P_opt:.1f} MW, "
          f"NPV*={NPV_opt/1e6:.2f} MUSD")
    if not in_range:
        print(f"    WARNING: optimum outside grid range — extrapolation, "
              f"treat with caution")

    return {
        "method": "quadratic",
        "e_opt": E_opt,
        "p_opt": P_opt,
        "npv_opt": NPV_opt,
        "e_opt_grid": best_grid["e_cap"],
        "p_opt_grid": best_grid["p_cap"],
        "npv_opt_grid": best_grid[npv_key],
        "r_squared": r_squared,
        "eigvals": eigvals.tolist(),
        "is_concave": True,
        "in_range": in_range,
        "label": label,
    }


def run_quadratic_fits(results: List[dict]) -> Dict[str, dict]:
    """Run quadratic fits for all three NPV models, return dict keyed by npv_key."""
    print("\n" + "=" * 78)
    print("QUADRATIC SURFACE FITTING")
    print("=" * 78)
    quad_fits: Dict[str, dict] = {}
    for label, key in [("Xu degradation", "npv_with_xu"),
                       ("Shi degradation", "npv_with_shi")]:
        qf = fit_quadratic_surface(results, key, label=label)
        if qf is not None:
            quad_fits[key] = qf
    return quad_fits


def plot_npv_three_models(results: List[dict], params: dict, n: int, ts: str,
                          quad_fits: Dict[str, dict] | None = None) -> None:
    """Jenna's requested figure: three NPV heatmaps side by side.

    One panel per degradation scenario: no-degradation, Xu, Shi.
    Every panel carries all three model optima as differently-coloured stars
    so the reader can immediately see whether the optimal design shifts between
    scenarios without needing a separate difference figure.

    Markers
    -------
    Blue star  : grid optimum of the no-degradation NPV surface
    Red star   : grid optimum of the Xu-degradation NPV surface
    Green star : grid optimum of the Shi-degradation NPV surface
    Solid circle : quadratic-fit continuous optimum (one per panel, own model)

    All three stars appear on all three panels.  On each panel, the star whose
    model matches the panel is drawn slightly larger to indicate "this is the
    optimum you are currently looking at."  The solid circle (if quad_fits is
    provided) shows the continuous optimum from the quadratic surface fit for
    that panel's own model only.

    Reference design (300 MWh / 150 MW) is shown as a white diamond on each
    panel for orientation.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available, skipping NPV comparison plot.")
        return

    BG   = "#f7f9fc"
    C_ND = "#2166ac"   # blue   — no degradation
    C_XU = "#b5351b"   # red    — Xu
    C_SH = "#4daf4a"   # green  — Shi

    E_arr = np.array(E_CAP_GRID, dtype=float)
    P_arr = np.array(P_CAP_GRID, dtype=float)
    e_ref, p_ref = params["e_cap"], params["p_cap"]

    # ── Pull the three NPV grids ───────────────────────────────────────────
    grid_nd  = _to_grid(results, "npv_no_deg")  / 1e6   # MUSD
    grid_xu  = _to_grid(results, "npv_with_xu") / 1e6
    grid_shi = _to_grid(results, "npv_with_shi") / 1e6

    # ── Compute all three optima once ──────────────────────────────────────
    def _opt(grid):
        i, j = np.unravel_index(np.nanargmax(grid), grid.shape)
        return float(E_arr[i]), float(P_arr[j]), float(grid[i, j])

    e_nd,  p_nd,  npv_nd  = _opt(grid_nd)
    e_xu,  p_xu,  npv_xu  = _opt(grid_xu)
    e_shi, p_shi, npv_shi = _opt(grid_shi)

    # Shared colour scale: use the full range across all three grids so the
    # three panels are directly comparable — same colour = same NPV value.
    vmin = min(np.nanmin(grid_nd), np.nanmin(grid_xu), np.nanmin(grid_shi))
    vmax = max(np.nanmax(grid_nd), np.nanmax(grid_xu), np.nanmax(grid_shi))

    # ── Figure ────────────────────────────────────────────────────────────
    cal_label = "+calendar" if INCLUDE_CALENDAR else "cycle-only"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"NPV design space: no-degradation vs Xu vs Shi  ({n}h, {cal_label})\n"
        f"Stars = grid optima.  Circle = quadratic-fit optimum.  "
        f"Diamond = WP2 reference ({e_ref:.0f} MWh / {p_ref:.0f} MW)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    panels = [
        (axes[0], grid_nd,  "NPV, no degradation [MUSD]",  C_ND, e_nd,  p_nd,  npv_nd,  "npv_no_deg"),
        (axes[1], grid_xu,  "NPV with Xu degradation [MUSD]", C_XU, e_xu,  p_xu,  npv_xu,  "npv_with_xu"),
        (axes[2], grid_shi, "NPV with Shi degradation [MUSD]", C_SH, e_shi, p_shi, npv_shi, "npv_with_shi"),
    ]

    # Use a fixed set of contour levels across all panels for comparability
    levels = np.linspace(vmin, vmax, 22)

    for ax, grid, title, own_color, own_e, own_p, own_npv, own_key in panels:
        ax.set_facecolor(BG)

        im = ax.contourf(P_arr, E_arr, grid, levels=levels, cmap="RdYlGn",
                         vmin=vmin, vmax=vmax)
        cs = ax.contour(P_arr, E_arr, grid, levels=10,
                        colors="k", linewidths=0.3, alpha=0.35)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")

        # ── Reference diamond ─────────────────────────────────────────────
        ax.scatter([p_ref], [e_ref], s=90, marker="D",
                   color="white", edgecolors="black", linewidths=0.9, zorder=6)

        # ── Three model optima as stars ───────────────────────────────────
        # All three appear on every panel.
        # The panel's "own" optimum is drawn larger (s=280) so it stands out.
        # The other two are drawn smaller (s=140) for context.
        star_specs = [
            (e_nd,  p_nd,  C_ND, "No-deg optimum",  e_nd  == own_e and p_nd  == own_p),
            (e_xu,  p_xu,  C_XU, "Xu optimum",      e_xu  == own_e and p_xu  == own_p),
            (e_shi, p_shi, C_SH, "Shi optimum",      e_shi == own_e and p_shi == own_p),
        ]
        for s_e, s_p, s_col, s_label, is_own in star_specs:
            size    = 300 if is_own else 130
            lw      = 0.9 if is_own else 0.6
            zorder  = 8   if is_own else 7
            ax.scatter([s_p], [s_e], s=size, marker="*",
                       color=s_col, edgecolors="black", linewidths=lw,
                       zorder=zorder)

        # ── Quadratic-fit continuous optimum (own model only) ─────────────
        if quad_fits is not None and own_key in quad_fits:
            qf = quad_fits[own_key]
            ax.scatter([qf["p_opt"]], [qf["e_opt"]],
                       s=110, marker="o", color=own_color,
                       edgecolors="black", linewidths=1.2, zorder=9)

        # ── Axis margin so corner markers are fully visible (3% of range) ──
        p_margin = 0.03 * (P_arr[-1] - P_arr[0])
        e_margin = 0.03 * (E_arr[-1] - E_arr[0])
        ax.set_xlim(P_arr[0] - p_margin, P_arr[-1] + p_margin)
        ax.set_ylim(E_arr[0] - e_margin, E_arr[-1] + e_margin)

        ax.set_xlabel("Power capacity P [MW]", fontsize=10)
        ax.set_ylabel("Energy capacity E [MWh]", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)

        # One shared colorbar per panel (same scale = directly comparable)
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("MUSD", fontsize=9)

    # ── Shared legend below all panels ────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="white", ec="black", label=f"Reference ({e_ref:.0f}/{p_ref:.0f})"),
        plt.scatter([], [], s=300, marker="*", color=C_ND, ec="black",
                    label=f"No-deg grid opt  ({e_nd:.0f} MWh / {p_nd:.0f} MW, "
                          f"{npv_nd:.1f} MUSD)"),
        plt.scatter([], [], s=300, marker="*", color=C_XU, ec="black",
                    label=f"Xu grid opt      ({e_xu:.0f} MWh / {p_xu:.0f} MW, "
                          f"{npv_xu:.1f} MUSD)"),
        plt.scatter([], [], s=300, marker="*", color=C_SH, ec="black",
                    label=f"Shi grid opt     ({e_shi:.0f} MWh / {p_shi:.0f} MW, "
                          f"{npv_shi:.1f} MUSD)"),
    ]

    # Add quadratic-fit legend entries if available
    if quad_fits is not None:
        for npv_key, color, short in [("npv_with_xu", C_XU, "Xu"),
                                       ("npv_with_shi", C_SH, "Shi")]:
            if npv_key in quad_fits:
                qf = quad_fits[npv_key]
                legend_handles.append(
                    plt.scatter([], [], s=140, marker="o", color=color, ec="black",
                                label=f"{short} quad opt ({qf['e_opt']:.0f} MWh / "
                                      f"{qf['p_opt']:.0f} MW, {qf['npv_opt']/1e6:.1f} MUSD"
                                      f"  R²={qf['r_squared']:.3f})"),
                )

    n_legend_cols = min(len(legend_handles), 4)
    fig.legend(handles=legend_handles, loc="lower center", ncol=n_legend_cols,
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.10))

    plt.tight_layout()
    out = OUTPUT_DIR / f"planB_2d_npv_comparison_{n}h_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  NPV comparison: {out.name}")
    plt.show()


def plot_sweep_slices(results: List[dict], params: dict, n: int, ts: str) -> None:
    """Plot key 1D cuts through the 2D NPV surface, including degradation deltas."""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.DataFrame(results)
    BG   = "#f7f9fc"
    C_ND = "#2166ac"   # blue   — no degradation
    C_XU = "#b5351b"   # red    — Xu
    C_SH = "#4daf4a"   # green  — Shi
    
    # Create a clean 2-panel slice figure (Left: NPV vs E_cap, Right: NPV vs P_cap)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor=BG)
    fig.patch.set_facecolor(BG)
    
    # --- PANEL 1: Fixed Optimal Power, vary E_cap ---
    # Find the absolute best power capacity from the Xu run to use as our cut line
    best_xu = max(results, key=lambda r: r["npv_with_xu"] if np.isfinite(r["npv_with_xu"]) else -np.inf)
    opt_p = best_xu["p_cap"]
    
    sub_e = df[df["p_cap"] == opt_p].sort_values("e_cap")
    
    ax1.set_facecolor(BG)
    ax1.plot(sub_e["e_cap"], sub_e["npv_no_deg"]/1e6, 'o-', color=C_ND, lw=2, ms=6,
             label="No Degradation")
    ax1.plot(sub_e["e_cap"], sub_e["npv_with_xu"]/1e6, 's-', color=C_XU, lw=2, ms=6,
             label="Xu Model")
    ax1.plot(sub_e["e_cap"], sub_e["npv_with_shi"]/1e6, '^-', color=C_SH, lw=2, ms=6,
             label="Shi Model")
    
    # Fill the gap to visually highlight the degradation cost penalty
    ax1.fill_between(sub_e["e_cap"], sub_e["npv_with_xu"]/1e6, sub_e["npv_no_deg"]/1e6, 
                     color=C_XU, alpha=0.10, label="Degradation Value Gap")
    
    ax1.set_title(f"NPV vs Energy Capacity (At Opt P = {opt_p} MW)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Energy Capacity [MWh]")
    ax1.set_ylabel("NPV [MUSD]")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()
    
    # --- PANEL 2: Fixed Optimal Energy, vary P_cap ---
    opt_e = best_xu["e_cap"]
    sub_p = df[df["e_cap"] == opt_e].sort_values("p_cap")
    
    ax2.set_facecolor(BG)
    ax2.plot(sub_p["p_cap"], sub_p["npv_no_deg"]/1e6, 'o-', color=C_ND, lw=2, ms=6,
             label="No Degradation")
    ax2.plot(sub_p["p_cap"], sub_p["npv_with_xu"]/1e6, 's-', color=C_XU, lw=2, ms=6,
             label="Xu Model")
    ax2.plot(sub_p["p_cap"], sub_p["npv_with_shi"]/1e6, '^-', color=C_SH, lw=2, ms=6,
             label="Shi Model")
    
    ax2.set_title(f"NPV vs Power Capacity (At Opt E = {opt_e} MWh)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Power Capacity [MW]")
    ax2.set_ylabel("NPV [MUSD]")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"planB_2d_slices_{ts}.png", dpi=300, facecolor=BG)
    plt.close()


def plot_sweep_2d(results: List[dict], params: dict, n: int, ts: str) -> None:
    """Heatmaps focusing purely on the degradation comparison between Xu and Shi."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    BG = "#f7f9fc"
    E = np.array(E_CAP_GRID, dtype=float)
    P = np.array(P_CAP_GRID, dtype=float)

    # 3-panel layout focusing purely on battery aging dynamics
    panels = [
        ("fd_xu_total",  "Annual Fraction Degradation (Xu)", "YlOrRd"),
        ("fd_shi_total", "Annual Fraction Degradation (Shi)", "YlOrRd"),
        ("efc",          "Equivalent Full Cycles (EFC)",      "Blues"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    
    fig.suptitle(
        f"Plan B 2D: Battery Degradation & Asset Cycling Comparison ({n} hours)",
        fontsize=14, fontweight="bold", y=0.98
    )

    for ax, (key, title, cmap) in zip(axes, panels):
        ax.set_facecolor(BG)
        data = _to_grid(results, key)

        im = ax.contourf(P, E, data, levels=20, cmap=cmap)
        cs = ax.contour(P, E, data, levels=10, colors="black", alpha=0.15, linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.3f" if "fd" in key else "%.0f")
        
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Power Capacity [MW]")
        ax.set_ylabel("Energy Capacity [MWh]")
        fig.colorbar(im, ax=ax, shrink=0.85, aspect=20)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"planB_2d_sweep_{ts}.png", dpi=300, facecolor=BG)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# Option C — Combined figure: difference heatmaps + line slices
# ════════════════════════════════════════════════════════════════════════════

def plot_npv_three_models_lifetime(
    results: List[dict], params: dict, n: int, ts: str
) -> None:
    """Lifetime sweep figure: 3 NPV heatmaps (row 1) + 2 line slices (row 2).

    Row 1 — three NPV heatmaps, identical structure to the annual version:
      - All three model optima shown as stars on every panel
      - Own model's star is larger; the other two are smaller for context
      - Shared colour scale across all three panels for direct comparison

    Row 2 — line slices through the reference design:
      - Left:  fixed E = nearest grid value to e_ref, sweep P
      - Right: fixed P = nearest grid value to p_ref, sweep E
      - Three lines per panel (no-deg / Xu / Shi), dotted verticals at optima

    Replacement heatmaps are NOT included here — they live in a separate
    optional figure if needed.  The slices replace them because they give
    the same directional insight in a format that is directly comparable
    to the annual sweep slices.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  matplotlib not available, skipping lifetime NPV plot.")
        return

    BG   = "#f7f9fc"
    C_ND = "#2166ac"
    C_XU = "#b5351b"
    C_SH = "#4daf4a"

    E_arr = np.array(E_CAP_GRID, dtype=float)
    P_arr = np.array(P_CAP_GRID, dtype=float)
    e_ref, p_ref = params["e_cap"], params["p_cap"]

    # ── Pull NPV grids ────────────────────────────────────────────────────
    grid_nd  = _to_grid(results, "npv_no_deg")  / 1e6
    grid_xu  = _to_grid(results, "npv_with_xu") / 1e6
    grid_shi = _to_grid(results, "npv_with_shi") / 1e6

    # ── Find all three optima ─────────────────────────────────────────────
    def _opt(grid):
        i, j = np.unravel_index(np.nanargmax(grid), grid.shape)
        return float(E_arr[i]), float(P_arr[j]), float(grid[i, j])

    e_nd, p_nd, npv_nd   = _opt(grid_nd)
    e_xu, p_xu, npv_xu   = _opt(grid_xu)
    e_shi, p_shi, npv_shi = _opt(grid_shi)

    # ── Locate slice axes through the Xu degradation optimum ─────────────
    # Slicing through the reference design (300/150) lands on filtered-out
    # NaN rows.  Instead, cut through the Xu optimal — that's the analysis-
    # relevant cross-section and guaranteed to have data.
    i_slice = int(np.argmin(np.abs(E_arr - e_xu)))   # row index for fixed E
    j_slice = int(np.argmin(np.abs(P_arr - p_xu)))   # col index for fixed P
    e_fix = E_arr[i_slice]
    p_fix = P_arr[j_slice]

    # Per-panel colour scales — shared scale crushes the degradation panels
    # into a narrow band because no-deg peaks at ~260 MUSD while Xu peaks at ~90.
    # Each panel gets its own range for meaningful colour differentiation.
    def _panel_range(grid):
        lo = np.nanmin(grid)
        hi = np.nanmax(grid)
        return lo, hi, np.linspace(lo, hi, 22)

    cal_label = "+calendar" if INCLUDE_CALENDAR else "cycle-only"

    # ── Figure layout: 2 rows × 3 cols ────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.30)

    fig.suptitle(
        f"Lifetime NPV design space: {N_YEARS}-year, three scenarios  "
        f"({n}h/yr, {cal_label})\n"
        f"Stars = each model's optimal design.  "
        f"Diamond = WP2 reference ({e_ref:.0f} MWh / {p_ref:.0f} MW).  "
        f"Battery replaced when SoH < {REPL_THRESHOLD*100:.0f}%.",
        fontsize=11, fontweight="bold", y=1.01,
    )

    # ── Helper to add margin so corner stars are not clipped ──────────────
    def _margin(ax):
        pm = 0.03 * (P_arr[-1] - P_arr[0])
        em = 0.03 * (E_arr[-1] - E_arr[0])
        ax.set_xlim(P_arr[0] - pm, P_arr[-1] + pm)
        ax.set_ylim(E_arr[0] - em, E_arr[-1] + em)

    # ── Row 0: three NPV heatmaps ─────────────────────────────────────────
    npv_panels = [
        (gs[0, 0], grid_nd,  "NPV, no degradation [MUSD]",  C_ND, e_nd,  p_nd),
        (gs[0, 1], grid_xu,  "NPV, Xu degradation [MUSD]",  C_XU, e_xu,  p_xu),
        (gs[0, 2], grid_shi, "NPV, Shi degradation [MUSD]", C_SH, e_shi, p_shi),
    ]

    # All three star specs — drawn on every panel
    star_specs = [
        (e_nd,  p_nd,  C_ND),
        (e_xu,  p_xu,  C_XU),
        (e_shi, p_shi, C_SH),
    ]

    for spec, grid, title, own_col, own_e, own_p in npv_panels:
        ax = fig.add_subplot(spec)
        ax.set_facecolor(BG)

        lo, hi, panel_levels = _panel_range(grid)
        im = ax.contourf(P_arr, E_arr, grid, levels=panel_levels,
                         cmap="RdYlGn", vmin=lo, vmax=hi)
        cs = ax.contour(P_arr, E_arr, grid, levels=10,
                        colors="k", linewidths=0.3, alpha=0.35)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")

        # Reference diamond
        ax.scatter([p_ref], [e_ref], s=90, marker="D",
                   color="white", edgecolors="black", linewidths=0.9, zorder=6)

        # Three stars: own model larger, others smaller
        for s_e, s_p, s_col in star_specs:
            is_own = (s_col == own_col and s_e == own_e and s_p == own_p)
            ax.scatter([s_p], [s_e],
                       s=150,
                       marker="*", color=s_col,
                       edgecolors="black",
                       linewidths=0.9,
                       zorder=8 if is_own else 7)

        _margin(ax)
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("MUSD", fontsize=9)
        ax.set_xlabel("Power capacity P [MW]", fontsize=10)
        ax.set_ylabel("Energy capacity E [MWh]", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)

    # ── Row 1, left: slice — fixed E, sweep P ────────────────────────────
    ax_bl = fig.add_subplot(gs[1, 0:2])   # span two columns for more room
    ax_bl.set_facecolor(BG)
    ax_bl.plot(P_arr, grid_nd[i_slice, :],  "o-", color=C_ND, lw=2, ms=6,
               label="No degradation")
    ax_bl.plot(P_arr, grid_xu[i_slice, :],  "s-", color=C_XU, lw=2, ms=6,
               label="With Xu degradation")
    ax_bl.plot(P_arr, grid_shi[i_slice, :], "^-", color=C_SH, lw=2, ms=6,
               label="With Shi degradation")

    opt_p_nd  = P_arr[np.nanargmax(grid_nd[i_slice, :])]
    opt_p_xu  = P_arr[np.nanargmax(grid_xu[i_slice, :])]
    opt_p_shi = P_arr[np.nanargmax(grid_shi[i_slice, :])]
    # Dotted verticals — colour-matched, no legend entries (title explains them)
    ax_bl.axvline(opt_p_nd,  color=C_ND, linestyle=":", alpha=0.7, lw=1.4)
    ax_bl.axvline(opt_p_xu,  color=C_XU, linestyle=":", alpha=0.7, lw=1.4)
    ax_bl.axvline(opt_p_shi, color=C_SH, linestyle=":", alpha=0.7, lw=1.4)
    ax_bl.axvline(p_ref, color="gray", linestyle="--", lw=0.9, alpha=0.7)

    shift_note = (f"  Deg. shifts opt P by {opt_p_xu - opt_p_nd:+.0f} MW"
                  if opt_p_nd != opt_p_xu else "  Deg. does not shift opt P")
    ax_bl.set_xlabel("Power capacity P [MW]", fontsize=11)
    ax_bl.set_ylabel("Lifetime NPV [MUSD]", fontsize=11)
    ax_bl.set_title(
        f"NPV vs P,  fixed E = {e_fix:.0f} MWh\n"
        f"Dotted = optimal P per model.{shift_note}",
        fontsize=10, fontweight="bold",
    )
    ax_bl.legend(fontsize=9)
    ax_bl.tick_params(labelsize=9)

    # ── Row 1, right: slice — fixed P, sweep E ───────────────────────────
    ax_br = fig.add_subplot(gs[1, 2])
    ax_br.set_facecolor(BG)
    ax_br.plot(E_arr, grid_nd[:, j_slice],  "o-", color=C_ND, lw=2, ms=6,
               label="No degradation")
    ax_br.plot(E_arr, grid_xu[:, j_slice],  "s-", color=C_XU, lw=2, ms=6,
               label="With Xu degradation")
    ax_br.plot(E_arr, grid_shi[:, j_slice], "^-", color=C_SH, lw=2, ms=6,
               label="With Shi degradation")

    opt_e_nd  = E_arr[np.nanargmax(grid_nd[:, j_slice])]
    opt_e_xu  = E_arr[np.nanargmax(grid_xu[:, j_slice])]
    opt_e_shi = E_arr[np.nanargmax(grid_shi[:, j_slice])]
    ax_br.axvline(opt_e_nd,  color=C_ND, linestyle=":", alpha=0.7, lw=1.4)
    ax_br.axvline(opt_e_xu,  color=C_XU, linestyle=":", alpha=0.7, lw=1.4)
    ax_br.axvline(opt_e_shi, color=C_SH, linestyle=":", alpha=0.7, lw=1.4)
    ax_br.axvline(e_ref, color="gray", linestyle="--", lw=0.9, alpha=0.7)

    # Shade between no-deg and Xu to show degradation cost
    ax_br.fill_between(
        E_arr,
        grid_xu[:, j_slice],
        grid_nd[:, j_slice],
        alpha=0.15, color=C_XU,
        label="Xu degradation cost\n(gap to blue line)",
    )

    ax_br.set_xlabel("Energy capacity E [MWh]", fontsize=11)
    ax_br.set_ylabel("Lifetime NPV [MUSD]", fontsize=11)
    ax_br.set_title(
        f"NPV vs E,  fixed P = {p_fix:.0f} MW\n"
        "Dotted = optimal E per model.  Shaded = Xu degradation cost.",
        fontsize=10, fontweight="bold",
    )
    ax_br.legend(fontsize=9)
    ax_br.tick_params(labelsize=9)

    # ── Shared legend at the bottom ───────────────────────────────────────
    # plt.scatter([], []) creates a zero-point artist for the legend icon only
    legend_handles = [
        mpatches.Patch(color="white", ec="black",
                       label=f"Reference ({e_ref:.0f}/{p_ref:.0f})"),
        plt.scatter([], [], s=300, marker="*", color=C_ND, ec="black",
                    label=f"No-deg optimum  "
                          f"({e_nd:.0f}/{p_nd:.0f} MW, {npv_nd:.1f} MUSD)"),
        plt.scatter([], [], s=300, marker="*", color=C_XU, ec="black",
                    label=f"Xu optimum      "
                          f"({e_xu:.0f}/{p_xu:.0f} MW, {npv_xu:.1f} MUSD)"),
        plt.scatter([], [], s=300, marker="*", color=C_SH, ec="black",
                    label=f"Shi optimum     "
                          f"({e_shi:.0f}/{p_shi:.0f} MW, {npv_shi:.1f} MUSD)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

    out = OUTPUT_DIR / f"planB_lifetime_npv_{n}h_{N_YEARS}yr_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Lifetime NPV plot: {out.name}")
    plt.show()

# ════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print(f"PLAN B: Parameter sweep, NPV vs (E_cap, P_cap)  [mode: {SWEEP_MODE}]")
    print("=" * 78)
    print(f"  E_cap grid:     {E_CAP_GRID} MWh")
    print(f"  P_cap grid:     {P_CAP_GRID} MW")
    print(f"  Grid points:    {len(E_CAP_GRID) * len(P_CAP_GRID)}")
    print(f"  Calendar aging: {INCLUDE_CALENDAR}")
    print(f"  Horizon:        {RUN_HOURS} hours")
    if SWEEP_MODE == "lifetime":
        print(f"  N years:        {N_YEARS}")
        print(f"  Repl threshold: SoH < {REPL_THRESHOLD*100:.0f}%")
        print(f"  NPV floor:      {LIFETIME_NPV_FLOOR*100:.0f}% of annual max")
        print(f"  Duration guard:  {LIFETIME_MIN_DURATION_H}–{LIFETIME_MAX_DURATION_H} h")
        print(f"  (Annual sweep runs first for smart filtering)")

    # ── Resolve which CSV to load, if any ────────────────────────────────
    csv_to_load: Path | None = None
    if REPLOT_CSV is not None:
        csv_to_load = Path(REPLOT_CSV)
    elif REPLOT_FROM_LAST:
        csv_to_load = find_latest_csv()

    # ── Branch: replot from CSV or run the sweep ──────────────────────────
    if csv_to_load is not None:
        print(f"\n[REPLOT MODE] Loading from CSV, LP sweep will be skipped.")
        results, params, n = load_and_verify(csv_to_load)
        stem_parts = csv_to_load.stem.split("_")
        ts = f"{stem_parts[-2]}_{stem_parts[-1]}_replot"

    elif SWEEP_MODE == "annual":
        print(f"\n[1/3] Loading data...")
        power_wind, price_eur, params = load_data(RUN_HOURS)
        n = len(price_eur)

        print(f"\n[2/3] Running annual 2D sweep "
              f"({len(E_CAP_GRID)*len(P_CAP_GRID)} points)...")
        t0 = time.perf_counter()
        results = run_sweep_2d(power_wind, price_eur, params)
        print(f"  Total sweep time: {time.perf_counter()-t0:.1f}s")

        print(f"\n[3/3] Saving results...")
        _, _, ts = save_results_2d(results, params, n)

    elif SWEEP_MODE == "lifetime":
        print(f"\n[1/4] Loading data...")
        power_wind, price_eur, params = load_data(RUN_HOURS)
        n = len(price_eur)

        # ── Stage 1: fast annual sweep for smart filtering ────────────────
        print(f"\n[2/4] Running annual sweep for smart filtering "
              f"({len(E_CAP_GRID)*len(P_CAP_GRID)} points)...")
        t0 = time.perf_counter()
        annual_results = run_sweep_2d(power_wind, price_eur, params)
        print(f"  Annual sweep: {time.perf_counter()-t0:.1f}s")

        # ── Stage 2: lifetime sweep on filtered points ────────────────────
        print(f"\n[3/4] Running lifetime 2D sweep (smart-filtered)...")
        t0 = time.perf_counter()
        results = run_sweep_20year(power_wind, price_eur, params,
                                   annual_results=annual_results)
        elapsed = time.perf_counter() - t0
        print(f"  Lifetime sweep: {elapsed/60:.1f} min")

        print(f"\n[4/4] Saving results...")
        _, _, ts = save_results_20year(results, params, n)

    else:
        raise ValueError(f"Unknown SWEEP_MODE: '{SWEEP_MODE}'. "
                         "Set to 'annual' or 'lifetime'.")

# ── Quadratic Surface Fitting ──────────────────────────────────────────
    quad_fits = run_quadratic_fits(results)

# ── Plotting Pipeline Execution ─────────────────────────────────────────
    if not SKIP_PLOT:
        if SWEEP_MODE == "lifetime":
            plot_npv_three_models_lifetime(results, params, n, ts)
        else:
            # The complete 4-figure portfolio for your thesis
            plot_npv_three_models(results, params, n, ts, quad_fits=quad_fits)
            plot_sweep_2d(results, params, n, ts)
            plot_sweep_slices(results, params, n, ts)

    # ── Print optimal design point ─────────────────────────────────────────
    print("\n" + "=" * 78)
    print("OPTIMAL DESIGN POINT")
    print("=" * 78)
    for label, key in [("Without degradation", "npv_no_deg"),
                       ("With Xu degradation", "npv_with_xu"),
                       ("With Shi degradation", "npv_with_shi")]:
        best = max(results, key=lambda r, k=key: r[k] if np.isfinite(r[k]) else -np.inf)
        ep_h = best["e_cap"] / best["p_cap"]
        line = (f"  {label:<25s}:  E={best['e_cap']:.0f} MWh  P={best['p_cap']:.0f} MW  "
                f"E/P={ep_h:.1f}h  NPV={best[key]/1e6:.2f} MUSD")
        if key in quad_fits and quad_fits[key].get("is_concave"):
            qf = quad_fits[key]
            line += (f"  |  quad: E*={qf['e_opt']:.0f}  P*={qf['p_opt']:.0f}  "
                     f"NPV*={qf['npv_opt']/1e6:.2f}  R²={qf['r_squared']:.3f}")
        print(line)

    print("\nDone.")
    return results


if __name__ == "__main__":
    main()