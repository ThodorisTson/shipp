"""Plan B — 2D parameter sweep: NPV vs (E_cap, P_cap) with degradation.

Extends planB_parameter_sweep.py (1D, E_cap only) to a 2D grid over
both energy capacity E_cap and power capacity P_cap.

What changed from the 1D version
---------------------------------
1. TOGGLES:  E_CAP_GRID and P_CAP_GRID are now independent lists.
             C_RATE removed — P_cap is no longer derived from E_cap.
2. run_sweep_2d:   nested loop over (E_cap, P_cap) pairs.
3. save_results_2d: CSV gains a p_cap column; report gains a 2D table.
4. plot_sweep_2d:  heatmaps over the (E, P) grid replacing line plots.
5. Bug fix:   fd_xu_cycle was incorrectly set to xu_result["fd"] (total,
              including calendar).  Corrected to xu_result["fd_cycle"].

What is IDENTICAL to the 1D version
-------------------------------------
- All imports
- load_data()              — unchanged
- evaluate_single_ecap()   — already took (e_cap, p_cap) separately; zero changes
- All degradation logic inside evaluate_single_ecap
- All NPV / revenue / capex formulas
- File paths and output directory

Usage
    python planB_2d_parameter_sweep.py
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

# E_cap sweep values [MWh]  — ±50% around WP2 reference (300 MWh)
E_CAP_GRID = [150, 200, 250, 300, 350, 400, 450]

# P_cap sweep values [MW]   — ±50% around WP2 reference (150 MW)
# Independent from E_cap: total grid points = len(E_CAP_GRID) × len(P_CAP_GRID)
P_CAP_GRID = [75, 100, 125, 150, 175, 200, 225]

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

    factor = npf.npv(DISCOUNT_RATE, np.ones(N_YEARS)) - 1

    # Revenue (annual, NPV-discounted)
    revenue_usd = 365.0 * 24.0 / n * factor * float(
        np.dot(price_eur * EUR_TO_USD, p_vec)) * DT

    # Capex
    capex_usd = p_cost * p_cap + e_cost * e_cap

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
    """Return the most recent planB_2d_sweep_*.csv in OUTPUT_DIR.

    Sorting is done on the timestamp embedded in the filename
    (YYYYMMDD_HHMMSS), so it is independent of filesystem modification times.
    Raises FileNotFoundError if no matching CSV exists yet.
    """
    candidates = sorted(OUTPUT_DIR.glob("planB_2d_sweep_*.csv"))
    # glob returns paths in arbitrary order; sort alphabetically — the
    # embedded timestamp makes alphabetical order identical to time order.
    if not candidates:
        raise FileNotFoundError(
            f"No planB_2d_sweep_*.csv files found in {OUTPUT_DIR}.\n"
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


def run_sweep_2d(
    power_wind: np.ndarray,
    price_eur:  np.ndarray,
    params:     dict,
) -> List[dict]:
    """Run the 2D parameter sweep over all (E_cap, P_cap) combinations.

    evaluate_single_ecap() is called identically to the 1D version —
    the only change is the loop structure.  Every result dict has an
    added 'p_cap' key which was already returned by evaluate_single_ecap.
    """
    total = len(E_CAP_GRID) * len(P_CAP_GRID)
    results = []
    done = 0
    t_sweep_start = time.perf_counter()

    for e_cap in E_CAP_GRID:
        for p_cap in P_CAP_GRID:
            done += 1
            # Clamp P_cap to grid connection limit
            p_cap_eff = min(p_cap, params["p_max"])

            elapsed = time.perf_counter() - t_sweep_start
            eta = (elapsed / done) * (total - done) if done > 1 else 0.0
            print(f"  [{done:3d}/{total}]  E={e_cap:5.0f} MWh  P={p_cap_eff:5.0f} MW  "
                  f"ETA={eta:.0f}s ...", end="", flush=True)

            r = evaluate_single_ecap(e_cap, p_cap_eff, power_wind, price_eur, params)
            results.append(r)

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


def plot_npv_three_models(results: List[dict], params: dict, n: int, ts: str) -> None:
    """Jenna's requested figure: three NPV heatmaps side by side.

    One panel per degradation scenario: no-degradation, Xu, Shi.
    Every panel carries all three model optima as differently-coloured stars
    so the reader can immediately see whether the optimal design shifts between
    scenarios without needing a separate difference figure.

    Star colours
    ------------
    Blue star  : optimum of the no-degradation NPV surface
    Red star   : optimum of the Xu-degradation NPV surface
    Green star : optimum of the Shi-degradation NPV surface

    All three stars appear on all three panels.  On each panel, the star whose
    model matches the panel is drawn slightly larger to indicate "this is the
    optimum you are currently looking at."

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
        f"Stars show each model's optimal design.  Diamond = WP2 reference "
        f"({e_ref:.0f} MWh / {p_ref:.0f} MW)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    panels = [
        (axes[0], grid_nd,  "NPV, no degradation [MUSD]",  C_ND, e_nd,  p_nd,  npv_nd),
        (axes[1], grid_xu,  "NPV with Xu degradation [MUSD]", C_XU, e_xu,  p_xu,  npv_xu),
        (axes[2], grid_shi, "NPV with Shi degradation [MUSD]", C_SH, e_shi, p_shi, npv_shi),
    ]

    # Use a fixed set of contour levels across all panels for comparability
    levels = np.linspace(vmin, vmax, 22)

    for ax, grid, title, own_color, own_e, own_p, own_npv in panels:
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

        # ── Axis margin so corner stars are fully visible ─────────────────
        p_margin = 0.04 * (P_arr[-1] - P_arr[0])
        e_margin = 0.04 * (E_arr[-1] - E_arr[0])
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
    # plt.scatter([], []) creates a zero-point artist used purely as a legend
    # handle — it draws nothing on the figure but gives a coloured star icon
    # in the legend box.  This is the standard matplotlib pattern for custom
    # scatter legend entries.
    legend_handles = [
        mpatches.Patch(color="white", ec="black", label=f"Reference ({e_ref:.0f}/{p_ref:.0f})"),
        plt.scatter([], [], s=300, marker="*", color=C_ND, ec="black",
                    label=f"No-deg optimum  ({e_nd:.0f} MWh / {p_nd:.0f} MW, "
                          f"{npv_nd:.1f} MUSD)"),
        plt.scatter([], [], s=300, marker="*", color=C_XU, ec="black",
                    label=f"Xu optimum      ({e_xu:.0f} MWh / {p_xu:.0f} MW, "
                          f"{npv_xu:.1f} MUSD)"),
        plt.scatter([], [], s=300, marker="*", color=C_SH, ec="black",
                    label=f"Shi optimum     ({e_shi:.0f} MWh / {p_shi:.0f} MW, "
                          f"{npv_shi:.1f} MUSD)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.07))

    plt.tight_layout()
    out = OUTPUT_DIR / f"planB_2d_npv_comparison_{n}h_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  NPV comparison: {out.name}")
    plt.show()


def plot_sweep_slices(results: List[dict], params: dict, n: int, ts: str) -> None:
    """Option A — Line slices through the reference design point.

    Produces two panels, each identical in style to the original 1D plot:
      Panel 1: fix E = nearest grid row to e_ref (300 MWh), sweep P_cap
      Panel 2: fix P = nearest grid col to p_ref (150 MW),  sweep E_cap

    Three lines per panel: no-degradation, Xu, Shi.
    This directly answers: does the model choice shift the optimal P for a
    given E, and the optimal E for a given P?
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping slice plots.")
        return

    BG   = "#f7f9fc"
    C_ND = "#2166ac"   # no-degradation: blue
    C_XU = "#b5351b"   # Xu:  red
    C_SH = "#4daf4a"   # Shi: green

    E_arr = np.array(E_CAP_GRID, dtype=float)
    P_arr = np.array(P_CAP_GRID, dtype=float)
    e_ref, p_ref = params["e_cap"], params["p_cap"]

    # ── Find the grid row/col nearest to the reference design ─────────────
    i_ref = int(np.argmin(np.abs(E_arr - e_ref)))   # row index for e_ref
    j_ref = int(np.argmin(np.abs(P_arr - p_ref)))   # col index for p_ref
    e_fix = E_arr[i_ref]                             # actual E used (may differ from e_ref)
    p_fix = P_arr[j_ref]                             # actual P used

    # Pull the three NPV grids once
    grid_nd  = _to_grid(results, "npv_no_deg")  / 1e6
    grid_xu  = _to_grid(results, "npv_with_xu") / 1e6
    grid_shi = _to_grid(results, "npv_with_shi") / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    cal_label = "+calendar" if INCLUDE_CALENDAR else "cycle-only"
    fig.suptitle(
        f"Plan B 2D: Line slices through reference design  ({n}h, {cal_label})",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: fix E ≈ e_ref, sweep P ───────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.plot(P_arr, grid_nd[i_ref, :],  "o-", color=C_ND, lw=2, ms=6, label="No degradation")
    ax.plot(P_arr, grid_xu[i_ref, :],  "s-", color=C_XU, lw=2, ms=6, label="With Xu degradation")
    ax.plot(P_arr, grid_shi[i_ref, :], "^-", color=C_SH, lw=2, ms=6, label="With Shi degradation")

    # Mark optimal P per model
    for data, col in [(grid_nd[i_ref, :], C_ND), (grid_xu[i_ref, :], C_XU),
                      (grid_shi[i_ref, :], C_SH)]:
        p_opt = P_arr[np.argmax(data)]
        ax.axvline(p_opt, color=col, linestyle=":", alpha=0.55, lw=1.2)

    ax.axvline(p_ref, color="gray", linestyle="--", lw=0.9, alpha=0.7, label=f"Reference P={p_ref:.0f} MW")
    ax.set_xlabel("Power capacity P [MW]", fontsize=11)
    ax.set_ylabel("NPV [MUSD]", fontsize=11)
    ax.set_title(f"Fixed E = {e_fix:.0f} MWh  (sweep P)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    # ── Panel 2: fix P ≈ p_ref, sweep E ───────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.plot(E_arr, grid_nd[:, j_ref],  "o-", color=C_ND, lw=2, ms=6, label="No degradation")
    ax2.plot(E_arr, grid_xu[:, j_ref],  "s-", color=C_XU, lw=2, ms=6, label="With Xu degradation")
    ax2.plot(E_arr, grid_shi[:, j_ref], "^-", color=C_SH, lw=2, ms=6, label="With Shi degradation")

    for data, col in [(grid_nd[:, j_ref], C_ND), (grid_xu[:, j_ref], C_XU),
                      (grid_shi[:, j_ref], C_SH)]:
        e_opt = E_arr[np.argmax(data)]
        ax2.axvline(e_opt, color=col, linestyle=":", alpha=0.55, lw=1.2)

    ax2.axvline(e_ref, color="gray", linestyle="--", lw=0.9, alpha=0.7, label=f"Reference E={e_ref:.0f} MWh")
    ax2.set_xlabel("Energy capacity E [MWh]", fontsize=11)
    ax2.set_ylabel("NPV [MUSD]", fontsize=11)
    ax2.set_title(f"Fixed P = {p_fix:.0f} MW  (sweep E)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.tick_params(labelsize=9)

    plt.tight_layout()
    out = OUTPUT_DIR / f"planB_2d_slices_{n}h_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Slice plot: {out.name}")
    plt.show()


def plot_sweep_2d(results: List[dict], params: dict, n: int, ts: str) -> None:
    """Heatmaps of NPV and degradation over the (E_cap, P_cap) grid."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  matplotlib not available, skipping plots.")
        return

    BG = "#f7f9fc"
    E  = np.array(E_CAP_GRID, dtype=float)
    P  = np.array(P_CAP_GRID, dtype=float)

    # Reference design (WP2 nominal)
    e_ref, p_ref = params["e_cap"], params["p_cap"]

    panels = [
        ("npv_no_deg",   "NPV without degradation [MUSD]",  "RdYlGn"),
        ("npv_with_xu",  "NPV with Xu degradation [MUSD]",   "RdYlGn"),
        ("fd_xu_total",  "Annual fd_Xu (total)",              "YlOrRd"),
        ("fd_shi_total", "Annual fd_Shi (cycle + calendar)",  "YlOrRd"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
    fig.patch.set_facecolor(BG)
    cal_label = "+calendar" if INCLUDE_CALENDAR else "cycle-only"
    fig.suptitle(
        f"Plan B 2D: NPV and degradation over (E_cap, P_cap) grid\n"
        f"({n} hours, {cal_label})",
        fontsize=13, fontweight="bold",
    )

    for ax, (key, title, cmap) in zip(axes.flat, panels):
        ax.set_facecolor(BG)
        data = _to_grid(results, key)

        # Scale NPV panels to MUSD
        if "npv" in key:
            data = data / 1e6

        im = ax.contourf(P, E, data, levels=20, cmap=cmap)
        cs = ax.contour(P, E, data, levels=10, colors="k",
                        linewidths=0.35, alpha=0.4)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")

        # Mark optimal point
        flat_idx = np.nanargmax(-data if "fd" in key else data)
        i_opt, j_opt = np.unravel_index(flat_idx, data.shape)
        ax.scatter(P[j_opt], E[i_opt], s=160, marker="*",
                   color="white", edgecolors="black", linewidths=0.8,
                   zorder=6, label=f"Optimum ({E[i_opt]:.0f}/{P[j_opt]:.0f})")

        # Mark WP2 reference design
        ax.scatter([p_ref], [e_ref], s=80, marker="D",
                   color="black", edgecolors="white", linewidths=0.8,
                   zorder=6, label=f"Reference ({e_ref:.0f}/{p_ref:.0f})")

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        ax.set_xlabel("Power capacity P [MW]", fontsize=10)
        ax.set_ylabel("Energy capacity E [MWh]", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    out = OUTPUT_DIR / f"planB_2d_sweep_{n}h_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Plot: {out.name}")

    # ── Second figure: NPV shift (degradation effect on optimal design) ──
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig2.patch.set_facecolor(BG)
    fig2.suptitle("Degradation effect: NPV shift from no-degradation baseline",
                  fontsize=12, fontweight="bold")

    grid_no  = _to_grid(results, "npv_no_deg")  / 1e6
    grid_xu  = _to_grid(results, "npv_with_xu") / 1e6
    grid_shi = _to_grid(results, "npv_with_shi") / 1e6

    for ax2, (shift, label) in zip(axes2, [
        (grid_xu  - grid_no, "NPV shift: Xu degradation [MUSD]"),
        (grid_shi - grid_no, "NPV shift: Shi degradation [MUSD]"),
    ]):
        ax2.set_facecolor(BG)
        im2 = ax2.contourf(P, E, shift, levels=20, cmap="RdBu_r")
        ax2.contour(P, E, shift, levels=[0], colors="black",
                    linewidths=1.2, linestyles="--")
        ax2.scatter([p_ref], [e_ref], s=80, marker="D",
                    color="black", edgecolors="white", linewidths=0.8, zorder=6)
        cbar2 = fig2.colorbar(im2, ax=ax2, pad=0.02)
        cbar2.ax.tick_params(labelsize=8)
        ax2.set_xlabel("Power capacity P [MW]", fontsize=10)
        ax2.set_ylabel("Energy capacity E [MWh]", fontsize=10)
        ax2.set_title(label, fontsize=10, fontweight="bold")
        ax2.tick_params(labelsize=9)

    plt.tight_layout()
    out2 = OUTPUT_DIR / f"planB_2d_shift_{n}h_{ts}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Shift plot: {out2.name}")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# Option C — Combined figure: difference heatmaps + line slices
# ════════════════════════════════════════════════════════════════════════════

def plot_sweep_combined(results: List[dict], params: dict, n: int, ts: str) -> None:
    """Option C — Redesigned 4-panel combined figure (2 × 2).

    Layout
    ------
    Top-left:  Degradation cost heatmap  (NPV_noDeg − NPV_Xu) [MUSD]
               Star = NPV_Xu optimum (450/175).  Shows WHERE degradation
               hurts most and how much it costs at the best design point.

    Top-right: Model gap heatmap  (NPV_Xu − NPV_Shi) [MUSD]
               Star = point of maximum model disagreement (argmin of gap).
               Tells you whether the two models steer you to different designs.
               Zero contour in dashed black (though it never crosses zero here).

    Bot-left:  Line slice — fixed E ≈ e_ref, sweep P.
               Three lines (noDeg/Xu/Shi). Dotted verticals at each optimum.
               Shows: does degradation shift the optimal P at this E?

    Bot-right: Line slice — fixed P ≈ p_ref, sweep E.
               Three lines. Shows: does degradation shift the optimal E?

    Star placement rules
    --------------------
    - Degradation cost panel:  star = NPV_Xu optimum (same design you'd build).
      This lets you read "the best design has X MUSD annual degradation cost."
    - Model gap panel:         star = argmin of model_gap (max disagreement).
      This marks where the two models most diverge in their penalty estimate.
    - Stars computed from data are NEVER placed on difference panels because
      min(degradation_cost) trivially = smallest battery, which is meaningless.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  matplotlib not available, skipping combined plot.")
        return

    BG   = "#f7f9fc"
    C_ND = "#2166ac"
    C_XU = "#b5351b"
    C_SH = "#4daf4a"

    E_arr = np.array(E_CAP_GRID, dtype=float)
    P_arr = np.array(P_CAP_GRID, dtype=float)
    e_ref, p_ref = params["e_cap"], params["p_cap"]

    i_ref = int(np.argmin(np.abs(E_arr - e_ref)))
    j_ref = int(np.argmin(np.abs(P_arr - p_ref)))
    e_fix = E_arr[i_ref]
    p_fix = P_arr[j_ref]

    # ── Pull all grids once ────────────────────────────────────────────────
    grid_nd   = _to_grid(results, "npv_no_deg")  / 1e6
    grid_xu   = _to_grid(results, "npv_with_xu") / 1e6
    grid_shi  = _to_grid(results, "npv_with_shi") / 1e6

    deg_cost  = grid_nd - grid_xu     # always ≥ 0, higher = more costly [MUSD]
    model_gap = grid_xu - grid_shi    # always ≤ 0, more negative = larger gap [MUSD]

    # ── Pre-compute the two meaningful star positions ──────────────────────
    # Star 1: NPV_Xu optimum — used on the degradation cost panel
    i_xu_opt, j_xu_opt = np.unravel_index(np.nanargmax(grid_xu), grid_xu.shape)
    e_xu_opt = E_arr[i_xu_opt]
    p_xu_opt = P_arr[j_xu_opt]
    deg_at_opt = float(deg_cost[i_xu_opt, j_xu_opt])   # cost at the optimal design

    # Star 2: maximum model disagreement — used on the model gap panel
    i_gap_max, j_gap_max = np.unravel_index(np.nanargmin(model_gap), model_gap.shape)
    e_gap_max = E_arr[i_gap_max]
    p_gap_max = P_arr[j_gap_max]
    gap_at_max = float(model_gap[i_gap_max, j_gap_max])

    # ── Helper: draw one heatmap panel ────────────────────────────────────
    def _heatmap(ax, data, title, cmap, unit,
                 star_e, star_p, star_label,
                 star_color="gold",
                 zero_contour=False):
        ax.set_facecolor(BG)
        im = ax.contourf(P_arr, E_arr, data, levels=20, cmap=cmap)
        cs = ax.contour(P_arr, E_arr, data, levels=10,
                        colors="k", linewidths=0.3, alpha=0.35)
        ax.clabel(cs, inline=True, fontsize=7,
                  fmt="%.2f" if np.nanmax(np.abs(data)) < 5 else "%.1f")

        if zero_contour:
            try:
                ax.contour(P_arr, E_arr, data, levels=[0],
                           colors="black", linewidths=1.4, linestyles="--")
            except Exception:
                pass  # no zero crossing in the data — skip silently

        # Reference diamond (always white so it's visible on any colormap)
        ax.scatter([p_ref], [e_ref], s=90, marker="D",
                   color="white", edgecolors="black", linewidths=0.9, zorder=6,
                   label=f"Reference ({e_ref:.0f}/{p_ref:.0f})")

        # Star at the explicitly provided position
        ax.scatter([star_p], [star_e], s=200, marker="*",
                   color=star_color, edgecolors="black", linewidths=0.7, zorder=7,
                   label=star_label)

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(unit, fontsize=9)
        ax.set_xlabel("Power capacity P [MW]", fontsize=10)
        ax.set_ylabel("Energy capacity E [MWh]", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.90,
                  handlelength=1.2, borderpad=0.6)
        ax.tick_params(labelsize=9)

        # ── Extend axis limits 7% beyond data range so boundary stars are
        # fully visible. Contours and data still end at the last data point.
        p_margin = 0.04 * (P_arr[-1] - P_arr[0])
        e_margin = 0.04 * (E_arr[-1] - E_arr[0])
        ax.set_xlim(P_arr[0] - p_margin, P_arr[-1] + p_margin)
        ax.set_ylim(E_arr[0] - e_margin, E_arr[-1] + e_margin)

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 13), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)

    cal_label = "+calendar" if INCLUDE_CALENDAR else "cycle-only"
    fig.suptitle(
        f"Plan B 2D: Design space analysis  ({n}h, {cal_label})\n"
        f"Reference: E={e_ref:.0f} MWh / P={p_ref:.0f} MW",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # ── Top-left: degradation cost surface ────────────────────────────────
    ax_tl = fig.add_subplot(gs[0, 0])
    _heatmap(
        ax_tl, deg_cost,
        title="Degradation cost:  NPV(no-deg) − NPV(Xu)  [MUSD]",
        cmap="YlOrRd",
        unit="MUSD  (larger = degradation hurts more here)",
        star_e=e_xu_opt, star_p=p_xu_opt,
        star_label=(f"NPV(Xu) optimum ({e_xu_opt:.0f}/{p_xu_opt:.0f})\n"
                    f"Deg cost = {deg_at_opt:.1f} MUSD"),
    )

    # ── Top-right: model gap surface ──────────────────────────────────────
    ax_tr = fig.add_subplot(gs[0, 1])
    _heatmap(
        ax_tr, model_gap,
        title="Model gap:  NPV(Xu) − NPV(Shi)  [MUSD]",
        cmap="RdBu",
        unit="MUSD  (more negative = Xu penalises more than Shi)",
        star_e=e_gap_max, star_p=p_gap_max,
        star_label=(f"Max disagreement ({e_gap_max:.0f}/{p_gap_max:.0f})\n"
                    f"Gap = {gap_at_max:.2f} MUSD"),
        star_color="orangered",
        zero_contour=True,
    )

    # ── Bottom-left: slice — fixed E, sweep P ────────────────────────────
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_bl.set_facecolor(BG)
    ax_bl.plot(P_arr, grid_nd[i_ref, :],  "o-", color=C_ND, lw=2, ms=6,
               label="No degradation")
    ax_bl.plot(P_arr, grid_xu[i_ref, :],  "s-", color=C_XU, lw=2, ms=6,
               label="With Xu degradation")
    ax_bl.plot(P_arr, grid_shi[i_ref, :], "^-", color=C_SH, lw=2, ms=6,
               label="With Shi degradation")

    opt_p_nd  = P_arr[np.argmax(grid_nd[i_ref, :])]
    opt_p_xu  = P_arr[np.argmax(grid_xu[i_ref, :])]
    opt_p_shi = P_arr[np.argmax(grid_shi[i_ref, :])]

    # Dotted verticals without legend entries — the title note explains them
    ax_bl.axvline(opt_p_nd,  color=C_ND, linestyle=":", alpha=0.7, lw=1.4)
    ax_bl.axvline(opt_p_xu,  color=C_XU, linestyle=":", alpha=0.7, lw=1.4)
    ax_bl.axvline(opt_p_shi, color=C_SH, linestyle=":", alpha=0.7, lw=1.4)
    ax_bl.axvline(p_ref, color="gray", linestyle="--", lw=0.9, alpha=0.7)

    ax_bl.set_xlabel("Power capacity P [MW]", fontsize=11)
    ax_bl.set_ylabel("NPV [MUSD]", fontsize=11)
    # Note the shift value directly in the title so no annotation is needed
    shift_note = (f"  |  Deg. shifts opt P by {opt_p_xu - opt_p_nd:+.0f} MW"
                  if opt_p_nd != opt_p_xu else "")
    ax_bl.set_title(
        f"NPV vs P,  fixed E = {e_fix:.0f} MWh\n"
        f"Dotted lines = optimal P per model{shift_note}",
        fontsize=10, fontweight="bold",
    )
    ax_bl.legend(fontsize=9)
    ax_bl.tick_params(labelsize=9)

    # ── Bottom-right: slice — fixed P, sweep E ───────────────────────────
    ax_br = fig.add_subplot(gs[1, 1])
    ax_br.set_facecolor(BG)
    ax_br.plot(E_arr, grid_nd[:, j_ref],  "o-", color=C_ND, lw=2, ms=6,
               label="No degradation")
    ax_br.plot(E_arr, grid_xu[:, j_ref],  "s-", color=C_XU, lw=2, ms=6,
               label="With Xu degradation")
    ax_br.plot(E_arr, grid_shi[:, j_ref], "^-", color=C_SH, lw=2, ms=6,
               label="With Shi degradation")

    opt_e_nd  = E_arr[np.argmax(grid_nd[:, j_ref])]
    opt_e_xu  = E_arr[np.argmax(grid_xu[:, j_ref])]
    opt_e_shi = E_arr[np.argmax(grid_shi[:, j_ref])]

    # Dotted verticals without legend entries
    ax_br.axvline(opt_e_nd,  color=C_ND, linestyle=":", alpha=0.7, lw=1.4)
    ax_br.axvline(opt_e_xu,  color=C_XU, linestyle=":", alpha=0.7, lw=1.4)
    ax_br.axvline(opt_e_shi, color=C_SH, linestyle=":", alpha=0.7, lw=1.4)
    ax_br.axvline(e_ref, color="gray", linestyle="--", lw=0.9, alpha=0.7)

    # Shade between no-deg and Xu — height = Xu degradation cost at each E
    ax_br.fill_between(
        E_arr,
        grid_xu[:, j_ref],
        grid_nd[:, j_ref],
        alpha=0.15, color=C_XU,
        label="Xu degradation cost\n(vertical gap to blue line)",
    )

    ax_br.set_xlabel("Energy capacity E [MWh]", fontsize=11)
    ax_br.set_ylabel("NPV [MUSD]", fontsize=11)
    ax_br.set_title(
        f"NPV vs E,  fixed P = {p_fix:.0f} MW\n"
        "Dotted lines = optimal E per model  |  Shaded = Xu degradation cost",
        fontsize=10, fontweight="bold",
    )
    ax_br.legend(fontsize=9)
    ax_br.tick_params(labelsize=9)

    plt.savefig(OUTPUT_DIR / f"planB_2d_combined_{n}h_{ts}.png",
                dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Combined plot: planB_2d_combined_{n}h_{ts}.png")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("PLAN B 2D: Parameter sweep, NPV vs (E_cap, P_cap) with degradation")
    print("=" * 78)
    print(f"  E_cap grid:     {E_CAP_GRID} MWh")
    print(f"  P_cap grid:     {P_CAP_GRID} MW")
    print(f"  Grid points:    {len(E_CAP_GRID) * len(P_CAP_GRID)}")
    print(f"  Calendar aging: {INCLUDE_CALENDAR}")
    print(f"  Horizon:        {RUN_HOURS} hours")

    # ── Resolve which CSV to load, if any ────────────────────────────────
    csv_to_load: Path | None = None
    if REPLOT_CSV is not None:
        csv_to_load = Path(REPLOT_CSV)
    elif REPLOT_FROM_LAST:
        csv_to_load = find_latest_csv()

    # ── Branch: replot from CSV or run the full sweep ─────────────────────
    if csv_to_load is not None:
        print(f"\n[REPLOT MODE] Loading from CSV, LP sweep will be skipped.")
        results, params, n = load_and_verify(csv_to_load)
        # Reconstruct timestamp from the filename for consistent output naming
        stem_parts = csv_to_load.stem.split("_")   # planB 2d sweep 8760h YYYYMMDD HHMMSS
        ts = f"{stem_parts[-2]}_{stem_parts[-1]}_replot"
    else:
        print(f"\n[1/3] Loading data...")
        power_wind, price_eur, params = load_data(RUN_HOURS)
        n = len(price_eur)

        print(f"\n[2/3] Running 2D sweep ({len(E_CAP_GRID) * len(P_CAP_GRID)} points)...")
        t0 = time.perf_counter()
        results = run_sweep_2d(power_wind, price_eur, params)
        total_s = time.perf_counter() - t0
        print(f"  Total sweep time: {total_s:.1f}s")

        print(f"\n[3/3] Saving results...")
        csv_path, report_path, ts = save_results_2d(results, params, n)

    # ── Plotting (always runs, whether from sweep or replot) ───────────────
    if not SKIP_PLOT:
        plot_npv_three_models(results, params, n, ts)   # Jenna: 3 NPV heatmaps, 3 stars each
        plot_sweep_slices(results, params, n, ts)
        plot_sweep_2d(results, params, n, ts)
        plot_sweep_combined(results, params, n, ts)

    # Print optimal design per NPV metric
    print("\n" + "=" * 78)
    print("OPTIMAL DESIGN POINT")
    print("=" * 78)
    for label, key in [("Without degradation", "npv_no_deg"),
                       ("With Xu degradation", "npv_with_xu"),
                       ("With Shi degradation", "npv_with_shi")]:
        best = max(results, key=lambda r: r[key])
        ep_h = best["e_cap"] / best["p_cap"]
        print(f"  {label:<25s}:  E={best['e_cap']:.0f} MWh  P={best['p_cap']:.0f} MW  "
              f"E/P={ep_h:.1f}h  NPV={best[key]/1e6:.2f} MUSD")

    print("\nDone.")
    return results


if __name__ == "__main__":
    main()
