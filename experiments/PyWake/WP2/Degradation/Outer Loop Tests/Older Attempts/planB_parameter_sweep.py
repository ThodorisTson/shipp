"""Plan B — Parameter sweep: NPV vs battery size with degradation.

Purpose
-------
For each battery size E_cap in a grid, solve the LP dispatch (Gurobi),
evaluate degradation post-hoc (Xu + Shi), and compute NPV with and
without degradation cost.  Plot NPV vs E_cap to identify how degradation
shifts the optimal battery size.

This directly answers Jenna's research question:
    "To what extent does explicitly accounting for battery degradation
     influence optimal design decisions in hybrid power plants?"

No gradients, no convergence, no cutting planes.  Just run, evaluate, plot.

Folder structure
    shipp/
    ├── degradation_xu.py, degradation_shi.py, ...
    ├── WP2_HPP.yaml, dk1_prices_2022.csv
    └── Outer Loop Tests/
        ├── planB_parameter_sweep.py     ← this script
        └── Results/

Usage
    python planB_parameter_sweep.py
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

# E_cap sweep values [MWh]
E_CAP_GRID = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]

# P_cap: C-rate determines P_cap = E_cap × C_RATE
C_RATE = 0.5              # P_cap = E_cap × 0.5  (2-hour battery)

# Include calendar aging in degradation evaluation?
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
    # Xu model (full)
    xu_result = analyze_degradation(
        storage_p=p_vec.tolist(),
        storage_e=e_vec.tolist(),
        e_cap_nominal=e_cap,
        battery_params=params["bat_params"],
        dt_hours=DT,
        T_cell_C=params["T_cell_C"],
    )
    fd_xu_cycle = xu_result["fd"]

    # Shi model (cycle-only)
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

    # Calendar aging (Xu model)
    t_total_s = n * DT * 3600.0
    sigma_mean = float(np.mean(e_vec)) / max(e_cap, 1e-9)
    fd_calendar = ft_calendar(t_total_s, sigma_mean, params["T_cell_C"])

    # Total fd
    fd_xu_total  = fd_xu_cycle + fd_calendar if INCLUDE_CALENDAR else fd_xu_cycle
    fd_shi_total = fd_shi_cycle + fd_calendar if INCLUDE_CALENDAR else fd_shi_cycle

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

def run_sweep(
    power_wind: np.ndarray,
    price_eur:  np.ndarray,
    params:     dict,
) -> List[dict]:
    """Run the parameter sweep over E_cap values."""
    results = []
    for i, e_cap in enumerate(E_CAP_GRID):
        p_cap = min(e_cap * C_RATE, params["p_max"])

        print(f"  [{i+1}/{len(E_CAP_GRID)}]  E_cap={e_cap:6.0f} MWh  "
              f"P_cap={p_cap:6.0f} MW  ...", end="", flush=True)

        r = evaluate_single_ecap(e_cap, p_cap, power_wind, price_eur, params)
        results.append(r)

        print(f"  rev={r['revenue_usd']/1e3:8.0f}k  "
              f"fd_xu={r['fd_xu_total']:.5f}  "
              f"fd_shi={r['fd_shi_total']:.5f}  "
              f"EFC={r['efc']:.0f}  "
              f"t={r['solve_s']:.1f}s")

    return results


def save_results(results: List[dict], params: dict, n: int) -> None:
    """Save sweep results to CSV and text report."""
    import pandas as pd
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── CSV ──────────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / f"planB_sweep_{n}h_{ts}.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path.name}")

    # ── Text report ──────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / f"planB_report_{n}h_{ts}.txt"

    lines = []
    w = lines.append

    w("=" * 90)
    w("PLAN B — Parameter sweep: NPV vs battery size with degradation")
    w("=" * 90)
    w("")
    w("CONFIGURATION")
    w(f"  Horizon:         {n} hours ({n/24:.1f} days)")
    w(f"  E_cap grid:      {E_CAP_GRID} MWh")
    w(f"  C-rate:           {C_RATE}  (P_cap = E_cap × {C_RATE})")
    w(f"  Calendar aging:  {INCLUDE_CALENDAR}")
    w(f"  SoC window:      {params['soc_min']*100:.0f}%–{params['soc_max']*100:.0f}%")
    w(f"  RTE(ac):         {params['eta_out']*100:.1f}%")
    w(f"  Shi k3:          {params['k3']:.4e}")
    w(f"  Shi k4:          {params['k4']:.4f}")
    w(f"  Cost:            {params['e_cost_eur_per_kwh']:.0f} EUR/kWh")

    w("")
    w("=" * 90)
    w("RESULTS")
    w("=" * 90)
    w("")
    w(f"  {'E_cap':>6s}  {'P_cap':>6s}  {'Revenue':>10s}  {'NPV':>10s}  "
      f"{'NPV_Xu':>10s}  {'NPV_Shi':>10s}  {'fd_Xu':>8s}  {'fd_Shi':>8s}  "
      f"{'fd_cal':>8s}  {'EFC':>5s}  {'Cyc':>4s}")
    w(f"  {'MWh':>6s}  {'MW':>6s}  {'kUSD':>10s}  {'kUSD':>10s}  "
      f"{'kUSD':>10s}  {'kUSD':>10s}  {'':>8s}  {'':>8s}  "
      f"{'':>8s}  {'':>5s}  {'':>4s}")
    w(f"  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  "
      f"{'─'*8}  {'─'*8}  {'─'*8}  {'─'*5}  {'─'*4}")

    for r in results:
        w(f"  {r['e_cap']:6.0f}  {r['p_cap']:6.0f}  "
          f"{r['revenue_usd']/1e3:10.0f}  "
          f"{r['npv_no_deg']/1e3:10.0f}  "
          f"{r['npv_with_xu']/1e3:10.0f}  "
          f"{r['npv_with_shi']/1e3:10.0f}  "
          f"{r['fd_xu_total']:8.5f}  "
          f"{r['fd_shi_total']:8.5f}  "
          f"{r['fd_calendar']:8.5f}  "
          f"{r['efc']:5.0f}  "
          f"{r['n_cycles']:4d}")

    # Optimal E_cap for each NPV metric
    w("")
    w("OPTIMAL BATTERY SIZE")
    w("")
    for label, key in [("Without degradation", "npv_no_deg"),
                       ("With Xu degradation", "npv_with_xu"),
                       ("With Shi degradation", "npv_with_shi")]:
        best = max(results, key=lambda r: r[key])
        w(f"  {label:<25s}:  E_cap = {best['e_cap']:.0f} MWh  "
          f"(NPV = {best[key]/1e6:.2f} MUSD)")

    w("")
    w("=" * 90)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {report_path.name}")

    return csv_path, report_path, ts


def plot_sweep(results: List[dict], params: dict, n: int, ts: str) -> None:
    """Plot NPV vs E_cap with and without degradation."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots.")
        return

    e_caps = [r["e_cap"] for r in results]
    npv_no  = [r["npv_no_deg"]   / 1e6 for r in results]
    npv_xu  = [r["npv_with_xu"]  / 1e6 for r in results]
    npv_shi = [r["npv_with_shi"] / 1e6 for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel 1: NPV vs E_cap ────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(e_caps, npv_no,  "o-", color="#2166ac", label="No degradation",
            markersize=6, linewidth=2)
    ax.plot(e_caps, npv_xu,  "s-", color="#b5351b", label="With Xu degradation",
            markersize=6, linewidth=2)
    ax.plot(e_caps, npv_shi, "^-", color="#4daf4a", label="With Shi degradation",
            markersize=6, linewidth=2)

    # Mark optima
    best_no  = e_caps[np.argmax(npv_no)]
    best_xu  = e_caps[np.argmax(npv_xu)]
    best_shi = e_caps[np.argmax(npv_shi)]
    ax.axvline(best_no,  color="#2166ac", linestyle=":", alpha=0.5)
    ax.axvline(best_xu,  color="#b5351b", linestyle=":", alpha=0.5)
    ax.axvline(best_shi, color="#4daf4a", linestyle=":", alpha=0.5)

    ax.set_xlabel("Battery capacity E_cap [MWh]")
    ax.set_ylabel("NPV [MUSD]")
    ax.set_title("NPV vs battery size", fontweight="bold")
    ax.legend(loc="best")

    # ── Panel 2: Degradation fd vs E_cap ─────────────────────────────────
    ax = axes[0, 1]
    fd_xu  = [r["fd_xu_total"]  for r in results]
    fd_shi = [r["fd_shi_total"] for r in results]
    fd_cal = [r["fd_calendar"]  for r in results]

    ax.plot(e_caps, fd_xu,  "s-", color="#b5351b", label="Xu total", markersize=5)
    ax.plot(e_caps, fd_shi, "^-", color="#4daf4a", label="Shi total", markersize=5)
    ax.plot(e_caps, fd_cal, "d-", color="#ff7f00", label="Calendar only",
            markersize=5, alpha=0.7)
    ax.set_xlabel("Battery capacity E_cap [MWh]")
    ax.set_ylabel("Annual fd (dimensionless)")
    ax.set_title("Degradation vs battery size", fontweight="bold")
    ax.legend(loc="best")

    # ── Panel 3: Revenue and Capex vs E_cap ──────────────────────────────
    ax = axes[1, 0]
    rev  = [r["revenue_usd"] / 1e6 for r in results]
    cap  = [r["capex_usd"]   / 1e6 for r in results]
    deg_xu  = [r["deg_cost_xu"]  / 1e6 for r in results]
    deg_shi = [r["deg_cost_shi"] / 1e6 for r in results]

    ax.plot(e_caps, rev, "o-", color="#2166ac", label="Revenue", markersize=5)
    ax.plot(e_caps, cap, "x-", color="gray", label="Capex", markersize=5)
    ax.plot(e_caps, deg_xu,  "s-", color="#b5351b", label="Deg cost (Xu)",
            markersize=5, alpha=0.7)
    ax.plot(e_caps, deg_shi, "^-", color="#4daf4a", label="Deg cost (Shi)",
            markersize=5, alpha=0.7)
    ax.set_xlabel("Battery capacity E_cap [MWh]")
    ax.set_ylabel("[MUSD]")
    ax.set_title("Revenue, capex, and degradation cost", fontweight="bold")
    ax.legend(loc="best")

    # ── Panel 4: EFC and throughput vs E_cap ──────────────────────────────
    ax = axes[1, 1]
    efc = [r["efc"] for r in results]
    tp  = [r["throughput_mwh"] / 1e3 for r in results]  # GWh

    ax2 = ax.twinx()
    l1 = ax.plot(e_caps, efc, "o-", color="#2166ac", label="EFC", markersize=5)
    l2 = ax2.plot(e_caps, tp, "s-", color="#b5351b", label="Throughput",
                  markersize=5)
    ax.set_xlabel("Battery capacity E_cap [MWh]")
    ax.set_ylabel("Equivalent full cycles")
    ax2.set_ylabel("Throughput [GWh]")
    ax.set_title("Cycling intensity vs battery size", fontweight="bold")
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc="best")

    cal_label = "+calendar" if INCLUDE_CALENDAR else "cycle-only"
    fig.suptitle(f"Plan B: Parameter sweep  ({n} hours, {cal_label})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    out = OUTPUT_DIR / f"planB_sweep_{n}h_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot: {out.name}")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("PLAN B — Parameter sweep: NPV vs battery size with degradation")
    print("=" * 78)
    print(f"  E_cap grid:     {E_CAP_GRID} MWh")
    print(f"  C-rate:         {C_RATE}  (P_cap = E_cap × {C_RATE})")
    print(f"  Calendar aging: {INCLUDE_CALENDAR}")
    print(f"  Horizon:        {RUN_HOURS} hours")

    print(f"\n[1/3] Loading data...")
    power_wind, price_eur, params = load_data(RUN_HOURS)
    n = len(price_eur)

    print(f"\n[2/3] Running sweep ({len(E_CAP_GRID)} points)...")
    t0 = time.perf_counter()
    results = run_sweep(power_wind, price_eur, params)
    total_s = time.perf_counter() - t0
    print(f"  Total sweep time: {total_s:.1f}s")

    print(f"\n[3/3] Saving results...")
    csv_path, report_path, ts = save_results(results, params, n)

    if not SKIP_PLOT:
        plot_sweep(results, params, n, ts)

    # Print summary
    print("\n" + "=" * 78)
    print("OPTIMAL BATTERY SIZE")
    print("=" * 78)
    for label, key in [("Without degradation", "npv_no_deg"),
                       ("With Xu degradation", "npv_with_xu"),
                       ("With Shi degradation", "npv_with_shi")]:
        best = max(results, key=lambda r: r[key])
        print(f"  {label:<25s}:  E_cap = {best['e_cap']:.0f} MWh  "
              f"(NPV = {best[key]/1e6:.2f} MUSD)")

    print("\nDone.")
    return results


if __name__ == "__main__":
    main()
