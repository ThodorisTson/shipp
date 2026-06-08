"""WP2 battery optimization + Shi degradation — v2 (multi-year, calendar-corrected).

Changes over v1 (run_battery_xu_shi_degradation.py)
----------------------------------------------------
1. Calendar correction (Option 2)
   The Shi accumulation framework is cycle-only (fd_calendar = 0.0 by design).
   For the WP2 site, calendar aging accounts for ~49-51% of total annual fd
   under the Xu model.  In v2, ft_calendar() from degradation_xu is added to
   the Shi result in the REPORTING PATH ONLY.  The gradient computation
   (phi_shi_prime_with_stress) is never touched — the dual-Phi convexity
   guarantee is fully preserved.

   Architectural note: dfd_calendar/dc_t = 0 for all t.  Calendar aging
   carries no dispatch-dependent information and cannot drive the outer loop.
   Separating it from fd_cycle makes this explicit rather than implicit.

2. Single-year FD gradient validation (v3.1)
   Rather than running one year and projecting EoL analytically (v1), v2
   loops over N_YEARS using the same representative 8760-hour dataset tiled
   each year.  This addresses two v1 limitations:
     - Stationarity: annual fd is recomputed each year from the actual
       dispatch given the degraded capacity (not assumed constant).
     - Capacity fade feedback: the inner LP is re-solved each year with
       E_eff = E_nominal * SoH(year-1), so dispatch adapts to the battery's
       current capacity.

3. Battery replacement at EOL_REPLACEMENT = 0.70 SoH
   When SoH drops below the 60% threshold within the multi-year loop, the
   battery is replaced: fd accumulator resets to 0, SoH resets to 1.0, and
   a replacement cost (lambda_E * E_nominal) is subtracted from the NPV.
   The number of replacements and the years they occur are recorded.

File structure
--------------
  CONFIG                   — toggles and thresholds (top of file)
  _shi_with_calendar_correction()  — single-year Shi + Xu calendar patching
  _build_storage_year()    — build Storage object for a given effective e_cap
  _single_year_solve()     — single LP year solve for FD validation
  main()                   — orchestrates single-year + multi-year runs

No changes required to:
  degradation_xu.py        — ft_calendar / sei_capacity_loss already in API
  degradation_shi.py       — analyze_degradation_shi unchanged
  degradation_subgradient.py — gradient path untouched
  degradation_plots.py     — works on the patched result dict
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from shipp.kernel import solve_lp_sparse
from shipp.kernel_pyomo import solve_lp_pyomo
from shipp.components import Storage, Production, TimeSeries

from wp2_common import quick_setup, get_wake_model

# Xu — reporting baseline + calendar correction helpers
from degradation_xu import (
    analyze_degradation,
    count_equivalent_full_cycles,
    rainflow_cycle_counting,
    ft_calendar,          # ← calendar correction for Shi reporting path
    sei_capacity_loss,    # ← recompute SoH after calendar correction
)

# Shi — cycle accumulation + gradient
from degradation_shi import analyze_degradation_shi
from degradation_subgradient import compute_subgradient, fit_shi_polynomial

from degradation_plots import (
    plot_degradation_analysis,
    print_degradation_report,
)

import xarray as xr
from py_wake.site import XRSite
import numpy_financial as npf

from matplotlib.patches import Patch

# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
HPP_YAML   = SCRIPT_DIR / "WP2_HPP.yaml"
PRICE_CSV  = SCRIPT_DIR / "dk1_prices_2022.csv"

RESULTS_DIR = SCRIPT_DIR / "Results"
PLOTS_DIR   = SCRIPT_DIR / "Degradation Plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

eur_to_usd    = 1.18
discount_rate = 0.03
dt            = 1.0      # hours

# Solver
pyo_solver = "gurobi"   # 'gurobi' recommended; 'none' = scipy sparse (no DoD, ≤6 mo)

# Horizon
RUN_FULL_YEAR     = True
N_DAYS_TEST       = 120
MAX_HOURS_SPARSE  = 180 * 24

# Problem settings
p_min      = 0.0
WAKE_MODEL = "Bastankhah"

# EoL thresholds for reporting (SoH fractions, e.g. 0.80 = 80% capacity remaining)
eol_thresholds = [0.80, 0.70, 0.60]

# ── v1 — finite-difference validation settings ──────────────────────────────
ECAP_EPS = 30.0    # perturbation size in MWh (used as fractional scale ±ECAP_EPS/e_cap)
N_YEARS  = 20     # kept only for dual scaling factor (npf.npv)

# Richardson extrapolation — three ε values (each half the previous)
RICH_EPS_FRACS = [0.10, 0.05, 0.025]   # fractional perturbations: 10%, 5%, 2.5%

VALIDATION_DIR = SCRIPT_DIR / "Gradient_Verification"
VALIDATION_DIR.mkdir(exist_ok=True)

# Output toggles
print_baseline_table = True
print_degr_reports   = True
SAVE_CSV             = True
SAVE_REPORT          = True
MAKE_PLOT            = False
show_plots           = False

run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

def _build_run_label(ts: str, price_csv: Path, p_cap: float, e_cap: float) -> str:
    stem    = price_csv.stem.lower()
    year    = ''.join(filter(str.isdigit, stem))[-4:]
    dataset = f"dk{year}"
    bat     = f"{int(round(p_cap))}mw_{int(round(e_cap))}mwh"
    return f"{ts}_{dataset}_{bat}_v2"

# =============================================================================
# Small structs
# =============================================================================

@dataclass
class HorizonData:
    n:         int
    ws:        np.ndarray
    wd:        np.ndarray
    ti:        Optional[np.ndarray]
    price_eur: np.ndarray

# =============================================================================
# Helpers — shared with v1
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
    ws = np.asarray(ts["wind_speed"],     dtype=float)
    wd = np.asarray(ts["wind_direction"], dtype=float)
    ti = None
    ti_dat = ts.get("turbulence_intensity")
    if isinstance(ti_dat, dict) and "data" in ti_dat:
        ti = np.asarray(ti_dat["data"], dtype=float)
    if ws.shape != wd.shape:
        raise ValueError("Wind speed and wind direction arrays have different lengths.")
    if ti is not None and ti.shape != ws.shape:
        ti = None
    return ws, wd, ti

def _load_prices() -> np.ndarray:
    df  = pd.read_csv(PRICE_CSV)
    col = _find_price_column(df)
    return df[col].astype(float).to_numpy()

def _run_pywake_power_MW(
    setup: dict,
    wd:    np.ndarray,
    ws:    np.ndarray,
    ti:    Optional[np.ndarray],
) -> np.ndarray:
    n    = len(ws)
    site = XRSite(ds=xr.Dataset(data_vars=dict(P=1)))
    wf_model = get_wake_model(WAKE_MODEL, site, setup["windturbine"])
    time_days = np.arange(n) / 24.0
    kwargs = {"x": setup["x"], "y": setup["y"], "wd": wd, "ws": ws, "time": time_days}
    if ti is not None:
        kwargs["TI"] = ti
    sim_res = wf_model(**kwargs)
    return sim_res.Power.sum(["wt"]).values / 1e6


def _build_shipp_components(
    setup:     dict,
    p_max_MW:  float,
) -> Tuple[Storage, Storage, float, float, float, float, float, float]:
    bat = setup["battery"]

    e_cap_MWh = float(bat["energy_capacity_Wh"]) / 1e6
    p_cap_MW  = float(bat["power_capacity_W"])   / 1e6

    rte_dc   = float(bat["rte_nominal"])
    pcu_eff  = float(bat["pcu_efficiency"])
    rte_ac   = rte_dc * (pcu_eff ** 2)

    e_cost_USD_per_MWh = float(bat["capex_EUR_per_kWh"]) * 1000.0 * eur_to_usd
    p_cost_USD_per_MW  = float(bat["capex_EUR_per_kW"])  * 1000.0 * eur_to_usd

    soc_min = float(bat.get("soc_min", 0.10))
    soc_max = float(bat.get("soc_max", 0.90))

    dod_yaml = 1.0 - soc_min
    if pyo_solver == "none":
        dod_eff = 1.0
        print(f"  ⚠ DoD ({dod_yaml:.0%}) ignored — scipy sparse requires dod=1.0.")
    else:
        dod_eff = dod_yaml

    print(f"  SoC window: {soc_min*100:.0f}% – {soc_max*100:.0f}%  "
          f"(DoD={dod_eff:.0%}, enforced by {pyo_solver})")

    stor = Storage(
        e_cap=e_cap_MWh,
        p_cap=p_cap_MW,
        eff_in=1.0,
        eff_out=rte_ac,
        e_cost=e_cost_USD_per_MWh,
        p_cost=p_cost_USD_per_MW,
        dod=dod_eff,
    )
    stor_null = Storage(e_cap=0.0, p_cap=0.0, eff_in=1.0, eff_out=1.0,
                        e_cost=0.0, p_cost=0.0)

    return (stor, stor_null, e_cap_MWh, p_cap_MW, rte_ac, e_cost_USD_per_MWh, p_cost_USD_per_MW, soc_min, soc_max)

def _solve_shipp(
    price_eur:    np.ndarray,
    power_wind_MW: np.ndarray,
    stor:         Storage,
    stor_null:    Storage,
    p_max_MW:     float,
    n:            int,
    soc_max:      float = 1.0,
):
    price_dam  = TimeSeries((price_eur * eur_to_usd).tolist(), dt)
    prod       = Production(TimeSeries(power_wind_MW.tolist(), dt), p_cost=0.0)
    prod_null  = Production(TimeSeries([0.0] * n, dt), p_cost=0.0)

    if pyo_solver == "none":
        os       = solve_lp_sparse(price_dam, prod, prod_null, stor, stor_null,
                                   discount_rate, N_YEARS, p_min, p_max_MW, n)
        os_fixed = solve_lp_sparse(price_dam, prod, prod_null, stor, stor_null,
                                   discount_rate, N_YEARS, p_min, p_max_MW, n,
                                   fixed_cap=True)
    else:
        os       = solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null,
                                  discount_rate, N_YEARS, p_min, p_max_MW, n,
                                  pyo_solver, soc_max1=soc_max)
        os_fixed = solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null,
                                  discount_rate, N_YEARS, p_min, p_max_MW, n,
                                  pyo_solver, fixed_cap=True, return_duals=True,
                                  soc_max1=soc_max)
    return os, os_fixed

# =============================================================================
# v2 helpers — calendar correction + multi-year loop
# =============================================================================

def _build_storage_year(
    e_cap_eff:         float,
    p_cap_MW:          float,
    rte_ac:            float,
    e_cost_USD_per_MWh: float,
    soc_min:           float,
    soc_max:           float,
) -> Storage:
    """Build a fixed-dispatch Storage object for a single year of the loop.

    Power capacity and costs are held at their nominal values — only energy
    capacity (e_cap_eff = E_nominal * SoH) changes year to year.
    """
    dod_eff = 1.0 - soc_min if pyo_solver != "none" else 1.0
    return Storage(
        e_cap=e_cap_eff,
        p_cap=p_cap_MW,
        eff_in=1.0,
        eff_out=rte_ac,
        e_cost=e_cost_USD_per_MWh,
        p_cost=0.0,     # sizing cost not included in per-year dispatch
        dod=dod_eff,
    )

def _shi_with_calendar_correction(
    storage_p:      List[float],
    storage_e:      List[float],
    e_cap_eff:      float,
    bat_params:     Dict,
    shi_fit,
    T_cell_C:       float,
    dt_hours:       float,
    eol_thresholds: List[float],
) -> Dict:
    """Run Shi degradation and add Xu calendar term to the reporting path only.

    The gradient path (phi_shi_prime_with_stress) is never touched.
    This mirrors the dual-Phi separation:
        Phi_xu  → reporting    (unchanged)
        Phi_shi → gradients    (unchanged)
        ft_calendar → reporting correction only (∂ft_cal/∂c_t = 0 for all t)

    sigma_mean and t_total_seconds are computed internally from storage_e and
    dt_hours — exactly as analyze_degradation() does — so partial-year series
    (e.g. 8752 h instead of 8760 h) are handled correctly without any
    hardcoding in the call site.

    The result dict is patched in place so all downstream functions
    (plot_degradation_analysis, print_degradation_report, CSV logging) work
    without changes — they see a fully populated fd_calendar field.

    Args:
        storage_p:      Battery power [MW] from LP solve.
        storage_e:      Battery SoC [MWh] from LP solve.
        e_cap_eff:      Effective energy capacity this year [MWh].
        bat_params:     Battery params dict (must have 'power_capacity_W').
        shi_fit:        ShiPolynomialFit from fit_shi_polynomial().
        T_cell_C:       Cell temperature [°C].
        dt_hours:       Timestep duration [h] — typically 1.0.
        eol_thresholds: SoH fractions to report EoL for.

    Returns:
        Patched result dict with:
            fd_calendar  — Xu calendar term (non-zero)
            fd           — fd_cycle + fd_calendar (corrected total)
            soh / eol_years / capacity_retention — recomputed from corrected fd
    """
    # ── Compute time quantities internally — same pattern as analyze_degradation
    n_steps         = len(storage_e)
    t_total_seconds = n_steps * dt_hours * 3600.0          # exact, not assumed 8760
    t_total_hours   = t_total_seconds / 3600.0
    e_arr           = np.asarray(storage_e, dtype=float)
    sigma_mean      = float(np.mean(e_arr)) / max(e_cap_eff, 1e-9)

    # ── Step 1: Shi cycle-only result (fd_calendar = 0.0 here by design) ──
    degr = analyze_degradation_shi(
        storage_p, storage_e, e_cap_eff, bat_params,
        shi_fit=shi_fit,
        T_cell_C=T_cell_C,
        dt_hours=dt_hours,
        eol_thresholds=eol_thresholds,
    )

    # ── Step 2: Xu calendar term — reporting path only ─────────────────────
    fd_cal       = ft_calendar(t_total_seconds, sigma_mean, T_cell_C)
    fd_corrected = degr["fd_shi"] + fd_cal
    cal_frac_pct = 100.0 * fd_cal / max(fd_corrected, 1e-30)

    # ── Step 3: Recompute SoH and EoL with corrected total fd ──────────────
    L_corr       = sei_capacity_loss(fd_corrected)
    cap_ret_corr = 1.0 - L_corr
    soh_corr     = cap_ret_corr * 100.0

    # Annualise fd before EoL bisection — same formula as degradation_xu.py L884:
    #   fd_per_yr = fd / (t_total_hours / 8760)
    # For a full year (8760 h) this is identity; for 8752 h it scales up by ~0.09%.
    fd_per_yr = fd_corrected / max(t_total_hours / 8760.0, 1e-9)
    eol_corr: Dict[float, Optional[float]] = {}
    for thr in eol_thresholds:
        # Guard: already below threshold at year 0
        if (1.0 - sei_capacity_loss(0.0)) < thr:
            eol_corr[thr] = 0.0
            continue
        lo, hi = 0.0, 200.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            # SoH still above threshold → EoL is later → raise lo
            # SoH below threshold   → EoL is earlier → lower hi
            # Matches degradation_xu.py lines 892-895 exactly.
            if 1.0 - sei_capacity_loss(fd_per_yr * mid) > thr:
                lo = mid
            else:
                hi = mid
        val = (lo + hi) / 2.0
        eol_corr[thr] = round(val, 2) if val < 190.0 else None

    # ── Step 4: Patch result dict ───────────────────────────────────────────
    degr["fd_calendar"]       = float(fd_cal)
    degr["fd"]                = float(fd_corrected)
    degr["capacity_retention"] = float(cap_ret_corr)
    degr["capacity_loss"]      = float(L_corr)
    degr["soh"]                = float(soh_corr)
    degr["capacity_fade_percent"] = float(L_corr * 100.0)
    degr["eol_years"]          = eol_corr
    degr["e_cap_degraded"]     = e_cap_eff * cap_ret_corr
    degr["p_cap_degraded"]     = (
        float(bat_params["power_capacity_W"]) / 1e6 * cap_ret_corr
    )
    degr["meta"]["calendar_correction"] = (
        "Xu ft_calendar added to Shi reporting path (dual-Phi: gradient unchanged)"
    )
    degr["meta"]["fd_calendar_note"] = (
        f"fd_cal={fd_cal:.4e}  ({cal_frac_pct:.1f}% of corrected total)  "
        f"sigma_mean={sigma_mean:.3f}"
    )
    return degr

def _single_year_solve(
    wind_8760:              np.ndarray,
    price_8760:             np.ndarray,
    stor_null,
    p_max_MW:               float,
    e_cap_eff:              float,
    p_cap_MW:               float,
    rte_ac:                 float,
    e_cost_USD_per_MWh:     float,
    bat_params:             Dict,
    shi_fit,
    soc_min:                float,
    soc_max:                float,
    T_cell_C:               float = 25.0,
) -> Dict:
    """Solve one LP year and return degradation + gradient.

    Used for finite-difference validation: call three times at
    soc_min ± eps and compare the central-difference slope to dDeg_dDoD.

    Returns dict with keys:
        fd          — annualised degradation fraction
        dDeg_dDoD   — inner-loop gradient [USD / MWh] (None if duals unavailable)
        degr        — full _shi_with_calendar_correction output dict
        os          — OptimizationSolution from the LP solve
    """
    n = len(wind_8760)

    stor_yr   = _build_storage_year(e_cap_eff, p_cap_MW, rte_ac,
                                    e_cost_USD_per_MWh, soc_min, soc_max)
    price_dam = TimeSeries((price_8760 * eur_to_usd).tolist(), dt)
    prod_yr   = Production(TimeSeries(wind_8760.tolist(), dt), p_cost=0.0)
    prod_null = Production(TimeSeries([0.0] * n, dt), p_cost=0.0)

    os_yr = solve_lp_pyomo(
        price_dam, prod_yr, prod_null, stor_yr, stor_null,
        discount_rate, N_YEARS, p_min, p_max_MW, n,
        pyo_solver, fixed_cap=True, soc_max1=soc_max, return_duals=True,
    )

    storage_e_yr = os_yr.storage_e[0].data
    storage_p_yr = os_yr.storage_p[0].data

    degr_yr = _shi_with_calendar_correction(
        storage_p_yr, storage_e_yr, e_cap_eff,
        bat_params, shi_fit, T_cell_C, dt, eol_thresholds,
    )

    scale = 1.0 / max(n / 8760.0, 1e-9)
    fd_yr = degr_yr["fd"] * scale

    dDeg_dDoD = None
    if os_yr.dual_prices is not None:
        dual_yr   = os_yr.dual_prices["dual_e_min1"]
        e_cap_yr  = os_yr.dual_prices["e_cap1"]
        cycles_yr = rainflow_cycle_counting(storage_e_yr, e_cap_yr)
        sg_yr     = compute_subgradient(
            storage_e=storage_e_yr,
            cycles=cycles_yr,
            dt_hours=dt,
            battery_replacement_cost_per_MWh=e_cost_USD_per_MWh,
            eff_in=1.0,
            eff_out=rte_ac,
            shi_fit=shi_fit,
        )
        factor        = npf.npv(discount_rate, np.ones(N_YEARS)) - 1
        dual_per_year = dual_yr / factor
        dDeg_dDoD = -e_cap_yr * float(
            np.dot(sg_yr["subgrad_combined"], dual_per_year)
        )

    return {"fd": fd_yr, "dDeg_dDoD": dDeg_dDoD, "degr": degr_yr, "os": os_yr}

# =============================================================================
# Console + CSV helpers
# =============================================================================

def _print_degr_block(degr: Dict, label: str, period_days: float) -> None:
    content = f"  DEGRADATION — {label}"
    # Auto-scale: wide enough for the label, minimum 72, always matches the border
    W = max(72, len(content) + 2)
    print("\n" + "╔" + "═" * W + "╗")
    print(f"║{content:<{W}}║")
    print("╚" + "═" * W + "╝")
    print_degradation_report(degr, period_days=period_days, enabled=True)

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("WP2 BATTERY OPTIMIZATION + DEGRADATION  v3.1 — single-year validation")
    print("Shi (2018) + Xu calendar correction | FD gradient validation")
    print("=" * 80)

    # ── 1. Load config + raw series ───────────────────────────────────────
    print("\n[1/5] Loading WP2 configuration...")
    setup = quick_setup(HPP_YAML, config={"interp_n": 2000}, verbose=False)
    hpp   = setup["hpp"]

    ws_all, wd_all, ti_all = _load_inputs(hpp)
    price_all               = _load_prices()

    # ── 2. Choose horizon ─────────────────────────────────────────────────
    print("\n[2/5] Loading electricity prices...")
    n = _choose_horizon(len(ws_all), len(price_all))
    ws        = ws_all[:n]
    wd        = wd_all[:n]
    ti        = ti_all[:n] if ti_all is not None else None
    price_eur = price_all[:n]

    print(f"  Horizon: {n:,} h ({n/24:.1f} days)")
    print(f"  Mean price: {float(np.mean(price_eur)):.2f} EUR/MWh")

    # ── 3. PyWake ─────────────────────────────────────────────────────────
    print("\n[3/5] Running PyWake time-series simulation...")
    power_wind_MW = _run_pywake_power_MW(setup, wd, ws, ti)
    print(f"  Wind mean: {float(np.mean(power_wind_MW)):.1f} MW | "
          f"peak: {float(np.max(power_wind_MW)):.1f} MW")

    # ── 4. SHIPP (single-year baseline) ───────────────────────────────────
    print("\n[4/5] Running SHIPP optimization (single-year baseline)...")
    p_max_MW = float(hpp["grid_connection_capacity"]) / 1e6

    (stor, stor_null, e_cap_MWh, p_cap_MW,
     rte_ac, e_cost_USD_per_MWh, p_cost_USD_per_MW, soc_min, soc_max) = _build_shipp_components(
        setup, p_max_MW
    )

    shi_fit = fit_shi_polynomial(soc_min, soc_max, verbose=True)

    print(f"  Battery: {p_cap_MW:.0f} MW / {e_cap_MWh:.0f} MWh | "
          f"Grid: {p_max_MW:.0f} MW | RTE(ac): {rte_ac*100:.1f}%")
    print(f"  Solver: {pyo_solver}")

    run_label = _build_run_label(run_ts, PRICE_CSV, p_cap_MW, e_cap_MWh)

    os, os_fixed = _solve_shipp(
        price_eur, power_wind_MW, stor, stor_null, p_max_MW, n, soc_max
    )

    np.save(RESULTS_DIR / "storage_e_fixed.npy",
            np.array(os_fixed.storage_e[0].data))
    np.save(RESULTS_DIR / "e_cap_fixed.npy",
            np.array([os_fixed.storage_list[0].e_cap]))

    # Baseline metrics
    revenues_res_only = (
        365.0 * 24.0 / n
        * np.dot(price_eur, np.minimum(power_wind_MW, p_max_MW)) * dt
    )
    os.get_added_npv(discount_rate, N_YEARS)
    os_fixed.get_added_npv(discount_rate, N_YEARS)
    period_days = n * dt / 24.0

    def _rev_inc_pct(rev: float) -> float:
        return 100.0 * (rev / revenues_res_only - 1.0)

    cycles_opt   = count_equivalent_full_cycles(
        os.storage_p[0].data, os.storage_e[0].data,
        os.storage_list[0].e_cap, dt_hours=dt
    )
    cycles_fixed = count_equivalent_full_cycles(
        os_fixed.storage_p[0].data, os_fixed.storage_e[0].data,
        os_fixed.storage_list[0].e_cap, dt_hours=dt
    )

    if print_baseline_table:
        print("\n" + "=" * 80)
        print("BASELINE RESULTS (single year)")
        print("=" * 80)
        print("                Revenue [kUSD]  Rev.inc%    p/e_cap [MW/MWh]    "
              "NPV [MUSD]  Cycles/yr")
        print("-" * 85)
        print(
            f"Sizing Opt.   {os.annual_revenue*1e-3:>10.1f}  "
            f"{_rev_inc_pct(os.annual_revenue):>8.2f}%  "
            f"{os.storage_list[0].p_cap:>7.1f}/{os.storage_list[0].e_cap:<7.1f}  "
            f"{os.npv:>10.1f}  {cycles_opt/period_days*365:>9.0f}"
        )
        print(
            f"Dispatch-fix  {os_fixed.annual_revenue*1e-3:>10.1f}  "
            f"{_rev_inc_pct(os_fixed.annual_revenue):>8.2f}%  "
            f"{os_fixed.storage_list[0].p_cap:>7.1f}/{os_fixed.storage_list[0].e_cap:<7.1f}  "
            f"{os_fixed.npv:>10.1f}  {cycles_fixed/period_days*365:>9.0f}"
        )
        print("-" * 85)

    # ── 5. Single-year degradation (Shi + calendar correction) ────────────
    print("\n[5/5] Single-year degradation analysis (Shi + Xu calendar)...")
    bat_params = setup["battery"]

    # Dispatch-fixed: Shi + calendar correction
    degr_fixed = _shi_with_calendar_correction(
        os_fixed.storage_p[0].data,
        os_fixed.storage_e[0].data,
        float(os_fixed.storage_list[0].e_cap),
        bat_params, shi_fit, 25.0,
        dt, eol_thresholds,
    )

    # Console report
    if print_degr_reports:
        _print_degr_block(degr_fixed, "DISPATCH-FIXED — 150 MW / 300 MWh", period_days)

    # Gap D verification (unchanged from v1)
    if os_fixed.dual_prices is not None:
        print("\n" + "=" * 60)
        print("GAP D VERIFICATION — dual prices + gradient signal")
        print("=" * 60)
        dual  = os_fixed.dual_prices["dual_e_min1"]
        e_cap = os_fixed.dual_prices["e_cap1"]
        print(f"  dual_e_min1: min={dual.min():.4e}  max={dual.max():.4e}  "
              f"mean={dual.mean():.4e}")
        print(f"  n_nonzero: {np.sum(dual != 0)} / {len(dual)}")
        cycles = rainflow_cycle_counting(os_fixed.storage_e[0].data, e_cap)
        sg = compute_subgradient(
            storage_e=os_fixed.storage_e[0].data,
            cycles=cycles,
            dt_hours=dt,
            battery_replacement_cost_per_MWh=e_cost_USD_per_MWh,
            eff_in=1.0,
            eff_out=rte_ac,
            shi_fit=shi_fit,
        )
        factor        = npf.npv(discount_rate, np.ones(N_YEARS)) - 1
        dual_per_year = dual / factor
        grad_deg_dod  = -e_cap * float(np.dot(sg["subgrad_combined"], dual_per_year))
        print(f"  subgrad_combined: [{sg['subgrad_combined'].min():.4e}, "
              f"{sg['subgrad_combined'].max():.4e}]")
        print(f"  dDeg/dDoD (chain rule): {grad_deg_dod:.6e}")
        print("=" * 60)
    else:
        print("\n  WARNING: dual_prices is None — dual extraction failed.")

# ── Finite-difference validation — frozen-dispatch SoC scaling ────────────
    print("\n" + "=" * 70)
    print("FINITE-DIFFERENCE VALIDATION  —  frozen-dispatch SoC scaling")
    print(f"Perturbation: scale entire SoC trajectory by (1 ± {ECAP_EPS/e_cap_MWh*100:.1f}%)")
    print("=" * 70)

    wind_tile  = power_wind_MW[:8760] if n >= 8760 else power_wind_MW
    price_tile = price_eur[:8760]     if n >= 8760 else price_eur

    # Base dispatch from the already-solved os_fixed (no LP re-solve needed)
    storage_e_base = np.array(os_fixed.storage_e[0].data)
    storage_p_base = np.array(os_fixed.storage_p[0].data)
    e_cap_base     = float(os_fixed.storage_list[0].e_cap)

    for yr_label, e_cap_nom, storage_e_nom in [
        ("Year 1  (SoH=100%)", e_cap_base, storage_e_base),
    ]:
        eps_frac = ECAP_EPS / e_cap_MWh   # fractional perturbation

        fd_vals = {}
        for tag, scale in [("lo",  1.0 - eps_frac),
                            ("mid", 1.0),
                            ("hi",  1.0 + eps_frac)]:
            storage_e_scaled = (storage_e_nom * scale).tolist()
            storage_p_scaled = (storage_p_base * scale).tolist()
            # e_cap stays FIXED — this is the key change
            degr_scaled = _shi_with_calendar_correction(
                storage_p_scaled, storage_e_scaled, e_cap_nom,   # ← e_cap_nom, not scaled
                bat_params, shi_fit, 25.0, dt, eol_thresholds,
            )
            fd_vals[tag] = degr_scaled["fd"]

        # ── FD slope ──────────────────────────────────────────────────────
        fd_slope_scale = (fd_vals["hi"] - fd_vals["lo"]) / (2.0 * eps_frac)

        # ── Analytical prediction (cycle-level, no timestep attribution) ──
        # Φ = k3 × δ^k4  →  Φ'(δ) × δ = k4 × Φ(δ)
        # Scaling all DoDs by (1+ε): ∂fd_cycle/∂ε = k4 × fd_cycle
        cycle_pred    = shi_fit.k4 * degr_fixed["fd_cycle"]
        calendar_pred = degr_fixed["fd_calendar"]   # linear in σ_mean, scales with ε
        total_pred    = cycle_pred + calendar_pred
        ratio_total   = fd_slope_scale / total_pred if abs(total_pred) > 1e-30 else None

        print(f"\n  {yr_label}")
        print(f"    E_cap fixed at          : {e_cap_nom:.2f} MWh  (NOT scaled)")
        print(f"    SoC scale perturbation  : ±{eps_frac*100:.2f}%")
        print(f"    fd(lo, mid, hi)         : "
              f"{fd_vals['lo']:.6f}  {fd_vals['mid']:.6f}  {fd_vals['hi']:.6f}")
        print(f"    FD  d(fd)/d(scale)      : {fd_slope_scale:.6e}")
        print(f"    Prediction — cycle only : {cycle_pred:.6e}"
              f"  [k4 × fd_cycle = {shi_fit.k4:.4f} × {degr_fixed['fd_cycle']:.6f}]")
        print(f"    Prediction — calendar   : {calendar_pred:.6e}  [≈ fd_calendar]")
        print(f"    Prediction — total      : {total_pred:.6e}")
        if ratio_total is not None:
            print(f"    Ratio (FD / total pred) : {ratio_total:.4f}  (target ≈ 1.0)")
        else:
            print(f"    Ratio                   : N/A")

    # ── Plot: compare FD slope vs predicted slope ──────────────────────
        pad    = eps_frac * 0.4
        x_line = np.linspace(1.0 - eps_frac - pad, 1.0 + eps_frac + pad, 80)

        # Both lines anchored at the same midpoint — only slopes differ
        y_fd_line    = fd_vals["mid"] + fd_slope_scale * (x_line - 1.0)
        y_pred_line  = fd_vals["mid"] + total_pred     * (x_line - 1.0)

        fig, ax = plt.subplots(figsize=(7, 5))

        # Prediction line (analytical)
        ax.plot(x_line, y_pred_line, '-', color='#d7191c', lw=2.2, zorder=3,
                label=f'Analytical prediction\nslope = {total_pred:.4e}  '
                      f'(k4×fd_cycle + fd_cal)')

        # FD slope line (measured — fitted through the three points)
        ax.plot(x_line, y_fd_line, '--', color='#2c7bb6', lw=2.0, zorder=3,
                label=f'FD measured slope\nslope = {fd_slope_scale:.4e}')

        # The three actual FD points
        ax.plot([1.0 - eps_frac, 1.0, 1.0 + eps_frac],
                [fd_vals["lo"], fd_vals["mid"], fd_vals["hi"]],
                'o', color='#2c3e50', ms=10, zorder=5,
                label='FD points (lo / mid / hi)')

        ax.set_xlabel('SoC scale factor  (1.0 = nominal dispatch)', fontsize=10)
        ax.set_ylabel('Annual degradation fraction  fd  [—]', fontsize=10)
        ratio_str = f"{ratio_total:.4f}" if ratio_total is not None else "N/A"
        ax.set_title(
            f'Subgradient validation - {yr_label}\n'
            f'FD slope = {fd_slope_scale:.4e}     '
            f'Predicted = {total_pred:.4e}     '
            f'Ratio = {ratio_str}',
            fontsize=9)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        yr_tag = "yr1" if yr_label.startswith("Year 1") else "yr2"
        plt.savefig(VALIDATION_DIR / f"subgrad_validation_{yr_tag}_{run_ts}.png", dpi=200)        
        plt.close()
        print(f"    ✓ Plot saved: subgrad_validation_{yr_tag}.png")

    print("=" * 70)

    # ── Richardson extrapolation — improved FD accuracy ───────────────────
    print("\n" + "=" * 70)
    print("RICHARDSON EXTRAPOLATION  —  O(ε²) → O(ε⁴) error reduction")
    print("=" * 70)

    rich_slopes = []
    for eps_f in RICH_EPS_FRACS:
        fd_lo = _shi_with_calendar_correction(
            (storage_p_base * (1.0 - eps_f)).tolist(),
            (storage_e_base * (1.0 - eps_f)).tolist(),
            e_cap_base, bat_params, shi_fit, 25.0, dt, eol_thresholds,
        )["fd"]
        fd_hi = _shi_with_calendar_correction(
            (storage_p_base * (1.0 + eps_f)).tolist(),
            (storage_e_base * (1.0 + eps_f)).tolist(),
            e_cap_base, bat_params, shi_fit, 25.0, dt, eol_thresholds,
        )["fd"]
        slope = (fd_hi - fd_lo) / (2.0 * eps_f)
        rich_slopes.append((eps_f, slope))
        print(f"  ε = {eps_f*100:.1f}%   FD slope = {slope:.8e}")

    # Richardson combination: cancel O(ε²) term
    # R = (4 × FD(ε/2) − FD(ε)) / 3
    r1 = (4.0 * rich_slopes[1][1] - rich_slopes[0][1]) / 3.0
    r2 = (4.0 * rich_slopes[2][1] - rich_slopes[1][1]) / 3.0

    print(f"\n  Richardson R1 (from 10%,  5%) : {r1:.8e}")
    print(f"  Richardson R2 (from  5%, 2.5%): {r2:.8e}")
    print(f"  Analytical prediction (total)  : {total_pred:.8e}")
    print(f"  Ratio R1 / pred                : {r1 / total_pred:.6f}")
    print(f"  Ratio R2 / pred                : {r2 / total_pred:.6f}")
    print(f"  R1 vs R2 consistency           : {abs(r1 - r2):.2e}  "
          f"(should be << {abs(rich_slopes[0][1] - rich_slopes[1][1]):.2e})")
    print("=" * 70)

    # ── Richardson convergence plot ───────────────────────────────────────
    # Use r2 (finest Richardson estimate) as the reference "true" derivative.
    # fd_errors measures how far each raw FD slope is from that converged value.
    # This shows O(ε²) convergence of the central difference toward the true slope.
    # A separate horizontal line shows the fixed calendar residual — the gap
    # between the converged FD and the analytical prediction (total_pred).
    true_slope = r2
    eps_vals   = [r[0] for r in rich_slopes]
    fd_errors  = [abs(r[1] - true_slope) for r in rich_slopes]
    calendar_residual = abs(true_slope - total_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(eps_vals, fd_errors, 'o--', color='#2c7bb6', lw=1.8, ms=8,
              label='Central difference  O(ε²)\n(error vs Richardson-converged value)')
    ax.axhline(calendar_residual, color='#d7191c', lw=1.8, ls='--',
               label=f'Calendar approx. residual = {calendar_residual:.2e}\n'
                     f'(converged FD vs analytical pred)')
    ax.set_xlabel('Perturbation size ε  [fraction of nominal dispatch]', fontsize=10)
    ax.set_ylabel('|FD slope − Richardson-converged value|', fontsize=10)
    ax.set_title('FD convergence — O(ε²) central difference\n'
                 'Blue converges to zero; red line = irreducible calendar residual',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(VALIDATION_DIR / f"richardson_convergence_{run_ts}.png", dpi=200)
    plt.close()
    print(f"  ✓ Plot saved: richardson_convergence_{run_ts}.png")

    # ── Save CSV ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if SAVE_CSV:
        # Base results
        base_csv  = RESULTS_DIR / f"battery_optimization_results_{run_label}.csv"
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
                "solver":               pyo_solver,
                "no_battery":           False,
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
                "solver":               pyo_solver,
                "no_battery":           False,
            },
        ]
        pd.DataFrame(base_rows).to_csv(
            base_csv, mode="a", header=not base_csv.exists(), index=False
        )

        # Single-year degradation (Shi + calendar correction)
        degr_csv = RESULTS_DIR / f"battery_degradation_results_{run_label}.csv"

        def _degr_row(d: Dict, case: str) -> Dict:
            stats = d.get("shi_cycle_stats", d.get("xu_cycle_stats", {}))
            return {
                "timestamp":          timestamp,
                "case":               case,
                "model":              d["meta"].get("model", "Shi2018"),
                "calendar_correction": d["meta"].get("calendar_correction", ""),
                "fd_total":           d["fd"],
                "fd_cycle":           d["fd_cycle"],
                "fd_calendar":        d["fd_calendar"],
                "fd_calendar_pct":    (
                    100.0 * d["fd_calendar"] / max(d["fd"], 1e-30)
                ),
                "soh_pct":            d["soh"],
                "capacity_fade_pct":  d["capacity_fade_percent"],
                "e_cap_degraded_MWh": d["e_cap_degraded"],
                "total_cycles_efc":   d["total_cycles"],
                "n_rainflow_cycles":  stats.get("n_rainflow_cycles", 0),
                "mean_dod_pct":       stats.get("mean_dod", 0) * 100,
                "mean_soc_pct":       stats.get("mean_soc", 0) * 100,
                "eol_80_yr":          d["eol_years"].get(0.80),
                "eol_70_yr":          d["eol_years"].get(0.70),
                "eol_60_yr":          d["eol_years"].get(0.60),
            }

        degr_rows = [_degr_row(degr_fixed, "dispatch_fixed")]
        pd.DataFrame(degr_rows).to_csv(
            degr_csv, mode="a", header=not degr_csv.exists(), index=False
        )

        print(f"\n  ✓ CSV: {base_csv.name}")
        print(f"  ✓ CSV: {degr_csv.name}")

    # ── Save text report ──────────────────────────────────────────────────
    if SAVE_REPORT:
        report_path = RESULTS_DIR / f"degradation_report_{run_label}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("WP2 BATTERY — Shi (2018) + Xu calendar | v2 REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated : {timestamp}\n")
            f.write(f"Dataset   : {PRICE_CSV.name}\n")
            f.write(f"Horizon   : {n:,} h  ({period_days:.1f} days)\n")
            f.write(f"Solver    : {pyo_solver}\n\n")
            f.write(f"Shi polynomial: k3={shi_fit.k3:.4e}  k4={shi_fit.k4:.4f}  "
                    f"R²={shi_fit.r2:.4f}  window=[{soc_min},{soc_max}]\n\n")

            for case_name, d in [("dispatch_fixed", degr_fixed)]:
                f.write(f"DEGRADATION ({case_name}) — Shi2018 + Xu calendar\n")
                f.write("-" * 60 + "\n")
                f.write(f"fd_total      : {d['fd']:.6f}\n")
                f.write(f"  fd_cycle    : {d['fd_cycle']:.6f}  "
                        f"({100*d['fd_cycle']/max(d['fd'],1e-30):.0f}%)\n")
                f.write(f"  fd_calendar : {d['fd_calendar']:.6f}  "
                        f"({100*d['fd_calendar']/max(d['fd'],1e-30):.0f}%)\n")
                f.write(f"SoH           : {d['soh']:.3f}%\n")
                for thr, yr in d["eol_years"].items():
                    f.write(f"EoL {thr*100:.0f}%      : {yr} yr\n")
                f.write("\n")

        print(f"  ✓ Report: {report_path.name}")

    # ── Plots ─────────────────────────────────────────────────────────────
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
        ax[0].set_xlabel("Time [days]"); ax[0].set_ylabel("Power [MW]")
        ax[0].set_title("Dispatch-Fixed: Power Export Profile")
        ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)
        ax[1].plot(time_vec, os_fixed.storage_e[0].data, linewidth=0.7)
        ax[1].set_xlabel("Time [days]"); ax[1].set_ylabel("SoC [MWh]")
        ax[1].set_title("Dispatch-Fixed: Battery State of Charge")
        ax[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            PLOTS_DIR / f"battery_baseline_results_{run_label}.png", dpi=200
        )

        # Single-year degradation detail
        plot_degradation_analysis(
            degr_fixed,
            storage_e=os_fixed.storage_e[0].data,
            time_vec=time_vec,
            save_path=str(
                PLOTS_DIR / f"battery_degradation_analysis_fixed_{run_label}.png"
            ),
            show=False, verbose=True, eol_thresholds=eol_thresholds,
        )

        if show_plots:
            plt.show()
        else:
            plt.close("all")

    print("\n" + "=" * 80)
    print("✓ COMPLETE — v3.1 (single-year | FD gradient validation)")    
    print("=" * 80)

if __name__ == "__main__":
    main()