"""WP2 battery optimization + Shi degradation — v5.3 (wp2_econ).

Multi-year, degradation-aware NPV sweep over battery energy capacity for the IEA Wind Task 50 reference site. Built on the validated Castillo gradient and the dual-Phi degradation framework 
(Xu reporting / Shi gradient).

What v5.3 changes
─────────────────
Shared economics moved to wp2_econ.py — the single source of truth for the conventions that must stay identical between this file and the Plan B sweep:
  - symmetric round-trip efficiency, eff_in = eff_out = sqrt(rte_ac) with the PCU included (eta_symmetric)
  - the SHIPP discount convention, years 1..N-1 (discount_weights / annuity_factor; see HORIZON below)
  - capex and replacement-cost scope, energy + power expansion (capex, replacement_cost)
  - the degradation penalty cost (degradation_cost)
Each file keeps its own multi-year orchestration; only the formulas are shared, so the two sweeps cannot silently diverge on efficiency, horizon, or cost basis.

Multi-year NPV
──────────────
Each year re-solves the LP at capacity faded to e_cap_nominal * SoH, so later years earn less. Per-year undiscounted revenue is extracted and discounted individually:

      NPV = -capex_initial + Σ_{t=1}^{N-1} rev_t * (1+r)^-t - PV(replacements)

Reported on two bases:
  - BATTERY-ONLY (marginal): total plant revenue − wind-only baseline; isolates the battery decision and captures curtailment recovery.
  - TOTAL PLANT (SHIPP-compatible): wind + battery; the basis the rc_e_cap1 gradient validates against.
(The SHIPP-arbitrage track, dot(price, storage_p), is added in a later step and is not present yet.)

HORIZON
───────
discount_weights() sums years 1..N-1, matching the SHIPP kernel factor npf.npv(r, ones(n_year)) - 1, which drops the final year — a known off-by-one flagged to the supervisor. 
The convention lives in one place in wp2_econ.py and flips there if confirmed.

Not edited (used as-is)
───────────────────────
degradation_xu.py, degradation_shi.py, degradation_subgradient.py, kernel_pyomo.py, the plotting modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time
import json

from shipp.kernel_pyomo import solve_lp_pyomo
from shipp.components import Storage, Production, TimeSeries

from degradation_xu import (
    analyze_degradation,
    count_equivalent_full_cycles,
    rainflow_cycle_counting,
    ft_calendar,
    sei_capacity_loss,
)
from degradation_shi import analyze_degradation_shi, phi_shi_prime_with_stress, s_soc, s_temp
from degradation_subgradient import compute_subgradient, fit_shi_polynomial
from wp2_econ import (eta_symmetric, capex, replacement_cost, annuity_factor, discount_weights, degradation_cost, revenue_annual, HEADLINE_BASIS)

import xarray as xr
from py_wake.site import XRSite
import numpy_financial as npf

from wp2_common import quick_setup, get_wake_model

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

discount_rate = 0.03
dt            = 1.0

# Solver
pyo_solver = "gurobi"

# Horizon
RUN_FULL_YEAR     = True
N_DAYS_TEST       = 30
MAX_HOURS_SPARSE  = 180 * 24

# Problem
p_min      = 0.0
WAKE_MODEL = "Bastankhah"

# EoL
eol_thresholds   = [0.80, 0.70, 0.60]
N_YEARS          = 20
EOL_REPLACEMENT  = 0.70
EOL_REPLACEMENT_TOL = 0.005

# ── Sweep configuration ────────────────────────────────────────────────────
SWEEP_MODE = True     # True = E_cap sweep (Phase 2);  False = single-point (legacy)

# E_CAP_GRID = [300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1100, 1300, 1500]  # MWh
E_CAP_GRID = [600, 800]  # MWh

# P_cap for the sweep.  None = use YAML value (150 MW WP2 reference).
# Override examples:  275 (Plan B optimal),  325 (grid connection limit)
P_CAP_SWEEP = 275

# Duration-based pruning:  skip E/P combos outside [MIN, MAX] hours
# Points outside this range are skipped (not cloned — 1D sweep doesn't need heatmap fill)
MIN_DURATION_H = 0.5    # E/P < 0.5h → power-bottlenecked, unrealistic
MAX_DURATION_H = 10.0   # E/P > 10h  → barely cycles, marginal revenue

# Verbosity during sweep:  False → one-line summary per E_cap (much faster output)
SWEEP_VERBOSE  = False

# ── Replot from saved CSV (skip the LP sweep entirely) ─────────────────
# Set REPLOT_FROM_LAST = True to find the most recent CSV and replot.
# Set REPLOT_CSV to an explicit path for a specific older run.
# If both are set, REPLOT_CSV takes priority.
# Leave both at defaults to run the full sweep normally.
REPLOT_FROM_LAST: bool = False
REPLOT_CSV: str | None = None

show_plots = False

run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
FILE_TAG = "v53"   

# =============================================================================
# Console logging — tee all output to a file
# =============================================================================

class TeeLogger:
    """Duplicates stdout/stderr to a log file while preserving console output."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._file = open(log_path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def start(self):
        sys.stdout = self
        sys.stderr = _TeeStream(self._stderr, self._file)
        return self

    def write(self, text):
        self._stdout.write(text)
        self._file.write(text)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def stop(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._file.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()


class _TeeStream:
    """Helper for stderr tee."""
    def __init__(self, original, log_file):
        self._orig = original
        self._file = log_file

    def write(self, text):
        self._orig.write(text)
        self._file.write(text)

    def flush(self):
        self._orig.flush()
        self._file.flush()


# =============================================================================
# Data loading helpers  (from v5.1, unchanged)
# =============================================================================

@dataclass
class HorizonData:
    n:         int
    ws:        np.ndarray
    wd:        np.ndarray
    ti:        Optional[np.ndarray]
    price_eur: np.ndarray


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


def _run_pywake_power_MW(setup, wd, ws, ti) -> np.ndarray:
    n    = len(ws)
    site = XRSite(ds=xr.Dataset(data_vars=dict(P=1)))
    wf_model = get_wake_model(WAKE_MODEL, site, setup["windturbine"])
    time_days = np.arange(n) / 24.0
    kwargs = {"x": setup["x"], "y": setup["y"], "wd": wd, "ws": ws, "time": time_days}
    if ti is not None:
        kwargs["TI"] = ti
    return wf_model(**kwargs).Power.sum(["wt"]).values / 1e6


# =============================================================================
# Core helpers  (from v5.1)
# =============================================================================

def _build_storage_year(
    e_cap_eff, p_cap_MW, rte_ac, e_cost_EUR_per_MWh, soc_min, soc_max
) -> Storage:
    dod_eff = 1.0 - soc_min if pyo_solver != "none" else 1.0
    eta = eta_symmetric(rte_ac)   # symmetric split: eff_in = eff_out = sqrt(rte_ac) = 0.9367
    return Storage(
        e_cap=e_cap_eff, p_cap=p_cap_MW, eff_in=eta, eff_out=eta,
        e_cost=e_cost_EUR_per_MWh, p_cost=0.0, dod=dod_eff,
    )


def _shi_with_calendar_correction(
    storage_p, storage_e, e_cap_eff, bat_params, shi_fit,
    T_cell_C, dt_hours, eol_thresholds,
) -> Dict:
    """Shi cycle degradation + Xu calendar term (reporting path only)."""
    n_steps         = len(storage_e)
    t_total_seconds = n_steps * dt_hours * 3600.0
    t_total_hours   = t_total_seconds / 3600.0
    e_arr           = np.asarray(storage_e, dtype=float)
    sigma_mean      = float(np.mean(e_arr)) / max(e_cap_eff, 1e-9)

    degr = analyze_degradation_shi(
        storage_p, storage_e, e_cap_eff, bat_params,
        shi_fit=shi_fit, T_cell_C=T_cell_C,
        dt_hours=dt_hours, eol_thresholds=eol_thresholds,
    )

    fd_cal       = ft_calendar(t_total_seconds, sigma_mean, T_cell_C)
    fd_corrected = degr["fd_shi"] + fd_cal
    L_corr       = sei_capacity_loss(fd_corrected)
    cap_ret_corr = 1.0 - L_corr
    soh_corr     = cap_ret_corr * 100.0

    fd_per_yr = fd_corrected / max(t_total_hours / 8760.0, 1e-9)
    eol_corr: Dict[float, Optional[float]] = {}
    for thr in eol_thresholds:
        if (1.0 - sei_capacity_loss(0.0)) < thr:
            eol_corr[thr] = 0.0
            continue
        lo, hi = 0.0, 200.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if 1.0 - sei_capacity_loss(fd_per_yr * mid) > thr:
                lo = mid
            else:
                hi = mid
        val = (lo + hi) / 2.0
        eol_corr[thr] = round(val, 2) if val < 190.0 else None

    degr["fd_calendar"]        = float(fd_cal)
    degr["fd"]                 = float(fd_corrected)
    degr["fd_cycle"]           = degr["fd_shi"]
    degr["capacity_retention"] = float(cap_ret_corr)
    degr["capacity_loss"]      = float(L_corr)
    degr["soh"]                = float(soh_corr)
    degr["capacity_fade_percent"] = float(L_corr * 100.0)
    degr["eol_years"]          = eol_corr
    degr["e_cap_degraded"]     = e_cap_eff * cap_ret_corr
    degr["p_cap_degraded"]     = float(bat_params["power_capacity_W"]) / 1e6 * cap_ret_corr
    degr["meta"]["calendar_correction"] = "Xu ft_calendar, reporting only"
    return degr


# =============================================================================
# Multi-year loop  (v5.3: MODIFIED for proper discounted NPV)
# =============================================================================

def _run_multiyear(
    wind_8760:          np.ndarray,
    price_8760:         np.ndarray,
    stor_null:          Storage,
    p_max_MW:           float,
    e_cap_nominal:      float,
    p_cap_MW:           float,
    rte_ac:             float,
    e_cost_EUR_per_MWh: float,
    p_cost_EUR_per_MW:  float,
    repl_e_EUR_per_MWh: float,
    repl_p_EUR_per_MW:  float,
    bat_params:         Dict,
    shi_fit,
    soc_min:            float,
    soc_max:            float,
    T_cell_C:           float = 25.0,
    verbose:            bool  = True,
) -> Dict:
    
    """Year-by-year degradation loop with capacity fade feedback and replacement.

    Each year:
      - re-solves the LP at capacity faded to e_cap_nominal * SoH,
      - extracts per-year UNDISCOUNTED revenue on two bases:
          battery-only (marginal) = total plant − wind-only baseline, total plant = wind + battery (SHIPP-compatible),
      - accumulates Shi cycle + Xu calendar degradation, updates SoH, and triggers a replacement when SoH falls below the threshold.

    Lifetime NPV is assembled through wp2_econ (capex, replacement_cost, discount_weights, annuity_factor) on the SHIPP year-1..N-1 convention:
          NPV = -capex_initial + Σ rev_t * weights[t-1] - PV(replacements)
    A no-degradation NPV (year-1 revenue × annuity) and a Plan B single-year extrapolation NPV are also returned for comparison.
    """

    n           = len(wind_8760)
    scale       = 1.0 / max(n / 8760.0, 1e-9)     # annualisation factor
    price_eur   = price_8760          # prices in EUR for revenue

    # Wind-only baseline: what the wind farm earns with NO battery (curtailed at p_max)
    wind_export_no_bat = np.minimum(wind_8760, p_max_MW)
    rev_wind_only = (365.0 * 24.0 / n) * float(np.dot(price_eur, wind_export_no_bat)) * dt

    fd_cumulative    = 0.0
    soh              = 1.0
    n_replacements   = 0
    replacement_years: List[int]   = []
    soh_trajectory:    List[tuple] = []
    annual_fd:         List[tuple] = []
    annual_gradient:   List        = []
    annual_soc:        List        = []
    annual_revenue:       List[float] = []   # battery-only marginal revenue per year
    annual_revenue_total: List[float] = []   # total plant revenue per year (wind+bat)
    annual_revenue_arb:   List[float] = []   # battery arbitrage revenue per year (price . storage_p)
    soc_frac_fixed:       float       = None

    if verbose:
        period_days = n * dt / 24.0
        print(f"\n{'─'*70}")
        print(f"Multi-year loop: {N_YEARS} yr | tile: {n} h ({period_days:.1f} d) | "
              f"E_cap={e_cap_nominal:.0f} MWh | P_cap={p_cap_MW:.0f} MW | "
              f"SoH threshold: {EOL_REPLACEMENT*100:.0f}%")
        print(f"{'─'*70}")
        print(f"  {'Year':>4}  {'SoH_%':>7}  {'fd_yr':>9}  {'fd_cum':>9}  "
              f"{'rev_kEUR':>10}  {'t_lp_s':>7}  {'Replaced':>8}")

    t_loop_start = time.perf_counter()
    t_lp_total   = 0.0

    for year in range(1, N_YEARS): # 19 revenue years (SHIPP convention)
        t_yr_start = time.perf_counter()

        # ── Effective capacity this year ──
        e_cap_eff = e_cap_nominal * soh
        e_start1_yr = soc_frac_fixed * e_cap_eff if soc_frac_fixed is not None else None

        # ── Re-solve LP with degraded capacity ──
        stor_yr  = _build_storage_year(
            e_cap_eff, p_cap_MW, rte_ac, e_cost_EUR_per_MWh, soc_min, soc_max
        )
        price_dam = TimeSeries((price_8760).tolist(), dt)
        prod_yr   = Production(TimeSeries(wind_8760.tolist(), dt), p_cost=0.0)
        prod_null = Production(TimeSeries([0.0] * n, dt), p_cost=0.0)

        t_lp_start = time.perf_counter()
        os_yr = solve_lp_pyomo(
            price_dam, prod_yr, prod_null, stor_yr, stor_null,
            discount_rate, N_YEARS, p_min, p_max_MW, n,
            pyo_solver, fixed_cap=True, soc_max1=soc_max,
            return_duals=True, e_start1=e_start1_yr,
        )
        t_lp_yr = time.perf_counter() - t_lp_start
        t_lp_total += t_lp_yr

        # ── Extract per-year revenues on three bases (via wp2_econ) ──────
        #   arbitrage : price . storage_p                 (battery-as-asset; SHIPP a_npv)
        #   marginal  : total plant - wind-only baseline  (+ curtailment recovery)
        #   total     : price . (wind_after_curtailment + battery)
        # All three come from the same dispatch at no extra cost. revenue_annual
        # returns arbitrage + marginal; total is reconstructed as marginal + wind-only.
        p_prod_yr = np.array(os_yr.production_p[0].data, dtype=float)
        p_bat_yr  = np.array(os_yr.storage_p[0].data,    dtype=float)
        rev_yr      = revenue_annual(price_eur, p_bat_yr, p_prod_yr,
                                     wind_8760, p_max_MW, n, dt)
        rev_arb_yr  = rev_yr["arbitrage"]
        rev_bat_yr  = rev_yr["marginal"]             # marginal == battery-only basis
        rev_total_yr = rev_bat_yr + rev_wind_only    # total = marginal + wind-only baseline
        annual_revenue_arb.append(rev_arb_yr)
        annual_revenue.append(rev_bat_yr)
        annual_revenue_total.append(rev_total_yr)

        # ── Annual degradation with calendar correction ──
        storage_e_yr = os_yr.storage_e[0].data
        if soc_frac_fixed is None:
            soc_frac_fixed = os_yr.soc_final / e_cap_eff
        annual_soc.append(np.array(storage_e_yr, dtype=float).copy())
        storage_p_yr = os_yr.storage_p[0].data

        degr_yr = _shi_with_calendar_correction(
            storage_p_yr, storage_e_yr, e_cap_eff,
            bat_params, shi_fit, T_cell_C, dt, eol_thresholds,
        )

        fd_yr          = degr_yr["fd"]          * scale
        fd_yr_cycle    = degr_yr["fd_cycle"]    * scale
        fd_yr_calendar = degr_yr["fd_calendar"] * scale
        fd_cumulative += fd_yr
        annual_fd.append((fd_yr, fd_yr_cycle, fd_yr_calendar))

        # ── Per-year gradient (Castillo, unchanged from v5.1) ──
        grad_yr = None
        if os_yr.dual_prices is not None:
            dual_yr   = os_yr.dual_prices["dual_e_min1"]
            e_cap_yr  = os_yr.dual_prices["e_cap1"]
            cycles_yr = rainflow_cycle_counting(storage_e_yr, e_cap_yr)
            sg_yr     = compute_subgradient(
                storage_e=storage_e_yr, cycles=cycles_yr, dt_hours=dt,
                battery_replacement_cost_per_MWh=repl_e_EUR_per_MWh,
                eff_in=eta_symmetric(rte_ac), eff_out=eta_symmetric(rte_ac), shi_fit=shi_fit,
            )

            factor_yr        = npf.npv(discount_rate, np.ones(N_YEARS)) - 1
            dual_per_year_yr = dual_yr / factor_yr

            # [0] OLD fused scalar (deprecated, kept for continuity)
            dDeg_dDoD_yr = -e_cap_yr * float(np.dot(sg_yr["subgrad_combined"], dual_per_year_yr))

            dods_yr     = np.array([c["dod"]      for c in cycles_yr])
            cnts_yr     = np.array([c["count"]    for c in cycles_yr])
            socm_yr     = np.array([c["soc_mean"] for c in cycles_yr])
            mean_dod_yr = float(np.average(dods_yr, weights=cnts_yr)) if len(dods_yr) > 0 else 0.0

            # TERM 1: dRev/dE_cap (exact, Castillo Theorem 1)
            dual_max_yr   = os_yr.dual_prices["dual_e_max1"]
            dRev_dEcap_yr = (
                soc_max * float(np.sum(dual_max_yr))
                - soc_min * float(np.sum(dual_yr))
            )

            # TERM 2: dDegCost/dE_cap (frozen-dispatch)
            if len(dods_yr) > 0:
                phi_prime_cyc = phi_shi_prime_with_stress(
                    dods_yr, socm_yr, T_cell_C, shi_fit.k3, shi_fit.k4
                )
                ddod_dEcap_yr     = -dods_yr / e_cap_yr
                dDegCost_dEcap_yr = repl_e_EUR_per_MWh * factor_yr * scale * float(
                    np.sum(cnts_yr * phi_prime_cyc * ddod_dEcap_yr)
                )
            else:
                dDegCost_dEcap_yr = 0.0

            # TERM 3: lambda_E
            lambda_E_yr   = e_cost_EUR_per_MWh # dCapex/dE: marginal capacity added at INITIAL capex (245), not replacement

            # Assembled gradient
            dNPV_dEcap_yr = dRev_dEcap_yr - dDegCost_dEcap_yr - lambda_E_yr
            rc_e_cap_yr   = float(os_yr.dual_prices.get("rc_e_cap1", 0.0))

            grad_yr = (
                dDeg_dDoD_yr, float(np.mean(np.abs(sg_yr["subgrad_combined"]))),
                mean_dod_yr, float(np.mean(np.abs(dual_per_year_yr))),
                float(e_cap_yr), float(sg_yr["cycle_coverage"]),
                sg_yr["subgrad_combined"].copy(),
                dRev_dEcap_yr, dDegCost_dEcap_yr, lambda_E_yr,
                dNPV_dEcap_yr, rc_e_cap_yr,
            )
        annual_gradient.append(grad_yr)

        # ── EFC for this year ──
        efc_yr = count_equivalent_full_cycles(
            storage_p_yr, storage_e_yr, e_cap_eff, dt_hours=dt
        )

        # ── Update SoH ──
        L_cum = sei_capacity_loss(fd_cumulative)
        soh   = 1.0 - L_cum
        soh_trajectory.append((year, soh * 100.0, fd_cumulative, n_replacements))

        eol_trigger = EOL_REPLACEMENT + EOL_REPLACEMENT_TOL

        if verbose:
            replaced_tag = "YES" if soh < eol_trigger else ""
            print(f"  {year:>4d}  {soh*100:>7.3f}  {fd_yr:>9.5f}  "
                  f"{fd_cumulative:>9.5f}  {rev_bat_yr*1e-3:>10.1f}  "
                  f"{t_lp_yr:>7.2f}  {replaced_tag:>8}")

        # ── Battery replacement ──
        if soh < eol_trigger:
            n_replacements += 1
            replacement_years.append(year)
            fd_cumulative = 0.0
            soh           = 1.0
            soc_frac_fixed = None
            if verbose:
                print(f"  *** Battery #{n_replacements} replaced at end of year {year} ***")

    t_loop_total = time.perf_counter() - t_loop_start

    # ═════════════════════════════════════════════════════════════════════════
    # DISCOUNTED LIFETIME NPV — DUAL BASIS (battery-only + total plant)
    # Economics via wp2_econ; SHIPP year-1..N-1 discount convention.
    # ═════════════════════════════════════════════════════════════════════════
    weights           = discount_weights(discount_rate, N_YEARS)   # 19 per-year factors
    factor            = annuity_factor(discount_rate, N_YEARS)      # = sum(weights)
    capex_initial     = capex(e_cap_nominal, p_cap_MW,
                              e_cost_EUR_per_MWh, p_cost_EUR_per_MW)
    capex_replacement = replacement_cost(e_cap_nominal, p_cap_MW,
                                         repl_e_EUR_per_MWh, repl_p_EUR_per_MW)

    # PV of replacement costs (shared by both bases)
    pv_replacements = sum(capex_replacement * weights[yr - 1]
                          for yr in replacement_years)

    # ── A. BATTERY-ONLY basis ────────────────────────────────────────────
    #   rev_battery = rev_total_plant − rev_wind_only
    #   Isolates the battery investment decision from the wind farm.

    pv_rev_bat        = float(np.dot(annual_revenue, weights))
    npv_bat_multiyear = -capex_initial + pv_rev_bat - pv_replacements
    npv_bat_no_deg    = -capex_initial + annual_revenue[0] * factor

    # Plan B style (single-year extrapolation)
    fd_yr1_total   = annual_fd[0][0]
    deg_cost_planB = degradation_cost(fd_yr1_total, e_cap_nominal,
                                      repl_e_EUR_per_MWh, factor)
    npv_bat_planB  = npv_bat_no_deg - deg_cost_planB

    # ── A2. ARBITRAGE basis (battery-as-asset: price . storage_p) ─────────
    #   SHIPP a_npv definition. Differs from marginal only by curtailment
    #   recovery (~0.02% at this site). HEADLINE_BASIS selects which is primary.
    pv_rev_arb        = float(np.dot(annual_revenue_arb, weights))
    npv_arb_multiyear = -capex_initial + pv_rev_arb - pv_replacements
    npv_arb_no_deg    = -capex_initial + annual_revenue_arb[0] * factor
    npv_arb_planB     = npv_arb_no_deg - deg_cost_planB   # same deg penalty (capacity-based)

    # ── B. TOTAL PLANT basis (wind + battery, SHIPP-compatible) ──────────
    #   Same as kernel's own NPV, but with proper per-year discounting.
    #   The gradient (rc_e_cap1) validates against this basis.
    pv_rev_total        = float(np.dot(annual_revenue_total, weights))

    npv_total_multiyear = -capex_initial + pv_rev_total - pv_replacements
    npv_total_no_deg    = -capex_initial + annual_revenue_total[0] * factor

    if verbose:
        print(f"{'─'*70}")
        print(f"  Loop time: {t_loop_total:.1f}s ({t_lp_total:.1f}s LP)")
        print(f"  Final SoH: {soh*100:.2f}% | Replacements: {n_replacements}")
        print(f"\n  ── NPV comparison (v5.3) ─────────────────────────────────")
        print(f"  Wind-only baseline : {rev_wind_only*1e-3:>9.0f} kEUR/yr  (constant)")
        print(f"  Battery capex      : {capex_initial*1e-6:>9.2f} MEUR")
        print(f"  PV(replacements)   : {pv_replacements*1e-6:>9.2f} MEUR")
        print(f"")
        print(f"  BATTERY-ONLY basis:")
        print(f"    PV(bat. rev)     : {pv_rev_bat*1e-6:>9.2f} MEUR")
        print(f"    NPV (multi-year) : {npv_bat_multiyear*1e-6:>9.2f} MEUR  ← corrected")
        print(f"    NPV (no-deg)     : {npv_bat_no_deg*1e-6:>9.2f} MEUR")
        print(f"    NPV (Plan B)     : {npv_bat_planB*1e-6:>9.2f} MEUR")
        print(f"    Bat rev yr1→yr{N_YEARS} : {annual_revenue[0]*1e-3:.0f} → "
              f"{annual_revenue[-1]*1e-3:.0f} kEUR  "
              f"({(annual_revenue[-1]/annual_revenue[0] - 1)*100:+.1f}%)")
        print(f"")
        print(f"  ARBITRAGE basis (battery-as-asset; SHIPP a_npv):")
        print(f"    PV(arb. rev)     : {pv_rev_arb*1e-6:>9.2f} MEUR")
        print(f"    NPV (multi-year) : {npv_arb_multiyear*1e-6:>9.2f} MEUR")
        print(f"    NPV (no-deg)     : {npv_arb_no_deg*1e-6:>9.2f} MEUR")
        print(f"    NPV (Plan B)     : {npv_arb_planB*1e-6:>9.2f} MEUR")
        print(f"")
        print(f"  TOTAL PLANT basis (SHIPP-compatible):")
        print(f"    PV(total rev)    : {pv_rev_total*1e-6:>9.2f} MEUR")
        print(f"    NPV (multi-year) : {npv_total_multiyear*1e-6:>9.2f} MEUR")
        print(f"    NPV (no-deg)     : {npv_total_no_deg*1e-6:>9.2f} MEUR")
        print(f"    Total rev yr1→yr{N_YEARS}: {annual_revenue_total[0]*1e-3:.0f} → "
              f"{annual_revenue_total[-1]*1e-3:.0f} kEUR  "
              f"({(annual_revenue_total[-1]/annual_revenue_total[0] - 1)*100:+.1f}%)")
        print(f"  ──────────────────────────────────────────────────────────")

    return {
        # ── Legacy fields (v5.1 compatible) ──
        "soh_trajectory":       soh_trajectory,
        "annual_fd":            annual_fd,
        "n_replacements":       n_replacements,
        "replacement_years":    replacement_years,
        "replacement_cost_EUR": n_replacements * capex_replacement,
        "final_soh":            soh,
        "final_soh_pct":        soh * 100.0,
        "e_cap_nominal":        e_cap_nominal,
        "annual_gradient":      annual_gradient,
        "annual_soc":           annual_soc,
        # ── v5.2: per-year revenues ──
        "annual_revenue_bat_eur":   annual_revenue,         # battery-only per year
        "annual_revenue_total_eur": annual_revenue_total,   # total plant per year
        "rev_wind_only_eur":        rev_wind_only,          # wind baseline (constant)
        # ── v5.2: battery-only NPV ──
        "capex_initial_EUR":        capex_initial,
        "pv_rev_bat_EUR":           pv_rev_bat,
        "pv_replacements_EUR":      pv_replacements,
        "npv_bat_multiyear_EUR":    npv_bat_multiyear,
        "npv_bat_no_deg_EUR":       npv_bat_no_deg,
        "npv_bat_planB_EUR":        npv_bat_planB,
        # ── arbitrage basis (battery-as-asset; SHIPP a_npv) ──
        "annual_revenue_arb_eur":   annual_revenue_arb,
        "pv_rev_arb_EUR":           pv_rev_arb,
        "npv_arb_multiyear_EUR":    npv_arb_multiyear,
        "npv_arb_no_deg_EUR":       npv_arb_no_deg,
        "npv_arb_planB_EUR":        npv_arb_planB,
        # ── v5.2: total-plant NPV (SHIPP-compatible, gradient-compatible) ──
        "pv_rev_total_EUR":         pv_rev_total,
        "npv_total_multiyear_EUR":  npv_total_multiyear,
        "npv_total_no_deg_EUR":     npv_total_no_deg,
    }


# =============================================================================
# Phase 2: E_cap parameter sweep
# =============================================================================

def _run_ecap_sweep(
    power_wind_MW:      np.ndarray,
    price_eur:          np.ndarray,
    p_max_MW:           float,
    p_cap_MW:           float,
    rte_ac:             float,
    e_cost_EUR_per_MWh: float,
    p_cost_EUR_per_MW:  float,
    repl_e_EUR_per_MWh: float,
    repl_p_EUR_per_MW:  float,
    bat_params:         Dict,
    shi_fit,
    soc_min:            float,
    soc_max:            float,
    T_cell_C:           float = 25.0,
) -> List[Dict]:
    """Sweep over E_cap values, running the full multi-year loop at each point.

    Features (v5.2):
      - Duration-based pruning: skips E/P ratios outside [MIN, MAX]
      - ETA timer: estimates remaining time from running average
      - Verbosity control: SWEEP_VERBOSE=False suppresses per-year detail
    """
    n = len(price_eur)
    stor_null = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
                        e_cost=0, p_cost=0)

    # ── Pre-filter grid by duration ──
    active_grid = []
    skipped = []
    for e_cap in E_CAP_GRID:
        duration_h = e_cap / p_cap_MW
        if duration_h < MIN_DURATION_H or duration_h > MAX_DURATION_H:
            skipped.append((e_cap, duration_h))
        else:
            active_grid.append(e_cap)

    n_active  = len(active_grid)
    n_skipped = len(skipped)
    n_total   = len(E_CAP_GRID)
    n_lp      = n_active * N_YEARS

    print(f"\n{'═'*70}")
    print(f"E_cap SWEEP: {n_total} grid points, {n_active} active, "
          f"{n_skipped} pruned")
    print(f"P_cap fixed at {p_cap_MW:.0f} MW | "
          f"Duration filter: [{MIN_DURATION_H:.1f}, {MAX_DURATION_H:.1f}] h")
    print(f"LP solves: {n_active} × {N_YEARS} yr = {n_lp}")
    if skipped:
        sk_str = ", ".join(f"{e:.0f} ({d:.1f}h)" for e, d in skipped)
        print(f"Skipped: {sk_str}")
    print(f"{'═'*70}")

    results = []
    t_sweep_start = time.perf_counter()
    times_so_far: List[float] = []

    for idx, e_cap in enumerate(active_grid):
        duration_h = e_cap / p_cap_MW

        # ETA
        if times_so_far:
            avg_s = np.mean(times_so_far)
            eta_s = avg_s * (n_active - idx)
            eta_str = f"ETA {eta_s/60:.0f}m" if eta_s > 120 else f"ETA {eta_s:.0f}s"
        else:
            eta_str = ""

        print(f"\n[{idx+1}/{n_active}] E_cap={e_cap:.0f} MWh  "
              f"E/P={duration_h:.1f}h  {eta_str}")

        t0 = time.perf_counter()
        my = _run_multiyear(
            wind_8760=power_wind_MW,
            price_8760=price_eur,
            stor_null=stor_null,
            p_max_MW=p_max_MW,
            e_cap_nominal=e_cap,
            p_cap_MW=p_cap_MW,
            repl_e_EUR_per_MWh=repl_e_EUR_per_MWh,
            repl_p_EUR_per_MW=repl_p_EUR_per_MW,
            rte_ac=rte_ac,
            e_cost_EUR_per_MWh=e_cost_EUR_per_MWh,
            p_cost_EUR_per_MW=p_cost_EUR_per_MW,
            bat_params=bat_params,
            shi_fit=shi_fit,
            soc_min=soc_min,
            soc_max=soc_max,
            T_cell_C=T_cell_C,
            verbose=SWEEP_VERBOSE,
        )
        elapsed = time.perf_counter() - t0
        times_so_far.append(elapsed)

        # ── One-line summary (always printed, even when verbose=False) ──
        fd_yr1 = my["annual_fd"][0][0]
        grad_yr1 = my["annual_gradient"][0]
        dNPV_yr1 = grad_yr1[10] if grad_yr1 is not None else None
        rev_fade = (my["annual_revenue_bat_eur"][-1] /
                    my["annual_revenue_bat_eur"][0] - 1) * 100 \
                   if my["annual_revenue_bat_eur"][0] != 0 else 0.0

        print(f"  NPV_bat={my['npv_bat_multiyear_EUR']*1e-6:>7.1f}M  "
              f"NPV_tot={my['npv_total_multiyear_EUR']*1e-6:>7.1f}M  "
              f"fd={fd_yr1:.5f}  repl={my['n_replacements']}  "
              f"fade={rev_fade:+.1f}%  t={elapsed:.1f}s")

        results.append({
            "e_cap":           e_cap,
            "p_cap":           p_cap_MW,
            # Battery-only basis
            "npv_bat_no_deg":      my["npv_bat_no_deg_EUR"],
            "npv_bat_multiyear":   my["npv_bat_multiyear_EUR"],
            "npv_bat_planB":       my["npv_bat_planB_EUR"],
            "rev_bat_yr1_eur":     my["annual_revenue_bat_eur"][0],
            "rev_bat_yrN_eur":     my["annual_revenue_bat_eur"][-1],
            "pv_rev_bat":          my["pv_rev_bat_EUR"],
            # Arbitrage basis (battery-as-asset; SHIPP a_npv)
            "npv_arb_no_deg":      my["npv_arb_no_deg_EUR"],
            "npv_arb_multiyear":   my["npv_arb_multiyear_EUR"],
            "npv_arb_planB":       my["npv_arb_planB_EUR"],
            "rev_arb_yr1_eur":     my["annual_revenue_arb_eur"][0],
            "rev_arb_yrN_eur":     my["annual_revenue_arb_eur"][-1],
            "pv_rev_arb":          my["pv_rev_arb_EUR"],
            # Total plant basis (SHIPP-compatible)
            "npv_total_no_deg":    my["npv_total_no_deg_EUR"],
            "npv_total_multiyear": my["npv_total_multiyear_EUR"],
            "rev_total_yr1_eur":   my["annual_revenue_total_eur"][0],
            "rev_total_yrN_eur":   my["annual_revenue_total_eur"][-1],
            "pv_rev_total":        my["pv_rev_total_EUR"],
            # Common
            "fd_yr1":              fd_yr1,
            "n_replacements":      my["n_replacements"],
            "replacement_yrs":     my["replacement_years"],
            "final_soh_pct":       my["final_soh_pct"],
            "capex_eur":           my["capex_initial_EUR"],
            "pv_replacements":     my["pv_replacements_EUR"],
            "dNPV_dEcap_yr1":      dNPV_yr1,
            "elapsed_s":           elapsed,
            "multiyear_raw":       my,
        })

    total_time = time.perf_counter() - t_sweep_start
    print(f"\n{'─'*70}")
    print(f"Sweep complete: {n_active} points in {total_time:.0f}s "
          f"({total_time/60:.1f}m)  "
          f"avg {np.mean(times_so_far):.1f}s/point")

    return results

# =============================================================================
# Sweep output: CSV, report, plots
# =============================================================================

def _save_sweep_csv(results: List[Dict], p_cap_MW: float) -> Path:
    """Save sweep results to CSV."""
    rows = []
    for r in results:
        rev_fade = (r["rev_bat_yrN_eur"] / r["rev_bat_yr1_eur"] - 1) * 100 \
                   if r["rev_bat_yr1_eur"] != 0 else 0.0
        rows.append({
            "e_cap_MWh":                r["e_cap"],
            "p_cap_MW":                 r["p_cap"],
            "ep_ratio_h":               r["e_cap"] / r["p_cap"],
            # Battery-only NPVs
            "npv_bat_noDeg_MEUR":       r["npv_bat_no_deg"]    * 1e-6,
            "npv_bat_multiyear_MEUR":   r["npv_bat_multiyear"] * 1e-6,
            "npv_bat_planB_MEUR":       r["npv_bat_planB"]     * 1e-6,
            "rev_bat_yr1_kEUR":         r["rev_bat_yr1_eur"]   * 1e-3,
            "rev_bat_yrN_kEUR":         r["rev_bat_yrN_eur"]   * 1e-3,
            "rev_bat_fade_pct":         rev_fade,
            # Arbitrage NPVs (battery-as-asset; SHIPP a_npv)
            "npv_arb_noDeg_MEUR":       r["npv_arb_no_deg"]    * 1e-6,
            "npv_arb_multiyear_MEUR":   r["npv_arb_multiyear"] * 1e-6,
            "npv_arb_planB_MEUR":       r["npv_arb_planB"]     * 1e-6,
            "rev_arb_yr1_kEUR":         r["rev_arb_yr1_eur"]   * 1e-3,
            "rev_arb_yrN_kEUR":         r["rev_arb_yrN_eur"]   * 1e-3,
            "pv_rev_arb_MEUR":          r["pv_rev_arb"]        * 1e-6,
            # Total plant NPVs
            "npv_total_noDeg_MEUR":     r["npv_total_no_deg"]    * 1e-6,
            "npv_total_multiyear_MEUR": r["npv_total_multiyear"] * 1e-6,
            # Common
            "fd_yr1":                   r["fd_yr1"],
            "n_replacements":           r["n_replacements"],
            "replacement_yrs":          ";".join(str(y) for y in r["replacement_yrs"]),  # years SoH crossed 0.705
            "final_soh_pct":            r["final_soh_pct"],
            "capex_MEUR":               r["capex_eur"]         * 1e-6,
            "pv_rev_bat_MEUR":          r["pv_rev_bat"]        * 1e-6,
            "pv_rev_total_MEUR":        r["pv_rev_total"]      * 1e-6,
            "pv_repl_MEUR":             r["pv_replacements"]   * 1e-6,
            "dNPV_dEcap_yr1":           r["dNPV_dEcap_yr1"],
            "elapsed_s":                r["elapsed_s"],
        })
    csv_path = RESULTS_DIR / f"{FILE_TAG}_sweep_{run_ts}_P{int(p_cap_MW)}MW.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  ✓ CSV: {csv_path.name}")
    return csv_path

def _save_soh_trajectory_csv(results: List[Dict], p_cap_MW: float) -> Path:
    """Long-format per-year SoH trajectory for every E_cap (tidy data for plotting)."""
    rows = []
    for r in results:
        traj = r["multiyear_raw"]["soh_trajectory"]   # list of (year, soh_pct, fd_cum, n_repl)
        for (year, soh_pct, fd_cum, n_repl) in traj:
            rows.append({
                "e_cap_MWh":     r["e_cap"],
                "p_cap_MW":      r["p_cap"],
                "year":          year,
                "soh_pct":       soh_pct,
                "fd_cumulative": fd_cum,
                "n_repl_so_far": n_repl,
            })
    path = RESULTS_DIR / f"{FILE_TAG}_soh_trajectory_{run_ts}_P{int(p_cap_MW)}MW.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

def _print_sweep_report(results: List[Dict]) -> None:
    """Print sweep summary table and optimal points for both NPV bases."""
    p = results[0]["p_cap"]

    # ── Battery-only table ──
    print(f"\n{'═'*115}")
    print(f"SWEEP RESULTS: BATTERY-ONLY NPV vs E_cap  "
          f"(N_YEARS={N_YEARS}, P_cap={p:.0f} MW)")
    print(f"{'═'*115}")
    print(f"  {'E_cap':>6}  {'E/P':>5}  {'NPV_noDeg':>10}  {'NPV_20yr':>10}  "
          f"{'NPV_planB':>10}  {'Δ(planB)':>9}  {'fd_yr1':>8}  {'#repl':>5}  "
          f"{'SoH_end':>7}  {'RevFade':>8}  {'dNPV/dE':>12}")
    print(f"  {'MWh':>6}  {'h':>5}  {'MEUR':>10}  {'MEUR':>10}  "
          f"{'MEUR':>10}  {'MEUR':>9}  {'':>8}  {'':>5}  "
          f"{'%':>7}  {'%':>8}  {'EUR/MWh':>12}")
    print(f"  {'─'*6}  {'─'*5}  {'─'*10}  {'─'*10}  "
          f"{'─'*10}  {'─'*9}  {'─'*8}  {'─'*5}  "
          f"{'─'*7}  {'─'*8}  {'─'*12}")

    for r in results:
        delta = (r["npv_bat_planB"] - r["npv_bat_multiyear"]) * 1e-6
        rev_fade = (r["rev_bat_yrN_eur"] / r["rev_bat_yr1_eur"] - 1) * 100 \
                   if r["rev_bat_yr1_eur"] != 0 else 0.0
        grad_str = f"{r['dNPV_dEcap_yr1']:+.2e}" if r["dNPV_dEcap_yr1"] is not None else "N/A"
        print(f"  {r['e_cap']:>6.0f}  {r['e_cap']/r['p_cap']:>5.1f}  "
              f"{r['npv_bat_no_deg']*1e-6:>10.2f}  {r['npv_bat_multiyear']*1e-6:>10.2f}  "
              f"{r['npv_bat_planB']*1e-6:>10.2f}  {delta:>+9.2f}  "
              f"{r['fd_yr1']:>8.5f}  {r['n_replacements']:>5d}  "
              f"{r['final_soh_pct']:>7.2f}  {rev_fade:>+8.1f}  "
              f"{grad_str:>12}")

    # ── Total plant table (compact) ──
    print(f"\n{'═'*80}")
    print(f"TOTAL PLANT NPV (SHIPP-compatible, gradient-compatible)")
    print(f"{'═'*80}")
    print(f"  {'E_cap':>6}  {'NPV_noDeg':>10}  {'NPV_20yr':>10}  "
          f"{'Rev_yr1':>10}  {'Rev_yrN':>10}  {'RevFade':>8}")
    print(f"  {'MWh':>6}  {'MEUR':>10}  {'MEUR':>10}  "
          f"{'kEUR':>10}  {'kEUR':>10}  {'%':>8}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")

    for r in results:
        rev_fade_t = (r["rev_total_yrN_eur"] / r["rev_total_yr1_eur"] - 1) * 100 \
                     if r["rev_total_yr1_eur"] != 0 else 0.0
        print(f"  {r['e_cap']:>6.0f}  "
              f"{r['npv_total_no_deg']*1e-6:>10.2f}  {r['npv_total_multiyear']*1e-6:>10.2f}  "
              f"{r['rev_total_yr1_eur']*1e-3:>10.0f}  {r['rev_total_yrN_eur']*1e-3:>10.0f}  "
              f"{rev_fade_t:>+8.1f}")

    # ── Optimal points (both bases) ──
    best_bat_nd = max(results, key=lambda r: r["npv_bat_no_deg"])
    best_bat_my = max(results, key=lambda r: r["npv_bat_multiyear"])
    best_bat_pb = max(results, key=lambda r: r["npv_bat_planB"])
    best_arb_nd = max(results, key=lambda r: r["npv_arb_no_deg"])
    best_arb_my = max(results, key=lambda r: r["npv_arb_multiyear"])
    best_arb_pb = max(results, key=lambda r: r["npv_arb_planB"])
    best_tot_nd = max(results, key=lambda r: r["npv_total_no_deg"])
    best_tot_my = max(results, key=lambda r: r["npv_total_multiyear"])

    print(f"\n  OPTIMAL DESIGN POINTS:")
    _headline_key = {"arbitrage": "npv_arb_multiyear",
                     "marginal":  "npv_bat_multiyear"}[HEADLINE_BASIS]
    _best_headline = max(results, key=lambda r: r[_headline_key])
    print(f"  >>> HEADLINE basis = '{HEADLINE_BASIS}'  →  optimal "
          f"E={_best_headline['e_cap']:>5.0f} MWh  "
          f"NPV={_best_headline[_headline_key]*1e-6:>8.2f} MEUR (multi-year deg)")
    print(f"  Arbitrage basis (battery-as-asset; SHIPP a_npv):")
    print(f"    No degradation : E={best_arb_nd['e_cap']:>5.0f} MWh  "
          f"NPV={best_arb_nd['npv_arb_no_deg']*1e-6:>8.2f} MEUR")
    print(f"    Multi-year deg : E={best_arb_my['e_cap']:>5.0f} MWh  "
          f"NPV={best_arb_my['npv_arb_multiyear']*1e-6:>8.2f} MEUR")
    print(f"    Plan B (1-yr)  : E={best_arb_pb['e_cap']:>5.0f} MWh  "
          f"NPV={best_arb_pb['npv_arb_planB']*1e-6:>8.2f} MEUR")
    print(f"  Battery-only (marginal) basis:")
    print(f"    No degradation : E={best_bat_nd['e_cap']:>5.0f} MWh  "
          f"NPV={best_bat_nd['npv_bat_no_deg']*1e-6:>8.2f} MEUR")
    print(f"    Multi-year deg : E={best_bat_my['e_cap']:>5.0f} MWh  "
          f"NPV={best_bat_my['npv_bat_multiyear']*1e-6:>8.2f} MEUR")
    print(f"    Plan B (1-yr)  : E={best_bat_pb['e_cap']:>5.0f} MWh  "
          f"NPV={best_bat_pb['npv_bat_planB']*1e-6:>8.2f} MEUR")
    shift_bat = best_bat_nd["e_cap"] - best_bat_my["e_cap"]
    print(f"    → Degradation shift: {shift_bat:+.0f} MWh "
          f"({shift_bat/best_bat_nd['e_cap']*100:+.0f}%)")
    print(f"  Total plant basis (SHIPP-compatible):")
    print(f"    No degradation : E={best_tot_nd['e_cap']:>5.0f} MWh  "
          f"NPV={best_tot_nd['npv_total_no_deg']*1e-6:>8.2f} MEUR")
    print(f"    Multi-year deg : E={best_tot_my['e_cap']:>5.0f} MWh  "
          f"NPV={best_tot_my['npv_total_multiyear']*1e-6:>8.2f} MEUR")
    shift_tot = best_tot_nd["e_cap"] - best_tot_my["e_cap"]
    print(f"    → Degradation shift: {shift_tot:+.0f} MWh "
          f"({shift_tot/best_tot_nd['e_cap']*100:+.0f}%)")


# =============================================================================
# Replot from saved CSV
# =============================================================================

def _find_latest_csv() -> Optional[Path]:
    """Find the most recent v53_sweep CSV in RESULTS_DIR."""
    csvs = sorted(RESULTS_DIR.glob(f"{FILE_TAG}_sweep_*.csv"), key=lambda p: p.stat().st_mtime)
    return csvs[-1] if csvs else None


def _load_sweep_from_csv(csv_path: Path) -> List[Dict]:
    """Load sweep results from a saved CSV for replotting."""
    df = pd.read_csv(csv_path)
    results = []
    for _, row in df.iterrows():
        results.append({
            "e_cap":               row["e_cap_MWh"],
            "p_cap":               row["p_cap_MW"],
            "npv_bat_no_deg":      row["npv_bat_noDeg_MEUR"] * 1e6,
            "npv_bat_multiyear":   row["npv_bat_multiyear_MEUR"] * 1e6,
            "npv_bat_planB":       row["npv_bat_planB_MEUR"] * 1e6,
            "rev_bat_yr1_eur":     row["rev_bat_yr1_kEUR"] * 1e3,
            "rev_bat_yrN_eur":     row["rev_bat_yrN_kEUR"] * 1e3,
            "pv_rev_bat":          row["pv_rev_bat_MEUR"] * 1e6,
            "npv_arb_no_deg":      row["npv_arb_noDeg_MEUR"] * 1e6,
            "npv_arb_multiyear":   row["npv_arb_multiyear_MEUR"] * 1e6,
            "npv_arb_planB":       row["npv_arb_planB_MEUR"] * 1e6,
            "rev_arb_yr1_eur":     row["rev_arb_yr1_kEUR"] * 1e3,
            "rev_arb_yrN_eur":     row["rev_arb_yrN_kEUR"] * 1e3,
            "pv_rev_arb":          row["pv_rev_arb_MEUR"] * 1e6,
            "npv_total_no_deg":    row["npv_total_noDeg_MEUR"] * 1e6,
            "npv_total_multiyear": row["npv_total_multiyear_MEUR"] * 1e6,
            "rev_total_yr1_eur":   0.0,
            "rev_total_yrN_eur":   0.0,
            "pv_rev_total":        row["pv_rev_total_MEUR"] * 1e6,
            "fd_yr1":              row["fd_yr1"],
            "n_replacements":      int(row["n_replacements"]),
            "replacement_yrs":     [],
            "final_soh_pct":       row["final_soh_pct"],
            "capex_eur":           row["capex_MEUR"] * 1e6,
            "pv_replacements":     row["pv_repl_MEUR"] * 1e6,
            "dNPV_dEcap_yr1":      row.get("dNPV_dEcap_yr1", None),
            "elapsed_s":           row.get("elapsed_s", 0.0),
        })
    print(f"  Loaded {len(results)} sweep points from {csv_path.name}")
    return results


# =============================================================================
# Plots
# =============================================================================


def _plot_sweep(results: List[Dict]) -> None:
    """Main thesis figure: Battery NPV(E_cap) with and without degradation."""

    E   = np.array([r["e_cap"]               for r in results])
    nd  = np.array([r["npv_bat_no_deg"]      for r in results]) * 1e-6
    my  = np.array([r["npv_bat_multiyear"]   for r in results]) * 1e-6
    pb  = np.array([r["npv_bat_planB"]       for r in results]) * 1e-6
    fd  = np.array([r["fd_yr1"]              for r in results])
    rev1 = np.array([r["rev_bat_yr1_eur"]   for r in results]) * 1e-6
    revN = np.array([r["rev_bat_yrN_eur"]   for r in results]) * 1e-6
    capx = np.array([r["capex_eur"]          for r in results]) * 1e-6
    pvr  = np.array([r["pv_rev_bat"]         for r in results]) * 1e-6
    pvpl = np.array([r["pv_replacements"]    for r in results]) * 1e-6
    p_cap = results[0]["p_cap"]

    # Matplotlib style
    BG     = "#f7f9fc"
    C_ND   = "#2166ac"
    C_MY   = "#b5351b"
    C_PB   = "#4daf4a"
    C_FILL = "#d73027"

    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    fig.suptitle(
        f"v5.3 Sweep: Battery NPV vs Capacity  "
        f"({N_YEARS}-yr, P={p_cap:.0f} MW, DK1 2022)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Panel 1: NPV curves ──────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1, facecolor=BG)
    ax1.plot(E, nd, "o-", color=C_ND, lw=2.5, ms=6, label="No degradation", zorder=3)
    ax1.plot(E, my, "s-", color=C_MY, lw=2.5, ms=6,
             label=f"Shi+Xu_cal multi-year ({N_YEARS} yr)", zorder=3)
    ax1.plot(E, pb, "^--", color=C_PB, lw=1.5, ms=5, alpha=0.8,
             label="Plan B (1-yr extrap.)", zorder=2)

    # Shade degradation cost
    ax1.fill_between(E, my, nd, alpha=0.12, color=C_FILL, label="Degradation cost")

    # Mark optima with legend entries
    i_nd = np.argmax(nd)
    i_my = np.argmax(my)
    ax1.axvline(E[i_nd], color=C_ND, ls=":", alpha=0.5, lw=1.2)
    ax1.axvline(E[i_my], color=C_MY, ls=":", alpha=0.5, lw=1.2)
    ax1.plot(E[i_nd], nd[i_nd], "*", color=C_ND, ms=16, zorder=5,
             label=f"No-deg optimum ({E[i_nd]:.0f} MWh)")
    ax1.plot(E[i_my], my[i_my], "*", color=C_MY, ms=16, zorder=5,
             label=f"Deg-aware optimum ({E[i_my]:.0f} MWh)")

    ax1.set_xlabel("Battery energy capacity E [MWh]", fontsize=11)
    ax1.set_ylabel("Battery NPV [MEUR]", fontsize=11)
    ax1.set_title("Battery NPV vs size", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right")

    # ── Panel 2: Revenue fade (no purple % line) ─────────────────────────
    ax2 = fig.add_subplot(2, 2, 2, facecolor=BG)
    ax2.plot(E, rev1, "o-", color=C_ND, lw=2, ms=5, label=f"Year 1 battery rev.")
    ax2.plot(E, revN, "s-", color=C_MY, lw=2, ms=5, label=f"Year {N_YEARS} battery rev.")
    ax2.fill_between(E, revN, rev1, alpha=0.12, color=C_FILL, label="Revenue lost to fade")

    ax2.set_xlabel("Battery energy capacity E [MWh]", fontsize=11)
    ax2.set_ylabel("Annual battery revenue [MEUR]", fontsize=11)
    ax2.set_title("Battery revenue fade over project life", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)

    # ── Panel 3: Degradation rate vs E_cap (fd only, no replacement bars) ─
    ax3 = fig.add_subplot(2, 2, 3, facecolor=BG)
    ax3.plot(E, fd, "s-", color=C_MY, lw=2, ms=5)
    ax3.set_xlabel("Battery energy capacity E [MWh]", fontsize=11)
    ax3.set_ylabel("Annual fractional degradation fd [-]", fontsize=11)
    ax3.set_title("Degradation rate vs battery size", fontsize=11, fontweight="bold")

    # ── Panel 4: Cost decomposition ──────────────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4, facecolor=BG)
    ax4.plot(E, pvr,  "o-", color=C_ND, lw=2, ms=5, label="PV(bat. revenues)")
    ax4.plot(E, capx, "x-", color="gray", lw=2, ms=5, label="Battery capex")
    ax4.plot(E, pvpl, "^-", color="#ff7f00", lw=2, ms=5, label="PV(replacements)")
    ax4.plot(E, my,   "s-", color=C_MY, lw=2, ms=5, label="Battery NPV (net)")

    ax4.set_xlabel("Battery energy capacity E [MWh]", fontsize=11)
    ax4.set_ylabel("[MEUR]", fontsize=11)
    ax4.set_title("Cost decomposition", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = PLOTS_DIR / f"{FILE_TAG}_sweep_{run_ts}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"  ✓ Plot: {out.name}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def _plot_planB_comparison(results: List[Dict]) -> None:
    """Direct comparison: multi-year NPV vs Plan B NPV."""
    E   = np.array([r["e_cap"]             for r in results])
    my  = np.array([r["npv_bat_multiyear"] for r in results]) * 1e-6
    pb  = np.array([r["npv_bat_planB"]     for r in results]) * 1e-6
    nd  = np.array([r["npv_bat_no_deg"]    for r in results]) * 1e-6
    p_cap = results[0]["p_cap"]

    BG = "#f7f9fc"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG)
    fig.suptitle(
        f"Multi-year vs Plan B Battery NPV  ({N_YEARS}-yr, P={p_cap:.0f} MW)",
        fontsize=13, fontweight="bold",
    )

    # Left: absolute NPV
    ax1.set_facecolor(BG)
    ax1.plot(E, nd, "o-", color="#2166ac", lw=2, ms=5, label="No degradation")
    ax1.plot(E, my, "s-", color="#b5351b", lw=2.5, ms=6, label="Multi-year (corrected)")
    ax1.plot(E, pb, "^--", color="#4daf4a", lw=1.5, ms=5, alpha=0.8, label="Plan B (1-yr extrap.)")

    i_my = np.argmax(my)
    i_pb = np.argmax(pb)
    ax1.plot(E[i_my], my[i_my], "*", color="#b5351b", ms=16, zorder=5,
             label=f"Multi-year optimum ({E[i_my]:.0f} MWh)")
    ax1.plot(E[i_pb], pb[i_pb], "*", color="#4daf4a", ms=16, zorder=5,
             label=f"Plan B optimum ({E[i_pb]:.0f} MWh)")

    ax1.set_xlabel("Battery energy capacity E [MWh]", fontsize=11)
    ax1.set_ylabel("Battery NPV [MEUR]", fontsize=11)
    ax1.set_title("Absolute battery NPV comparison", fontweight="bold")
    ax1.legend(fontsize=8)

    # Right: overestimation by Plan B
    overest = pb - my
    ax2.set_facecolor(BG)
    bar_w = np.min(np.diff(E)) * 0.7 if len(E) > 1 else 50
    ax2.bar(E, overest, width=bar_w, color="#d73027", alpha=0.7)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("Battery energy capacity E [MWh]", fontsize=11)
    ax2.set_ylabel("Plan B overestimation [MEUR]", fontsize=11)
    ax2.set_title("Plan B bias (positive = optimistic)", fontweight="bold")

    plt.tight_layout()
    out = PLOTS_DIR / f"{FILE_TAG}_planB_comparison_{run_ts}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"  ✓ Plot: {out.name}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    global run_ts

    print("=" * 80)
    print("WP2 BATTERY OPTIMIZATION + DEGRADATION  v5.3 (wp2_econ integration)")
    print("Shi (2018) + Xu calendar | multi-year NPV | E_cap parameter sweep")
    print("=" * 80)

    # ── Check for replot mode (skip entire LP pipeline) ──────────────────
    csv_to_load = None
    if REPLOT_CSV is not None:
        csv_to_load = Path(REPLOT_CSV)
        if not csv_to_load.exists():
            raise FileNotFoundError(f"REPLOT_CSV not found: {csv_to_load}")
    elif REPLOT_FROM_LAST:
        csv_to_load = _find_latest_csv()
        if csv_to_load is None:
            raise FileNotFoundError(
                "REPLOT_FROM_LAST=True but no v52_sweep_*.csv found in "
                f"{RESULTS_DIR}. Run the full sweep first."
            )

    if csv_to_load is not None:
        print(f"\n[REPLOT MODE] Loading from CSV — LP sweep will be skipped.")
        print(f"  Source: {csv_to_load.name}")
        results = _load_sweep_from_csv(csv_to_load)
        # Use timestamp from the CSV filename for plot filenames
        stem = csv_to_load.stem  # e.g. v52_sweep_20260526_131059_P275MW
        parts = stem.split("_")
        run_ts = f"{parts[2]}_{parts[3]}_replot"
        _print_sweep_report(results)
        _plot_sweep(results)
        _plot_planB_comparison(results)
        print(f"\n{'═'*80}")
        print("✓ v5.3 REPLOT COMPLETE")
        print(f"{'═'*80}")
        return

    # ── 1. Load config + raw series ──
    print("\n[1/4] Loading WP2 configuration...")
    setup = quick_setup(HPP_YAML, config={"interp_n": 2000}, verbose=False)
    hpp   = setup["hpp"]

    ws_all, wd_all, ti_all = _load_inputs(hpp)
    price_all               = _load_prices()

    # ── 2. Choose horizon ──
    print("\n[2/4] Preparing data...")
    n = _choose_horizon(len(ws_all), len(price_all))
    ws        = ws_all[:n]
    wd        = wd_all[:n]
    ti        = ti_all[:n] if ti_all is not None else None
    price_eur = price_all[:n]
    print(f"  Horizon: {n:,} h ({n/24:.1f} days)")
    print(f"  Mean price: {float(np.mean(price_eur)):.2f} EUR/MWh")

    # ── 3. PyWake ──
    print("\n[3/4] Running PyWake simulation...")
    power_wind_MW = _run_pywake_power_MW(setup, wd, ws, ti)
    print(f"  Wind mean: {float(np.mean(power_wind_MW)):.1f} MW | "
          f"peak: {float(np.max(power_wind_MW)):.1f} MW")

    # ── 4. Battery params ──
    bat     = setup["battery"]
    p_max_MW = float(hpp["grid_connection_capacity"]) / 1e6

    e_cap_yaml = float(bat["energy_capacity_Wh"]) / 1e6
    p_cap_yaml = float(bat["power_capacity_W"])   / 1e6

    rte_dc  = float(bat["rte_nominal"])
    pcu_eff = float(bat["pcu_efficiency"])
    rte_ac  = rte_dc * (pcu_eff ** 2)

    e_cost_EUR_per_MWh = float(bat["capex_EUR_per_kWh"]) * 1000.0
    p_cost_EUR_per_MW  = float(bat["capex_EUR_per_kW"])  * 1000.0
    repl_e_EUR_per_MWh = float(bat["repl_energy_EUR_per_kWh"]) * 1000.0 # energy expansion: replacement + deg valuation
    repl_p_EUR_per_MW  = float(bat["repl_power_EUR_per_kW"])   * 1000.0 # power expansion: replacement only

    soc_min = float(bat.get("soc_min", 0.10))
    soc_max = float(bat.get("soc_max", 0.90))

    shi_fit = fit_shi_polynomial(soc_min, soc_max, verbose=True)

    bat_params = setup["battery"]

    p_cap_sweep = P_CAP_SWEEP if P_CAP_SWEEP is not None else p_cap_yaml

    print(f"  Battery (YAML): {p_cap_yaml:.0f} MW / {e_cap_yaml:.0f} MWh")
    print(f"  Sweep P_cap: {p_cap_sweep:.0f} MW")
    print(f"  Grid: {p_max_MW:.0f} MW | RTE(ac): {rte_ac*100:.1f}%")
    print(f"  SoC: {soc_min*100:.0f}%–{soc_max*100:.0f}%")
    print(f"  e_cost: {e_cost_EUR_per_MWh:.0f} EUR/MWh = "
          f"{e_cost_EUR_per_MWh/1000:.0f} EUR/kWh")
    print(f"  N_YEARS: {N_YEARS} | EOL: {EOL_REPLACEMENT*100:.0f}%")

    if not SWEEP_MODE:
        # ── Single-point mode (legacy) ──
        print("\n  SWEEP_MODE=False → single-point at YAML capacity.")
        stor_null = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
                            e_cost=0, p_cost=0)
        my = _run_multiyear(
            wind_8760=power_wind_MW, price_8760=price_eur,
            stor_null=stor_null, p_max_MW=p_max_MW,
            e_cap_nominal=e_cap_yaml, p_cap_MW=p_cap_yaml,
            rte_ac=rte_ac, e_cost_EUR_per_MWh=e_cost_EUR_per_MWh,
            p_cost_EUR_per_MW=p_cost_EUR_per_MW,
            repl_e_EUR_per_MWh=repl_e_EUR_per_MWh,
            repl_p_EUR_per_MW=repl_p_EUR_per_MW,
            bat_params=bat_params, shi_fit=shi_fit,
            soc_min=soc_min, soc_max=soc_max,
        )
        print("\n✓ Single-point complete.")
        return

    # ── Sweep mode ──
    print(f"\n[4/4] Running E_cap sweep...")
    results = _run_ecap_sweep(
        power_wind_MW=power_wind_MW,
        price_eur=price_eur,
        p_max_MW=p_max_MW,
        p_cap_MW=p_cap_sweep,
        rte_ac=rte_ac,
        e_cost_EUR_per_MWh=e_cost_EUR_per_MWh,
        p_cost_EUR_per_MW=p_cost_EUR_per_MW,
        repl_e_EUR_per_MWh=repl_e_EUR_per_MWh,
        repl_p_EUR_per_MW=repl_p_EUR_per_MW,
        bat_params=bat_params,
        shi_fit=shi_fit,
        soc_min=soc_min,
        soc_max=soc_max,
    )

    # ── Results ──
    _print_sweep_report(results)
    _save_sweep_csv(results, p_cap_sweep)
    _save_soh_trajectory_csv(results, p_cap_sweep)   # long-format SoH-vs-year per E

    # ── Save config sidecar ──
    config_path = RESULTS_DIR / f"{FILE_TAG}_sweep_{run_ts}_P{int(p_cap_sweep)}MW.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "version":       "v5.3",
            "revenue_basis": f"headline={HEADLINE_BASIS}; arbitrage + marginal + total all reported",
            "E_CAP_GRID":    E_CAP_GRID,
            "P_CAP_SWEEP":   p_cap_sweep,
            "N_YEARS":       N_YEARS,
            "discount_rate": discount_rate,
            "soc_min":       soc_min,
            "soc_max":       soc_max,
            "eta_symmetric": eta_symmetric(rte_ac),
            "round_trip_ac": rte_ac,            "e_cost_eur_kwh": e_cost_EUR_per_MWh / 1000.0,
            "k3":            shi_fit.k3,
            "k4":            shi_fit.k4,
            "RUN_HOURS":     n,
            "EOL_REPLACEMENT": EOL_REPLACEMENT,
        }, f, indent=2)
    print(f"  ✓ Config: {config_path.name}")

    # ── Plots ──
    _plot_sweep(results)
    _plot_planB_comparison(results)

    print(f"\n{'═'*80}")
    print("✓ v5.3 COMPLETE (Castillo + Sweep)")
    print(f"{'═'*80}")


if __name__ == "__main__":
    log_path = RESULTS_DIR / f"{FILE_TAG}_log_{run_ts}.txt"
    with TeeLogger(log_path):
        main()
    # Print outside tee so we know the file is written
    print(f"\n  ✓ Full log saved: {log_path.name}")