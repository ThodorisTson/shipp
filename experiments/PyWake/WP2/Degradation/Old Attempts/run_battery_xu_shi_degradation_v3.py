"""WP2 battery optimization + Shi degradation,  v2 (multi-year, calendar-corrected).

Changes over v1 (run_battery_xu_shi_degradation.py)
----------------------------------------------------
1. Calendar correction (Option 2)
   The Shi accumulation framework is cycle-only (fd_calendar = 0.0 by design).
   For the WP2 site, calendar aging accounts for ~49-51% of total annual fd
   under the Xu model.  In v2, ft_calendar() from degradation_xu is added to
   the Shi result in the REPORTING PATH ONLY.  The gradient computation
   (phi_shi_prime_with_stress) is never touched,  the dual-Phi convexity
   guarantee is fully preserved.

   Architectural note: dfd_calendar/dc_t = 0 for all t.  Calendar aging
   carries no dispatch-dependent information and cannot drive the outer loop.
   Separating it from fd_cycle makes this explicit rather than implicit.

2. Multi-year degradation loop (MULTI_YEAR = True)
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
  CONFIG                  ,  toggles and thresholds (top of file)
  _shi_with_calendar_correction() ,  single-year Shi + Xu calendar patching
  _build_storage_year()   ,  build Storage object for a given effective e_cap
  _run_multiyear()        ,  year-by-year loop: LP re-solve + degradation
  main()                  ,  orchestrates single-year + multi-year runs

No changes required to:
  degradation_xu.py       ,  ft_calendar / sei_capacity_loss already in API
  degradation_shi.py      ,  analyze_degradation_shi unchanged
  degradation_subgradient.py,  gradient path untouched
  degradation_plots.py    ,  works on the patched result dict
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

# Xu,  reporting baseline + calendar correction helpers
from degradation_xu import (
    analyze_degradation,
    count_equivalent_full_cycles,
    rainflow_cycle_counting,
    ft_calendar,          # ← calendar correction for Shi reporting path
    sei_capacity_loss,    # ← recompute SoH after calendar correction
)

# Shi,  cycle accumulation + gradient
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

# ── v2 additions ──────────────────────────────────────────────────────────────
MULTI_YEAR       = True     # False → single-year run identical to v1
N_YEARS          = 20       # project lifetime for multi-year loop
EOL_REPLACEMENT  = 0.70     # replace battery when SoH drops below this threshold
EOL_REPLACEMENT_TOL   = 0.005    # tolerance band: replace if SoH < threshold + tol (e.g. 0.005 → triggers at 70.5% instead of strictly 70%)

# ─────────────────────────────────────────────────────────────────────────────

# Output toggles
print_baseline_table = True
print_degr_reports   = True
SAVE_CSV             = True
SAVE_REPORT          = True
MAKE_PLOT            = True
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
# Helpers,  shared with v1
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
        print(f"  ⚠ DoD ({dod_yaml:.0%}) ignored,  scipy sparse requires dod=1.0.")
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
# v2 helpers,  calendar correction + multi-year loop
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

    Power capacity and costs are held at their nominal values,  only energy
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
    dt_hours,  exactly as analyze_degradation() does,  so partial-year series
    (e.g. 8752 h instead of 8760 h) are handled correctly without any
    hardcoding in the call site.

    The result dict is patched in place so all downstream functions
    (plot_degradation_analysis, print_degradation_report, CSV logging) work
    without changes,  they see a fully populated fd_calendar field.

    Args:
        storage_p:      Battery power [MW] from LP solve.
        storage_e:      Battery SoC [MWh] from LP solve.
        e_cap_eff:      Effective energy capacity this year [MWh].
        bat_params:     Battery params dict (must have 'power_capacity_W').
        shi_fit:        ShiPolynomialFit from fit_shi_polynomial().
        T_cell_C:       Cell temperature [°C].
        dt_hours:       Timestep duration [h],  typically 1.0.
        eol_thresholds: SoH fractions to report EoL for.

    Returns:
        Patched result dict with:
            fd_calendar ,  Xu calendar term (non-zero)
            fd          ,  fd_cycle + fd_calendar (corrected total)
            soh / eol_years / capacity_retention,  recomputed from corrected fd
    """
    # ── Compute time quantities internally,  same pattern as analyze_degradation
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

    # ── Step 2: Xu calendar term,  reporting path only ─────────────────────
    fd_cal       = ft_calendar(t_total_seconds, sigma_mean, T_cell_C)
    fd_corrected = degr["fd_shi"] + fd_cal
    cal_frac_pct = 100.0 * fd_cal / max(fd_corrected, 1e-30)

    # ── Step 3: Recompute SoH and EoL with corrected total fd ──────────────
    L_corr       = sei_capacity_loss(fd_corrected)
    cap_ret_corr = 1.0 - L_corr
    soh_corr     = cap_ret_corr * 100.0

    # Annualise fd before EoL bisection,  same formula as degradation_xu.py L884:
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


def _run_multiyear(
    wind_8760:          np.ndarray,
    price_8760:         np.ndarray,
    stor_null:          Storage,
    p_max_MW:           float,
    e_cap_nominal:      float,
    p_cap_MW:           float,
    rte_ac:             float,
    e_cost_USD_per_MWh: float,
    p_cost_USD_per_MW:  float,
    bat_params:         Dict,
    shi_fit,
    soc_min:            float,
    soc_max:            float,
    T_cell_C:           float = 25.0,
) -> Dict:
    """Year-by-year degradation loop with capacity fade feedback and replacement.

    Design decisions:
    - Representative year assumption: the same 8760-h wind and price dataset
      is reused every year.  This is the standard approach in energy storage
      planning studies and is consistent with HyDesign.
    - Capacity fade feedback: the LP is re-solved each year with
      E_eff = E_nominal * SoH(year-1).  Dispatch adapts to the reduced capacity.
    - Battery replacement: when SoH < EOL_REPLACEMENT, fd resets to 0.0 and
      SoH resets to 1.0 (fresh battery).  Replacement cost is tracked separately.
    - Calendar correction: applied each year via _shi_with_calendar_correction().

    Args:
        wind_8760:   8760-hour wind power series [MW].
        price_8760:  8760-hour price series [EUR/MWh].
        stor_null:   Empty storage for the null (wind-only) LP.
        p_max_MW:    Grid connection limit [MW].
        e_cap_nominal: Nameplate energy capacity [MWh].
        p_cap_MW:    Power capacity [MW].
        rte_ac:      AC round-trip efficiency.
        e_cost_USD_per_MWh: Energy capital cost [USD/MWh],  for replacement.
        bat_params:  Battery params dict.
        shi_fit:     ShiPolynomialFit for this SoC window.
        soc_min, soc_max: Operating limits from YAML.
        T_cell_C:    Cell temperature [°C].

    Returns:
        Dict with:
            soh_trajectory      list of (year, soh_pct, fd_cumulative, n_replacements)
            annual_fd           list of annual fd (corrected) per year
            n_replacements      int
            replacement_years   list of years where replacement occurred
            replacement_cost_USD float,  total replacement CAPEX
            final_soh           float,  SoH at end of project lifetime
    """
    n           = len(wind_8760)
    period_days = n * dt / 24.0
    if n != 8760:
        print(f"  Note: annual tile is {n} h ({period_days:.1f} days),  "
              f"fd annualised by ×{8760/n:.5f} each year.")

    fd_cumulative    = 0.0
    soh              = 1.0
    n_replacements   = 0
    replacement_years: List[int]   = []
    soh_trajectory:    List[tuple] = []
    annual_fd:         List[tuple] = []   # (fd_total_annualised, fd_cycle, fd_calendar)
    annual_gradient: List = []
    annual_soc:      List = []
    soc_frac_fixed: float = None  # None → year 1 free; set after yr1 to optimal SoC% 

    print(f"\n{'─'*70}")
    print(f"Multi-year loop: {N_YEARS} years | "
          f"tile: {n} h ({period_days:.1f} d) | "
          f"replacement threshold: SoH < {EOL_REPLACEMENT*100:.0f}%")
    print(f"{'─'*70}")
    print(f"  {'Year':>4}  {'SoH_%':>7}  {'fd_yr':>9}  {'fd_cum':>9}  "
          f"{'t_lp_s':>7}  {'t_deg_s':>7}  {'t_yr_s':>7}  {'Replaced':>8}")
    print(f"  {'────':>4}  {'──────':>7}  {'──────':>9}  {'──────':>9}  "
          f"{'───────':>7}  {'───────':>7}  {'───────':>7}  {'────────':>8}")

    t_loop_start = time.perf_counter()
    t_lp_total   = 0.0
    t_degr_total = 0.0

    for year in range(1, N_YEARS + 1):
        t_yr_start = time.perf_counter()

        # ── Effective capacity this year ──────────────────────────────────
        e_cap_eff = e_cap_nominal * soh
        e_start1_yr = soc_frac_fixed * e_cap_eff if soc_frac_fixed is not None else None


        # ── Re-solve LP with degraded capacity (fixed dispatch, no sizing) ─
        stor_yr  = _build_storage_year(
            e_cap_eff, p_cap_MW, rte_ac, e_cost_USD_per_MWh, soc_min, soc_max
        )
        price_dam = TimeSeries((price_8760 * eur_to_usd).tolist(), dt)
        prod_yr   = Production(TimeSeries(wind_8760.tolist(), dt), p_cost=0.0)
        prod_null = Production(TimeSeries([0.0] * n, dt), p_cost=0.0)

        t_lp_start = time.perf_counter()

        if pyo_solver == "none":
            os_yr = solve_lp_sparse(
                price_dam, prod_yr, prod_null, stor_yr, stor_null,
                discount_rate, N_YEARS, p_min, p_max_MW, n, fixed_cap=True,
            )
        else:
            os_yr = solve_lp_pyomo(
                price_dam, prod_yr, prod_null, stor_yr, stor_null,
                discount_rate, N_YEARS, p_min, p_max_MW, n,
                pyo_solver, fixed_cap=True, soc_max1=soc_max,
                return_duals=True, e_start1=e_start1_yr,
            )

        t_lp_yr = time.perf_counter() - t_lp_start
        t_lp_total += t_lp_yr

        # ── Annual degradation with calendar correction ───────────────────
        storage_e_yr = os_yr.storage_e[0].data
        if soc_frac_fixed is None:
            soc_frac_fixed = os_yr.soc_final / e_cap_eff   # lock SoC% after year 1
        annual_soc.append(np.array(storage_e_yr, dtype=float).copy())
        storage_p_yr = os_yr.storage_p[0].data

        t_deg_start = time.perf_counter()

        degr_yr = _shi_with_calendar_correction(
            storage_p_yr, storage_e_yr, e_cap_eff,
            bat_params, shi_fit, T_cell_C,
            dt, eol_thresholds,
        )

        t_deg_yr = time.perf_counter() - t_deg_start
        t_degr_total += t_deg_yr

        # Annualise fd before accumulating,  same formula as degradation_xu L884.
        # For a full 8760-h tile this is identity; for partial years it scales up.
        scale          = 1.0 / max(n / 8760.0, 1e-9)
        fd_yr          = degr_yr["fd"]          * scale
        fd_yr_cycle    = degr_yr["fd_cycle"]    * scale
        fd_yr_calendar = degr_yr["fd_calendar"] * scale
        fd_cumulative += fd_yr
        annual_fd.append((fd_yr, fd_yr_cycle, fd_yr_calendar))

        # ── Per-year inner-loop gradient ──────────────────────────────────
        grad_yr = None
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
            factor_yr        = npf.npv(discount_rate, np.ones(N_YEARS)) - 1
            dual_per_year_yr = dual_yr / factor_yr
            dDeg_dDoD_yr     = -e_cap_yr * float(
                np.dot(sg_yr["subgrad_combined"], dual_per_year_yr)
            )
            dods_yr     = np.array([c["dod"]   for c in cycles_yr])
            cnts_yr     = np.array([c["count"] for c in cycles_yr])
            mean_dod_yr = float(np.average(dods_yr, weights=cnts_yr)) if len(dods_yr) > 0 else 0.0
            grad_yr = (
                dDeg_dDoD_yr,                                        # [0] dDeg/dDoD
                float(np.mean(np.abs(sg_yr["subgrad_combined"]))),   # [1] mean_abs_subgrad
                mean_dod_yr,                                         # [2] mean_dod (weighted)
                float(np.mean(np.abs(dual_per_year_yr))),            # [3] mean_abs_dual
                float(e_cap_yr),                                     # [4] e_cap_eff
                float(sg_yr["cycle_coverage"]),                      # [5] cycle_coverage
                sg_yr["subgrad_combined"].copy(),                    # [6] full time series
            )
        annual_gradient.append(grad_yr)   # None if duals unavailable

        # ── Update SoH via SEI curve on cumulative fd ─────────────────────
        L_cum = sei_capacity_loss(fd_cumulative)
        soh   = 1.0 - L_cum

        soh_trajectory.append((year, soh * 100.0, fd_cumulative, n_replacements))
        
        # ── Effective replacement trigger (threshold + tolerance band) ────
        eol_trigger = EOL_REPLACEMENT + EOL_REPLACEMENT_TOL

        t_yr = time.perf_counter() - t_yr_start
        replaced_tag = "YES" if soh < eol_trigger else ""
        print(f"  {year:>4d}  {soh*100:>7.3f}  {fd_yr:>9.5f}  "
              f"{fd_cumulative:>9.5f}  "
              f"{t_lp_yr:>7.2f}  {t_deg_yr:>7.3f}  {t_yr:>7.2f}  {replaced_tag:>8}")

        # ── Battery replacement check ─────────────────────────────────────
        if soh < eol_trigger:
            n_replacements += 1
            replacement_years.append(year)
            fd_cumulative = 0.0     # reset accumulator,  fresh battery
            soh           = 1.0
            print(f"  *** Battery #{n_replacements} replaced at end of year {year} "
              f"(SoH fell below {eol_trigger*100:.1f}%  "
              f"[{EOL_REPLACEMENT*100:.0f}% + {EOL_REPLACEMENT_TOL*100:.1f}% tol]) ***")
            soc_frac_fixed = None   # new battery: year 1 picks SoC% freely again

    t_loop_total = time.perf_counter() - t_loop_start
    pct_lp   = 100.0 * t_lp_total   / max(t_loop_total, 1e-9)
    pct_degr = 100.0 * t_degr_total / max(t_loop_total, 1e-9)

    print(f"{'─'*70}")
    print(f"  Final SoH: {soh*100:.2f}%  |  Replacements: {n_replacements}  |  "
          f"Replacement years: {replacement_years}")
    print(f"\n  ── Timing summary ────────────────────────────────────────────")
    print(f"  Total loop          : {t_loop_total:>8.1f} s  ({t_loop_total/60:.1f} min)")
    print(f"  LP solves (total)   : {t_lp_total:>8.1f} s  ({pct_lp:.0f}% of loop)")
    print(f"  Degradation (total) : {t_degr_total:>8.1f} s  ({pct_degr:.0f}% of loop)")
    print(f"  Other overhead      : {t_loop_total - t_lp_total - t_degr_total:>8.1f} s")
    print(f"  Per-year average    : {t_loop_total/N_YEARS:>8.1f} s/yr")
    print(f"  ─────────────────────────────────────────────────────────────")

    return {
        "soh_trajectory":    soh_trajectory,
        "annual_fd":         annual_fd,
        "n_replacements":    n_replacements,
        "replacement_years": replacement_years,
        "replacement_cost_USD": n_replacements * (e_cap_nominal * e_cost_USD_per_MWh + p_cap_MW * p_cost_USD_per_MW),
        "final_soh":         soh,
        "final_soh_pct":     soh * 100.0,
        "e_cap_nominal":     e_cap_nominal,
        "annual_gradient":   annual_gradient,
        "annual_soc":        annual_soc,
    }


# =============================================================================
# Plotting helpers,  multi-year trajectory
# =============================================================================
def _split_battery_segments(years_full, soh_full, replacement_years):
    """Split SoH trajectory into one segment per battery generation.
    Fully data-driven,  works for 0, 1, or N replacements at any year."""
    repl_set = set(replacement_years)
    segments = []
    cur_x, cur_y = [], []

    for yr, soh in zip(years_full, soh_full):
        cur_x.append(yr)
        cur_y.append(soh)

        if yr in repl_set:
            segments.append((list(cur_x), list(cur_y)))
            cur_x = [yr]      # same x-anchor at replacement year
            cur_y = [100.0]   # fresh battery starts at 100%

    if cur_x:
        segments.append((cur_x, cur_y))

    return segments

def _plot_gradient_analysis(
    multiyear: Dict,
    run_label: str,
) -> None:
    """Standalone 2×2 inner-loop gradient analysis figure.

    Top-left    dDeg/dDoD trajectory over project years
    Top-right   dDeg/dDoD vs SoH scatter with year labels
    Bottom-left Subgradient spread (subgrad_max - subgrad_min) per year
    Bottom-right cycle_coverage per year
    """
    ann_grad = multiyear.get("annual_gradient", [])
    traj     = multiyear["soh_trajectory"]

    years    = [r[0] for r in traj]
    soh_pct  = [r[1] for r in traj]

    # Unpack gradient scalars,  None-safe
    dDeg_dDoD        = [g[0] if g else None for g in ann_grad]
    mean_abs_subgrad = [g[1] if g else None for g in ann_grad]
    mean_dod         = [g[2] if g else None for g in ann_grad]
    mean_abs_dual    = [g[3] if g else None for g in ann_grad]
    e_cap_eff_yr     = [g[4] if g else None for g in ann_grad]
    cycle_cov        = [g[5] if g else None for g in ann_grad]

    # Alignment factor,  the core finding
    alignment = []
    for dd, mas, mad, ec in zip(dDeg_dDoD, mean_abs_subgrad, mean_abs_dual, e_cap_eff_yr):
        if dd is not None and mas is not None and mad is not None and ec is not None:
            denom = ec * mas * mad
            alignment.append(dd / denom if abs(denom) > 1e-9 else None)
        else:
            alignment.append(None)

    # Regime classification,  threshold from Pearson analysis
    ALIGN_THRESHOLD = 3000.0
    regime_colors = []
    for yr, al in zip(years, alignment):
        if al is None:
            regime_colors.append("#4C72B0")
        elif al >= ALIGN_THRESHOLD:
            regime_colors.append("#e74c3c")   # high-alignment
        else:
            regime_colors.append("#4C72B0")   # low-alignment

    # Filter out None years for scatter
    valid = [(yr, soh, dd) for yr, soh, dd in zip(years, soh_pct, dDeg_dDoD)
             if dd is not None]
    v_years, v_soh, v_dDeg = zip(*valid) if valid else ([], [], [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Inner-Loop Gradient Analysis  |  {N_YEARS} yr", fontsize=11, y=1.01)

    # ── Top-left: dDeg/dDoD trajectory ───────────────────────────────────
    ax = axes[0, 0]
    valid_years  = [yr for yr, dd in zip(years, dDeg_dDoD) if dd is not None]
    valid_dDeg   = [dd for dd in dDeg_dDoD if dd is not None]
    valid_rcols  = [c  for c, dd in zip(regime_colors, dDeg_dDoD) if dd is not None]
    bar_colors   = [
        "#e74c3c" if yr in multiyear["replacement_years"] else c
        for yr, c in zip(valid_years, valid_rcols)
    ]
    ax.bar(valid_years, valid_dDeg, width=0.7, color=bar_colors, alpha=0.82)
    for yr in multiyear["replacement_years"]:
        ax.axvline(yr + 0.5, linestyle="-.", linewidth=1.4, color="red", alpha=0.8)
    if valid_dDeg:
        mean_g = np.mean(valid_dDeg)
        ax.axhline(mean_g, linestyle="--", linewidth=1.2, color="#2c3e50",
                   alpha=0.8, label=f"Mean: {mean_g:.3e}")
    ax.set_xlabel("Project year")
    ax.set_ylabel("dDeg/dDoD  [USD / MWh]")
    ax.set_title("dDeg/dDoD per Year") # Note: red = high-alignment regime, blue = low-alignment regime
    ax.set_xlim(0.5, N_YEARS + 0.5)
    ax.set_xticks(range(2, N_YEARS + 1, 2))

    ax.legend(handles=[
        plt.Line2D([0],[0], color="none"),   # spacer
        Patch(facecolor="#e74c3c", alpha=0.82, label="High-alignment"),
        Patch(facecolor="#4C72B0", alpha=0.82, label="Low-alignment"),
        plt.Line2D([0],[0], linestyle="--", color="#2c3e50", label=f"Mean: {mean_g:.3e}" if valid_dDeg else "Mean: N/A"),
    ], fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")

    # ── Top-right: alignment vs dDeg/dDoD scatter (the money plot) ───────
    ax = axes[0, 1]
    valid_align = [(yr, al, dd, c) for yr, al, dd, c
                   in zip(years, alignment, dDeg_dDoD, regime_colors)
                   if al is not None and dd is not None]
    va_yrs, va_al, va_dd, va_cols = zip(*valid_align) if valid_align else ([], [], [], [])
    ax.scatter(va_al, va_dd, color=va_cols, s=65, zorder=3,
               edgecolors="white", linewidths=0.5, alpha=0.88)
    for yr, al, dd in zip(va_yrs, va_al, va_dd):
        ax.annotate(str(yr), (al, dd), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points", alpha=0.85)
    if len(va_al) >= 2:
        m, b = np.polyfit(va_al, va_dd, 1)
        xs = np.linspace(min(va_al), max(va_al), 60)
        r  = np.corrcoef(va_al, va_dd)[0, 1]
        ax.plot(xs, m * np.array(xs) + b, "--", color="#2c3e50",
                linewidth=1.3, alpha=0.7, label=f"r = {r:+.3f}")
    ax.axvline(ALIGN_THRESHOLD, linestyle=":", linewidth=1.2,
               color="grey", alpha=0.6, label=f"Regime boundary = {ALIGN_THRESHOLD:.0f}")
    ax.set_xlabel("Alignment factor  =  dDeg/dDoD / (e_cap × |subgrad| × |dual|)")
    ax.set_ylabel("dDeg/dDoD  [USD / MWh]")
    ax.set_title(f"Alignment Factor vs dDeg/dDoD  (r = {r:+.3f})") # alignment = dDeg_dDoD / (e_cap × |subgrad| × |dual|) explains the oscillation
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # ── Bottom-left: subgradient spread ───────────────────────────────────
    ax = axes[1, 0]
    valid_mas_yrs = [yr for yr, m in zip(years, mean_abs_subgrad) if m is not None]
    valid_mas     = [m for m in mean_abs_subgrad if m is not None]
    ax.bar(valid_mas_yrs, valid_mas, width=0.7, color="#8e44ad", alpha=0.80)
    for yr in multiyear["replacement_years"]:
        ax.axvline(yr + 0.5, linestyle="-.", linewidth=1.4, color="red", alpha=0.8)
    ax.set_xlabel("Project year")
    ax.set_ylabel("mean |subgrad_combined|  [USD / MWh]")
    ax.set_xlim(0.5, N_YEARS + 0.5)
    ax.set_xticks(range(2, N_YEARS + 1, 2))
    ax.grid(True, alpha=0.25, axis="y")

    valid_dod_yrs = [yr for yr, d in zip(years, mean_dod) if d is not None]
    valid_dod     = [d * 100 for d in mean_dod if d is not None]
    if valid_dod:
        ax_dod = ax.twinx()
        ax_dod.plot(valid_dod_yrs, valid_dod, "s--", color="#e67e22",
                    linewidth=1.3, markersize=4, alpha=0.85, label="mean DoD [%]")
        ax_dod.set_ylabel("mean DoD [%]", color="#e67e22", fontsize=9)
        ax_dod.tick_params(axis="y", colors="#e67e22", labelsize=8)
        ax_dod.set_ylim(0, max(valid_dod) * 2.2)
        ax_dod.legend(fontsize=8, loc="lower right")
    ax.set_title("Mean |Subgradient| and Mean DoD per Year") # neither scalar tracks the gradient oscillation — alignment (dot product structure) does

    # ── Bottom-right: scalar product vs gradient,  normalised ─────────────
    ax = axes[1, 1]

    valid_comp_yrs = [yr for yr, ec in zip(years, e_cap_eff_yr) if ec is not None]
    comp_ecap = [ec for ec in e_cap_eff_yr    if ec is not None]
    comp_mas  = [m  for m  in mean_abs_subgrad if m  is not None]
    comp_mad  = [d  for d  in mean_abs_dual    if d  is not None]
    comp_grad = [dd for dd in dDeg_dDoD        if dd is not None]

    if comp_ecap and comp_mas and comp_mad and comp_grad:
        # Scalar product of all three components
        comp_product = [ec * m * d for ec, m, d
                        in zip(comp_ecap, comp_mas, comp_mad)]

        # Normalise each series to its own mean
        def _norm(lst):
            mu = np.mean(lst)
            return [v / mu for v in lst]

        n_grad    = _norm(comp_grad)
        n_mas     = _norm(comp_mas)
        n_mad     = _norm(comp_mad)
        n_product = _norm(comp_product)

        # Shade second battery lifetime
        for yr in multiyear["replacement_years"]:
            ax.axvspan(yr, N_YEARS + 0.5, alpha=0.07, color="#e74c3c",
                       label="2nd battery lifetime")

        ax.plot(valid_comp_yrs, n_grad,    "o-",  color="#2c3e50",
                linewidth=2.0, markersize=5, zorder=4,
                label="dDeg/dDoD  (normalised)")
        ax.plot(valid_comp_yrs, n_product, "s--", color="#3498db",
                linewidth=1.3, markersize=4, alpha=0.85,
                label="e_cap × |subgrad| × |dual|  (normalised)")
        ax.plot(valid_comp_yrs, n_mas,     "^--", color="#8e44ad",
                linewidth=1.3, markersize=4, alpha=0.85,
                label="mean |subgrad|  (normalised)")
        ax.plot(valid_comp_yrs, n_mad,     "D--", color="#27ae60",
                linewidth=1.3, markersize=4, alpha=0.85,
                label="mean |dual|  (normalised)")

        ax.axhline(1.0, linestyle=":", linewidth=1.0, color="grey", alpha=0.5)
        for yr in multiyear["replacement_years"]:
            ax.axvline(yr + 0.5, linestyle="-.", linewidth=1.4,
                       color="red", alpha=0.8)

    ax.set_xlabel("Project year")
    ax.set_ylabel("Value / mean  (normalised to 1.0)")
    ax.set_title("Scalar Components vs Gradient — Normalised") # |subgrad|, |dual|, and their scalar product are flat across years 
    ax.set_xlim(0.5, N_YEARS + 0.5) # gradient oscillates 2.5× despite flat scalar components — alignment drives it
    ax.set_xticks(range(2, N_YEARS + 1, 2))
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    save_path = PLOTS_DIR / f"gradient_analysis_{run_label}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  ✓ Plot: {save_path.name}")
    if not show_plots:
        plt.close("all")

def _plot_subgradient_timeseries(
    multiyear: Dict,
    run_label: str,
) -> None:
    """Two-panel subgradient time series: year 1 vs last year before replacement.

    Each panel shows subgrad_combined (left y-axis) and SoC profile (right y-axis)
    over the 8760-hour horizon, so the relationship between dispatch and gradient
    signal is visible directly.
    """
    ann_grad = multiyear.get("annual_gradient", [])
    ann_soc  = multiyear.get("annual_soc", [])
    traj     = multiyear["soh_trajectory"]

    if not ann_grad or not ann_soc:
        print("  ⚠ Skipping subgradient time series,  data unavailable.")
        return

    # Year 1 = index 0
    # Last year before replacement = replacement_year itself (EoL year)
    repl_years = multiyear["replacement_years"]
    if repl_years:
        eol_yr      = repl_years[0]
        eol_idx     = eol_yr - 1      # 0-indexed
        panel_years = [1, eol_yr]
        panel_idxs  = [0, eol_idx]
    else:
        # No replacement,  use year 1 and final year
        panel_years = [1, traj[-1][0]]
        panel_idxs  = [0, len(traj) - 1]

    time_days = np.arange(len(ann_soc[0])) / 24.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(                                                    # comparing year 1 (fresh battery, full capacity) vs EoL year (degraded capacity)
        f"Subgradient Time Series: Year 1 vs Year {panel_years[1]}", # left axis = per-timestep subgradient signal; right axis = SoC [MWh]
        fontsize=11, y=1.01, 
    )

    colours = ["#4C72B0", "#e74c3c"]
    for ax, yr, idx, col in zip(axes, panel_years, panel_idxs, colours):
        if idx >= len(ann_grad) or ann_grad[idx] is None:
            ax.text(0.5, 0.5, f"Year {yr},  gradient unavailable",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        subgrad = ann_grad[idx][6]
        grad_val = ann_grad[idx][0]   # dDeg/dDoD scalar for this year
        soc     = ann_soc[idx]
        soh_p   = traj[idx][1]
        t       = time_days[:len(subgrad)]

        # Left axis,  subgradient
        ax.plot(t, subgrad, linewidth=0.6, color=col, alpha=0.85,
            label=f"yr {yr}  |  SoH = {soh_p:.1f}%  |  dDeg/dDoD = {grad_val:.2e}")
        ax.set_ylabel("subgrad_combined\n[USD / MWh]", color=col)
        ax.tick_params(axis="y", labelcolor=col)
        ax.axhline(0, linewidth=0.8, color="grey", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.20)
        ax.legend(loc="upper left", fontsize=8)

        # Right axis,  SoC
        ax2 = ax.twinx()
        ax2.plot(t[:len(soc)], soc, linewidth=0.5, color="#95a5a6",
                 alpha=0.60, label="SoC [MWh]")
        ax2.set_ylabel("SoC [MWh]", color="#7f8c8d")
        ax2.tick_params(axis="y", labelcolor="#7f8c8d")
        ax2.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time [days]")
    plt.tight_layout()
    save_path = PLOTS_DIR / f"subgradient_timeseries_{run_label}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  ✓ Plot: {save_path.name}")
    if not show_plots:
        plt.close("all")

def _plot_multiyear_trajectory(
    multiyear: Dict,
    run_label: str,
) -> None:
    """2×2 multi-year degradation figure.

    Top-left    SoH trajectory from year 0 with EoL crossings annotated
    Top-right   Annual fd total bars (Shi cycles + Xu calendar), y-axis zoomed
                to show the slow acceleration from capacity fade feedback
    Bottom-left Effective capacity (MWh) over the project lifetime
    Bottom-right fd decomposition: exact per-year cycle vs calendar stacked bars
    """
    traj    = multiyear["soh_trajectory"]
    years   = [r[0] for r in traj]
    soh_pct = [r[1] for r in traj]

    # Unpack tuple annual_fd: (fd_total, fd_cycle, fd_calendar)
    ann_fd_total    = [t[0] for t in multiyear["annual_fd"]]
    ann_fd_cycle    = [t[1] for t in multiyear["annual_fd"]]
    ann_fd_calendar = [t[2] for t in multiyear["annual_fd"]]

    # Prepend year-0 origin
    years_full = [0] + years
    soh_full   = [100.0] + soh_pct

# Effective capacity: E_nominal × (SoH/100), reset at replacement
    e_cap_nominal_approx = multiyear.get("e_cap_nominal", 300.0)
    e_cap_eff_list = [
        soh_p / 100.0 * e_cap_nominal_approx
        for (_, soh_p, _, _) in traj
    ]

    # EoL crossing interpolator
    def _eol_crossing(soh_list, year_list, threshold):
        for i in range(1, len(soh_list)):
            if soh_list[i-1] >= threshold >= soh_list[i]:
                frac = (soh_list[i-1] - threshold) / max(soh_list[i-1] - soh_list[i], 1e-9)
                return year_list[i-1] + frac * (year_list[i] - year_list[i-1])
        return None

    thr_colours = {80.0: "#e74c3c", 70.0: "#e67e22"}
    thr_labels  = {80.0: "80% [IEC/EV]", 70.0: "70% [warranty]"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Multi-year Degradation — Shi (2018) + Xu calendar  |  {N_YEARS} yr",
        fontsize=11, y=1.01,
    )

    # ── Top-left: SoH trajectory ─────────────────────────────────────────
    ax = axes[0, 0]
    
    segments = _split_battery_segments(years_full, soh_full, multiyear["replacement_years"])
    for (sx, sy) in segments:
        ax.plot(sx, sy,
                linewidth=2.0, color="#4C72B0",
                linestyle="-",
                marker="o", markersize=3.5, zorder=3,
                label="_nolegend_")

    # ── Connector: vertical dotted line between end of Battery i and start of Battery i+1
    for i in range(len(segments) - 1):
        x_conn  = segments[i][0][-1]   # replacement year (same for both endpoints)
        y_bot   = segments[i][1][-1]   # EoL SoH of outgoing battery (~70%)
        y_top   = segments[i+1][1][0]  # 100.0,  fresh battery anchor
        ax.plot([x_conn, x_conn], [y_bot, y_top],
                linewidth=2.0, color="#4C72B0",
                linestyle="-", alpha=1.0, zorder=4)

    crossing_lines = []
    for thr, col in thr_colours.items():
        ln = ax.axhline(thr, linestyle="--", linewidth=1.2, color=col, alpha=0.85)
        crossing_lines.append((ln, thr_labels[thr]))
        cross = _eol_crossing(soh_full, years_full, thr)
        if cross is not None and cross <= N_YEARS:
            ax.axvline(cross, linestyle=":", linewidth=1.0, color=col, alpha=0.4, zorder=1)
            ha = "right" if cross > N_YEARS * 0.6 else "left"
            offset = -0.4 if ha == "right" else 0.4
            ax.annotate(
                f"{thr:.0f}%  yr {cross:.1f}",
                xy=(cross, thr), xytext=(cross + offset, thr + 2.0),
                fontsize=7.5, color=col, fontweight="bold", ha=ha,
                arrowprops=dict(arrowstyle="-", color=col, lw=0.6),
            )

    repl_handle = None
    for yr in multiyear["replacement_years"]:
        repl_handle = ax.axvline(yr, linestyle="-.", linewidth=1.5,
                                 color="red", alpha=0.85, zorder=2)
        ax.text(yr - 0.3, min(soh_pct) - 3.5, f"Replace\nyr {yr}",
                fontsize=7, color="red", ha="right", va="top")

    # Build legend manually,  single SoH entry + threshold lines + replacement
    from matplotlib.lines import Line2D
    leg_handles = [Line2D([0], [0], color="#4C72B0", lw=2, marker="o",
                          markersize=4, label="Simulated SoH")]
    for ln, lbl in crossing_lines:
        leg_handles.append(Line2D([0], [0], linestyle="--",
                                   color=ln.get_color(), lw=1.2, label=lbl))
    if repl_handle is not None:
        leg_handles.append(Line2D([0], [0], linestyle="-.", color="red",
                                   lw=1.5, label="Battery replacement"))

    min_soh = min(soh_pct) if soh_pct else 55.0
    ax.set_xlim(0, N_YEARS)
    ax.set_xticks(range(0, N_YEARS + 1, 2))
    ax.set_ylim(max(min_soh - 8, 50), 103)
    ax.set_xlabel("Project year")
    ax.set_ylabel("State of Health [%]")
    ax.set_title("State of Health Trajectory") # capacity fade feedback: LP re-solved each year with degraded e_cap_eff
    ax.legend(handles=leg_handles, fontsize=8, loc="lower left")    # Place legend in lower-left where SoH line hasn't reached yet
    ax.grid(True, alpha=0.25)

    # ── Top-right: annual fd bars,  zoomed to show acceleration ──────────
    ax = axes[0, 1]
    ax.bar(years, ann_fd_total, color="#55A868", alpha=0.80, width=0.7,
           label="Annual fd\n(Shi cycles + Xu calendar)")

    for yr in multiyear["replacement_years"]:
        ax.axvline(yr + 0.5, linestyle="-.", linewidth=1.4, color="red", alpha=0.8)

    # Note: fd accelerates over the project lifetime due to capacity fade → deeper relative cycles
    # Note: pct_acc = 100 * (ann_fd_total[-1] - ann_fd_total[0]) / ann_fd_total[0] — printable if needed

    ax.set_xlabel("Project year")
    ax.set_ylabel("Annual fd (Shi + calendar)")
    ax.set_title("Annual Degradation Rate  (fd)") # Note: fd = Shi cycle accumulation + Xu calendar correction (reporting path only)
    ax.set_xlim(0.5, N_YEARS + 0.5)
    ax.set_xticks(range(2, N_YEARS + 1, 2)) #even numbering 2,4,....,20
    # Zoom y-axis tightly to show the slow acceleration
    fd_min = min(ann_fd_total) * 0.995
    fd_max = max(ann_fd_total) * 1.010
    ax.set_ylim(fd_min, fd_max)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25, axis="y")

    # ── Bottom-left: year-over-year SoH loss rate ────────────────────────
    ax = axes[1, 0]

    # ΔSoH per year,  derived from the full trajectory including year-0.
    # For post-replacement years the reference is 100% (fresh battery), not the previous year's degraded SoH.
    post_repl_years = {yr + 1 for yr in multiyear["replacement_years"]}
    delta_soh = []
    for i, yr in enumerate(years):
        if yr in post_repl_years:
            delta_soh.append(100.0 - soh_full[i + 1])   # loss from fresh battery
        else:
            delta_soh.append(soh_full[i] - soh_full[i + 1])

    # All years are now valid,  post-replacement years use 100% as reference
    loss_years  = years
    loss_values = delta_soh
    bar_colors  = ["#27ae60" if yr in post_repl_years
                   else "#e74c3c" if yr in multiyear["replacement_years"]
                   else "#4C72B0" for yr in loss_years]
    ax.bar(loss_years, loss_values, width=0.7, color=bar_colors, alpha=0.82)

    # Mean loss rate over normal operating years (exclude the EoL year itself
    # as it may be a partial year below threshold, and post-replacement year 1
    # is included since it represents genuine first-year loss of a fresh battery)
    normal_losses = [d for yr, d in zip(loss_years, loss_values)
                     if yr not in multiyear["replacement_years"]]
    if normal_losses:
        mean_loss = np.mean(normal_losses)
        ax.axhline(mean_loss, linestyle="--", linewidth=1.3,
                   color="#2c3e50", alpha=0.8,
                   label=f"Mean loss rate: {mean_loss:.3f} %/yr")

    # Mark replacement events
    for yr in multiyear["replacement_years"]:
        ax.axvline(yr + 0.5, linestyle="-.", linewidth=1.4, color="red", alpha=0.8)
        idx = loss_years.index(yr)
        ax.text(yr, loss_values[idx] + 0.05,
                f"EoL yr {yr}", fontsize=7, color="red", ha="center")

    # Add legend entry for post-replacement bar colour
    from matplotlib.patches import Patch
    extra_handles = [
        Patch(facecolor="#27ae60", alpha=0.82,
              label="Post-replacement yr\n(ref = 100% fresh battery)"),
        Patch(facecolor="#e74c3c", alpha=0.82, label="EoL year"),
        Patch(facecolor="#4C72B0", alpha=0.82, label="Normal operating year"),
    ]

    ax.set_xlabel("Project year")
    ax.set_ylabel("Annual SoH loss [%/yr]")
    ax.set_title("Annual SoH Loss Rate") # increasing trend = degradation accelerating as capacity fades and cycles deepen
    ax.set_xlim(0.5, N_YEARS + 0.5)
    ax.set_xticks(range(2, N_YEARS + 1, 2)) #even numbering 2,4,....,20
    ax.set_ylim(0, max(loss_values) * 1.35 if loss_values else 10)
    mean_handle, _ = ax.get_legend_handles_labels()
    ax.legend(handles=mean_handle + extra_handles, fontsize=7.5, loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")

    # ── Bottom-right: exact cycle vs calendar stacked bars ────────────────
    ax = axes[1, 1]
    ax.bar(years, ann_fd_cycle, width=0.7, color="#3498db", alpha=0.82,
           label="Cycle fd  (Shi Φ accumulation)")
    ax.bar(years, ann_fd_calendar, width=0.7, color="#e67e22", alpha=0.82,
           bottom=ann_fd_cycle, label="Calendar fd  (Xu ft_calendar)")

    for yr in multiyear["replacement_years"]:
        ax.axvline(yr + 0.5, linestyle="-.", linewidth=1.4, color="red", alpha=0.8)

    # Mean fractions over the run
    mean_cal_pct = 100.0 * np.mean(ann_fd_calendar) / max(np.mean(ann_fd_total), 1e-9)
    mean_cyc_pct = 100.0 - mean_cal_pct
    ax.set_xlabel("Project year")
    ax.set_ylabel("Annual fd contribution")
    ax.set_title(f"fd Decomposition — Cycle {mean_cyc_pct:.0f}%  /  Calendar {mean_cal_pct:.0f}%") # cycle fd from Shi Φ accumulation; calendar fd from Xu ft_calendar (reporting path only)
    ax.set_xlim(0.5, N_YEARS + 0.5)
    ax.set_xticks(range(2, N_YEARS + 1, 2)) #even numbering 2,4,....,20
    ax.set_ylim(0, max(ann_fd_total) * 1.12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    save_path = PLOTS_DIR / f"multiyear_trajectory_{run_label}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  ✓ Plot: {save_path.name}")
    if not show_plots:
        plt.close("all")


# =============================================================================
# Console + CSV helpers
# =============================================================================

def _print_degr_block(degr: Dict, label: str, period_days: float) -> None:
    content = f"  DEGRADATION,  {label}"
    # Auto-scale: wide enough for the label, minimum 72, always matches the border
    W = max(72, len(content) + 2)
    print("\n" + "╔" + "═" * W + "╗")
    print(f"║{content:<{W}}║")
    print("╚" + "═" * W + "╝")
    print_degradation_report(degr, period_days=period_days, enabled=True)


def _save_multiyear_csv(
    multiyear:  Dict,
    run_label:  str,
    timestamp:  str,
) -> None:
    traj = multiyear["soh_trajectory"]
    ann  = multiyear["annual_fd"]
    rows = []
    
    ann_grad = multiyear.get("annual_gradient", [None] * len(ann))
    for i, ((year, soh_pct, fd_cum, n_rep), (fd_yr, fd_cyc, fd_cal)) in enumerate(zip(traj, ann)):
        grad = ann_grad[i]
        rows.append({
            "timestamp":             timestamp,
            "run_label":             run_label,
            "year":                  year,
            "soh_pct":               round(soh_pct, 4),
            "fd_annual":             round(fd_yr, 6),
            "fd_cycle":              round(fd_cyc, 6),
            "fd_calendar":           round(fd_cal, 6),
            "fd_calendar_pct":       round(100.0 * fd_cal / max(fd_yr, 1e-30), 1),
            "fd_cumulative":         round(fd_cum, 6),
            "n_replacements":        n_rep,
            "replacement_this_year": year in multiyear["replacement_years"],
            "dDeg_dDoD":             round(grad[0], 6) if grad else None,
            "mean_abs_subgrad":      round(grad[1], 6) if grad else None,
            "mean_dod":              round(grad[2], 4) if grad else None,
            "mean_abs_dual":         round(grad[3], 6) if grad else None,
            "e_cap_eff":             round(grad[4], 4) if grad else None,
            "cycle_coverage":        round(grad[5], 4) if grad else None,
        })

    csv_path = RESULTS_DIR / f"multiyear_trajectory_{run_label}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  ✓ CSV: {csv_path.name}")

def _save_gradient_timeseries_csv(
    multiyear:  Dict,
    run_label:  str,
    timestamp:  str,
) -> None:
    """Save full per-timestep subgrad_combined arrays for all years to CSV.

    Produces one row per timestep per year,  shape (N_YEARS × n_timesteps).
    Columns: timestamp, run_label, year, timestep, soh_pct, subgrad_combined.
    The scalar summary (dDeg_dDoD etc.) stays in the multiyear_trajectory CSV.
    """
    ann_grad = multiyear.get("annual_gradient")
    if not ann_grad or all(g is None for g in ann_grad):
        print("  ⚠ No gradient time series to save,  duals unavailable.")
        return

    traj = multiyear["soh_trajectory"]
    rows = []
    for (year, soh_pct, fd_cum, _n_rep), grad in zip(traj, ann_grad):
        if grad is None:
            continue
        subgrad = grad[6]   # full np.ndarray, shape (n_timesteps,)
        for t, val in enumerate(subgrad):
            rows.append({
                "timestamp":        timestamp,
                "run_label":        run_label,
                "year":             year,
                "timestep":         t,
                "soh_pct":          round(soh_pct, 4),
                "subgrad_combined": round(float(val), 8),
            })

    csv_path = RESULTS_DIR / f"gradient_timeseries_{run_label}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  ✓ CSV: {csv_path.name}  "
          f"({len(rows):,} rows,  {len(ann_grad)} yr × {len(ann_grad[0][6])} timesteps)")
# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("WP2 BATTERY OPTIMIZATION + DEGRADATION  v2")
    print("Shi (2018) + Xu calendar correction | multi-year | battery replacement")
    print("=" * 80)

    # ── 1. Load config + raw series ───────────────────────────────────────
    print("\n[1/6] Loading WP2 configuration...")
    setup = quick_setup(HPP_YAML, config={"interp_n": 2000}, verbose=False)
    hpp   = setup["hpp"]

    ws_all, wd_all, ti_all = _load_inputs(hpp)
    price_all               = _load_prices()

    # ── 2. Choose horizon ─────────────────────────────────────────────────
    print("\n[2/6] Loading electricity prices...")
    n = _choose_horizon(len(ws_all), len(price_all))
    ws        = ws_all[:n]
    wd        = wd_all[:n]
    ti        = ti_all[:n] if ti_all is not None else None
    price_eur = price_all[:n]

    print(f"  Horizon: {n:,} h ({n/24:.1f} days)")
    print(f"  Mean price: {float(np.mean(price_eur)):.2f} EUR/MWh")

    if MULTI_YEAR and n != 8760:
        print(f"  ⚠  MULTI_YEAR=True requires exactly 8760 h. "
              f"Got {n} h,  multi-year loop will use this as the annual tile.")

    # ── 3. PyWake ─────────────────────────────────────────────────────────
    print("\n[3/6] Running PyWake time-series simulation...")
    power_wind_MW = _run_pywake_power_MW(setup, wd, ws, ti)
    print(f"  Wind mean: {float(np.mean(power_wind_MW)):.1f} MW | "
          f"peak: {float(np.max(power_wind_MW)):.1f} MW")

    # ── 4. SHIPP (single-year baseline) ───────────────────────────────────
    print("\n[4/6] Running SHIPP optimization (single-year baseline)...")
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
    print("\n[5/6] Single-year degradation analysis (Shi + Xu calendar)...")
    bat_params = setup["battery"]

    # Dispatch-fixed: Shi + calendar correction
    degr_fixed = _shi_with_calendar_correction(
        os_fixed.storage_p[0].data,
        os_fixed.storage_e[0].data,
        float(os_fixed.storage_list[0].e_cap),
        bat_params, shi_fit, 25.0,
        dt, eol_thresholds,
    )

    # Sizing-opt: guard for no-battery case (common with 2019 prices)
    opt_e_cap  = float(os.storage_list[0].e_cap)
    no_battery = opt_e_cap < 1.0
    if no_battery:
        print(
            f"\n  ⚠  Sizing optimizer chose no battery "
            f"(e_cap = {opt_e_cap:.3f} MWh < 1 MWh).\n"
            f"     Not economically viable at this price level. Degradation skipped."
        )
        degr_opt = None
    else:
        degr_opt   = _shi_with_calendar_correction(
            os.storage_p[0].data,
            os.storage_e[0].data,
            opt_e_cap,
            bat_params, shi_fit, 25.0,
            dt, eol_thresholds,
        )

    # Console report
    if print_degr_reports:
        _print_degr_block(degr_fixed, "DISPATCH-FIXED,  150 MW / 300 MWh", period_days)
        if degr_opt is not None:
            _print_degr_block(degr_opt, "SIZING OPT,  optimal capacity", period_days)
        else:
            print("\n  [Sizing-opt degradation skipped,  no battery installed]")

    # Gap D verification (unchanged from v1)
    if os_fixed.dual_prices is not None:
        print("\n" + "=" * 60)
        print("GAP D VERIFICATION,  dual prices + gradient signal")
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
        print("\n  WARNING: dual_prices is None,  dual extraction failed.")

    # ── 6. Multi-year loop ────────────────────────────────────────────────
    multiyear = None
    if MULTI_YEAR:
        print(f"\n[6/6] Multi-year degradation loop ({N_YEARS} years)...")
        # Use the full 8760-h series (or the truncated one if RUN_FULL_YEAR=False)
        wind_tile  = power_wind_MW[:8760] if n >= 8760 else power_wind_MW
        price_tile = price_eur[:8760]     if n >= 8760 else price_eur

        multiyear = _run_multiyear(
            wind_8760=wind_tile,
            price_8760=price_tile,
            stor_null=stor_null,
            p_max_MW=p_max_MW,
            e_cap_nominal=e_cap_MWh,
            p_cap_MW=p_cap_MW,
            rte_ac=rte_ac,
            e_cost_USD_per_MWh=e_cost_USD_per_MWh,
            p_cost_USD_per_MW=p_cost_USD_per_MW,
            bat_params=bat_params,
            shi_fit=shi_fit,
            soc_min=soc_min,
            soc_max=soc_max,
            T_cell_C=25.0,
        )

        print(f"\n  Multi-year summary:")
        print(f"    Final SoH      : {multiyear['final_soh_pct']:.2f}%")
        print(f"    Replacements   : {multiyear['n_replacements']}")
        print(f"    Replacement yrs: {multiyear['replacement_years']}")
        print(f"    Replacement cost: "
              f"{multiyear['replacement_cost_USD']*1e-6:.2f} M USD")

        # ── Multi-year NPV correction ─────────────────────────────────────
        # The single-year LP NPV is correct for revenues and initial CAPEX.
        # The only missing term is the discounted replacement CAPEX(es).
        capex_replacement_USD = (e_cap_MWh * e_cost_USD_per_MWh
                                 + p_cap_MW * p_cost_USD_per_MW)
        pv_replacements_USD = sum(
            capex_replacement_USD / (1.0 + discount_rate) ** yr
            for yr in multiyear["replacement_years"]
        )
        npv_multiyear_MUSD = os_fixed.npv - pv_replacements_USD * 1e-6
        multiyear["npv_singleyr_MUSD"]    = os_fixed.npv
        multiyear["pv_replacements_MUSD"] = pv_replacements_USD * 1e-6
        multiyear["npv_multiyear_MUSD"]   = npv_multiyear_MUSD

        print(f"\n  Multi-year NPV:")
        print(f"    NPV (single-year LP)  : {os_fixed.npv:>9.2f} M USD")
        print(f"    PV of replacement(s)  : {pv_replacements_USD*1e-6:>9.2f} M USD"
              f"  (yr {multiyear['replacement_years']}, r={discount_rate*100:.0f}%)")
        print(f"    ─────────────────────────────────────────")
        print(f"    NPV (multi-year)      : {npv_multiyear_MUSD:>9.2f} M USD")
        print(f"    Δ NPV                 : {npv_multiyear_MUSD - os_fixed.npv:>9.2f} M USD")

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
        if degr_opt is not None:
            degr_rows.append(_degr_row(degr_opt, "sizing_opt"))
        pd.DataFrame(degr_rows).to_csv(
            degr_csv, mode="a", header=not degr_csv.exists(), index=False
        )

        print(f"\n  ✓ CSV: {base_csv.name}")
        print(f"  ✓ CSV: {degr_csv.name}")

        # Multi-year trajectory
        if multiyear is not None:
            _save_multiyear_csv(multiyear, run_label, timestamp)
            _save_gradient_timeseries_csv(multiyear, run_label, timestamp)

    # ── Save text report ──────────────────────────────────────────────────
    if SAVE_REPORT:
        report_path = RESULTS_DIR / f"degradation_report_{run_label}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("WP2 BATTERY,  Shi (2018) + Xu calendar | v2 REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated : {timestamp}\n")
            f.write(f"Dataset   : {PRICE_CSV.name}\n")
            f.write(f"Horizon   : {n:,} h  ({period_days:.1f} days)\n")
            f.write(f"Solver    : {pyo_solver}\n\n")
            f.write(f"Shi polynomial: k3={shi_fit.k3:.4e}  k4={shi_fit.k4:.4f}  "
                    f"R²={shi_fit.r2:.4f}  window=[{soc_min},{soc_max}]\n\n")

            for case_name, d in [("dispatch_fixed", degr_fixed),
                                  ("sizing_opt",     degr_opt)]:
                if d is None:
                    continue
                f.write(f"DEGRADATION ({case_name}),  Shi2018 + Xu calendar\n")
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

            if multiyear is not None:
                f.write("MULTI-YEAR SUMMARY\n" + "-" * 60 + "\n")
                f.write(f"N_years          : {N_YEARS}\n")
                f.write(f"EoL replacement  : SoH < {EOL_REPLACEMENT*100:.0f}%\n")
                f.write(f"Replacements     : {multiyear['n_replacements']}\n")
                f.write(f"Replacement years: {multiyear['replacement_years']}\n")
                f.write(f"Replacement cost : "
                        f"{multiyear['replacement_cost_USD']*1e-6:.2f} M USD\n")
                f.write(f"NPV (single-year LP)  : {multiyear['npv_singleyr_MUSD']:.2f} M USD\n")
                f.write(f"PV of replacement(s)  : {multiyear['pv_replacements_MUSD']:.2f} M USD"
                        f"  (yr {multiyear['replacement_years']}, r={discount_rate*100:.0f}%)\n")
                f.write(f"NPV (multi-year)      : {multiyear['npv_multiyear_MUSD']:.2f} M USD\n")
                f.write(f"Delta NPV             : "
                        f"{multiyear['npv_multiyear_MUSD'] - multiyear['npv_singleyr_MUSD']:.2f} M USD\n")
                f.write(f"Final SoH        : {multiyear['final_soh_pct']:.2f}%\n\n")
                f.write(f"{'Year':>5}  {'SoH_%':>8}  {'fd_annual':>10}  "
                        f"{'fd_cumul':>10}  {'dDeg/dDoD':>12}  {'mean_DoD%':>10}  {'replaced':>8}\n")
                f.write("-" * 72 + "\n")
                ann_fd   = multiyear["annual_fd"]
                ann_grad = multiyear.get("annual_gradient", [None] * len(ann_fd))
                for (yr, soh_p, fd_c, _n_rep), (fd_y, fd_cyc, fd_cal), grad in zip(
                    multiyear["soh_trajectory"], ann_fd, ann_grad
                ):
                    replaced = "YES" if yr in multiyear["replacement_years"] else ""
                    grad_str = f"{grad[0]:.4e}" if grad is not None else "     N/A"
                    dod_str  = f"{grad[2]*100:.1f}" if grad is not None else "   N/A"
                    f.write(f"{yr:>5d}  {soh_p:>8.3f}  {fd_y:>10.6f}  "
                            f"{fd_c:>10.6f}  {grad_str:>12}  {dod_str:>10}  {replaced:>8}\n")

                valid_grads = [g[0] for g in ann_grad if g is not None]
                if valid_grads:
                    f.write(f"\ndDeg/dDoD over {N_YEARS} yr:  "
                            f"min={min(valid_grads):.4e}  "
                            f"max={max(valid_grads):.4e}  "
                            f"mean={sum(valid_grads)/len(valid_grads):.4e}\n")

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
        if degr_opt is not None:
            plot_degradation_analysis(
                degr_opt,
                storage_e=os.storage_e[0].data,
                time_vec=time_vec,
                save_path=str(
                    PLOTS_DIR / f"battery_degradation_analysis_sizing_{run_label}.png"
                ),
                show=False, verbose=True, eol_thresholds=eol_thresholds,
            )

        # Multi-year trajectory plot
        if multiyear is not None:
            _plot_multiyear_trajectory(multiyear, run_label)
            _plot_gradient_analysis(multiyear, run_label)
            _plot_subgradient_timeseries(multiyear, run_label)

        if show_plots:
            plt.show()
        else:
            plt.close("all")

    print("\n" + "=" * 80)
    print("✓ COMPLETE,  v2 (Shi + Xu calendar | multi-year | battery replacement)")
    print("=" * 80)


if __name__ == "__main__":
    main()