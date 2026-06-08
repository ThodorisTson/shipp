"""WP2 battery optimization + Shi degradation,  v5 (Castillo outer-loop, VALIDATED).

STATUS: Frozen validated milestone. Gradient validated at n=8760 / N_YEARS=20:
  - dRev & lambda_E vs Gurobi reduced cost (rc_e_cap1) — match to 6 s.f.
  - dDegCost vs frozen-dispatch finite difference — match to 0.003%.
Constraint: requires exactly 8760-h input; MULTI_YEAR flag gates the loop.
v5.1 supersedes this with a horizon-agnostic unified loop (any window, any lifetime).

v5 builds directly on v4 (multi-year, calendar-corrected) and adds the analytic outer-loop sizing gradient dNPV/dE_cap, derived from Castillo et al.
(2008) sensitivity analysis.  No v4 behaviour in the REPORTING path (fd, SoH, EoL, NPV) is altered; v5 only ADDS gradient diagnostics.

New in v5 vs v4
---------------
Outer-loop gradient (Castillo Theorem 1 / Example 5).  After each inner LP solve, v5 assembles the sizing gradient from quantities the LP already produces:

    dNPV/dE_cap = dRevenue*/dE_cap  -  dDegCost/dE_cap  -  lambda_E

  - dRevenue*/dE_cap : EXACT within the LP.  E_cap enters the LP only through the right-hand side of the SoC capacity constraints, so by Castillo's Theorem 1 
    the revenue sensitivity is a weighted sum of the LP dual variables on those constraints (coeff soc_max on the upper bound, soc_min on the lower bound).  
    This is a free by-product of the solve and replaces the earlier (incorrect) assumption that this term is ~0 under frozen dispatch.
  - dDegCost/dE_cap : FROZEN-DISPATCH approximation.  Dispatch (c_t, d_t) is held fixed, so cycle throughput is fixed and only the fractional cycle depth rescales 
    with capacity: ddelta_i/dE_cap = -delta_i/E_cap.  The Shi cost is B*Phi(delta) with E entering only inside delta, so the product rule has a single term, '
    B*Phi'(delta_i)*(-delta_i/E_cap), using the same phi_shi_prime_with_stress kernel as the validated subgradient.
  - lambda_E : annualised energy capex coefficient (trivial).

Cross-check: the reduced cost of the fixed-bounds e_cap1 variable (rc_e_cap1, extracted in kernel_pyomo) equals dObjective/dE_cap and is carried through for 
an independent check against the assembled gradient.

The old fused scalar dDeg_dDoD (conceptually incorrect: it dotted the degradation subgradient against the revenue dual) is RETAINED unchanged at tuple index [0] 
so existing CSV columns, console tables, and gradient plots receive identical input.  The clean v5 terms occupy new tuple indices [7]-[11] and new CSV columns; 
nothing consumes them for decisions yet (the outer loop is not yet closed).

Inherited from v4
-----------------
1. Calendar correction (Option 2)
   The Shi accumulation framework is cycle-only (fd_calendar = 0.0 by design). For the WP2 site, calendar aging accounts for ~49-51% of total annual fd under 
   the Xu model.  ft_calendar() from degradation_xu is added to the Shi result in the REPORTING PATH ONLY.  The gradient computation is never touched, 
   so the dual-Phi convexity guarantee is fully preserved.

   Architectural note: dfd_calendar/dc_t = 0 for all t.  Calendar aging carries no dispatch-dependent information and cannot drive the outer loop.

2. Multi-year degradation loop (MULTI_YEAR = True)
   Loops over N_YEARS using the same representative 8760-hour dataset tiled each year.  Annual fd is recomputed each year from the actual dispatch given the 
   degraded capacity; the inner LP is re-solved each year with E_eff = E_nominal * SoH(year-1).

3. Battery replacement at EOL_REPLACEMENT = 0.70 SoH
   When SoH drops below the 70% threshold, the battery is replaced: fd resets to 0, SoH resets to 1.0, and a replacement cost is subtracted from NPV.

File structure
--------------
  CONFIG                          — toggles and thresholds (top of file)
  _shi_with_calendar_correction() — single-year Shi + Xu calendar patching
  _build_storage_year()           — build Storage object for a given effective e_cap
  _run_multiyear()                — year-by-year loop: LP re-solve + degradation
                                    + v5 Castillo outer-loop gradient assembly
  _save_multiyear_csv()           — save per-year trajectory CSV
  _save_gradient_timeseries_csv() — save per-timestep subgrad CSV
  main()                          — orchestrates single-year + multi-year runs

  Plotting delegated to:
  degradation_plots.py            — single-year degradation figures (unchanged)
  degradation_plots_multiyear.py  — multi-year trajectory + gradient figures

No changes required to:
  degradation_xu.py            — ft_calendar / sei_capacity_loss already in API
  degradation_shi.py           — phi_shi_prime_with_stress reused by v5 gradient
  degradation_subgradient.py   — subgradient path untouched
  degradation_plots.py         — single-year plots unchanged
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
from degradation_shi import analyze_degradation_shi, phi_shi_prime_with_stress, s_soc, s_temp
from degradation_subgradient import compute_subgradient, fit_shi_polynomial

from degradation_plots import (
    plot_degradation_analysis,
    print_degradation_report,
)

import xarray as xr
from py_wake.site import XRSite
import numpy_financial as npf

from degradation_plots_multiyear import (
    plot_gradient_analysis,
    plot_subgradient_timeseries,
    plot_multiyear_trajectory,
)

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


def _build_run_label(ts: str, price_csv: Path, p_cap: float, e_cap: float, soc_min: float = 0.10, soc_max: float = 0.90) -> str:
    stem    = price_csv.stem.lower()
    year    = ''.join(filter(str.isdigit, stem))[-4:]
    dataset = f"dk{year}"
    bat     = f"{int(round(p_cap))}mw_{int(round(e_cap))}mwh"
    soc_tag = f"soc{int(soc_min*100)}_{int(soc_max*100)}"
    return f"{ts}_{dataset}_{bat}_{soc_tag}_v5"


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

    Power capacity and costs are held at their nominal values,  only energy capacity (e_cap_eff = E_nominal * SoH) changes year to year.
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

def _fd_validate_degradation_term(
    dods: np.ndarray,
    counts: np.ndarray,
    soc_means: np.ndarray,
    e_cap: float,
    e_cost_USD_per_MWh: float,
    factor: float,
    shi_fit,
    T_cell_C: float = 25.0,
    eps_list: Tuple[float, ...] = (0.01, 0.005),
) -> None:
    """Frozen-dispatch finite-difference check of the analytic dDegCost/dE_cap.

    Validates ONLY the degradation channel — the one term rc cannot check,
    since the LP is blind to degradation.

    Frozen dispatch: cycle ENERGY throughput is held fixed, so the fractional
    cycle depth scales as delta(E') = delta * E / E' = delta / (1 ± eps).
    We rebuild the TOTAL degradation cost at E, E(1+eps), E(1-eps) using the
    SAME stress-weighted Shi cost as the analytic term, then central-difference.

    Analytic term under test:
        dDegCost/dE = e_cost * factor * sum_i [ count_i * Phi'(delta_i) * (-delta_i/E) ]

    Cost used for the FD (integral of that derivative's kernel):
        Cost(E) = e_cost * factor * sum_i [ count_i * k3 * delta_i(E)^k4
                                            * S_sigma(soc_i) * S_T(T) ]

    A PASS is FD ≈ analytic to a few % AND stable across the two eps values
    (Richardson sanity: shrinking eps should not change the estimate).
    """
    if len(dods) == 0:
        print("[1.5] no cycles — degradation FD skipped.")
        return

    k3, k4 = shi_fit.k3, shi_fit.k4
    stress = s_soc(soc_means) * s_temp(T_cell_C)   # per-cycle S_sigma * S_T (same defaults as analytic)
    B_fac  = e_cost_USD_per_MWh * factor            # common lifetime-NPV prefactor

    def deg_cost(e_prime: float) -> float:
        # Frozen dispatch: throughput fixed → delta rescales by E/e_prime
        delta_p = dods * (e_cap / e_prime)
        delta_p = np.clip(delta_p, 1e-9, 1.0)       # guard the power law domain
        return B_fac * float(np.sum(counts * k3 * delta_p ** k4 * stress))

    # Analytic term (recomputed here so the function is self-contained)
    phi_prime = phi_shi_prime_with_stress(dods, soc_means, T_cell_C, k3, k4)
    analytic  = B_fac * float(np.sum(counts * phi_prime * (-dods / e_cap)))

    print(f"[1.5] degradation-term FD check at E_cap={e_cap:.2f} MWh")
    print(f"      analytic dDegCost/dE = {analytic:+.6f}")
    for eps in eps_list:
        dE   = eps * e_cap
        cp   = deg_cost(e_cap + dE)
        cm   = deg_cost(e_cap - dE)
        fd   = (cp - cm) / (2.0 * dE)               # central difference
        rel  = (fd - analytic) / analytic if analytic != 0 else float("nan")
        print(f"      eps={eps:6.3f}  dE={dE:7.3f}  FD={fd:+.6f}  "
              f"rel.err={rel:+.3%}")

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

    sigma_mean and t_total_seconds are computed internally from storage_e and dt_hours,  exactly as analyze_degradation() does,  so partial-year series
    (e.g. 8752 h instead of 8760 h) are handled correctly without any hardcoding in the call site.

    The result dict is patched in place so all downstream functions (plot_degradation_analysis, print_degradation_report, CSV logging) work 
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
    - Representative year assumption: the same 8760-h wind and price dataset is reused every year.  This is the standard approach in energy storage 
      planning studies and is consistent with HyDesign.
    - Capacity fade feedback: the LP is re-solved each year with E_eff = E_nominal * SoH(year-1).  Dispatch adapts to the reduced capacity.
    - Battery replacement: when SoH < EOL_REPLACEMENT, fd resets to 0.0 and SoH resets to 1.0 (fresh battery).  Replacement cost is tracked separately.
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
        if os_yr.dual_prices is not None:                       # duals exist only if LP solved with return_duals=True
            dual_yr   = os_yr.dual_prices["dual_e_min1"]        # LP shadow price of lower SoC bound, per timestep [USD/MWh]
            e_cap_yr  = os_yr.dual_prices["e_cap1"]             # this year's EFFECTIVE capacity E_eff = E_nom*SoH [MWh]
            cycles_yr = rainflow_cycle_counting(storage_e_yr, e_cap_yr)  # extract cycles from the FROZEN SoC trace
            sg_yr     = compute_subgradient(                    # Shi per-timestep subgradient on that frozen trace
                storage_e=storage_e_yr,
                cycles=cycles_yr,
                dt_hours=dt,
                battery_replacement_cost_per_MWh=e_cost_USD_per_MWh,
                eff_in=1.0,
                eff_out=rte_ac,
                shi_fit=shi_fit,
            )
            factor_yr        = npf.npv(discount_rate, np.ones(N_YEARS)) - 1  # NPV annuity factor (discounts a flat stream)
            dual_per_year_yr = dual_yr / factor_yr              # de-annualise the dual back to a single-year shadow price

            # [0] OLD fused scalar — KEPT UNCHANGED so existing CSV/plots/table still receive identical input.  Conceptually wrong (mixes the degradation subgradient 
            # with the revenue dual) but retained as a fallback until the clean terms below are FD-validated.
            dDeg_dDoD_yr     = -e_cap_yr * float(np.dot(sg_yr["subgrad_combined"], dual_per_year_yr))

            dods_yr     = np.array([c["dod"]      for c in cycles_yr])   # cycle depths δ_i, as FRACTION of capacity (0–1)
            cnts_yr     = np.array([c["count"]    for c in cycles_yr])   # 1.0 full cycle, 0.5 half cycle (rainflow weights)
            socm_yr     = np.array([c["soc_mean"] for c in cycles_yr])   # mean SoC of each cycle (feeds the S_σ stress term)
            mean_dod_yr = float(np.average(dods_yr, weights=cnts_yr)) if len(dods_yr) > 0 else 0.0

            # ── Castillo outer-loop terms (additive, Phase 1: FROZEN DISPATCH) ──
            # Revenue sensitivity (Castillo Thm 1 / Example 5): E_cap enters the LP only via the RHS of e_max1 (coeff soc_max) and e_min1 (coeff 1-dod, 
            # which in v4 == soc_min, verified).  Both soc_min/soc_max are function arguments of _run_multiyear, so in scope here.
            dual_max_yr   = os_yr.dual_prices["dual_e_max1"]    # shadow price of UPPER SoC bound, per timestep [USD/MWh]
            dRev_dEcap_yr = (
                soc_max * float(np.sum(dual_max_yr))           # upper-bound RHS = E_cap*soc_max → chain coeff soc_max
                - soc_min * float(np.sum(dual_yr))             # lower-bound RHS = E_cap*soc_min → chain coeff soc_min # lower-bound: raising E_cap RAISES the floor E_cap*soc_min, tightening the ≥ constraint → revenue LOSS → SUBTRACT.
            )                                                   # units: [USD/MWh] = revenue change per MWh of capacity

            # Degradation sensitivity, frozen dispatch.  The Shi cost is B·Φ(δ) with E entering ONLY inside δ (no leading-E factor — see subgrad formula B·τ·Φ'/2η), 
            #  so the product rule has ONE term: dCost_i/dE = B·Φ'(δ_i)·(dδ_i/dE),  dδ_i/dE = -δ_i/E  (frozen).
            # Φ' is taken from phi_shi_prime_with_stress — the SAME kernel the validated subgradient uses (incl. S_σ, S_T) — NOT a hand-rolled 
            # k3·k4·δ^(k4-1), to avoid silent divergence from the subgradient.
            if len(dods_yr) > 0:
                phi_prime_cyc = phi_shi_prime_with_stress(      # Φ'(δ)·S_σ·S_T — SAME kernel as the subgradient
                    dods_yr, socm_yr, T_cell_C, shi_fit.k3, shi_fit.k4   # (reused so the two can never silently diverge)
                )
                ddod_dEcap_yr     = -dods_yr / e_cap_yr         # dδ_i/dE = -δ_i/E  (the frozen-dispatch chain factor)
                dDegCost_dEcap_yr = e_cost_USD_per_MWh * factor_yr * float( # B · Σ_i [ count_i · Φ'(δ_i) · (dδ_i/dE) ] ×factor_yr lifts this single-year degradation sensitivity to the SAME lifetime-NPV basis as the revenue dual & objective
                    np.sum(cnts_yr * phi_prime_cyc * ddod_dEcap_yr)      # sum over cycles, weighted by rainflow count
                )                                               # units: [USD/MWh]; sign is typically NEGATIVE
            else:                                               # (bigger E → shallower δ → less degradation cost)
                dDegCost_dEcap_yr = 0.0                         # no cycles this year → no capacity-driven deg change

            if year == 1:   # STEP 1.5: validate degradation term once (frozen-dispatch FD)
                _fd_validate_degradation_term(
                    dods=dods_yr, counts=cnts_yr, soc_means=socm_yr,
                    e_cap=e_cap_yr, e_cost_USD_per_MWh=e_cost_USD_per_MWh,
                    factor=factor_yr, shi_fit=shi_fit, T_cell_C=T_cell_C,
                )
                
            # Annualised energy capex coefficient λ_E and the assembled outer gradient.
            lambda_E_yr   = e_cost_USD_per_MWh  # cost of one extra MWh of capacity, annuitised [USD/MWh] & objective's capex term is e_cost*e_cap1 (one-time, unscaled), so dObjective/dE_cap from capex = -e_cost. Matches kernel.
            dNPV_dEcap_yr = dRev_dEcap_yr - dDegCost_dEcap_yr - lambda_E_yr # dNPV/dE = (revenue gained) − (degradation cost change) − (capex of the extra capacity)
            rc_e_cap_yr   = float(os_yr.dual_prices.get("rc_e_cap1", 0.0)) # solver's own dObj/dE_cap — independent check

            # # ── STEP 1.3 BASIS DIAGNOSTIC (temporary — remove after validation) ──
            # _ratio = (rc_e_cap_yr / dNPV_dEcap_yr) if dNPV_dEcap_yr != 0.0 else float("nan")
            # print(f"[1.3] yr={year:>2d}  e_cap={e_cap_yr:6.1f}  "
            #       f"rc={rc_e_cap_yr:+12.1f}  dNPV={dNPV_dEcap_yr:+12.1f}  "
            #       f"ratio(rc/dNPV)={_ratio:+7.3f}")
            # print(f"      dRev={dRev_dEcap_yr:+12.1f}  "
            #       f"dDeg={dDegCost_dEcap_yr:+12.1f}  lambdaE={lambda_E_yr:+10.1f}  "
            #       f"factor={factor_yr:.3f}")
            
            grad_yr = (
                dDeg_dDoD_yr,                                        # [0] OLD fused (kept)
                float(np.mean(np.abs(sg_yr["subgrad_combined"]))),   # [1] mean_abs_subgrad
                mean_dod_yr,                                         # [2] mean_dod (weighted)
                float(np.mean(np.abs(dual_per_year_yr))),            # [3] mean_abs_dual
                float(e_cap_yr),                                     # [4] e_cap_eff
                float(sg_yr["cycle_coverage"]),                      # [5] cycle_coverage
                sg_yr["subgrad_combined"].copy(),                    # [6] full time series
                dRev_dEcap_yr,                                       # [7] NEW dRev/dE
                dDegCost_dEcap_yr,                                   # [8] NEW dDegCost/dE
                lambda_E_yr,                                         # [9] NEW λ_E
                dNPV_dEcap_yr,                                       # [10] NEW full gradient
                rc_e_cap_yr,                                         # [11] NEW rc cross-check
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
            "dDeg_dDoD_DEPRECATED":  round(grad[0], 6) if grad else None,   # old fused scalar, kept for continuity
            "mean_abs_subgrad":      round(grad[1], 6) if grad else None,
            "mean_dod":              round(grad[2], 4) if grad else None,
            "mean_abs_dual":         round(grad[3], 6) if grad else None,
            "e_cap_eff":             round(grad[4], 4) if grad else None,
            "cycle_coverage":        round(grad[5], 4) if grad else None,
            "dRev_dEcap":            round(grad[7], 4) if grad else None,    # validated revenue sensitivity
            "dDegCost_dEcap":        round(grad[8], 6) if grad else None,    # validated degradation sensitivity
            "lambda_E":              round(grad[9], 4) if grad else None,    # capex coefficient
            "dNPV_dEcap":            round(grad[10], 4) if grad else None,   # VALIDATED full outer gradient
            "rc_e_cap1":             round(grad[11], 4) if grad else None,   # solver reduced-cost cross-check
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
    print("WP2 BATTERY OPTIMIZATION + DEGRADATION  v5 (Castillo outer-loop)")
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

    run_label = _build_run_label(run_ts, PRICE_CSV, p_cap_MW, e_cap_MWh, soc_min, soc_max)

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
            f.write("WP2 BATTERY,  Shi (2018) + Xu calendar | v5 REPORT  (Castillo outer-loop)\n")
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
                        f"{'fd_cumul':>10}  {'dNPV/dEcap':>14}  {'mean_DoD%':>10}  {'replaced':>8}\n")
                f.write("-" * 72 + "\n")
                ann_fd   = multiyear["annual_fd"]
                ann_grad = multiyear.get("annual_gradient", [None] * len(ann_fd))
                for (yr, soh_p, fd_c, _n_rep), (fd_y, fd_cyc, fd_cal), grad in zip(
                    multiyear["soh_trajectory"], ann_fd, ann_grad
                ):
                    replaced = "YES" if yr in multiyear["replacement_years"] else ""
                    grad_str = f"{grad[10]:+.4e}" if grad is not None else "      N/A"   # [10] = validated dNPV/dEcap
                    dod_str  = f"{grad[2]*100:.1f}" if grad is not None else "   N/A"
                    f.write(f"{yr:>5d}  {soh_p:>8.3f}  {fd_y:>10.6f}  "
                            f"{fd_c:>10.6f}  {grad_str:>12}  {dod_str:>10}  {replaced:>8}\n")

                valid_grads = [g[10] for g in ann_grad if g is not None]
                if valid_grads:
                    f.write(f"\ndNPV/dEcap over {N_YEARS} yr:  "
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
            plot_multiyear_trajectory(multiyear, run_label,
                plots_dir=PLOTS_DIR, n_years=N_YEARS,
                eol_replacement=EOL_REPLACEMENT, show=show_plots)
            plot_gradient_analysis(multiyear, run_label,
                plots_dir=PLOTS_DIR, n_years=N_YEARS, show=show_plots)
            plot_subgradient_timeseries(multiyear, run_label,
                plots_dir=PLOTS_DIR, show=show_plots)

        if show_plots:
            plt.show()
        else:
            plt.close("all")

    print("\n" + "=" * 80)
    print("✓ COMPLETE,  v5 (Shi + Xu calendar | multi-year | replacement | Castillo gradient)")
    print("=" * 80)


if __name__ == "__main__":
    main()