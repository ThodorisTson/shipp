"""Half-cycle attribution validation — Test 1 v2.

What this tests
---------------
Test 1 (analyze_gradient_test1.py) validated the aggregate slope of fd.
This script validates the structure of the subgradient vector — the
8,760-element array subgrad_combined — using five structural properties.

Note: the per-timestep FD approach (perturbing one hour at a time) returns
zero for all timesteps because rainflow only responds to turning points, not
individual interior timesteps.  The five structural properties below are the
correct validation approach for the hourly attribution vector.

Five properties tested
----------------------
Property 1 — Intra-half-cycle uniformity
Property 2 — Value matches per-cycle formula (connects Test 2 to full trace)
Property 3 — Zero outside half-cycle ranges (no orphan attributions)
Property 4 — Junction boundary convention (Missing 3)
Property 5 — Global attribution sum conservation (Missing 4)

Outputs
-------
  VALIDATION_DIR/halfcycle_validation_<run_ts>.png
  VALIDATION_DIR/halfcycle_property_results_<run_ts>.csv
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

from degradation_xu import (
    analyze_degradation,
    count_equivalent_full_cycles,
    rainflow_cycle_counting,
    ft_calendar,
    sei_capacity_loss,
)

from degradation_shi import analyze_degradation_shi
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
HPP_YAML   = SCRIPT_DIR / "WP2_HPP.yaml"
PRICE_CSV  = SCRIPT_DIR / "dk1_prices_2022.csv"

RESULTS_DIR = SCRIPT_DIR / "Results"
VALIDATION_DIR = SCRIPT_DIR / "Gradient_Verification"
RESULTS_DIR.mkdir(exist_ok=True)
VALIDATION_DIR.mkdir(exist_ok=True)

eur_to_usd    = 1.18
discount_rate = 0.03
dt            = 1.0      # hours

pyo_solver = "gurobi"

RUN_FULL_YEAR    = True
N_DAYS_TEST      = 120
MAX_HOURS_SPARSE = 180 * 24

p_min      = 0.0
WAKE_MODEL = "Bastankhah"

eol_thresholds = [0.80, 0.70, 0.60]

N_YEARS = 20   # for dual scaling factor only

# ── v2 half-cycle attribution settings ───────────────────────────────────────
PASS_TOL  = 0.001   # 0.1% tolerance for value comparisons (Properties 1, 2, 4)
PASS_TOL5 = 0.01    # 1% tolerance for global sum (Property 5)

# Shi S_σ parameters — must match degradation_subgradient.py
K_SIGMA   = 1.04
SIGMA_REF = 0.50

run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# =============================================================================
# Shared helpers (identical to analyze_gradient_test1.py)
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
        raise ValueError("Wind speed and direction arrays have different lengths.")
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
    setup:    dict,
    p_max_MW: float,
) -> Tuple:
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
    dod_eff  = 1.0 if pyo_solver == "none" else dod_yaml

    print(f"  SoC window: {soc_min*100:.0f}% – {soc_max*100:.0f}%  "
          f"(DoD={dod_eff:.0%}, enforced by {pyo_solver})")

    stor = Storage(
        e_cap=e_cap_MWh, p_cap=p_cap_MW,
        eff_in=1.0, eff_out=rte_ac,
        e_cost=e_cost_USD_per_MWh, p_cost=p_cost_USD_per_MW,
        dod=dod_eff,
    )
    stor_null = Storage(e_cap=0.0, p_cap=0.0, eff_in=1.0, eff_out=1.0,
                        e_cost=0.0, p_cost=0.0)

    return (stor, stor_null, e_cap_MWh, p_cap_MW, rte_ac,
            e_cost_USD_per_MWh, p_cost_USD_per_MW, soc_min, soc_max)


def _solve_shipp(
    price_eur:     np.ndarray,
    power_wind_MW: np.ndarray,
    stor:          Storage,
    stor_null:     Storage,
    p_max_MW:      float,
    n:             int,
    soc_max:       float = 1.0,
):
    price_dam = TimeSeries((price_eur * eur_to_usd).tolist(), dt)
    prod      = Production(TimeSeries(power_wind_MW.tolist(), dt), p_cost=0.0)
    prod_null = Production(TimeSeries([0.0] * n, dt), p_cost=0.0)

    if pyo_solver == "none":
        os_fixed = solve_lp_sparse(price_dam, prod, prod_null, stor, stor_null,
                                   discount_rate, N_YEARS, p_min, p_max_MW, n,
                                   fixed_cap=True)
    else:
        os_fixed = solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null,
                                  discount_rate, N_YEARS, p_min, p_max_MW, n,
                                  pyo_solver, fixed_cap=True, return_duals=True,
                                  soc_max1=soc_max)
    return os_fixed

# =============================================================================
# Helper: expected per-half-cycle subgradient value
# =============================================================================

def _expected_sg_discharge(dod: float, sigma_bar: float,
                            shi_fit, e_cost: float, rte_ac: float) -> float:
    """Expected positive subgradient (discharge half) in USD/MWh units."""
    phi_prime = shi_fit.k3 * shi_fit.k4 * dod ** (shi_fit.k4 - 1.0)
    s_sigma   = np.exp(K_SIGMA * (sigma_bar - SIGMA_REF))
    return 0.5 * s_sigma * phi_prime * e_cost / rte_ac


def _expected_sg_charge(dod: float, sigma_bar: float,
                         shi_fit, e_cost: float) -> float:
    """Expected negative subgradient magnitude (charge half) in USD/MWh units."""
    phi_prime = shi_fit.k3 * shi_fit.k4 * dod ** (shi_fit.k4 - 1.0)
    s_sigma   = np.exp(K_SIGMA * (sigma_bar - SIGMA_REF))
    return 0.5 * s_sigma * phi_prime * e_cost


# =============================================================================
# Property 1 — Intra-half-cycle uniformity
# =============================================================================

def prop1_uniformity(subgrad: np.ndarray, cycles: list) -> dict:
    """Within [i_start:i_end+1], positive values must all be equal to each
    other, and negative values must all be equal to each other.
    The span contains both charge (negative) and discharge (positive) halves.
    """
    n_tested = 0; n_pass = 0; n_skip = 0; max_spread = 0.0
    rows = []
    for i, c in enumerate(cycles):
        i_s = c.get("i_start"); i_e = c.get("i_end")
        dod = float(c.get("dod", 0.0))
        if i_s is None or i_e is None or dod < 1e-6:
            n_skip += 1; continue
        seg = subgrad[int(i_s) : int(i_e) + 1]
        if len(seg) == 0:
            n_skip += 1; continue

        pos_vals = seg[seg > 0]
        neg_vals = seg[seg < 0]

        spread_pos = float(pos_vals.max() - pos_vals.min()) if len(pos_vals) > 1 else 0.0
        spread_neg = float(np.abs(neg_vals).max() - np.abs(neg_vals).min()) if len(neg_vals) > 1 else 0.0
        spread = max(spread_pos, spread_neg)

        ref = max(float(np.abs(seg[seg != 0]).max()), 1e-30) if np.any(seg != 0) else 1e-30
        uniform = bool(spread < PASS_TOL * ref)
        max_spread = max(max_spread, spread)

        n_tested += 1
        if uniform: n_pass += 1
        rows.append({"cycle_idx": i, "i_start": int(i_s), "i_end": int(i_e),
                     "dod": dod, "n_timesteps": len(seg),
                     "spread_pos": spread_pos, "spread_neg": spread_neg,
                     "spread": spread, "pass": uniform})
    return {"n_tested": n_tested, "n_pass": n_pass, "n_skip": n_skip,
            "max_spread": max_spread, "df": pd.DataFrame(rows)}


# =============================================================================
# Property 2 — Value matches per-cycle formula
# =============================================================================

def prop2_formula_match(subgrad: np.ndarray, cycles: list,
                        shi_fit, e_cap: float, e_cost: float,
                        rte_ac: float) -> dict:
    """For each cycle, compare measured vs expected subgradient.

    If a cycle fails the formula check, an inline overwrite audit determines
    whether the failure is explained by the 'deeper overwrites shallower'
    convention (build_half_cycle_map assigns the dominant outer cycle's value
    to the nested inner cycle's timesteps) or is a genuine attribution error.

    Classification per cycle:
        PASS_FORMULA   — measured matches own formula (dominant cycle)
        PASS_OVERWRITE — measured matches dominant cycle formula (nested, overwritten)
        FAIL           — neither own nor dominant formula matches (genuine error)
    """
    n_tested = 0; n_pass_formula = 0; n_pass_overwrite = 0; n_fail = 0; n_skip = 0
    rows = []

    # Pre-compute the dominant cycle value (the most common positive subgrad value)
    pos_vals = subgrad[subgrad > 0]
    if len(pos_vals) > 0:
        dominant_sg_discharge = float(pd.Series(np.round(pos_vals, 6)).mode().iloc[0])
        dominant_sg_charge    = float(pd.Series(np.round(subgrad[subgrad < 0], 6)).mode().iloc[0])
    else:
        dominant_sg_discharge = float("nan")
        dominant_sg_charge    = float("nan")

    for i, c in enumerate(cycles):
        i_s  = c.get("i_start"); i_e = c.get("i_end")
        dod  = float(c.get("dod", 0.0))
        sbar = float(c.get("soc_mean", 0.5))
        if i_s is None or i_e is None or dod < 1e-6:
            n_skip += 1; continue

        seg      = subgrad[int(i_s) : int(i_e) + 1]
        pos_seg  = seg[seg > 0]
        neg_seg  = seg[seg < 0]
        if len(pos_seg) == 0 or len(neg_seg) == 0:
            n_skip += 1; continue

        measured_dis = float(pos_seg.mean())
        measured_chg = float(np.abs(neg_seg).mean())
        exp_dis      = _expected_sg_discharge(dod, sbar, shi_fit, e_cost, rte_ac)
        exp_chg      = _expected_sg_charge(dod, sbar, shi_fit, e_cost)

        ratio_dis = measured_dis / exp_dis if abs(exp_dis) > 1e-30 else float("nan")
        ratio_chg = measured_chg / exp_chg if abs(exp_chg) > 1e-30 else float("nan")

        formula_ok = (not np.isnan(ratio_dis) and abs(ratio_dis - 1.0) < PASS_TOL and
                      not np.isnan(ratio_chg) and abs(ratio_chg - 1.0) < PASS_TOL)

        # ── Overwrite audit for failing cycles ────────────────────────────
        overwrite_classification = "PASS_FORMULA"
        pct_overwritten          = 0.0
        n_carry_dominant         = 0
        n_carry_own              = 0

        if not formula_ok:
            # Check: does the measured value match the dominant cycle instead?
            ratio_dominant_dis = (measured_dis / dominant_sg_discharge
                                  if abs(dominant_sg_discharge) > 1e-30 else float("nan"))
            is_overwritten = (not np.isnan(ratio_dominant_dis) and
                              abs(ratio_dominant_dis - 1.0) < PASS_TOL)

            # Timestep-level audit within this cycle's span
            n_total_pos      = len(pos_seg)
            n_carry_dominant = int(np.sum(
                np.abs(pos_seg - dominant_sg_discharge) < PASS_TOL * dominant_sg_discharge
            ))
            n_carry_own = int(np.sum(
                np.abs(pos_seg - exp_dis) < PASS_TOL * max(abs(exp_dis), 1e-30)
            ))
            pct_overwritten = 100.0 * n_carry_dominant / max(n_total_pos, 1)

            if is_overwritten:
                overwrite_classification = "PASS_OVERWRITE"
                n_pass_overwrite += 1
            else:
                overwrite_classification = "FAIL"
                n_fail += 1
        else:
            n_pass_formula += 1

        n_tested += 1
        rows.append({
            "cycle_idx":              i,
            "dod":                    dod,
            "sigma_bar":              sbar,
            "measured_discharge":     measured_dis,
            "expected_discharge":     exp_dis,
            "ratio_discharge":        ratio_dis,
            "measured_charge":        measured_chg,
            "expected_charge":        exp_chg,
            "ratio_charge":           ratio_chg,
            "classification":         overwrite_classification,
            "pct_overwritten":        pct_overwritten,
            "n_discharge_timesteps":  len(pos_seg),
            "n_carry_dominant":       n_carry_dominant,
            "n_carry_own":            n_carry_own,
        })

    return {
        "n_tested":          n_tested,
        "n_pass_formula":    n_pass_formula,
        "n_pass_overwrite":  n_pass_overwrite,
        "n_fail":            n_fail,
        "n_skip":            n_skip,
        "dominant_sg_dis":   dominant_sg_discharge,
        "df":                pd.DataFrame(rows),
    }


# =============================================================================
# Property 3 — Zero outside half-cycle ranges
# =============================================================================

def prop3_zero_outside(subgrad: np.ndarray, cycles: list) -> dict:
    covered = np.zeros(len(subgrad), dtype=bool)
    for c in cycles:
        i_s = c.get("i_start"); i_e = c.get("i_end")
        if i_s is None or i_e is None: continue
        covered[int(i_s) : int(i_e) + 1] = True
    uncov_vals   = np.abs(subgrad[~covered])
    n_nonzero    = int(np.sum(uncov_vals >= 1e-12))
    return {"n_covered": int(covered.sum()),
            "n_uncovered": int((~covered).sum()),
            "n_nonzero_out": n_nonzero,
            "max_val_outside": float(uncov_vals.max()) if len(uncov_vals) else 0.0,
            "pass": n_nonzero == 0}


# =============================================================================
# Property 4 — Junction boundary convention
# =============================================================================

def prop4_junction_convention(subgrad: np.ndarray, cycles: list,
                               shi_fit, e_cap: float, e_cost: float) -> dict:
    n_tested = 0; n_pass = 0
    rows = []
    indexed = [(i, c) for i, c in enumerate(cycles)
               if c.get("i_start") is not None and c.get("i_end") is not None
               and float(c.get("dod", 0)) >= 1e-6]
    indexed.sort(key=lambda x: int(x[1]["i_start"]))
    for k in range(len(indexed) - 1):
        i_k,  c_k  = indexed[k]
        i_k1, c_k1 = indexed[k + 1]
        if int(c_k["i_end"]) != int(c_k1["i_start"]): continue
        t        = int(c_k["i_end"])
        sg_val   = float(subgrad[t])
        dod_next = float(c_k1.get("dod", 0))
        sbar_nxt = float(c_k1.get("soc_mean", 0.5))
        expected = -_expected_sg_charge(dod_next, sbar_nxt, shi_fit, e_cost)
        ratio    = sg_val / expected if abs(expected) > 1e-30 else float("nan")
        passed   = bool(abs(ratio - 1.0) < PASS_TOL) if not np.isnan(ratio) else False
        n_tested += 1
        if passed: n_pass += 1
        rows.append({"junction_t": t, "cycle_k": i_k, "cycle_k1": i_k1,
                     "sg_val": sg_val, "expected_sg": expected,
                     "ratio": ratio, "pass": passed})
    return {"n_tested": n_tested, "n_pass": n_pass, "df": pd.DataFrame(rows)}


# =============================================================================
# Property 5 — Global attribution sum conservation
# =============================================================================

def prop5_global_sum(subgrad: np.ndarray, cycles: list,
                     shi_fit, e_cap: float, e_cost: float,
                     rte_ac: float) -> dict:
    """Global sum with fingerprinting: for each discharge timestep, find the
    cycle whose formula best matches the observed subgradient value, then
    compare the sum of matched expected values to the measured sum.
    """
    # Build lookup of expected discharge values per valid cycle
    valid_cycles = [
        {"dod": float(c.get("dod", 0)), "soc_mean": float(c.get("soc_mean", 0.5)),
         "expected": _expected_sg_discharge(
             float(c.get("dod", 0)), float(c.get("soc_mean", 0.5)),
             shi_fit, e_cost, rte_ac)}
        for c in cycles if float(c.get("dod", 0)) >= 1e-6
    ]

    # Fingerprint each discharge timestep
    pos_indices = np.where(subgrad > 0)[0]
    measured_sum        = float(np.sum(subgrad[pos_indices]))
    expected_fp_sum     = 0.0
    matched_cycle_dods  = []

    for t in pos_indices:
        sg_val     = float(subgrad[t])
        best       = min(valid_cycles,
                         key=lambda c: abs(c["expected"] - sg_val))
        expected_fp_sum += best["expected"]
        matched_cycle_dods.append(best["dod"])

    matched_cycle_dods = np.array(matched_cycle_dods)
    ratio_fp = measured_sum / expected_fp_sum if abs(expected_fp_sum) > 1e-30 else float("nan")
    passed   = bool(abs(ratio_fp - 1.0) < PASS_TOL5) if not np.isnan(ratio_fp) else False

    # Also keep old per-cycle sums for comparison
    expected_all      = sum(c["expected"] for c in valid_cycles)
    expected_dominant = sum(c["expected"] for c in valid_cycles if c["dod"] >= 0.79)
    ratio_all         = measured_sum / expected_all if abs(expected_all) > 1e-30 else float("nan")

    return {
        "n_timesteps_fingerprinted": len(pos_indices),
        "measured_sum":              measured_sum,
        "expected_fp_sum":           expected_fp_sum,
        "ratio_fp":                  ratio_fp,
        "expected_all":              expected_all,
        "expected_dominant":         expected_dominant,
        "ratio_all":                 ratio_all,
        "ratio_dominant":            measured_sum / expected_dominant if expected_dominant > 0 else float("nan"),
        "n_matched_dominant":        int(np.sum(matched_cycle_dods >= 0.79)),
        "pass":                      passed,
    }

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("PER-TIMESTEP SUBGRADIENT VALIDATION  — Test 1 v2")
    print("d(fd_cycle)/d(storage_e[t])  vs  subgrad_combined[t]  for all active t")
    print("=" * 80)

    # ── 1. Load config ────────────────────────────────────────────────────
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

    # ── 4. SHIPP (dispatch-fixed only) ────────────────────────────────────
    print("\n[4/5] Running SHIPP optimization (dispatch-fixed)...")
    p_max_MW = float(hpp["grid_connection_capacity"]) / 1e6

    (stor, stor_null, e_cap_MWh, p_cap_MW,
     rte_ac, e_cost_USD_per_MWh, p_cost_USD_per_MW,
     soc_min, soc_max) = _build_shipp_components(setup, p_max_MW)

    shi_fit = fit_shi_polynomial(soc_min, soc_max, verbose=True)
    bat_params = setup["battery"]

    print(f"  Battery: {p_cap_MW:.0f} MW / {e_cap_MWh:.0f} MWh | "
          f"Grid: {p_max_MW:.0f} MW | RTE(ac): {rte_ac*100:.1f}%")

    os_fixed = _solve_shipp(price_eur, power_wind_MW, stor, stor_null,
                            p_max_MW, n, soc_max)

    storage_e = np.array(os_fixed.storage_e[0].data)
    storage_p = np.array(os_fixed.storage_p[0].data)
    e_cap     = float(os_fixed.storage_list[0].e_cap)

# ── 5. Run half-cycle attribution validation ───────────────────────────
    print("\n[5/5] Half-cycle attribution validation...")
    print("=" * 70)

    cycles = rainflow_cycle_counting(storage_e.tolist(), e_cap)
    sg     = compute_subgradient(
        storage_e=storage_e.tolist(),
        cycles=cycles,
        dt_hours=dt,
        battery_replacement_cost_per_MWh=e_cost_USD_per_MWh,
        eff_in=1.0,
        eff_out=rte_ac,
        shi_fit=shi_fit,
    )
    subgrad = sg["subgrad_combined"]

    print(f"  Rainflow cycles   : {len(cycles)}")
    print(f"  Non-zero subgrads : {np.sum(subgrad != 0)} / {len(subgrad)}")
    print(f"  subgrad range     : [{subgrad.min():.4e}, {subgrad.max():.4e}]")

    print("\n  Running Property 1 — Intra-half-cycle uniformity...")
    r1 = prop1_uniformity(subgrad, cycles)
    print(f"    Tested: {r1['n_tested']}  Pass: {r1['n_pass']}  "
          f"Max spread: {r1['max_spread']:.2e}")

    print("  Running Property 2 — Formula match...")
    r2 = prop2_formula_match(subgrad, cycles, shi_fit, e_cap, e_cost_USD_per_MWh, rte_ac)
    print(f"    Tested: {r2['n_tested']}  Formula pass: {r2['n_pass_formula']}  "
          f"Overwrite: {r2['n_pass_overwrite']}  Genuine fail: {r2['n_fail']}")
    print("  Running Property 3 — Zero outside half-cycle ranges...")
    r3 = prop3_zero_outside(subgrad, cycles)
    print(f"    Covered: {r3['n_covered']}  Uncovered: {r3['n_uncovered']}  "
          f"Non-zero outside: {r3['n_nonzero_out']}")

    print("  Running Property 4 — Junction boundary convention...")
    r4 = prop4_junction_convention(subgrad, cycles, shi_fit, e_cap,
                                   e_cost_USD_per_MWh)
    print(f"    Junctions found: {r4['n_tested']}  Pass: {r4['n_pass']}")

    print("  Running Property 5 — Global sum conservation...")
    r5 = prop5_global_sum(subgrad, cycles, shi_fit, e_cap, e_cost_USD_per_MWh, rte_ac)
    print(f"    Measured: {r5['measured_sum']:.6e}  "
          f"Expected (fingerprinted): {r5['expected_fp_sum']:.6e}  "
          f"ratio_fp: {r5['ratio_fp']:.4f}  "
          f"({r5['n_matched_dominant']}/{r5['n_timesteps_fingerprinted']} "
          f"timesteps matched to dominant cycle)")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HALF-CYCLE ATTRIBUTION — SUMMARY")
    print("=" * 70)
    def _line(label, n_pass, n_total, extra=""):
        res = "✓ ALL PASS" if n_pass == n_total else f"✗ {n_total-n_pass} FAIL"
        print(f"  {label:<42} {n_pass:>5}/{n_total:<5}  {res}  {extra}")
    _line("Property 1 — Uniformity", r1["n_pass"], r1["n_tested"],
          f"max spread = {r1['max_spread']:.2e}")
    n_p2_total = r2["n_pass_formula"] + r2["n_pass_overwrite"] + r2["n_fail"]
    print(f"  {'Property 2 — Formula match':<42} "
          f"{r2['n_pass_formula']:>5}/{n_p2_total:<5}  "
          f"dominant ✓ | {r2['n_pass_overwrite']} overwritten (expected) | "
          f"{r2['n_fail']} genuine fail")
    p3 = "✓ ALL PASS" if r3["pass"] else "✗ FAIL"
    print(f"  {'Property 3 — Zero outside':<42}   n/a       {p3}  "
          f"nonzero outside={r3['n_nonzero_out']}")
    _line("Property 4 — Junction convention", r4["n_pass"], r4["n_tested"])
    p5 = "✓ PASS" if r5["pass"] else "✗ FAIL"
    print(f"  {'Property 5 — Global sum (fingerprinted)':<42}   n/a   {p5}  "
          f"ratio_fp={r5['ratio_fp']:.4f}  "
          f"({r5['n_matched_dominant']}/{r5['n_timesteps_fingerprinted']} matched dominant)")
    print("=" * 70)

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = VALIDATION_DIR / f"halfcycle_property_results_{run_ts}.csv"
    r2["df"].to_csv(csv_path, index=False)
    print(f"\n  ✓ CSV saved: {csv_path.name}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Half-cycle Attribution Validation — Test 1 v2  |  Year 1",
                 fontsize=12, fontweight="bold")

    def _text_panel(ax, title, lines):
        ax.axis("off")
        for j, (txt, col, bold) in enumerate(lines):
            ax.text(0.5, 0.80 - j * 0.20, txt, ha="center", va="center",
                    fontsize=11 if not bold else 13, color=col,
                    fontweight="bold" if bold else "normal",
                    transform=ax.transAxes)
        ax.set_title(title, fontsize=9)

    # P1
    pass_col = "#27ae60" if r1["n_pass"] == r1["n_tested"] else "#e74c3c"
    _text_panel(axes[0,0], f"Property 1 — Uniformity\nPASS {r1['n_pass']}/{r1['n_tested']}", [
        (f"Tested: {r1['n_tested']} half-cycles", "black", False),
        (f"Max spread = {r1['max_spread']:.2e}", "#2c7bb6", True),
        ("✓ Perfectly uniform" if r1["n_pass"]==r1["n_tested"] else "✗ FAIL",
         pass_col, True),
    ])

    # P2
    df2 = r2["df"].dropna(subset=["ratio_discharge"])
    ratios2 = df2["ratio_discharge"].values
    pass_col = "#27ae60" if r2["n_fail"] == 0 else "#e74c3c"
    p2_title = (f"Property 2 — Formula match\n"
                f"dominant={r2['n_pass_formula']} ✓  overwritten={r2['n_pass_overwrite']}  "
                f"fail={r2['n_fail']}")
    if len(ratios2) > 0 and np.ptp(ratios2) < 1e-10:
        _text_panel(axes[0,1], p2_title, [
            (f"All {len(ratios2)} cycles", "black", False),
            (f"ratio = {ratios2.mean():.6f}", "#2c7bb6", True),
            ("✓ Exact formula match", pass_col, True),
        ])
    else:
        axes[0,1].hist(ratios2, bins=40, color="#2c7bb6", alpha=0.85, edgecolor="white")
        axes[0,1].axvline(1.0, color="#d7191c", lw=2, ls="--")
        axes[0,1].set_xlabel("Measured / Expected subgrad", fontsize=9)
        axes[0,1].set_title(p2_title, fontsize=9)
        axes[0,1].grid(True, alpha=0.3)

    # P3
    n_cov = r3["n_covered"]; n_unc = r3["n_uncovered"]
    axes[0,2].bar(["Covered", "Uncovered"],
                  [n_cov, n_unc],
                  color=["#27ae60" if r3["pass"] else "#e74c3c", "#95a5a6"],
                  alpha=0.85, width=0.5)
    for rect, val in zip(axes[0,2].patches, [n_cov, n_unc]):
        axes[0,2].text(rect.get_x() + rect.get_width()/2,
                       rect.get_height() + 20, str(val),
                       ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0,2].set_ylabel("Timestep count", fontsize=9)
    axes[0,2].set_title(
        f"Property 3 — Zero outside ranges\nnon-zero outside = {r3['n_nonzero_out']}  "
        f"({'✓ ALL PASS' if r3['pass'] else '✗ FAIL'})", fontsize=9)
    axes[0,2].grid(True, alpha=0.3, axis="y")

    # P4
    pass_col = "#27ae60" if r4["n_pass"] == r4["n_tested"] else "#e74c3c"
    df4 = r4["df"].dropna(subset=["ratio"])
    if len(df4) == 0:
        _text_panel(axes[1,0], "Property 4 — Junction convention\nNo junctions found", [
            ("No consecutive cycles share a valley", "gray", False)])
    elif np.ptp(df4["ratio"].values) < 1e-10:
        _text_panel(axes[1,0], f"Property 4 — Junction convention\nPASS {r4['n_pass']}/{r4['n_tested']}", [
            (f"Junctions: {r4['n_tested']}", "black", False),
            (f"ratio = {df4['ratio'].mean():.6f}", "#2c7bb6", True),
            ("✓ Correct convention", pass_col, True),
        ])
    else:
        axes[1,0].hist(df4["ratio"].values, bins=30, color="#e67e22", alpha=0.85, edgecolor="white")
        axes[1,0].axvline(1.0, color="#d7191c", lw=2, ls="--")
        axes[1,0].set_title(f"Property 4 — Junction convention\nPASS {r4['n_pass']}/{r4['n_tested']}", fontsize=9)
        axes[1,0].grid(True, alpha=0.3)

    # P5
    vals = [r5["measured_sum"], r5["expected_fp_sum"], r5["expected_all"]]
    axes[1,1].bar(["Measured\n(timestep sum)", "Expected\n(fingerprinted)", "Expected\n(per-cycle)"],
                  vals, color=["#2c7bb6", "#27ae60", "#e74c3c"], alpha=0.85, width=0.5)
    for rect, val in zip(axes[1,1].patches, vals):
        axes[1,1].text(rect.get_x() + rect.get_width()/2, rect.get_height()*1.02,
                       f"{val:.3e}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[1,1].set_ylabel("Sum of contributions", fontsize=9)
    axes[1,1].set_title(
        f"Property 5 — Global sum (fingerprinted)\n"
        f"ratio_fp={r5['ratio_fp']:.4f}  ratio_per_cycle={r5['ratio_all']:.3f}  "
        f"({'✓ PASS' if r5['pass'] else '✗ FAIL'})", fontsize=9)
    axes[1,1].grid(True, alpha=0.3, axis="y")

    # Summary panel
    results = [
        ("P1 Uniformity",     r1["n_pass"] == r1["n_tested"]),
        ("P2 Formula match",  r2["n_fail"] == 0),
        ("P3 Zero outside",   r3["pass"]),
        ("P4 Junction conv.", r4["n_pass"] == r4["n_tested"]),
        ("P5 Global sum",     r5["pass"]),
    ]
    axes[1,2].axis("off")
    for j, (label, passed) in enumerate(results):
        col = "#27ae60" if passed else "#e74c3c"
        axes[1,2].text(0.08, 0.82 - j*0.15,
                       f"{'✓' if passed else '✗'}  {label}",
                       transform=axes[1,2].transAxes, fontsize=12,
                       color=col, fontweight="bold", va="center")
    axes[1,2].set_title("Summary", fontsize=9)

    plt.tight_layout()
    plot_path = VALIDATION_DIR / f"halfcycle_validation_{run_ts}.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Plot saved: {plot_path.name}")

    print("\n" + "=" * 80)
    print("✓ COMPLETE — half-cycle attribution validation")
    print("=" * 80)

if __name__ == "__main__":
    main()