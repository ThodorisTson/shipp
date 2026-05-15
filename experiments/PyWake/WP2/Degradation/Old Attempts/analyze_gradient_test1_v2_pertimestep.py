"""Per-timestep subgradient validation — Test 1 v2.

What this tests
---------------
Test 1 (analyze_gradient_test1.py) validated the *aggregate* slope of fd
across the full 8,760-hour year.  This script goes one level deeper: for
each active timestep t (where subgrad_combined[t] != 0), it numerically
computes the partial derivative of fd_cycle with respect to storage_e[t]
alone — perturbing just that one hour, leaving all others fixed — and checks
whether it matches the subgradient value the code assigned to that hour.

This tests the rainflow half-cycle attribution: not just whether the Phi'(δ)
formula is correct (Test 2 / analyze_rainflow_year1_cycles.py covers that),
but whether the correct timesteps are being assigned the correct values.

Method
------
For each active timestep t:

    FD_t = (fd_cycle(e + eps×1_t) - fd_cycle(e - eps×1_t)) / (2 × eps)

where 1_t is a vector that is 1 at position t and 0 everywhere else.
This is compared to subgrad_combined[t].  The ratio FD_t / subgrad_combined[t]
should equal 1.0 if the attribution is correct.

Note: fd_cycle only (not fd_total) is used.  Calendar aging has zero
dependence on individual timesteps by design (dual-Phi architecture), so
including it would add noise without signal.

Outputs
-------
  VALIDATION_DIR/pertimestep_scatter_<run_ts>.png  — subgrad vs FD per timestep
  VALIDATION_DIR/pertimestep_ratio_hist_<run_ts>.png — ratio distribution
  VALIDATION_DIR/pertimestep_results_<run_ts>.csv  — full per-timestep table
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

# ── v2 per-timestep settings ──────────────────────────────────────────────────
EPS_TIMESTEP  = 1.0    # MWh — single-element perturbation size
MAX_TIMESTEPS = None   # None = test all non-zero subgradient timesteps (~1,751)
PASS_TOL      = 0.01   # ratio must be within 1% of 1.0 to count as PASS

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
# Per-timestep validation functions
# =============================================================================

def _fd_cycle_at_t(
    storage_e:  np.ndarray,
    storage_p:  np.ndarray,
    e_cap:      float,
    t:          int,
    eps:        float,
    bat_params: Dict,
    shi_fit,
) -> float:
    """Central difference of fd_cycle with respect to storage_e[t].

    Perturbs only position t by ±eps MWh, holds e_cap and all other
    timesteps fixed.  Returns d(fd_cycle)/d(storage_e[t]).

    Uses fd_cycle only — calendar term has zero timestep dependence by design.
    """
    def _fd_cycle(e_arr: np.ndarray) -> float:
        degr = analyze_degradation_shi(
            storage_p.tolist(), e_arr.tolist(), e_cap, bat_params,
            shi_fit=shi_fit, T_cell_C=25.0, dt_hours=dt,
            eol_thresholds=eol_thresholds,
        )
        return float(degr["fd_shi"])

    e_hi = storage_e.copy(); e_hi[t] += eps
    e_lo = storage_e.copy(); e_lo[t] -= eps

    return (_fd_cycle(e_hi) - _fd_cycle(e_lo)) / (2.0 * eps)


def _get_active_timesteps(subgrad: np.ndarray) -> np.ndarray:
    """Return indices where subgrad_combined is non-zero."""
    return np.where(subgrad != 0.0)[0]


def _run_pertimestep_validation(
    storage_e:  np.ndarray,
    storage_p:  np.ndarray,
    e_cap:      float,
    subgrad:    np.ndarray,
    bat_params: Dict,
    shi_fit,
    eps:        float,
    max_ts:     Optional[int],
) -> pd.DataFrame:
    """Run the per-timestep FD check across all active timesteps.

    For each active timestep t:
        FD_t  = d(fd_cycle)/d(storage_e[t])  [numerical, central difference]
        sg_t  = subgrad_combined[t]           [analytical, from compute_subgradient]
        ratio = FD_t / sg_t                   [should be 1.0]

    Returns a DataFrame with columns:
        t, subgrad_t, fd_partial_t, ratio, pass
    """
    active = _get_active_timesteps(subgrad)
    if max_ts is not None:
        active = active[:max_ts]

    n_test = len(active)
    print(f"  Active (non-zero) timesteps : {n_test}")
    print(f"  Perturbation size           : ±{eps:.1f} MWh")
    print(f"  Pass tolerance              : |ratio − 1| < {PASS_TOL*100:.0f}%")
    print(f"  Running {n_test} central difference evaluations (2 calls each)...")

    t_start = time.perf_counter()

    rows = []
    for i, t in enumerate(active):
        fd_t  = _fd_cycle_at_t(storage_e, storage_p, e_cap, int(t),
                                eps, bat_params, shi_fit)
        sg_t  = float(subgrad[t])
        ratio = fd_t / sg_t if abs(sg_t) > 1e-30 else float("nan")
        passed = abs(ratio - 1.0) < PASS_TOL if not np.isnan(ratio) else False

        rows.append({
            "t":            int(t),
            "subgrad_t":    sg_t,
            "fd_partial_t": fd_t,
            "ratio":        ratio,
            "pass":         passed,
        })

        if (i + 1) % 200 == 0 or (i + 1) == n_test:
            elapsed = time.perf_counter() - t_start
            print(f"    {i+1:>5}/{n_test}  elapsed: {elapsed:.1f}s  "
                  f"last ratio: {ratio:.6f}")

    elapsed = time.perf_counter() - t_start
    print(f"  ✓ Done in {elapsed:.1f}s")

    return pd.DataFrame(rows)


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

    # ── 5. Compute base subgradient ───────────────────────────────────────
    print("\n[5/5] Per-timestep subgradient validation...")
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

    print(f"\n  subgrad_combined range : [{subgrad.min():.4e}, {subgrad.max():.4e}]")
    print(f"  Non-zero timesteps     : {np.sum(subgrad != 0)} / {len(subgrad)}")

    # ── Run per-timestep validation ───────────────────────────────────────
    print()
    df = _run_pertimestep_validation(
        storage_e=storage_e,
        storage_p=storage_p,
        e_cap=e_cap,
        subgrad=subgrad,
        bat_params=bat_params,
        shi_fit=shi_fit,
        eps=EPS_TIMESTEP,
        max_ts=MAX_TIMESTEPS,
    )

    # ── Print summary ─────────────────────────────────────────────────────
    n_tested = len(df)
    n_pass   = df["pass"].sum()
    n_fail   = n_tested - n_pass
    mean_r   = df["ratio"].mean()
    std_r    = df["ratio"].std()
    max_dev  = (df["ratio"] - 1.0).abs().max()

    print("\n" + "=" * 70)
    print("PER-TIMESTEP VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Timesteps tested  : {n_tested}")
    print(f"  PASS (|r−1|<{PASS_TOL*100:.0f}%) : {n_pass}  ({100*n_pass/n_tested:.1f}%)")
    print(f"  FAIL              : {n_fail}")
    print(f"  Mean ratio        : {mean_r:.6f}  (target = 1.0)")
    print(f"  Std of ratio      : {std_r:.2e}")
    print(f"  Max |ratio − 1|   : {max_dev:.2e}")
    if n_fail == 0:
        print(f"  ✓ ALL PASS")
    else:
        print(f"  ✗ {n_fail} FAILURES — check df for details")
        fails = df[~df["pass"]].head(10)
        print(fails.to_string(index=False))
    print("=" * 70)

    # ── Save results CSV ──────────────────────────────────────────────────
    csv_path = VALIDATION_DIR / f"pertimestep_results_{run_ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Results CSV: {csv_path.name}")

    # ── Plot A: scatter — subgrad vs FD partial ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    # Separate charge (positive subgrad) and discharge (negative subgrad)
    pos = df[df["subgrad_t"] > 0]
    neg = df[df["subgrad_t"] < 0]

    ax.scatter(pos["subgrad_t"], pos["fd_partial_t"],
               s=12, alpha=0.6, color="#2c7bb6", label="Charge timesteps")
    ax.scatter(neg["subgrad_t"], neg["fd_partial_t"],
               s=12, alpha=0.6, color="#d7191c", label="Discharge timesteps")

    # 45° reference line through the data range
    all_vals = np.concatenate([df["subgrad_t"].values, df["fd_partial_t"].values])
    vmin, vmax = all_vals.min(), all_vals.max()
    pad = (vmax - vmin) * 0.05
    ref = np.linspace(vmin - pad, vmax + pad, 80)
    ax.plot(ref, ref, 'k--', lw=1.2, alpha=0.6, label='45° line (perfect match)')

    ax.set_xlabel("subgrad_combined[t]  [analytical]", fontsize=10)
    ax.set_ylabel("d(fd_cycle)/d(storage_e[t])  [FD numerical]", fontsize=10)
    ax.set_title(
        f"Per-timestep subgradient validation — Year 1\n"
        f"n = {n_tested} active timesteps   "
        f"mean ratio = {mean_r:.6f}   max |r−1| = {max_dev:.2e}",
        fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_path = VALIDATION_DIR / f"pertimestep_scatter_{run_ts}.png"
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    print(f"  ✓ Plot saved: {scatter_path.name}")

    # ── Plot B: histogram of ratios ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["ratio"], bins=50, color="#2c7bb6", alpha=0.85, edgecolor="white")
    ax.axvline(1.0,  color="#d7191c", lw=2.0, ls="--", label="Target ratio = 1.0")
    ax.axvline(mean_r, color="#2ecc71", lw=1.5, ls=":",
               label=f"Mean ratio = {mean_r:.6f}")
    ax.set_xlabel("FD partial / subgrad_combined  [ratio]", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(
        f"Per-timestep ratio distribution — Year 1\n"
        f"n = {n_tested}   std = {std_r:.2e}   max |r−1| = {max_dev:.2e}",
        fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = VALIDATION_DIR / f"pertimestep_ratio_hist_{run_ts}.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"  ✓ Plot saved: {hist_path.name}")

    print("\n" + "=" * 80)
    print("✓ COMPLETE — per-timestep subgradient validation")
    print("=" * 80)


if __name__ == "__main__":
    main()
