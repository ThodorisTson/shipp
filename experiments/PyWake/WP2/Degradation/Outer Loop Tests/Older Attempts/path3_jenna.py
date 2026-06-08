"""
Path 3 — Jenna/SHIPP Version: Degradation-Aware NLP Dispatch
=============================================================

Uses SHIPP kernel functions (build_lp_obj_revenues, build_lp_cst_sparse) for
the LP baseline, then adds real rainflow Shi degradation via scipy minimize.

Variable layout (lp_alt formulation, 5*n + 6 total):
    x[0:n]          stor1_p     battery power (positive=discharge)
    x[n:2*n]        stor2_p     null storage (always 0)
    x[2*n:3*n]      p_curtailed curtailed power (always 0, no production)
    x[3*n:4*n+1]    stor1_e     battery energy state (n+1 values)
    x[4*n+1:5*n+2]  stor2_e     null storage energy (always 0)
    x[5*n+2:5*n+6]  capacities  [p_cap1, e_cap1, p_cap2, e_cap2] (fixed)

Usage:
    python path3_jenna.py --year 2022 --month 7
    python path3_jenna.py --year 2019 --month 7 --solver slsqp --max-iter 200
    python path3_jenna.py --year 2019 --month 7 --tr-radius 0.1 --max-iter 50
    python path3_jenna.py --test

Author: Thodoris Tsonopoulos -- MSc Thesis, TU Delft Wind Energy
Based on: Jenna Iori, example_degradation.py (SHIPP feature_degradation)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linprog, minimize, LinearConstraint

# -- SHIPP imports ---------------------------------------------------------
from shipp.kernel import build_lp_obj_revenues, build_lp_cst_sparse
from shipp.components import Storage

# -- Degradation model imports ---------------------------------------------
_SCRIPT_DIR = Path(__file__).parent
_PARENT_DIR = _SCRIPT_DIR.parent

for p in [str(_SCRIPT_DIR), str(_PARENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from degradation_xu import (
    XuModelParams,
    XU_LMO,
    ShiPolynomialFit,
    fit_shi_polynomial,
    phi_shi,
    phi_shi_prime,
    rainflow_cycle_counting,
    s_soc,
    s_temp,
    ft_calendar,
)

_RUNNER_NAME = "path3_jenna"


# ==========================================================================
# 1. Degradation functions  (from validated prototype)
# ==========================================================================

def compute_f_deg(
    storage_e: np.ndarray,
    e_cap: float,
    shi_fit: ShiPolynomialFit,
    T_C: float = 25.0,
    p: XuModelParams = XU_LMO,
) -> Tuple[float, List[Dict]]:
    """Total Shi cycling degradation."""
    cycles = rainflow_cycle_counting(storage_e, e_cap)
    k3, k4 = shi_fit.k3, shi_fit.k4
    S_T = float(s_temp(T_C, p))

    f_total = 0.0
    for c in cycles:
        delta = c["dod"]
        if delta < 1e-12:
            continue
        phi_val = float(phi_shi(delta, k3, k4))
        S_sigma = float(s_soc(c["soc_mean"], p))
        f_total += phi_val * S_sigma * S_T * c["count"]

    return f_total, cycles


def compute_df_de(
    storage_e: np.ndarray,
    cycles: List[Dict],
    e_cap: float,
    shi_fit: ShiPolynomialFit,
    T_C: float = 25.0,
    p: XuModelParams = XU_LMO,
) -> np.ndarray:
    """Sparse subgradient df_deg/de_t -- nonzero at turning points only."""
    n = len(storage_e)
    df_de = np.zeros(n, dtype=np.float64)

    k3, k4 = shi_fit.k3, shi_fit.k4
    S_T = float(s_temp(T_C, p))

    for cyc in cycles:
        delta = cyc["dod"]
        if delta < 1e-12:
            continue

        phi_prime = float(phi_shi_prime(delta, k3, k4))
        S_sigma   = float(s_soc(cyc["soc_mean"], p))
        kernel    = phi_prime * S_sigma * S_T * cyc["count"]

        i_s, i_e = cyc["i_start"], cyc["i_end"]
        i0 = max(min(i_s, i_e), 0)
        i1 = min(max(i_s, i_e), n - 1)
        segment = storage_e[i0 : i1 + 1]

        peak_idx   = i0 + int(np.argmax(segment))
        trough_idx = i0 + int(np.argmin(segment))

        df_de[peak_idx]   += kernel
        df_de[trough_idx] -= kernel

    df_de /= e_cap
    return df_de


# ==========================================================================
# 2. NLP objective + gradient
# ==========================================================================

def make_nlp_functions(
    vec_obj: np.ndarray,
    n: int,
    e_cap_nominal: float,
    shi_fit: ShiPolynomialFit,
    w_deg: float,
    T_C: float = 25.0,
):
    """Build objective and gradient closures for minimize().

    Energy state of storage 1 lives at x[3*n : 4*n+1].
    """
    e_slice = slice(3*n, 4*n + 1) #battery energy state

    _state = {"n_obj": 0, "last_f_deg": 0.0} 

    def objective(x):
        e = x[e_slice]
        lp_cost = np.dot(vec_obj, x) #LP cost
        f_deg, _ = compute_f_deg(e, e_cap_nominal, shi_fit, T_C)
        _state["n_obj"] += 1
        _state["last_f_deg"] = f_deg
        return lp_cost + w_deg * f_deg #deg penalty

    def gradient(x):
        e = x[e_slice]
        f_deg, cycles = compute_f_deg(e, e_cap_nominal, shi_fit, T_C)
        df_de = compute_df_de(e, cycles, e_cap_nominal, shi_fit, T_C)

        grad = vec_obj.copy() #we start with vec_obj
        grad[e_slice] += w_deg * df_de #then add w_deg x df_de at each energy position
        return grad

    return objective, gradient, _state


# ==========================================================================
# 3. Build LP problem (reusable by alpha sweep)
# ==========================================================================

def build_lp_problem(prices, config):
    """Build SHIPP LP matrices, solve LP, return matrices + LP results."""
    E_CAP   = config["e_cap"]
    P_CAP   = config["p_cap"]
    SOC_MIN = config["soc_min"]
    SOC_MAX = config["soc_max"]
    dt      = config["dt"]
    T       = len(prices)

    stor = Storage( #build the storage objects
        e_cap=E_CAP, p_cap=P_CAP,
        eff_in=config["eff_in"], eff_out=config["eff_out"],
        e_cost=0, p_cost=0, dod=1.0,
    )
    stor_null = Storage(e_cap=0, p_cap=0, eff_in=1, eff_out=1, e_cost=0, p_cost=0)

    power = np.zeros(T) #pure arbitage, batttery only trades on th grid 
    options = dict(formulation='lp_alt', fixed_cap=True)

    vec_obj = build_lp_obj_revenues(prices, T, options) #objective vector from prices
    mat_eq, vec_eq, mat_ineq, vec_ineq, bounds_lower, bounds_upper = \
        build_lp_cst_sparse( #build inequality constraints and variable bounds 
            power, dt, -P_CAP, P_CAP, T,
            stor, stor_null,
            stor1_p_cap_max=P_CAP, stor2_p_cap_max=0,
            stor1_e_cap_max=E_CAP, stor2_e_cap_max=0,
            options=options,
        )

    e1_slice = slice(3*T, 4*T + 1)
    bounds_lower[e1_slice] = SOC_MIN * E_CAP #Tighten the SHIPP set energy bounds to [SoC_min,SoC_max] for our E_cap
    bounds_upper[e1_slice] = SOC_MAX * E_CAP
    bounds_list = list(zip(bounds_lower, bounds_upper))

    print(f"  Solving LP (HiGHS)...", end=" ", flush=True) #Solve LP with ,linprog and HiGHS
    t0 = time.perf_counter()
    res_lp = linprog(
        vec_obj,
        A_ub=mat_ineq.toarray(), b_ub=vec_ineq,
        A_eq=mat_eq.toarray(), b_eq=vec_eq,
        bounds=bounds_list, method='highs',
    )
    t_lp = time.perf_counter() - t0
    if not res_lp.success:
        raise RuntimeError(f"LP failed: {res_lp.message}")
    print(f"{t_lp:.1f} s")

    x_lp = res_lp.x
    lp_revenue = float(np.sum(prices[:T] * x_lp[0:T] * dt)) #Compute LP revenue

    lp_mats = dict( #pack patrixces and LP solution for NLP
        vec_obj=vec_obj, mat_eq=mat_eq, vec_eq=vec_eq,
        mat_ineq=mat_ineq, vec_ineq=vec_ineq,
        bounds_list=bounds_list, e1_slice=e1_slice,
        T=T, x_lp=x_lp, t_lp=t_lp,
    )
    lp_data = dict(p=x_lp[0:T], e=x_lp[e1_slice], revenue=lp_revenue) #pack power, energy and revenue into a dic for comparison
    return lp_mats, lp_data


# ==========================================================================
# 4. Run single period (LP + NLP + compare + save)
# ==========================================================================

def run_single_period(
    prices: np.ndarray,
    config: Dict,
    shi_fit: ShiPolynomialFit,
    max_iter: int,
    results_dir: Path,
    prefix: str,
    year: int,
    month_label: str,
    solver: str = "trust-constr",
    tr_radius: float = None,
    verbose: bool = True,
) -> Dict:
    """Full LP -> NLP pipeline for one price slice using SHIPP kernel."""

    E_CAP = config["e_cap"]
    B     = config["replacement_cost"]
    dt    = config["dt"]
    T     = len(prices)
    w_deg = B * E_CAP #cost per unit of degradation

    print(f"\n  Period: {month_label}  ({T} hours)")
    print(f"  Price: mean={prices.mean():.1f}  std={prices.std():.1f}  "
          f"range=[{prices.min():.1f}, {prices.max():.1f}]")

    # -- Build + solve LP --------------------------------------------------
    lp_mats, lp_data = build_lp_problem(prices, config)
    print(f"  SHIPP layout: {len(lp_mats['vec_obj'])} vars  (5*{T}+6 = {5*T+6})")

    f_lp, cyc_lp = compute_f_deg(lp_data["e"], E_CAP, shi_fit) #post-hoc evaluation of LP degradation
    deg_cost_lp = w_deg * f_lp
    lp_data["deg_cost"] = deg_cost_lp
    lp_data["n_cycles"] = len(cyc_lp)

    print(f"  LP Revenue:  {lp_data['revenue']:>12,.2f}   "
          f"Deg cost: {deg_cost_lp:>10,.2f}   "
          f"Net: {lp_data['revenue'] - deg_cost_lp:>12,.2f}   "
          f"Cycles: {len(cyc_lp)}")

    # -- NLP with degradation ----------------------------------------------
    obj_fn, grad_fn, state = make_nlp_functions(
        lp_mats["vec_obj"], T, E_CAP, shi_fit, w_deg,
    )
    x0 = lp_mats["x_lp"].copy() #start the NLP from the LP solution

    solver_label = solver
    if tr_radius and solver == "trust-constr":
        solver_label += f" tr={tr_radius}"

    print(f"  Solving NLP ({solver_label}, max_iter={max_iter})...", flush=True)
    t0 = time.perf_counter()

    if solver == "trust-constr":#bounds handle SoC limits and power limits
        constraints = [
            LinearConstraint(lp_mats["mat_ineq"], ub=lp_mats["vec_ineq"]), #  mat_ineq @ x ≤ vec_ineq
            LinearConstraint(lp_mats["mat_eq"],
                             lb=lp_mats["vec_eq"], ub=lp_mats["vec_eq"]), #mat_eq @ x = vec_eq (energy balance)
        ]
        _iter = {"n": 0}
        _history = []  # convergence trajectory
        def callback_tc(xk, opt_state):
            _iter["n"] += 1
            obj_v = opt_state.fun if hasattr(opt_state, 'fun') else float('nan')
            cv = opt_state.constr_violation if hasattr(opt_state, 'constr_violation') else float('nan')
            opt_val = opt_state.optimality if hasattr(opt_state, 'optimality') else float('nan')
            _history.append(dict(
                iter=_iter["n"], obj=float(obj_v), cv=float(cv),
                optimality=float(opt_val), f_deg=float(state['last_f_deg']),
            ))
            if verbose and _iter["n"] % 10 == 0:
                print(f"    iter {_iter['n']:4d}  obj={obj_v:+.4e}  "
                      f"cv={cv:.2e}  opt={opt_val:.2e}  "
                      f"f_deg={state['last_f_deg']:.4e}")
            return False

        tc_opts = dict(sparse_jacobian=True, verbose=2 if verbose else 0, # we tell the solver that the constraint Jacobian is sparse
                       maxiter=max_iter)
        if tr_radius is not None:
            tc_opts["initial_tr_radius"] = tr_radius # if tr_radious is set is controls the initial region size

        res_nlp = minimize(obj_fn, x0, jac=grad_fn, constraints=constraints,
                           bounds=lp_mats["bounds_list"], method='trust-constr',
                           tol=1e-6, options=tc_opts, callback=callback_tc)

    elif solver == "slsqp": #SLSQP solver path
        A_eq_d = lp_mats["mat_eq"].toarray() #toarray converts sparse to dense which is why the SLSQP uses more memory
        A_ub_d = lp_mats["mat_ineq"].toarray()
        constraints_sq = [
            {"type": "eq", # fun(x) = mat_eq @ x − vec_eq = 0
             "fun": lambda x: A_eq_d @ x - lp_mats["vec_eq"], 
             "jac": lambda x: A_eq_d},
            {"type": "ineq", # fun(x) = vec_ineq − mat_ineq @ x ≥ 0
             "fun": lambda x: lp_mats["vec_ineq"] - A_ub_d @ x,
             "jac": lambda x: -A_ub_d},
        ]
        _iter = {"n": 0}
        _history = []
        def callback_sq(xk):
            _iter["n"] += 1
            obj_v = float(obj_fn(xk))
            _history.append(dict(
                iter=_iter["n"], obj=obj_v, cv=0.0,
                optimality=float('nan'), f_deg=float(state['last_f_deg']),
            ))
            if verbose and _iter["n"] % 25 == 0:
                print(f"    iter {_iter['n']:4d}  obj={obj_v:+.4e}  "
                      f"f_deg={state['last_f_deg']:.4e}")

        res_nlp = minimize(obj_fn, x0, jac=grad_fn, constraints=constraints_sq,
                           bounds=lp_mats["bounds_list"], method='SLSQP',
                           options=dict(maxiter=max_iter, disp=verbose, ftol=1e-9),
                           callback=callback_sq)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    t_nlp = time.perf_counter() - t0

    # -- Extract + compare -------------------------------------------------
    x_nlp = res_nlp.x
    nlp_p = x_nlp[0:T]
    nlp_e = x_nlp[lp_mats["e1_slice"]]
    nlp_revenue = float(np.sum(prices[:T] * nlp_p * dt))

    f_nlp, cyc_nlp = compute_f_deg(nlp_e, E_CAP, shi_fit)
    deg_cost_nlp = w_deg * f_nlp

    net_lp  = lp_data["revenue"] - deg_cost_lp
    net_nlp = nlp_revenue - deg_cost_nlp

    print(f"\n  NLP status: {res_nlp.message}")
    print(f"  NLP time:   {t_nlp:.1f} s ({t_nlp/60:.1f} min)  iters: {res_nlp.nit}")

    sac = lp_data["revenue"] - nlp_revenue
    sav = deg_cost_lp - deg_cost_nlp

    print(f"\n  {'Metric':<22s} {'LP':>12s} {'NLP':>12s} {'Delta':>10s}")
    print(f"  {'---'*18}")
    print(f"  {'Revenue [EUR]':<22s} {lp_data['revenue']:>12,.0f} "
          f"{nlp_revenue:>12,.0f} {nlp_revenue-lp_data['revenue']:>+10,.0f}")
    print(f"  {'Deg cost [EUR]':<22s} {deg_cost_lp:>12,.0f} "
          f"{deg_cost_nlp:>12,.0f} {deg_cost_nlp-deg_cost_lp:>+10,.0f}")
    print(f"  {'Net utility [EUR]':<22s} {net_lp:>12,.0f} "
          f"{net_nlp:>12,.0f} {net_nlp-net_lp:>+10,.0f}")
    print(f"  {'Cycles':<22s} {len(cyc_lp):>12d} {len(cyc_nlp):>12d}")
    if sac > 0:
        print(f"  Sacrifice: {sac:>+,.0f}  Saving: {sav:>+,.0f}  "
              f"Ratio: {sav/sac:.2f}x")

    nlp_result = dict(
        p=nlp_p, e=nlp_e,
        revenue=nlp_revenue, f_deg=f_nlp,
        deg_cost_EUR=deg_cost_nlp, n_cycles=len(cyc_nlp),
        status_msg=res_nlp.message, n_iter=res_nlp.nit,
        history=_history,
    )

    _save_results(results_dir, prefix, year, month_label,
                  lp_data, nlp_result, prices, config,
                  {"LP": lp_mats["t_lp"], "NLP": t_nlp},
                  solver_info=solver_label)

    return nlp_result


# ==========================================================================
# 5. Save / plot
# ==========================================================================

def _save_results(results_dir, prefix, year, month_label,
                  lp_data, nlp_result, prices, config, timings,
                  solver_info=""):
    import json as _json

    results_dir.mkdir(exist_ok=True)
    T = len(prices)

    rev_lp, rev_nlp = lp_data["revenue"], nlp_result["revenue"]
    deg_lp, deg_nlp = lp_data["deg_cost"], nlp_result["deg_cost_EUR"]
    net_lp  = rev_lp - deg_lp
    net_nlp = rev_nlp - deg_nlp
    sac = rev_lp - rev_nlp
    sav = deg_lp - deg_nlp

    # -- 1. Single .npz with all arrays + metadata ---------------------------
    npz_path = results_dir / f"{prefix}_results.npz"
    np.savez_compressed(
        npz_path,
        lp_e=lp_data["e"], lp_p=lp_data["p"],
        nlp_e=nlp_result["e"], nlp_p=nlp_result["p"],
        prices=prices,
    )
    print(f"  Saved: {npz_path.name}")

    # -- 2. Enhanced convergence JSON with summary header --------------------
    summary = dict(
        runner=_RUNNER_NAME, solver=solver_info,
        year=year, period=month_label, hours=T,
        e_cap=config["e_cap"], p_cap=config["p_cap"],
        soc_min=config["soc_min"], soc_max=config["soc_max"],
        replacement_cost=config["replacement_cost"],
        lp_revenue=round(rev_lp, 2),
        nlp_revenue=round(rev_nlp, 2),
        lp_deg_cost=round(deg_lp, 2),
        nlp_deg_cost=round(deg_nlp, 2),
        lp_net=round(net_lp, 2),
        nlp_net=round(net_nlp, 2),
        revenue_sacrifice=round(sac, 2),
        degradation_saving=round(sav, 2),
        ratio=round(sav / sac, 2) if sac > 0 else 0.0,
        nlp_status=nlp_result.get("status_msg", "N/A"),
        nlp_iters=nlp_result.get("n_iter", 0),
        nlp_n_cycles=nlp_result.get("n_cycles", 0),
        lp_n_cycles=lp_data.get("n_cycles", 0),
        timings={k: round(v, 2) for k, v in timings.items()},
    )

    history = nlp_result.get("history", [])
    json_data = dict(summary=summary, convergence=history)
    json_path = results_dir / f"{prefix}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(json_data, f, indent=1)
    print(f"  Saved: {json_path.name} ({len(history)} iterations)")

    # -- 3. Console-friendly summary (replaces .txt file) --------------------
    print(f"\n  {'Metric':<28s} {'LP':>12s} {'NLP':>12s} {'Delta':>10s}")
    print(f"  {'---'*18}")
    print(f"  {'Revenue [EUR]':<28s} {rev_lp:>12,.2f} {rev_nlp:>12,.2f} "
          f"{rev_nlp-rev_lp:>+10,.2f}")
    print(f"  {'Deg cost [EUR]':<28s} {deg_lp:>12,.2f} {deg_nlp:>12,.2f} "
          f"{deg_nlp-deg_lp:>+10,.2f}")
    print(f"  {'Net utility [EUR]':<28s} {net_lp:>12,.2f} {net_nlp:>12,.2f} "
          f"{net_nlp-net_lp:>+10,.2f}")
    if sac > 0:
        print(f"  Sacrifice: {sac:>+,.0f}  Saving: {sav:>+,.0f}  "
              f"Ratio: {sav/sac:.2f}x")

    # -- 4. Comparison plot (3-panel) ----------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        fig.patch.set_facecolor("#f7f9fc")
        for ax in axes:
            ax.set_facecolor("#f7f9fc")
        fig.suptitle(f"Path 3 SHIPP ({solver_info})  |  DK1 {year} {month_label}",
                     fontsize=13, fontweight="bold")
        hours = np.arange(T)

        axes[0].plot(hours, prices, color="#2166ac", lw=0.5, alpha=0.8)
        axes[0].set_ylabel("Price [EUR/MWh]")
        axes[0].set_title("Day-Ahead Price", fontweight="bold")

        axes[1].plot(hours, lp_data["e"][:T], color="#2166ac", lw=0.6,
                     label="LP", alpha=0.8)
        axes[1].plot(hours, nlp_result["e"][:T], color="#b5351b", lw=0.6,
                     label="NLP", alpha=0.8)
        axes[1].set_ylabel("Stored Energy [MWh]")
        axes[1].set_title("Battery SoC", fontweight="bold")
        axes[1].legend(fontsize=9)

        diff = nlp_result["p"] - lp_data["p"]
        axes[2].fill_between(hours, diff, 0, where=diff > 0,
                             color="#2166ac", alpha=0.3, label="NLP > LP")
        axes[2].fill_between(hours, diff, 0, where=diff < 0,
                             color="#b5351b", alpha=0.3, label="NLP < LP")
        axes[2].set_ylabel("Delta Power [MW]")
        axes[2].set_xlabel("Hour")
        axes[2].set_title("Dispatch Difference", fontweight="bold")
        axes[2].legend(fontsize=9)

        plt.tight_layout()
        png = results_dir / f"{prefix}_comparison.png"
        fig.savefig(png, dpi=150, bbox_inches="tight",
                    facecolor="#f7f9fc", edgecolor="none")
        plt.close(fig)
        print(f"  Saved: {png.name}")

        # -- 5. Convergence trajectory plot (4-panel) ------------------------
        if len(history) > 2:
            fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
            fig2.patch.set_facecolor("#f7f9fc")
            for ax in axes2.flat:
                ax.set_facecolor("#f7f9fc")
            fig2.suptitle(f"Convergence  |  DK1 {year} {month_label}  |  {solver_info}",
                         fontsize=13, fontweight="bold")

            iters = [h["iter"] for h in history]
            objs  = [h["obj"] for h in history]
            cvs   = [h["cv"] for h in history]
            opts  = [h["optimality"] for h in history]
            fdegs = [h["f_deg"] for h in history]

            # Panel 1: Objective value
            axes2[0,0].plot(iters, objs, color="#2166ac", lw=0.8)
            axes2[0,0].set_ylabel("Objective")
            axes2[0,0].set_title("Objective Value", fontweight="bold")
            axes2[0,0].ticklabel_format(axis='y', style='scientific', scilimits=(-3,3))

            # Panel 2: Objective change per iteration (Δobj)
            if len(objs) > 1:
                dobj = np.abs(np.diff(objs))
                dobj_safe = np.where(dobj > 0, dobj, 1e-20)
                axes2[0,1].semilogy(iters[1:], dobj_safe, color="#2166ac", lw=0.6, alpha=0.7)
                axes2[0,1].set_ylabel("|ΔObj|")
                axes2[0,1].set_title("Objective Change per Iter", fontweight="bold")

            # Panel 3: Optimality
            opts_clean = [o for o in opts if not (o != o)]  # remove NaN
            if opts_clean:
                axes2[1,0].semilogy(iters[:len(opts_clean)], opts_clean, color="#b5351b", lw=0.8)
                axes2[1,0].axhline(1e-6, ls='--', color='gray', lw=0.7, label='tol=1e-6')
                axes2[1,0].set_ylabel("Optimality")
                axes2[1,0].set_title("KKT Optimality", fontweight="bold")
                axes2[1,0].legend(fontsize=8)

            # Panel 4: f_deg trajectory
            axes2[1,1].plot(iters, fdegs, color="#2166ac", lw=0.8)
            axes2[1,1].set_ylabel("f_deg")
            axes2[1,1].set_title("Degradation Fraction", fontweight="bold")
            axes2[1,1].ticklabel_format(axis='y', style='scientific', scilimits=(-3,3))

            for ax in axes2[1,:]:
                ax.set_xlabel("Iteration")

            plt.tight_layout()
            conv_png = results_dir / f"{prefix}_convergence.png"
            fig2.savefig(conv_png, dpi=150, bbox_inches="tight",
                        facecolor="#f7f9fc", edgecolor="none")
            plt.close(fig2)
            print(f"  Saved: {conv_png.name}")
    except Exception as exc:
        print(f"  Plot failed: {exc}")


# ==========================================================================
# 6. Data loading + CLI
# ==========================================================================

_DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]


def month_hour_range(month: int) -> Tuple[int, int]:
    start = sum(_DAYS_PER_MONTH[:month-1]) * 24
    end   = start + _DAYS_PER_MONTH[month-1] * 24
    return start, end


# Pre-defined week slots for sub-monthly testing
# Each entry: (start_day_of_year_0indexed, n_days, label)
WEEK_SLOTS = {
    "W1":  (0,   7, "Jan 1-7 (2019-High)"),      # h0-168
    "W7":  (42,  7, "Feb 12-18 (2022-Low)"),      # h1008-1176
    "W13": (84,  7, "Mar 26-Apr 1 (2022-Mod)"),   # h2016-2184
    "W28": (189, 7, "Jul 9-15 (2019-Low)"),       # h4536-4704
    "W34": (231, 7, "Aug 20-26 (2022-High)"),     # h5544-5712
    "W35": (238, 7, "Aug 27-Sep 2 (2019-Mod)"),   # h5712-5880
}


def slot_hour_range(slot_key: str) -> Tuple[int, int, str]:
    """Return (h_start, h_end, label) for a predefined week slot."""
    day0, n_days, label = WEEK_SLOTS[slot_key]
    return day0 * 24, (day0 + n_days) * 24, label


def load_dk1_prices(year: int) -> np.ndarray:
    for base in [_SCRIPT_DIR, _PARENT_DIR, _SCRIPT_DIR.parent]:
        csv_path = base / f"dk1_prices_{year}.csv"
        if csv_path.exists():
            break
    else:
        raise FileNotFoundError(f"dk1_prices_{year}.csv not found")

    import pandas as pd
    df = pd.read_csv(csv_path)
    for col in ["price", "price [EUR/MWh]", "Price", "EUR/MWh", "DK1"]:
        if col in df.columns:
            prices = df[col].values.astype(float)
            break
    else:
        prices = df.select_dtypes(include=[np.number]).iloc[:, -1].values

    prices = prices[:8760]
    if len(prices) < 8760:
        raise ValueError(f"Need 8760 hours, got {len(prices)}")

    nan_count = np.isnan(prices).sum()
    if nan_count > 0:
        print(f"  Warning: {nan_count} NaN prices replaced with mean")
        prices[np.isnan(prices)] = np.nanmean(prices)

    return prices


def main(): # parses command line arguments, set us config (batt parameters), fits Shi polynominal and toutes to either test or real data
    parser = argparse.ArgumentParser(
        description="Path 3 Jenna/SHIPP -- Degradation-Aware NLP")
    parser.add_argument("--year", type=int, default=2022, choices=[2019, 2022])
    parser.add_argument("--month", type=str, default="7",
                        help="Month (1-12), 'all', or 'full'")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--solver", type=str, default="trust-constr",
                        choices=["trust-constr", "slsqp"])
    parser.add_argument("--tr-radius", type=float, default=None,
                        help="Initial trust region radius (trust-constr only)")
    parser.add_argument("--test", action="store_true",
                        help="Small synthetic test (120h)")
    parser.add_argument("--start-hour", type=int, default=None,
                        help="Start hour (0-indexed) for arbitrary slice")
    parser.add_argument("--n-hours", type=int, default=168,
                        help="Number of hours to run (default: 168 = 1 week)")
    parser.add_argument("--slot", type=str, default=None,
                        choices=list(WEEK_SLOTS.keys()) + ["all"],
                        help="Predefined week slot (W1,W7,W13,W28,W34,W35), or 'all'")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = _SCRIPT_DIR / "Results_Path3"
    results_dir.mkdir(exist_ok=True)

    solver_tag = args.solver
    if args.tr_radius:
        solver_tag += f"_tr{args.tr_radius}"

    print("=" * 72)
    print(f"Path 3 Jenna/SHIPP  |  DK1 {args.year}  |  month={args.month}")
    print(f"Solver: {solver_tag}   max_iter: {args.max_iter}")
    print(f"Timestamp: {timestamp}")
    print("=" * 72)

    config = dict(
        e_cap=300.0, p_cap=150.0,
        soc_min=0.10, soc_max=0.90,
        eff_in=0.95, eff_out=0.95,
        dt=1.0,
        replacement_cost=150_000.0,
    )

    print(f"\n  Battery: {config['e_cap']} MWh / {config['p_cap']} MW")
    print(f"  SoC: [{config['soc_min']}, {config['soc_max']}]")
    print(f"  w_deg = {config['replacement_cost']*config['e_cap']:,.0f} EUR")

    shi_fit = fit_shi_polynomial(
        soc_min=config["soc_min"], soc_max=config["soc_max"],
        source=f"Path3 SHIPP {args.year}", verbose=True,
    )

    def _run(prices_slice, label, pfx):
        run_single_period(prices_slice, config, shi_fit, args.max_iter,
                          results_dir, pfx, args.year, label,
                          solver=args.solver, tr_radius=args.tr_radius)

    if args.test:
        T = 120
        rng = np.random.RandomState(42)
        hours = np.arange(T) % 24
        prices = 30 + 20*np.sin(2*np.pi*(hours-6)/24) + rng.normal(0, 5, T)
        prices = np.maximum(prices, 0)
        _run(prices, f"Test ({T}h)",
             f"{timestamp}_{_RUNNER_NAME}_test_{solver_tag}")
        return

    prices_full = load_dk1_prices(args.year)
    print(f"  Loaded {len(prices_full)} hours of DK1 {args.year}")

    t_total_start = time.perf_counter()

    if args.slot is not None:
        # Predefined week slots
        slots = list(WEEK_SLOTS.keys()) if args.slot == "all" else [args.slot]
        for sk in slots:
            h0, h1, slabel = slot_hour_range(sk)
            label = f"{sk} {slabel} (h{h0}-{h1})"
            pfx = f"{timestamp}_{_RUNNER_NAME}_dk{args.year}_{sk}_{solver_tag}"
            print(f"\n{'---'*24}")
            print(f"  Slot {sk}: {slabel}")
            print(f"{'---'*24}")
            _run(prices_full[h0:h1], label, pfx)

    elif args.start_hour is not None:
        # Arbitrary hour range
        h0 = args.start_hour
        h1 = min(h0 + args.n_hours, 8760)
        label = f"Custom h{h0}-{h1} ({h1-h0}h)"
        pfx = f"{timestamp}_{_RUNNER_NAME}_dk{args.year}_h{h0}_{h1-h0}h_{solver_tag}"
        _run(prices_full[h0:h1], label, pfx)

    elif args.month == "full":
        _run(prices_full, "Full Year",
             f"{timestamp}_{_RUNNER_NAME}_dk{args.year}_full_{solver_tag}")

    elif args.month == "all":
        for m in range(1, 13):
            h0, h1 = month_hour_range(m)
            label = f"{_MONTH_NAMES[m-1]} (h{h0}-{h1})"
            pfx = f"{timestamp}_{_RUNNER_NAME}_dk{args.year}_m{m:02d}_{solver_tag}"
            print(f"\n{'---'*24}")
            print(f"  Month {m}/12: {_MONTH_NAMES[m-1]}")
            print(f"{'---'*24}")
            _run(prices_full[h0:h1], label, pfx)
    else:
        m = int(args.month)
        if not 1 <= m <= 12:
            raise ValueError(f"Month must be 1-12, got {m}")
        h0, h1 = month_hour_range(m)
        label = f"{_MONTH_NAMES[m-1]} (h{h0}-{h1})"
        pfx = f"{timestamp}_{_RUNNER_NAME}_dk{args.year}_m{m:02d}_{solver_tag}"
        _run(prices_full[h0:h1], label, pfx)

    t_total = time.perf_counter() - t_total_start
    print(f"\n{'==='*24}")
    print(f"Done.  Wall time: {t_total:.1f} s ({t_total/60:.1f} min)")
    print(f"{'==='*24}")


if __name__ == "__main__":
    main()