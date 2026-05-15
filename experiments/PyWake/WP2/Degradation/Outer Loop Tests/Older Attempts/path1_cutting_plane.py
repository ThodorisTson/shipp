"""Path 1 — Cutting-plane dispatch with rainflow-based Shi degradation.

Purpose
-------
Solve the degradation-aware dispatch problem using iterative cutting
planes (Kelley's method / Benders decomposition).  Each iteration:

    1. Solve the LP dispatch (Gurobi)  →  SoC trajectory + dispatch
    2. Compute Shi degradation cost f_d via rainflow cycle counting
    3. Compute per-timestep subgradient (Shi Eq. 17–18)
    4. Add a cutting plane  θ ≥ f_d + g·(x − xᵏ)  to the LP
    5. Re-solve until convergence

The LP stays linear (Gurobi) throughout — the nonlinear degradation is
handled entirely through the external oracle + linear cuts.  This uses
the ACTUAL rainflow cycle-based degradation, not a smooth proxy.

Key advantage over Path 2
    Uses the validated Shi subgradient and rainflow cycle counting from
    the existing codebase.  The degradation cost is physically accurate.

Folder structure
    shipp/
    ├── degradation_xu.py
    ├── degradation_shi.py
    ├── degradation_subgradient.py
    ├── wp2_common.py
    ├── WP2_HPP.yaml / dk1_prices_2022.csv
    └── Outer Loop Tests/
        ├── path1_cutting_plane.py      ← this script
        └── Results/

Usage
    python path1_cutting_plane.py                # 168 h (1 week)
    python path1_cutting_plane.py --hours 100
    python path1_cutting_plane.py --full          # 8760 h
    python path1_cutting_plane.py --max-cuts 30   # more iterations
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ─── Path setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "Results"
OUTPUT_DIR.mkdir(exist_ok=True)

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import numpy as np
import numpy_financial as npf
import pyomo.environ as pyo

# ─── SHIPP imports (installed package) ──────────────────────────────────────
from shipp.kernel_pyomo import solve_lp_pyomo
from shipp.components import Storage, Production, TimeSeries

# ─── WP2 local modules (from parent directory) ─────────────────────────────
from degradation_xu import rainflow_cycle_counting
from degradation_subgradient import compute_subgradient, fit_shi_polynomial

# ─── Data files (in parent directory) ───────────────────────────────────────
HPP_YAML = PARENT_DIR / "WP2_HPP.yaml"
PRICE_CSV = PARENT_DIR / "dk1_prices_2022.csv"


# ════════════════════════════════════════════════════════════════════════════
# 0.  QUICK TOGGLES  —  change these instead of using command-line flags
# ════════════════════════════════════════════════════════════════════════════

RUN_HOURS   = 8760        # 100, 168, 8760 (or any custom value)
RUN_FULL    = False       # True overrides RUN_HOURS → 8760
C_DEG       = 10.0         # degradation cost multiplier (try 1, 5, 10)
MAX_CUTS    = 100         # max cutting-plane iterations
CONV_TOL    = 0.005       # relative gap tolerance (0.001 = 0.1%, 0.005 = 0.5%)
SKIP_PLOT   = False       # True to skip matplotlib plots


# ════════════════════════════════════════════════════════════════════════════
# 0b. FIXED PARAMETERS  (normally don't change)
# ════════════════════════════════════════════════════════════════════════════

EUR_TO_USD    = 1.18
DISCOUNT_RATE = 0.03
N_YEARS       = 20
DT            = 1.0       # hours

# Battery defaults (overridden by YAML)
E_CAP_MWH   = 300.0
P_CAP_MW    = 150.0
ETA_IN      = 1.0
RTE_DC      = 0.95
PCU_EFF     = 0.975
SOC_MIN     = 0.10
SOC_MAX     = 0.90
P_MIN_MW    = 0.0
P_MAX_MW    = 300.0


# ════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING  (same pipeline as Path 2 / run_battery v4)
# ════════════════════════════════════════════════════════════════════════════

def load_data(n_hours: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load wind power and price data from WP2 pipeline.

    Returns (power_wind_MW, price_eur, params_dict).
    """
    params = {
        "e_cap": E_CAP_MWH, "p_cap": P_CAP_MW,
        "eta_in": ETA_IN, "eta_out": RTE_DC * PCU_EFF**2,
        "soc_min": SOC_MIN, "soc_max": SOC_MAX,
        "e_cost_usd_per_mwh": 0.0, "p_cost_usd_per_mw": 0.0,
        "p_max": P_MAX_MW,
    }

    import pandas as pd
    import xarray as xr
    from py_wake.site import XRSite
    from wp2_common import quick_setup, get_wake_model

    print("  Loading WP2 configuration from YAML...")
    setup = quick_setup(HPP_YAML, config={"interp_n": 2000}, verbose=False)
    hpp = setup["hpp"]

    # Wind resource
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

    # PyWake
    print("  Running PyWake simulation...")
    site = XRSite(ds=xr.Dataset(data_vars=dict(P=1)))
    wf_model = get_wake_model("Bastankhah", site, setup["windturbine"])
    time_days = np.arange(n) / 24.0
    kwargs = {"x": setup["x"], "y": setup["y"],
              "wd": wd, "ws": ws, "time": time_days}
    if ti is not None:
        kwargs["TI"] = ti
    sim_res = wf_model(**kwargs)
    power_wind_MW = sim_res.Power.sum(["wt"]).values / 1e6

    # Prices
    df = pd.read_csv(PRICE_CSV)
    for col in ("price_eur_mwh", "price_eur_per_mwh", "Price", "price"):
        if col in df.columns:
            break
    price_eur = df[col].astype(float).to_numpy()[:n]

    # Battery params from YAML
    bat = setup["battery"]
    rte_dc = float(bat["rte_nominal"])
    pcu_eff = float(bat["pcu_efficiency"])

    params["e_cap"]   = float(bat["energy_capacity_Wh"]) / 1e6
    params["p_cap"]   = float(bat["power_capacity_W"])   / 1e6
    params["eta_out"] = rte_dc * pcu_eff**2
    params["soc_min"] = float(bat.get("soc_min", SOC_MIN))
    params["soc_max"] = float(bat.get("soc_max", SOC_MAX))
    params["e_cost_usd_per_mwh"] = float(bat["capex_EUR_per_kWh"]) * 1000.0 * EUR_TO_USD
    params["p_cost_usd_per_mw"]  = float(bat["capex_EUR_per_kW"])  * 1000.0 * EUR_TO_USD
    params["p_max"] = float(hpp["grid_connection_capacity"]) / 1e6

    # Shi polynomial fit
    shi_fit = fit_shi_polynomial(params["soc_min"], params["soc_max"], verbose=True)
    params["shi_fit"] = shi_fit
    params["k3"] = shi_fit.k3
    params["k4"] = shi_fit.k4

    print(f"  Battery: {params['p_cap']:.0f} MW / {params['e_cap']:.0f} MWh")
    print(f"  Grid:    {params['p_max']:.0f} MW")
    print(f"  RTE(ac): {params['eta_out']*100:.1f}%")
    print(f"  SoC:     {params['soc_min']*100:.0f}%–{params['soc_max']*100:.0f}%")
    print(f"  Shi fit: k3={params['k3']:.4e}, k4={params['k4']:.4f}")

    return power_wind_MW[:n], price_eur[:n], params


# ════════════════════════════════════════════════════════════════════════════
# 2.  BASELINE LP  (Gurobi, same as kernel_pyomo)
# ════════════════════════════════════════════════════════════════════════════

def solve_baseline_lp(
    power_wind: np.ndarray,
    price_eur:  np.ndarray,
    params:     dict,
) -> dict:
    """Solve the dispatch LP using the existing SHIPP formulation."""
    n       = len(price_eur)
    e_cap   = params["e_cap"]
    p_cap   = params["p_cap"]
    eta_out = params["eta_out"]
    soc_max = params["soc_max"]
    soc_min = params["soc_min"]
    dod     = 1.0 - soc_min
    p_max   = params["p_max"]

    stor = Storage(e_cap=e_cap, p_cap=p_cap, eff_in=ETA_IN, eff_out=eta_out,
                   e_cost=params["e_cost_usd_per_mwh"],
                   p_cost=params["p_cost_usd_per_mw"], dod=dod)
    stor_null = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
                        e_cost=0, p_cost=0)

    price_dam = TimeSeries((price_eur * EUR_TO_USD).tolist(), DT)
    prod      = Production(TimeSeries(power_wind.tolist(), DT), p_cost=0.0)
    prod_null = Production(TimeSeries([0.0] * n, DT), p_cost=0.0)

    t0 = time.perf_counter()
    os_res = solve_lp_pyomo(
        price_dam, prod, prod_null, stor, stor_null,
        DISCOUNT_RATE, N_YEARS, P_MIN_MW, p_max, n,
        "gurobi", fixed_cap=True, soc_max1=soc_max,
    )
    wall_s = time.perf_counter() - t0

    e_vec = np.array(os_res.storage_e[0].data, dtype=float)
    p_vec = np.array(os_res.storage_p[0].data, dtype=float)

    factor = npf.npv(DISCOUNT_RATE, np.ones(N_YEARS)) - 1
    revenue_usd = 365.0 * 24.0 / n * factor * float(
        np.dot(price_eur * EUR_TO_USD, p_vec)
    ) * DT

    return {
        "revenue_usd": revenue_usd,
        "e_vec": e_vec,
        "p_vec": p_vec,
        "npv":   os_res.npv,
        "time_s": wall_s,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3.  DEGRADATION ORACLE  (rainflow + Shi subgradient)
# ════════════════════════════════════════════════════════════════════════════

def degradation_oracle(
    e_vec:     np.ndarray,
    charge:    np.ndarray,
    discharge: np.ndarray,
    params:    dict,
) -> dict:
    """Compute Shi degradation cost and per-timestep subgradient.

    This is the 'oracle' that the cutting-plane method queries.
    It calls the validated rainflow_cycle_counting and compute_subgradient
    from the existing codebase.

    Args:
        e_vec:     SoC trajectory [MWh], length n+1 (E[0]..E[n]).
        charge:    Charge power [MW], length n, non-negative.
        discharge: Discharge power [MW], length n, non-negative.
        params:    Battery parameters.

    Returns:
        dict with:
            fd               — dimensionless fractional degradation
            fd_cost          — degradation cost in USD (fd × E_B)
            subgrad_charge   — ∂fd/∂charge[t], length n (Shi Eq. 17)
            subgrad_discharge— ∂fd/∂discharge[t], length n (Shi Eq. 18)
            n_cycles         — number of rainflow cycles identified
    """
    n       = len(charge)    # number of power timesteps
    e_cap   = params["e_cap"]
    eta_in  = params["eta_in"]
    eta_out = params["eta_out"]
    k3      = params["k3"]
    k4      = params["k4"]
    shi_fit = params["shi_fit"]
    E_B     = params["e_cost_usd_per_mwh"] * e_cap   # total replacement cost [$]

    # ── Step 1: Rainflow cycle counting ──────────────────────────────────
    cycles = rainflow_cycle_counting(e_vec, e_cap)

    # ── Step 2: Compute Shi fd ───────────────────────────────────────────
    #   fd = Σ Φ(dod_i) × count_i  where count = 0.5 for half, 1.0 for full
    fd = 0.0
    for c in cycles:
        dod = c["dod"]
        cnt = c["count"]
        fd += k3 * dod**k4 * cnt

    # ── Step 3: Call validated compute_subgradient ────────────────────────
    sg = compute_subgradient(
        storage_e=e_vec,
        cycles=cycles,
        dt_hours=DT,
        battery_replacement_cost_per_MWh=params["e_cost_usd_per_mwh"],
        eff_in=eta_in,
        eff_out=eta_out,
        shi_fit=shi_fit,
    )

    # compute_subgradient returns arrays of length len(e_vec) = n+1,
    # indexed by SoC point.  The last entry (index n) is a padding
    # artifact from np.diff(e, append=e[-1]) and is always zero.
    # We slice to [:n] to match the n power timesteps.
    #
    # subgrad_charge[t]    = ∂fd/∂charge[t]     (Shi Eq. 17, non-negative)
    # subgrad_discharge[t] = ∂fd/∂discharge[t]  (Shi Eq. 18, non-negative)
    #
    # These use the SAME rainflow cycles as fd above, guaranteeing
    # consistency between the function value and the gradient.

    return {
        "fd":                fd,
        "fd_cost":           fd * E_B,
        "subgrad_charge":    sg["subgrad_charge"][:n],
        "subgrad_discharge": sg["subgrad_discharge"][:n],
        "n_cycles":          len(cycles),
    }


# ════════════════════════════════════════════════════════════════════════════
# 4.  CUTTING-PLANE LP SOLVER
# ════════════════════════════════════════════════════════════════════════════

def solve_cutting_plane(
    power_wind:  np.ndarray,
    price_eur:   np.ndarray,
    params:      dict,
    max_cuts:    int   = MAX_CUTS,
    conv_tol:    float = CONV_TOL,
    c_deg_scale: float = 1.0,
) -> dict:
    """Solve dispatch with degradation via iterative cutting planes.

    The LP is solved with Gurobi.  After each solve, the degradation
    oracle computes the true Shi cost and subgradient.  A linear cut
    is added to the LP and it is re-solved.  The process repeats until
    the epigraph variable θ matches the oracle's cost (convergence).

    Args:
        power_wind:   Wind power [MW], length n.
        price_eur:    Day-ahead price [EUR/MWh], length n.
        params:       Battery and grid parameters.
        max_cuts:     Maximum cutting-plane iterations.
        conv_tol:     Relative convergence tolerance.
        c_deg_scale:  Degradation cost multiplier (1.0 = full cost).

    Returns:
        dict with final dispatch, convergence history, etc.
    """
    n       = len(price_eur)
    e_cap   = params["e_cap"]
    p_cap   = params["p_cap"]
    eta_in  = params["eta_in"]
    eta_out = params["eta_out"]
    soc_min = params["soc_min"]
    soc_max = params["soc_max"]
    p_max   = params["p_max"]
    E_B     = params["e_cost_usd_per_mwh"] * e_cap * c_deg_scale

    price_usd = price_eur * EUR_TO_USD
    power_res = power_wind.copy()

    # NPV / scaling factors (same as kernel_pyomo)
    factor    = npf.npv(DISCOUNT_RATE, np.ones(N_YEARS)) - 1
    rev_scale = 365.0 * 24.0 / n * factor
    deg_scale = 365.0 * 24.0 / n * factor * E_B   # $/fd_dimensionless

    e_min = e_cap * soc_min
    e_max = e_cap * soc_max

    # ── Build Pyomo LP model ─────────────────────────────────────────────
    model = pyo.ConcreteModel()
    N   = list(range(n))
    Np1 = list(range(n + 1))
    model.N   = pyo.Set(initialize=N)
    model.Np1 = pyo.Set(initialize=Np1)

    # Decision variables: split charge/discharge
    model.charge    = pyo.Var(model.N, bounds=(0, p_cap), initialize=0)
    model.discharge = pyo.Var(model.N, bounds=(0, p_cap), initialize=0)
    model.e         = pyo.Var(model.Np1, bounds=(e_min, e_max),
                              initialize=e_cap * 0.5)

    # Epigraph variable for degradation cost (dimensionless fd)
    model.theta = pyo.Var(bounds=(0, None), initialize=0)

    # Energy balance (equality)
    def rule_energy_balance(m, i):
        return (m.e[i+1] == m.e[i]
                + DT * eta_in * m.charge[i]
                - DT / eta_out * m.discharge[i])
    model.energy_balance = pyo.Constraint(model.N, rule=rule_energy_balance)

    # Periodic SoC
    model.periodic = pyo.Constraint(expr=model.e[0] == model.e[n])

    # Grid constraints
    def rule_grid_max(m, i):
        return m.discharge[i] - m.charge[i] <= max(p_max - power_res[i], 0)
    model.grid_max = pyo.Constraint(model.N, rule=rule_grid_max)

    def rule_grid_min(m, i):
        return m.discharge[i] - m.charge[i] >= P_MIN_MW - power_res[i]
    model.grid_min = pyo.Constraint(model.N, rule=rule_grid_min)

    # Capex (constant)
    capex = params["p_cost_usd_per_mw"] * p_cap + params["e_cost_usd_per_mwh"] * e_cap

    # Objective: max  Revenue − Capex − deg_scale × θ
    revenue_expr = rev_scale * sum(
        price_usd[i] * (model.discharge[i] - model.charge[i])
        for i in N
    )
    model.obj = pyo.Objective(
        expr=revenue_expr - capex - deg_scale * model.theta,
        sense=pyo.maximize,
    )

    # Cutting plane container
    model.cuts = pyo.ConstraintList()

    solver = pyo.SolverFactory("gurobi")

    # ── Cutting-plane iteration ──────────────────────────────────────────
    history = []
    t_total_start = time.perf_counter()

    for k in range(max_cuts + 1):
        t_iter = time.perf_counter()

        # Solve LP
        results = solver.solve(model, tee=False)
        solve_ok = (results.solver.status == pyo.SolverStatus.ok)

        # Extract current solution
        e_vec       = np.array([pyo.value(model.e[i])         for i in Np1])
        p_charge    = np.array([pyo.value(model.charge[i])    for i in N])
        p_discharge = np.array([pyo.value(model.discharge[i]) for i in N])
        theta_val   = float(pyo.value(model.theta))
        obj_val     = float(pyo.value(model.obj))

        # Query degradation oracle
        oracle = degradation_oracle(e_vec, p_charge, p_discharge, params)
        fd_actual = oracle["fd"]

        # Convergence: relative gap between θ and actual fd
        gap = fd_actual - theta_val
        rel_gap = gap / max(fd_actual, 1e-12)

        iter_time = time.perf_counter() - t_iter

        history.append({
            "iter":       k,
            "theta":      theta_val,
            "fd_actual":  fd_actual,
            "fd_cost":    oracle["fd_cost"],
            "gap":        gap,
            "rel_gap":    rel_gap,
            "obj":        obj_val,
            "n_cycles":   oracle["n_cycles"],
            "time_s":     iter_time,
        })

        # Print iteration
        print(f"    iter {k:2d}:  θ={theta_val:.8f}  fd={fd_actual:.8f}  "
              f"gap={rel_gap:+.4f}  cycles={oracle['n_cycles']:4d}  "
              f"t={iter_time:.2f}s")

        # Check convergence
        if k > 0 and abs(rel_gap) < conv_tol:
            print(f"    ✓ Converged at iteration {k}  (rel_gap={rel_gap:.6f})")
            break

        if k == max_cuts:
            print(f"    ⚠ Max iterations ({max_cuts}) reached, "
                  f"rel_gap={rel_gap:.6f}")
            break

        # ── Add cutting plane ────────────────────────────────────────────
        # subgrad_charge[t]    = ∂fd/∂charge[t]    (Shi Eq. 17)
        # subgrad_discharge[t] = ∂fd/∂discharge[t]  (Shi Eq. 18)
        # Both from validated compute_subgradient using rainflow cycles.
        #
        # Cutting plane (tangent to convex fd at current dispatch):
        #   θ ≥ fd_k + Σ_t [ sg_c[t]·(charge[t] − charge_k[t])
        #                   + sg_d[t]·(discharge[t] − discharge_k[t]) ]
        #
        # Rearranged:
        #   θ ≥ (fd_k − Σ sg_c·charge_k − Σ sg_d·discharge_k)
        #       + Σ sg_c·charge + Σ sg_d·discharge
        #
        
        # compute_subgradient returns ∂(cost)/∂power [$/MW].
        # θ is dimensionless fd, so we need ∂fd/∂power.
        # Divide by E_B = B × E_cap to remove the cost scaling.
        E_B_oracle = params["e_cost_usd_per_mwh"] * params["e_cap"]
        sg_c = oracle["subgrad_charge"]    / E_B_oracle   # now ∂fd/∂charge [1/MW]
        sg_d = oracle["subgrad_discharge"] / E_B_oracle   # now ∂fd/∂discharge [1/MW]

        rhs_const = fd_actual - float(
            np.dot(sg_c, p_charge) + np.dot(sg_d, p_discharge)
        )

        cut_expr = rhs_const + sum(
            sg_c[i] * model.charge[i] + sg_d[i] * model.discharge[i]
            for i in range(n)
            if (sg_c[i] > 1e-15 or sg_d[i] > 1e-15)
        )

        model.cuts.add(model.theta >= cut_expr)

    # ── Final results ────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_total_start
    p_net = p_discharge - p_charge

    revenue_usd = rev_scale * float(np.dot(price_usd, p_net)) * DT
    deg_cost_usd = fd_actual * E_B * 365.0 * 24.0 / n * factor

    return {
        "revenue_usd":  revenue_usd,
        "e_vec":        e_vec,
        "p_net":        p_net,
        "p_charge":     p_charge,
        "p_discharge":  p_discharge,
        "npv":          obj_val,
        "time_s":       total_time,
        "fd":           fd_actual,
        "fd_cost_usd":  deg_cost_usd,
        "n_iterations": len(history),
        "converged":    abs(history[-1]["rel_gap"]) < conv_tol if history else False,
        "history":      history,
    }


# ════════════════════════════════════════════════════════════════════════════
# 5.  COMPARISON + CSV EXPORT
# ════════════════════════════════════════════════════════════════════════════

def compare_results(
    lp:     dict,
    cp:     dict,
    params: dict,
    n:      int,
) -> None:
    """Print side-by-side comparison of baseline LP vs cutting-plane."""

    print("\n" + "=" * 78)
    print("COMPARISON:  Baseline LP  vs  Cutting-plane (Shi rainflow)")
    print("=" * 78)

    print(f"\n  {'Metric':<36s}  {'LP':>14s}  {'Cut-plane':>14s}  {'Delta':>10s}")
    print(f"  {'─'*36}  {'─'*14}  {'─'*14}  {'─'*10}")

    def row(label, lp_val, cp_val, fmt=".2f", pct=False):
        delta = cp_val - lp_val
        if pct and lp_val != 0:
            ds = f"{delta/abs(lp_val)*100:+.2f}%"
        else:
            ds = f"{delta:+{fmt}}"
        print(f"  {label:<36s}  {lp_val:>14{fmt}}  {cp_val:>14{fmt}}  {ds:>10s}")

    capex = params["p_cost_usd_per_mw"] * params["p_cap"] + params["e_cost_usd_per_mwh"] * params["e_cap"]

    row("Revenue [kUSD]",
        lp["revenue_usd"]/1e3, cp["revenue_usd"]/1e3, ".1f", pct=True)
    row("NPV (rev−capex) [MUSD]",
        (lp["revenue_usd"]-capex)/1e6, (cp["revenue_usd"]-capex)/1e6, ".3f", pct=True)
    row("NPV (rev−capex−deg) [MUSD]",
        (lp["revenue_usd"]-capex)/1e6,
        (cp["revenue_usd"]-capex-cp["fd_cost_usd"])/1e6, ".3f")
    row("Degradation cost [kUSD]",
        0.0, cp["fd_cost_usd"]/1e3, ".1f")
    row("Shi fd (dimensionless)",
        0.0, cp["fd"], ".8f")
    row("Solve time [s]",
        lp["time_s"], cp["time_s"], ".2f")

    lp_tp  = float(np.sum(np.abs(lp["p_vec"]))) * DT
    cp_tp  = float(np.sum(cp["p_charge"] + cp["p_discharge"])) * DT
    row("Energy throughput [MWh]", lp_tp, cp_tp, ".0f", pct=True)

    print(f"\n  Cutting-plane iterations: {cp['n_iterations']}")
    print(f"  Converged: {cp['converged']}")

    # Print convergence history
    if cp["history"]:
        print(f"\n  {'Iter':>4s}  {'θ':>12s}  {'fd':>12s}  {'gap':>10s}  {'cycles':>6s}")
        for h in cp["history"]:
            print(f"  {h['iter']:4d}  {h['theta']:12.8f}  {h['fd_actual']:12.8f}  "
                  f"{h['rel_gap']:+10.6f}  {h['n_cycles']:6d}")


def save_csv(
    lp:         dict,
    cp:         dict,
    params:     dict,
    power_wind: np.ndarray,
    price_eur:  np.ndarray,
    n:          int,
    c_deg_scale: float,
    max_cuts:    int,
    conv_tol:    float,
) -> None:
    """Save timestep and summary CSVs."""
    import pandas as pd
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    e_cap = params["e_cap"]

    # LP: decompose p_vec into charge/discharge
    lp_p = lp["p_vec"]
    lp_charge    = np.where(lp_p < 0, -lp_p, 0.0)
    lp_discharge = np.where(lp_p > 0,  lp_p, 0.0)

    rows = []
    for i in range(n):
        rows.append({
            "hour":                i,
            "price_eur":           round(float(price_eur[i]), 2),
            "wind_MW":             round(float(power_wind[i]), 2),
            "lp_soc_pct":          round(float(lp["e_vec"][i] / e_cap * 100), 3),
            "lp_charge_MW":        round(float(lp_charge[i]), 3),
            "lp_discharge_MW":     round(float(lp_discharge[i]), 3),
            "cp_soc_pct":          round(float(cp["e_vec"][i] / e_cap * 100), 3),
            "cp_charge_MW":        round(float(cp["p_charge"][i]), 3),
            "cp_discharge_MW":     round(float(cp["p_discharge"][i]), 3),
            "delta_soc_pct":       round(float((cp["e_vec"][i] - lp["e_vec"][i]) / e_cap * 100), 3),
        })

    # Format c_deg for filename: 1.0 → "cdeg1", 5.0 → "cdeg5", 0.5 → "cdeg0p5"
    cdeg_tag = f"cdeg{c_deg_scale:g}".replace(".", "p")

    ts_csv = OUTPUT_DIR / f"path1_timesteps_{n}h_{cdeg_tag}_{ts}.csv"
    pd.DataFrame(rows).to_csv(ts_csv, index=False)
    print(f"  Timestep CSV: {ts_csv.name}")

    # Summary
    capex = params["p_cost_usd_per_mw"] * params["p_cap"] + params["e_cost_usd_per_mwh"] * e_cap
    lp_tp = float(np.sum(np.abs(lp_p))) * DT
    cp_tp = float(np.sum(cp["p_charge"] + cp["p_discharge"])) * DT

    summary = {
        # ── Run configuration ──
        "timestamp":            ts,
        "n_hours":              n,
        "c_deg_scale":          c_deg_scale,
        "max_cuts":             max_cuts,
        "conv_tol":             conv_tol,
        # ── Battery parameters ──
        "e_cap_MWh":            e_cap,
        "p_cap_MW":             params["p_cap"],
        "soc_min":              params["soc_min"],
        "soc_max":              params["soc_max"],
        "eta_in":               params["eta_in"],
        "eta_out":              round(params["eta_out"], 6),
        "k3":                   params["k3"],
        "k4":                   params["k4"],
        "capex_usd":            round(capex, 0),
        # ── Baseline LP ──
        "lp_revenue_usd":       round(lp["revenue_usd"], 0),
        "lp_npv_usd":           round(lp["revenue_usd"] - capex, 0),
        "lp_throughput_MWh":    round(lp_tp, 0),
        "lp_solve_s":           round(lp["time_s"], 3),
        # ── Cutting-plane ──
        "cp_revenue_usd":       round(cp["revenue_usd"], 0),
        "cp_npv_usd":           round(cp["revenue_usd"] - capex, 0),
        "cp_npv_with_deg_usd":  round(cp["revenue_usd"] - capex - cp["fd_cost_usd"], 0),
        "cp_fd":                round(cp["fd"], 10),
        "cp_fd_cost_usd":       round(cp["fd_cost_usd"], 0),
        "cp_throughput_MWh":    round(cp_tp, 0),
        "cp_solve_s":           round(cp["time_s"], 3),
        "cp_iterations":        cp["n_iterations"],
        "cp_converged":         cp["converged"],
        "cp_final_gap":         round(cp["history"][-1]["rel_gap"], 6) if cp["history"] else None,
        # ── Deltas ──
        "delta_revenue_pct":    round((cp["revenue_usd"]/max(lp["revenue_usd"],1)-1)*100, 4),
        "delta_throughput_pct": round((cp_tp/max(lp_tp,1)-1)*100, 4),
    }

    # Convergence history
    for h in cp["history"]:
        summary[f"iter{h['iter']}_theta"] = round(h["theta"], 10)
        summary[f"iter{h['iter']}_fd"]    = round(h["fd_actual"], 10)

    sum_csv = OUTPUT_DIR / f"path1_summary_{n}h_{cdeg_tag}_{ts}.csv"
    pd.DataFrame([summary]).to_csv(sum_csv, index=False)
    print(f"  Summary CSV:  {sum_csv.name}")

    # ── Text report ──────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / f"path1_report_{n}h_{cdeg_tag}_{ts}.txt"
    _write_report(report_path, lp, cp, params, n, c_deg_scale, max_cuts, conv_tol)
    print(f"  Report TXT:   {report_path.name}")


def _write_report(
    path:        Path,
    lp:          dict,
    cp:          dict,
    params:      dict,
    n:           int,
    c_deg_scale: float,
    max_cuts:    int,
    conv_tol:    float,
) -> None:
    """Write a complete text report capturing all console output."""
    e_cap = params["e_cap"]
    capex = params["p_cost_usd_per_mw"] * params["p_cap"] + params["e_cost_usd_per_mwh"] * e_cap
    lp_tp = float(np.sum(np.abs(lp["p_vec"]))) * DT
    cp_tp = float(np.sum(cp["p_charge"] + cp["p_discharge"])) * DT

    lines = []
    w = lines.append   # shorthand

    w("=" * 78)
    w("PATH 1 — Cutting-plane dispatch with Shi rainflow degradation")
    w("=" * 78)
    w("")
    w("CONFIGURATION")
    w(f"  Horizon:         {n} hours ({n/24:.1f} days)")
    w(f"  c_deg:           {c_deg_scale}")
    w(f"  Max cuts:        {max_cuts}")
    w(f"  Tolerance:       {conv_tol}  ({conv_tol*100:.1f}%)")
    w(f"  E_cap:           {params['e_cap']:.0f} MWh")
    w(f"  P_cap:           {params['p_cap']:.0f} MW")
    w(f"  SoC window:      {params['soc_min']*100:.0f}%–{params['soc_max']*100:.0f}%")
    w(f"  RTE(ac):         {params['eta_out']*100:.1f}%")
    w(f"  Shi k3:          {params['k3']:.4e}")
    w(f"  Shi k4:          {params['k4']:.4f}")
    w(f"  Capex:           {capex/1e6:.3f} MUSD")

    w("")
    w("=" * 78)
    w("RESULTS COMPARISON")
    w("=" * 78)

    def row(label, lp_val, cp_val, fmt=".2f", pct=False):
        delta = cp_val - lp_val
        if pct and lp_val != 0:
            ds = f"{delta/abs(lp_val)*100:+.2f}%"
        else:
            ds = f"{delta:+{fmt}}"
        w(f"  {label:<36s}  {lp_val:>14{fmt}}  {cp_val:>14{fmt}}  {ds:>10s}")

    w(f"")
    w(f"  {'Metric':<36s}  {'LP':>14s}  {'Cut-plane':>14s}  {'Delta':>10s}")
    w(f"  {'─'*36}  {'─'*14}  {'─'*14}  {'─'*10}")

    row("Revenue [kUSD]",
        lp["revenue_usd"]/1e3, cp["revenue_usd"]/1e3, ".1f", pct=True)
    row("NPV (rev-capex) [MUSD]",
        (lp["revenue_usd"]-capex)/1e6, (cp["revenue_usd"]-capex)/1e6, ".3f", pct=True)
    row("NPV (rev-capex-deg) [MUSD]",
        (lp["revenue_usd"]-capex)/1e6,
        (cp["revenue_usd"]-capex-cp["fd_cost_usd"])/1e6, ".3f")
    row("Degradation cost [kUSD]",
        0.0, cp["fd_cost_usd"]/1e3, ".1f")
    row("Shi fd (dimensionless)",
        0.0, cp["fd"], ".8f")
    row("Solve time [s]",
        lp["time_s"], cp["time_s"], ".2f")
    row("Energy throughput [MWh]",
        lp_tp, cp_tp, ".0f", pct=True)

    w(f"")
    w(f"  Cutting-plane iterations: {cp['n_iterations']}")
    w(f"  Converged: {cp['converged']}")
    if cp["history"]:
        final = cp["history"][-1]
        w(f"  Final gap: {final['rel_gap']:+.6f}  ({final['rel_gap']*100:+.3f}%)")

    w("")
    w("=" * 78)
    w("CONVERGENCE HISTORY")
    w("=" * 78)
    w(f"")
    w(f"  {'Iter':>4s}  {'theta':>14s}  {'fd_actual':>14s}  {'gap':>12s}  "
      f"{'cycles':>6s}  {'time_s':>7s}")
    w(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*12}  {'─'*6}  {'─'*7}")
    for h in cp["history"]:
        w(f"  {h['iter']:4d}  {h['theta']:14.10f}  {h['fd_actual']:14.10f}  "
          f"{h['rel_gap']:+12.8f}  {h['n_cycles']:6d}  {h['time_s']:7.2f}")

    w("")
    w("=" * 78)

    path.write_text("\n".join(lines), encoding="utf-8")



def plot_comparison(
    lp:     dict,
    cp:     dict,
    params: dict,
    n:      int,
    c_deg:  float = 1.0,
) -> None:
    """Plot LP vs cutting-plane dispatch comparison."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots.")
        return

    e_cap = params["e_cap"]
    t = np.arange(n)

    # Use GridSpec: top 3 panels share x-axis (hours), bottom panel independent
    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 2, 1.5, 1.5], hspace=0.35)
    ax_soc   = fig.add_subplot(gs[0])
    ax_power = fig.add_subplot(gs[1], sharex=ax_soc)
    ax_delta = fig.add_subplot(gs[2], sharex=ax_soc)
    ax_conv  = fig.add_subplot(gs[3])   # independent x-axis

    # SoC
    ax_soc.plot(t, lp["e_vec"][:n] / e_cap * 100, label="LP baseline",
                color="#2166ac", linewidth=1)
    ax_soc.plot(t, cp["e_vec"][:n] / e_cap * 100, label="Cutting-plane + Shi",
                color="#b5351b", linewidth=1, linestyle="--")
    ax_soc.set_ylabel("SoC [%]")
    ax_soc.legend(loc="upper right")
    ax_soc.set_title("State of Charge comparison")

    # Power
    ax_power.plot(t, lp["p_vec"], label="LP net power",
                  color="#2166ac", linewidth=0.8, alpha=0.7)
    ax_power.plot(t, cp["p_net"], label="CP net power",
                  color="#b5351b", linewidth=0.8, alpha=0.7)
    ax_power.axhline(0, color="gray", linewidth=0.5)
    ax_power.set_ylabel("Battery power [MW]")
    ax_power.legend(loc="upper right")

    # ΔSoC
    ax_delta.fill_between(t, (cp["e_vec"][:n] - lp["e_vec"][:n]) / e_cap * 100,
                          alpha=0.4, color="#b5351b")
    ax_delta.axhline(0, color="gray", linewidth=0.5)
    ax_delta.set_ylabel("ΔSoC (CP − LP) [%]")
    ax_delta.set_xlabel("Hour")

    # Convergence (independent x-axis: iteration number)
    iters  = [h["iter"] for h in cp["history"]]
    thetas = [h["theta"] for h in cp["history"]]
    fds    = [h["fd_actual"] for h in cp["history"]]
    ax_conv.plot(iters, thetas, "o-", label="θ (LP estimate)", color="#2166ac", markersize=5)
    ax_conv.plot(iters, fds,    "s-", label="fd (oracle)",     color="#b5351b", markersize=5)
    ax_conv.set_ylabel("Degradation fd")
    ax_conv.set_xlabel("Cutting-plane iteration")
    ax_conv.legend(loc="center right")
    ax_conv.set_title("Convergence: θ → fd")
    # ax_conv.set_xticks(iters)
    ax_conv.set_xlim(-0.5, max(iters) + 0.5)

    fig.suptitle(f"Path 1: Cutting-plane (Shi rainflow) vs baseline LP  "
                 f"({n} hours, c_deg={c_deg})",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.93)

    from datetime import datetime
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    cdeg_tag = f"cdeg{c_deg:g}".replace(".", "p")
    out = OUTPUT_DIR / f"path1_comparison_{n}h_{cdeg_tag}_{ts_str}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {out.name}")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Path 1: Cutting-plane dispatch with Shi rainflow degradation"
    )
    parser.add_argument("--hours", type=int, default=None,
                        help=f"Number of timesteps (toggle: {RUN_HOURS})")
    parser.add_argument("--full", action="store_true", default=None,
                        help="Run full year (8760 hours)")
    parser.add_argument("--max-cuts", type=int, default=None,
                        help=f"Max cutting-plane iterations (toggle: {MAX_CUTS})")
    parser.add_argument("--tol", type=float, default=None,
                        help=f"Convergence tolerance (toggle: {CONV_TOL})")
    parser.add_argument("--c-deg", type=float, default=None,
                        help=f"Degradation cost scale factor (toggle: {C_DEG})")
    parser.add_argument("--no-plot", action="store_true", default=None,
                        help="Skip matplotlib plots")
    args = parser.parse_args()

    # Resolve: CLI flags override toggles at top of file
    use_full    = args.full     if args.full     is not None else RUN_FULL
    n_hours     = 8760          if use_full else (args.hours    or RUN_HOURS)
    max_cuts    = args.max_cuts if args.max_cuts is not None else MAX_CUTS
    conv_tol    = args.tol      if args.tol      is not None else CONV_TOL
    c_deg       = args.c_deg    if args.c_deg    is not None else C_DEG
    skip_plot   = args.no_plot  if args.no_plot  is not None else SKIP_PLOT

    print("=" * 78)
    print("PATH 1 — Cutting-plane dispatch with Shi rainflow degradation")
    print("=" * 78)
    print(f"  Configuration:")
    print(f"    Horizon:       {n_hours} hours ({n_hours/24:.1f} days)")
    print(f"    c_deg:         {c_deg}")
    print(f"    Max cuts:      {max_cuts}")
    print(f"    Tolerance:     {conv_tol}  ({conv_tol*100:.1f}%)")
    print(f"    Skip plot:     {skip_plot}")
    print(f"    Source:        {'CLI override' if any(v is not None for v in [args.hours, args.full, args.max_cuts, args.tol, args.c_deg]) else 'file toggles'}")

    # ── Load data ────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading data...")
    power_wind, price_eur, params = load_data(n_hours)
    n = len(price_eur)
    print(f"  Loaded {n} hours of data")

    # ── Baseline LP ──────────────────────────────────────────────────────
    print(f"\n[2/4] Solving baseline LP (Gurobi)...")
    lp = solve_baseline_lp(power_wind, price_eur, params)
    print(f"  Done in {lp['time_s']:.2f}s | Revenue: {lp['revenue_usd']/1e3:.1f} kUSD")

    # ── Cutting-plane solve ──────────────────────────────────────────────
    print(f"\n[3/4] Solving with cutting planes...")
    cp = solve_cutting_plane(
        power_wind, price_eur, params,
        max_cuts=max_cuts,
        conv_tol=conv_tol,
        c_deg_scale=c_deg,
    )
    print(f"  Done in {cp['time_s']:.2f}s | Revenue: {cp['revenue_usd']/1e3:.1f} kUSD")

    # ── Compare ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Comparison...")
    compare_results(lp, cp, params, n)
    save_csv(lp, cp, params, power_wind, price_eur, n, c_deg, max_cuts, conv_tol)

    if not skip_plot:
        plot_comparison(lp, cp, params, n, c_deg=c_deg)

    print("\nDone.")
    return lp, cp


if __name__ == "__main__":
    main()