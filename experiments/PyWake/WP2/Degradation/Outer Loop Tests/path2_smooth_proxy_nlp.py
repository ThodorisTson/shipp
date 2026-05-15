"""Path 2 — Smooth proxy NLP: monolithic dispatch + degradation with IPOPT.

Purpose
-------
Test whether IPOPT can solve a dispatch optimization that includes a
degradation penalty directly in the objective function.  The penalty
uses a smooth per-timestep energy-throughput proxy (NOT rainflow-cycle-
based) with the Shi polynomial stress function Phi(d) = k3 * d^k4.

Architecture
    Baseline LP  →  Gurobi, same relaxed formulation as kernel_pyomo.py
    NLP          →  IPOPT, split charge/discharge variables,
                    exact energy balance (equalities, no big-M),
                    + smooth degradation penalty in objective

Key question answered
    "Can it be solved?  In reasonable time?" — Jenna, 25 Apr 2026

Usage
    python path2_smooth_proxy_nlp.py               # 168 h (1 week)
    python path2_smooth_proxy_nlp.py --hours 100    # custom
    python path2_smooth_proxy_nlp.py --full          # 8760 h (full year)

Requires
    - SHIPP installed (pip install -e .)  with Gurobi + IPOPT available
    - WP2_HPP.yaml, dk1_prices_2022.csv in parent directory (../shipp/)
    - wp2_common.py, degradation_*.py in parent directory (auto-imported)

Folder structure
    shipp/
    ├── wp2_common.py
    ├── degradation_xu.py
    ├── degradation_shi.py
    ├── degradation_subgradient.py
    ├── WP2_HPP.yaml
    ├── dk1_prices_2022.csv
    └── Outer Loop Tests/
        ├── path2_smooth_proxy_nlp.py    ← this script
        └── Results/                     ← outputs saved here

Output
    Console comparison table + optional matplotlib plots.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

# ─── Path setup ─────────────────────────────────────────────────────────────
# This script lives in  .../shipp/Outer Loop Tests/
# All WP2 data files and local modules live one level up in  .../shipp/
SCRIPT_DIR  = Path(__file__).parent            # .../Outer Loop Tests/
PARENT_DIR  = SCRIPT_DIR.parent                # .../shipp/
OUTPUT_DIR  = SCRIPT_DIR / "Results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Add parent to sys.path so wp2_common, degradation_*.py are importable
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import numpy as np
import numpy_financial as npf
import pyomo.environ as pyo

# ─── SHIPP imports (installed package) ──────────────────────────────────────
from shipp.kernel_pyomo import solve_lp_pyomo
from shipp.components import Storage, Production, TimeSeries

# ─── Data files (in parent directory) ───────────────────────────────────────
HPP_YAML   = PARENT_DIR / "WP2_HPP.yaml"
PRICE_CSV  = PARENT_DIR / "dk1_prices_2022.csv"


# ════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

# Economics
EUR_TO_USD    = 1.18
DISCOUNT_RATE = 0.03
N_YEARS       = 20
DT            = 1.0       # hours

# Battery (WP2 defaults — overridden by YAML if available)
E_CAP_MWH   = 300.0
P_CAP_MW    = 150.0
ETA_IN      = 1.0         # charge efficiency
RTE_DC      = 0.95        # DC round-trip efficiency  (from YAML: rte_nominal)
PCU_EFF     = 0.975       # power conversion unit efficiency
SOC_MIN     = 0.10
SOC_MAX     = 0.90

# Shi polynomial (default from 10-90% SoC window fit)
K3_DEFAULT  = 3.24e-5
K4_DEFAULT  = 1.179

# Grid
P_MIN_MW    = 0.0
P_MAX_MW    = 300.0       # grid connection capacity

# Degradation cost scaling
# c_deg = fraction of replacement cost used as penalty weight.
# 1.0 means full replacement cost.  Tune this to explore sensitivity.
C_DEG_SCALE = 1.0


# ════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_data(n_hours: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load wind power and price data from WP2 pipeline.

    Returns (power_wind_MW, price_eur, params_dict).
    Falls back to synthetic data if WP2 files are not found.
    """
    params = {
        "e_cap": E_CAP_MWH,
        "p_cap": P_CAP_MW,
        "eta_in": ETA_IN,
        "eta_out": RTE_DC * PCU_EFF**2,
        "soc_min": SOC_MIN,
        "soc_max": SOC_MAX,
        "e_cost_usd_per_mwh": 0.0,  # placeholder
        "p_cost_usd_per_mw":  0.0,
        "p_max": P_MAX_MW,
        "k3": K3_DEFAULT,
        "k4": K4_DEFAULT,
    }

    try:
        import pandas as pd
        import xarray as xr
        from py_wake.site import XRSite
        from wp2_common import quick_setup, get_wake_model
        from degradation_subgradient import fit_shi_polynomial

        print("  Loading WP2 configuration from YAML...")
        setup = quick_setup(HPP_YAML, config={"interp_n": 2000}, verbose=False)
        hpp   = setup["hpp"]

        # Wind resource
        ts  = hpp["site"]["energy_resource"]["time_series"]["wind_resource"]
        ws  = np.asarray(ts["wind_speed"],     dtype=float)
        wd  = np.asarray(ts["wind_direction"], dtype=float)
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
        site     = XRSite(ds=xr.Dataset(data_vars=dict(P=1)))
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
        rte_dc  = float(bat["rte_nominal"])
        pcu_eff = float(bat["pcu_efficiency"])
        rte_ac  = rte_dc * pcu_eff**2

        params["e_cap"]   = float(bat["energy_capacity_Wh"]) / 1e6
        params["p_cap"]   = float(bat["power_capacity_W"])   / 1e6
        params["eta_out"] = rte_ac
        params["soc_min"] = float(bat.get("soc_min", SOC_MIN))
        params["soc_max"] = float(bat.get("soc_max", SOC_MAX))
        params["e_cost_usd_per_mwh"] = float(bat["capex_EUR_per_kWh"]) * 1000.0 * EUR_TO_USD
        params["p_cost_usd_per_mw"]  = float(bat["capex_EUR_per_kW"])  * 1000.0 * EUR_TO_USD
        params["p_max"] = float(hpp["grid_connection_capacity"]) / 1e6

        # Shi polynomial fit for this SoC window
        shi_fit = fit_shi_polynomial(params["soc_min"], params["soc_max"],
                                     verbose=True)
        params["k3"] = shi_fit.k3
        params["k4"] = shi_fit.k4

        print(f"  Battery: {params['p_cap']:.0f} MW / {params['e_cap']:.0f} MWh")
        print(f"  Grid:    {params['p_max']:.0f} MW")
        print(f"  RTE(ac): {params['eta_out']*100:.1f}%")
        print(f"  SoC:     {params['soc_min']*100:.0f}%–{params['soc_max']*100:.0f}%")
        print(f"  Shi fit: k3={params['k3']:.4e}, k4={params['k4']:.4f}")

        return power_wind_MW[:n], price_eur[:n], params

    except Exception as exc:
        print(f"  WP2 data unavailable ({exc}), using synthetic data")
        return _synthetic_data(n_hours, params)


def _synthetic_data(n: int, params: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Generate synthetic wind + price for testing without WP2 files."""
    np.random.seed(42)
    t = np.arange(n)

    # Sinusoidal wind with noise (mean ~120 MW, cap at p_max)
    wind = 120.0 + 60.0 * np.sin(2 * np.pi * t / 168) + 20.0 * np.random.randn(n)
    wind = np.clip(wind, 0, params["p_max"])

    # Price with daily pattern + volatility (mean ~60 EUR/MWh)
    price = 60.0 + 30.0 * np.sin(2 * np.pi * t / 24 - np.pi/2) + 10.0 * np.random.randn(n)
    price = np.clip(price, 1.0, 300.0)

    params["e_cost_usd_per_mwh"] = 200.0 * 1000.0 * EUR_TO_USD  # 200 EUR/kWh
    params["p_cost_usd_per_mw"]  = 0.0

    print(f"  Synthetic: {n} hours, wind mean={wind.mean():.1f} MW, "
          f"price mean={price.mean():.1f} EUR/MWh")
    return wind, price, params


# ════════════════════════════════════════════════════════════════════════════
# 2.  BASELINE LP  (Gurobi, same as kernel_pyomo.py)
# ════════════════════════════════════════════════════════════════════════════

def solve_baseline_lp(
    power_wind: np.ndarray,
    price_eur:  np.ndarray,
    params:     dict,
) -> dict:
    """Solve the dispatch-fixed LP using the existing SHIPP formulation.

    Returns dict with keys: revenue, e_vec, p_vec, obj, time_s.
    """
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
                   p_cost=params["p_cost_usd_per_mw"],
                   dod=dod)
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

    # Revenue in USD (annual, discounted)
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
# 3.  NLP WITH SMOOTH DEGRADATION PROXY  (IPOPT)
# ════════════════════════════════════════════════════════════════════════════

def solve_nlp_with_degradation(
    power_wind: np.ndarray,
    price_eur:  np.ndarray,
    params:     dict,
    c_deg_scale: float = C_DEG_SCALE,
) -> dict:
    """Solve dispatch NLP with smooth degradation proxy using IPOPT.

    The formulation uses split charge/discharge variables and an exact
    energy balance (equalities).  The degradation proxy penalizes each
    timestep's normalised energy throughput via Phi(delta) = k3 * delta^k4.

    This is NOT cycle-based degradation.  It is a smooth proxy that
    captures the nonlinear DoD-stress relationship and tests NLP tractability.

    Args:
        power_wind:  Wind power time series [MW].
        price_eur:   Day-ahead price [EUR/MWh].
        params:      Battery and grid parameters.
        c_deg_scale: Fraction of replacement cost used as degradation penalty.

    Returns:
        dict with revenue, e_vec, p_charge, p_discharge, obj, time_s,
        degradation_cost.
    """
    n       = len(price_eur)
    e_cap   = params["e_cap"]
    p_cap   = params["p_cap"]
    eta_in  = params["eta_in"]
    eta_out = params["eta_out"]
    soc_min = params["soc_min"]
    soc_max = params["soc_max"]
    p_max   = params["p_max"]
    k3      = params["k3"]
    k4      = params["k4"]
    e_cost  = params["e_cost_usd_per_mwh"]

    price_usd = price_eur * EUR_TO_USD
    power_res = power_wind.copy()

    # NPV discount factor (same as kernel_pyomo.py)
    factor = npf.npv(DISCOUNT_RATE, np.ones(N_YEARS)) - 1

    # Revenue scaling: annualise from n hours to 8760 hours
    rev_scale = 365.0 * 24.0 / n * factor

    # Degradation cost: battery replacement cost × NPV factor × annual scaling
    # E_B in Shi's notation = e_cost_per_MWh × E_cap (total replacement cost)
    E_B       = e_cost * e_cap * c_deg_scale
    deg_scale = 365.0 * 24.0 / n * factor * E_B

    # SoC bounds
    e_min = e_cap * (1.0 - (1.0 - soc_min))  # = e_cap * soc_min
    e_max = e_cap * soc_max

    # ── Build Pyomo model ────────────────────────────────────────────────
    model = pyo.ConcreteModel()
    N = list(range(n))
    Np1 = list(range(n + 1))
    model.N   = pyo.Set(initialize=N)
    model.Np1 = pyo.Set(initialize=Np1)

    # Decision variables: split charge/discharge (both non-negative)
    model.charge    = pyo.Var(model.N, bounds=(0, p_cap), initialize=0)
    model.discharge = pyo.Var(model.N, bounds=(0, p_cap), initialize=0)
    model.e         = pyo.Var(model.Np1, bounds=(e_min, e_max),
                              initialize=e_cap * 0.5)

    # ── Energy balance (equality constraints) ────────────────────────────
    def rule_energy_balance(m, i):
        return (m.e[i+1] == m.e[i]
                + DT * eta_in  * m.charge[i]
                - DT / eta_out * m.discharge[i])
    model.energy_balance = pyo.Constraint(model.N, rule=rule_energy_balance)

    # ── Periodic SoC constraint ──────────────────────────────────────────
    model.periodic = pyo.Constraint(expr=model.e[0] == model.e[n])

    # ── Grid constraints ─────────────────────────────────────────────────
    def rule_grid_max(m, i):
        # Net power to grid = discharge - charge + wind <= p_max
        return m.discharge[i] - m.charge[i] <= max(p_max - power_res[i], 0)
    model.grid_max = pyo.Constraint(model.N, rule=rule_grid_max)

    def rule_grid_min(m, i):
        # Net power >= p_min - wind  (p_min = 0 typically)
        return m.discharge[i] - m.charge[i] >= P_MIN_MW - power_res[i]
    model.grid_min = pyo.Constraint(model.N, rule=rule_grid_min)

    # ── Objective: revenue − capex − degradation proxy ───────────────────
    #
    # Revenue term:  rev_scale * Σ price_i * (discharge_i - charge_i)
    #
    # Degradation proxy (per-timestep throughput stress):
    #   deg_scale * Σ [ Phi(delta_charge_i)/2 + Phi(delta_discharge_i)/2 ]
    #
    # where delta_charge_i   = dt * eta_in * charge_i / e_cap
    #       delta_discharge_i = dt / (eta_out * e_cap) * discharge_i
    #       Phi(delta) = k3 * delta^k4
    #
    # The /2 is because each timestep is a half-cycle (Shi Eq. 6).
    #
    # Note: k4 > 1 (=1.179), so x^k4 is smooth and convex for x >= 0.
    #       Pyomo + IPOPT handle this via automatic differentiation.

    # Pre-compute normalisation constants
    norm_c = DT * eta_in  / e_cap     # charge normalisation
    norm_d = DT / (eta_out * e_cap)   # discharge normalisation

    # Capex (fixed battery, so this is a constant offset)
    capex = params["p_cost_usd_per_mw"] * p_cap + e_cost * e_cap

    # Build objective expression
    revenue_expr = rev_scale * sum(
        price_usd[i] * (model.discharge[i] - model.charge[i])
        for i in N
    )

    # Degradation proxy: sum of Phi(delta)/2 for each timestep
    # Using (a * x)^k4 = a^k4 * x^k4 to keep Pyomo expression clean
    coeff_c = k3 * norm_c**k4 / 2.0   # constant factor for charge terms
    coeff_d = k3 * norm_d**k4 / 2.0   # constant factor for discharge terms

    # CRITICAL: for x^k4 with non-integer k4, x must be strictly positive
    # for IPOPT's log-barrier.  The bounds=(0, p_cap) handles this, but
    # we add a tiny epsilon shift to avoid numerical issues at x=0.
    eps = 1e-8

    deg_expr = deg_scale * sum(
        coeff_c * (model.charge[i]    + eps)**k4
      + coeff_d * (model.discharge[i] + eps)**k4
        for i in N
    )

    model.obj = pyo.Objective(
        expr=revenue_expr - capex - deg_expr,
        sense=pyo.maximize,
    )

    # ── Solve ────────────────────────────────────────────────────────────
    solver = pyo.SolverFactory("ipopt")
    solver.options["max_iter"]         = 10000
    solver.options["tol"]              = 1e-6
    solver.options["acceptable_tol"]   = 1e-4
    solver.options["print_level"]      = 5     # moderate output

    t0 = time.perf_counter()
    results = solver.solve(model, tee=True)
    wall_s = time.perf_counter() - t0

    # ── Check solution status ────────────────────────────────────────────
    ok = (results.solver.status == pyo.SolverStatus.ok and
          results.solver.termination_condition == pyo.TerminationCondition.optimal)
    if not ok:
        print(f"\n  ⚠ IPOPT status: {results.solver.status}, "
              f"termination: {results.solver.termination_condition}")

    # ── Extract results ──────────────────────────────────────────────────
    e_vec       = np.array([pyo.value(model.e[i])         for i in Np1])
    p_charge    = np.array([pyo.value(model.charge[i])    for i in N])
    p_discharge = np.array([pyo.value(model.discharge[i]) for i in N])
    p_net       = p_discharge - p_charge   # same sign convention as LP

    # Revenue (same calculation as baseline)
    revenue_usd = rev_scale * float(np.dot(price_usd, p_net)) * DT

    # Degradation cost (from objective)
    deg_cost = float(pyo.value(deg_expr))

    # Proxy fd (dimensionless, for comparison with Shi fd)
    proxy_fd = float(pyo.value(deg_expr)) / (deg_scale) if deg_scale > 0 else 0

    return {
        "revenue_usd":  revenue_usd,
        "e_vec":        e_vec,
        "p_net":        p_net,
        "p_charge":     p_charge,
        "p_discharge":  p_discharge,
        "npv":          float(pyo.value(model.obj)),
        "time_s":       wall_s,
        "deg_cost_usd": deg_cost,
        "proxy_fd":     proxy_fd,
        "converged":    ok,
    }


# ════════════════════════════════════════════════════════════════════════════
# 4.  COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def compare_results(
    lp:     dict,
    nlp:    dict,
    params: dict,
    n:      int,
) -> None:
    """Print side-by-side comparison of LP baseline vs NLP with degradation."""

    print("\n" + "=" * 78)
    print("COMPARISON:  Baseline LP  vs  NLP with smooth degradation proxy")
    print("=" * 78)

    print(f"\n  {'Metric':<32s}  {'LP (Gurobi)':>16s}  {'NLP (IPOPT)':>16s}  {'Delta':>10s}")
    print(f"  {'─'*32}  {'─'*16}  {'─'*16}  {'─'*10}")

    def row(label, lp_val, nlp_val, fmt=".2f", pct=False):
        delta = nlp_val - lp_val
        if pct and lp_val != 0:
            delta_str = f"{delta/abs(lp_val)*100:+.2f}%"
        else:
            delta_str = f"{delta:+{fmt}}"
        print(f"  {label:<32s}  {lp_val:>16{fmt}}  {nlp_val:>16{fmt}}  {delta_str:>10s}")

    row("Revenue [kUSD]",
        lp["revenue_usd"] / 1e3, nlp["revenue_usd"] / 1e3, ".1f", pct=True)

    # Comparable NPV: Revenue − Capex (same formula for both, ignoring deg penalty)
    capex = params["p_cost_usd_per_mw"] * params["p_cap"] + params["e_cost_usd_per_mwh"] * params["e_cap"]
    lp_npv_comparable  = lp["revenue_usd"]  - capex
    nlp_npv_comparable = nlp["revenue_usd"] - capex
    nlp_npv_with_deg   = nlp_npv_comparable - nlp["deg_cost_usd"]

    row("NPV (rev−capex) [MUSD]",
        lp_npv_comparable / 1e6, nlp_npv_comparable / 1e6, ".3f", pct=True)
    row("NPV (rev−capex−deg) [MUSD]",
        lp_npv_comparable / 1e6, nlp_npv_with_deg / 1e6, ".3f")
    row("Degradation cost [kUSD]",
        0.0, nlp["deg_cost_usd"] / 1e3, ".1f")
    row("Solve time [s]",
        lp["time_s"], nlp["time_s"], ".2f")

    # Energy throughput
    lp_throughput  = float(np.sum(np.abs(lp["p_vec"]))) * DT
    nlp_throughput = float(np.sum(nlp["p_charge"] + nlp["p_discharge"])) * DT

    row("Energy throughput [MWh]",
        lp_throughput, nlp_throughput, ".0f", pct=True)

    # Peak charge/discharge
    lp_max_charge    = float(np.min(lp["p_vec"]))
    lp_max_discharge = float(np.max(lp["p_vec"]))
    nlp_max_charge   = float(np.max(nlp["p_charge"]))
    nlp_max_discharge= float(np.max(nlp["p_discharge"]))

    print(f"\n  {'Peak charge [MW]':<32s}  {abs(lp_max_charge):>16.1f}  "
          f"{nlp_max_charge:>16.1f}")
    print(f"  {'Peak discharge [MW]':<32s}  {lp_max_discharge:>16.1f}  "
          f"{nlp_max_discharge:>16.1f}")

    # SoC statistics
    lp_soc  = lp["e_vec"]  / params["e_cap"] * 100
    nlp_soc = nlp["e_vec"] / params["e_cap"] * 100

    print(f"\n  {'SoC mean [%]':<32s}  {np.mean(lp_soc):>16.1f}  "
          f"{np.mean(nlp_soc):>16.1f}")
    print(f"  {'SoC std [%]':<32s}  {np.std(lp_soc):>16.1f}  "
          f"{np.std(nlp_soc):>16.1f}")

    # NLP-specific
    print(f"\n  NLP proxy fd:          {nlp['proxy_fd']:>10.6f}")
    print(f"  NLP converged:         {nlp['converged']}")

    # Scalability note
    print(f"\n  Horizon: {n} timesteps")
    print(f"  Variables: {2*n} power + {n+1} energy = {3*n+1} total (NLP)")
    print(f"  Constraints: {n} energy_balance + {2*n} grid + 1 periodic = {3*n+1}")


def save_csv(
    lp:          dict,
    nlp:         dict,
    params:      dict,
    power_wind:  np.ndarray,
    price_eur:   np.ndarray,
    n:           int,
    c_deg_scale: float,
) -> None:
    """Save timestep-level and summary CSV files for detailed analysis.

    Creates two files in OUTPUT_DIR:
        path2_timesteps_{n}h_{timestamp}.csv  — per-timestep data
        path2_summary_{n}h_{timestamp}.csv    — scalar metrics
    """
    import pandas as pd
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    e_cap = params["e_cap"]
    k3    = params["k3"]
    k4    = params["k4"]
    eta_in  = params["eta_in"]
    eta_out = params["eta_out"]

    # ── Timestep CSV ─────────────────────────────────────────────────────
    # LP: decompose single p_vec into charge/discharge
    lp_p = lp["p_vec"]
    lp_charge    = np.where(lp_p < 0, -lp_p, 0.0)    # positive when charging
    lp_discharge = np.where(lp_p > 0,  lp_p, 0.0)

    # Per-timestep normalised DoD proxy (same formula as NLP objective)
    norm_c = DT * eta_in  / e_cap
    norm_d = DT / (eta_out * e_cap)

    def proxy_stress(charge, discharge):
        """Per-timestep Phi contribution (half-cycle each)."""
        phi_c = k3 * (norm_c * charge)**k4  / 2.0
        phi_d = k3 * (norm_d * discharge)**k4 / 2.0
        return phi_c + phi_d

    lp_stress  = proxy_stress(lp_charge, lp_discharge)
    nlp_stress = proxy_stress(nlp["p_charge"], nlp["p_discharge"])

    rows = []
    for i in range(n):
        rows.append({
            "hour":                i,
            "price_eur":           round(float(price_eur[i]), 2),
            "price_usd":           round(float(price_eur[i] * EUR_TO_USD), 2),
            "wind_MW":             round(float(power_wind[i]), 2),
            # LP
            "lp_soc_pct":          round(float(lp["e_vec"][i] / e_cap * 100), 3),
            "lp_p_net_MW":         round(float(lp_p[i]), 3),
            "lp_charge_MW":        round(float(lp_charge[i]), 3),
            "lp_discharge_MW":     round(float(lp_discharge[i]), 3),
            "lp_proxy_stress":     round(float(lp_stress[i]), 10),
            # NLP
            "nlp_soc_pct":         round(float(nlp["e_vec"][i] / e_cap * 100), 3),
            "nlp_p_net_MW":        round(float(nlp["p_net"][i]), 3),
            "nlp_charge_MW":       round(float(nlp["p_charge"][i]), 3),
            "nlp_discharge_MW":    round(float(nlp["p_discharge"][i]), 3),
            "nlp_proxy_stress":    round(float(nlp_stress[i]), 10),
            # Deltas
            "delta_soc_pct":       round(float((nlp["e_vec"][i] - lp["e_vec"][i]) / e_cap * 100), 3),
            "delta_p_net_MW":      round(float(nlp["p_net"][i] - lp_p[i]), 3),
            "delta_stress":        round(float(nlp_stress[i] - lp_stress[i]), 10),
        })

    ts_csv = OUTPUT_DIR / f"path2_timesteps_{n}h_{ts}.csv"
    pd.DataFrame(rows).to_csv(ts_csv, index=False)
    print(f"  Timestep CSV: {ts_csv.name}  ({n} rows)")

    # ── Summary CSV ──────────────────────────────────────────────────────
    capex = params["p_cost_usd_per_mw"] * params["p_cap"] + params["e_cost_usd_per_mwh"] * params["e_cap"]

    lp_throughput  = float(np.sum(np.abs(lp_p))) * DT
    nlp_throughput = float(np.sum(nlp["p_charge"] + nlp["p_discharge"])) * DT

    summary = {
        "timestamp":              ts,
        "n_hours":                n,
        "c_deg_scale":            c_deg_scale,
        "e_cap_MWh":              params["e_cap"],
        "p_cap_MW":               params["p_cap"],
        "soc_min":                params["soc_min"],
        "soc_max":                params["soc_max"],
        "k3":                     params["k3"],
        "k4":                     params["k4"],
        "eta_in":                 params["eta_in"],
        "eta_out":                params["eta_out"],
        "capex_usd":              round(capex, 0),
        # LP
        "lp_revenue_usd":         round(lp["revenue_usd"], 0),
        "lp_npv_usd":             round(lp["revenue_usd"] - capex, 0),
        "lp_throughput_MWh":      round(lp_throughput, 0),
        "lp_proxy_fd_total":      round(float(np.sum(lp_stress)), 8),
        "lp_soc_mean_pct":        round(float(np.mean(lp["e_vec"][:n] / e_cap * 100)), 2),
        "lp_soc_std_pct":         round(float(np.std(lp["e_vec"][:n] / e_cap * 100)), 2),
        "lp_solve_s":             round(lp["time_s"], 3),
        # NLP
        "nlp_revenue_usd":        round(nlp["revenue_usd"], 0),
        "nlp_npv_usd":            round(nlp["revenue_usd"] - capex, 0),
        "nlp_npv_with_deg_usd":   round(nlp["revenue_usd"] - capex - nlp["deg_cost_usd"], 0),
        "nlp_deg_cost_usd":       round(nlp["deg_cost_usd"], 0),
        "nlp_throughput_MWh":     round(nlp_throughput, 0),
        "nlp_proxy_fd_total":     round(nlp["proxy_fd"], 8),
        "nlp_soc_mean_pct":       round(float(np.mean(nlp["e_vec"][:n] / e_cap * 100)), 2),
        "nlp_soc_std_pct":        round(float(np.std(nlp["e_vec"][:n] / e_cap * 100)), 2),
        "nlp_solve_s":            round(nlp["time_s"], 3),
        "nlp_converged":          nlp["converged"],
        # Deltas
        "delta_revenue_pct":      round((nlp["revenue_usd"] / max(lp["revenue_usd"], 1) - 1) * 100, 4),
        "delta_throughput_pct":   round((nlp_throughput / max(lp_throughput, 1) - 1) * 100, 4),
    }

    sum_csv = OUTPUT_DIR / f"path2_summary_{n}h_{ts}.csv"
    pd.DataFrame([summary]).to_csv(sum_csv, index=False)
    print(f"  Summary CSV:  {sum_csv.name}")


def plot_comparison(
    lp:     dict,
    nlp:    dict,
    params: dict,
    n:      int,
) -> None:
    """Plot LP vs NLP dispatch and SoC side by side."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots.")
        return

    t = np.arange(n)
    e_cap = params["e_cap"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # SoC
    ax = axes[0]
    ax.plot(t, lp["e_vec"][:n]  / e_cap * 100, label="LP baseline",
            color="#2166ac", linewidth=1)
    ax.plot(t, nlp["e_vec"][:n] / e_cap * 100, label="NLP + degradation",
            color="#b5351b", linewidth=1, linestyle="--")
    ax.set_ylabel("SoC [%]")
    ax.legend(loc="upper right")
    ax.set_title("State of Charge comparison")
    ax.set_ylim(params["soc_min"]*100 - 2, params["soc_max"]*100 + 2)

    # Power
    ax = axes[1]
    ax.plot(t, lp["p_vec"],  label="LP net power",
            color="#2166ac", linewidth=0.8, alpha=0.7)
    ax.plot(t, nlp["p_net"], label="NLP net power",
            color="#b5351b", linewidth=0.8, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("Battery power [MW]")
    ax.legend(loc="upper right")

    # Difference
    ax = axes[2]
    delta_soc = (nlp["e_vec"][:n] - lp["e_vec"][:n]) / e_cap * 100
    ax.fill_between(t, delta_soc, alpha=0.4, color="#b5351b")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("ΔSoC (NLP − LP) [%]")
    ax.set_xlabel("Hour")

    fig.suptitle(f"Path 2: Smooth proxy NLP vs baseline LP  ({n} hours)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Save with timestamp
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"path2_comparison_{n}h_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {out.name}")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Path 2: Smooth proxy NLP test for monolithic dispatch + degradation"
    )
    parser.add_argument("--hours", type=int, default=168,
                        help="Number of timesteps (default: 168 = 1 week)")
    parser.add_argument("--full", action="store_true",
                        help="Run full year (8760 hours)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip matplotlib plots")
    parser.add_argument("--c-deg", type=float, default=C_DEG_SCALE,
                        help="Degradation cost scale factor (default: 1.0)")
    args = parser.parse_args()

    n_hours = 8760 if args.full else args.hours

    print("=" * 78)
    print("PATH 2 — Smooth proxy NLP: monolithic dispatch + degradation")
    print("=" * 78)
    print(f"  Horizon:  {n_hours} hours ({n_hours/24:.1f} days)")
    print(f"  c_deg:    {args.c_deg}")

    # ── Load data ────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading data...")
    power_wind, price_eur, params = load_data(n_hours)
    n = len(price_eur)
    print(f"  Loaded {n} hours of data")

    # ── Baseline LP ──────────────────────────────────────────────────────
    print(f"\n[2/4] Solving baseline LP (Gurobi)...")
    lp = solve_baseline_lp(power_wind, price_eur, params)
    print(f"  Done in {lp['time_s']:.2f}s | Revenue: {lp['revenue_usd']/1e3:.1f} kUSD")

    # ── NLP with degradation ─────────────────────────────────────────────
    print(f"\n[3/4] Solving NLP with degradation proxy (IPOPT)...")
    nlp = solve_nlp_with_degradation(
        power_wind, price_eur, params, c_deg_scale=args.c_deg,
    )
    print(f"  Done in {nlp['time_s']:.2f}s | Revenue: {nlp['revenue_usd']/1e3:.1f} kUSD")

    # ── Compare ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Comparison...")
    compare_results(lp, nlp, params, n)

    save_csv(lp, nlp, params, power_wind, price_eur, n, c_deg_scale=args.c_deg)

    if not args.no_plot:
        plot_comparison(lp, nlp, params, n)

    print("\nDone.")
    return lp, nlp


if __name__ == "__main__":
    main()