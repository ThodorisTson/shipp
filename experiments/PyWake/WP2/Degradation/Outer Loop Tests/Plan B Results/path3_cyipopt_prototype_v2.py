"""
Path 3 — CyIpopt Prototype: Degradation-Aware NLP Dispatch
===========================================================

Black-box rainflow + gradient callback for IPOPT.

Architecture:
    max  J = Revenue(c,d) − w · f_deg(e)

    Revenue = Σ_t  price_t · (d_t − c_t) · Δt        (linear)
    f_deg   = Σ_i  Φ_shi(δ_i) · count_i               (convex, non-smooth)

    s.t.  e[t+1] = e[t] + Δt·η_in·c[t] − (Δt/η_out)·d[t]   (energy balance)
          e[0]   = e[T]                                        (periodicity)
          e[t]   ∈ [soc_min·E, soc_max·E]                     (SoC bounds)
          c[t]   ∈ [0, P_cap]                                  (charge limit)
          d[t]   ∈ [0, P_cap]                                  (discharge limit)

Key structural difference from v4:
    - In v4: e_t is a DEPENDENT variable (computed from c,d via energy balance).
      Shi subgradients ∂f/∂c_t and ∂f/∂d_t propagate through the chain rule.
    - Here:  e_t is a DECISION variable (energy balance is a CONSTRAINT).
      Degradation gradient lives on ∂f/∂e_t only (sparse: nonzero at turning
      points only). IPOPT propagates to c,d via constraint duals automatically.

Dependencies:
    pip install cyipopt numpy rainflow scipy
    Requires degradation_xu.py in the same directory (or on PYTHONPATH).

Usage:
    python path3_cyipopt_prototype.py

Author: Thodoris Tsonopoulos — MSc Thesis, TU Delft Wind Energy
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Import from existing degradation pipeline
# ---------------------------------------------------------------------------
# degradation_xu.py etc. live one directory up from this script
_SCRIPT_DIR = Path(__file__).parent
_PARENT_DIR = _SCRIPT_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

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
)


# ═══════════════════════════════════════════════════════════════════════════
# NEW: ∂f_deg/∂e_t  —  SoC-space subgradient for CyIpopt
# ═══════════════════════════════════════════════════════════════════════════

def compute_df_de(
    storage_e: np.ndarray,
    cycles: List[Dict],
    e_cap: float,
    shi_fit: ShiPolynomialFit,
    T_C: float = 25.0,
    p: XuModelParams = XU_LMO,
) -> np.ndarray:
    """Compute ∂f_deg/∂e_t — sparse subgradient at turning points only.

    Unlike compute_subgradient() (which gives dense ∂f/∂c_t, ∂f/∂d_t over
    ALL timesteps), this returns nonzeros only at local extrema of the SoC
    profile — typically ~400–800 out of 8,761 entries.

    Sign convention:
        At a local maximum (peak):   ∂f/∂e_t > 0  (raising peak deepens cycles)
        At a local minimum (trough): ∂f/∂e_t < 0  (raising trough shallows cycles)
        Interior (monotone segment): ∂f/∂e_t = 0  (invisible to rainflow)

    Relationship to Shi Eqs. 17–18:
        IPOPT propagates ∂f/∂e_t to c_t, d_t automatically via the energy
        balance constraint duals (Lagrange multipliers). No manual chain
        rule needed — that's the whole point of treating e as a decision
        variable with dynamics as constraints.

    Args:
        storage_e: Battery energy [MWh], shape (T+1,).
        cycles:    From rainflow_cycle_counting(storage_e, e_cap).
        e_cap:     Nominal capacity [MWh].
        shi_fit:   Fitted Shi polynomial (k3, k4).
        T_C:       Cell temperature [°C] (default 25°C → S_T = 1.0).
        p:         Xu model params (for S_σ, S_T stress factors).

    Returns:
        df_de: shape (T+1,), units [1/MWh]. Multiply by w = B·E_cap to get
               [EUR/MWh], commensurable with electricity prices.
    """
    n = len(storage_e)
    df_de = np.zeros(n, dtype=np.float64)

    k3, k4 = shi_fit.k3, shi_fit.k4
    S_T = float(s_temp(T_C, p))          # scalar, constant at 25°C → 1.0

    # Diagnostic counters
    _n_shifted = 0
    _n_total   = 0

    for c in cycles:
        delta = c["dod"]                  # normalised DoD ∈ (0, 1]
        count = c["count"]               # 0.5 (half) or 1.0 (full)
        i_s   = c["i_start"]
        i_e   = c["i_end"]
        sigma = c["soc_mean"]            # mean SoC (for S_σ stress)

        if delta < 1e-12:
            continue

        # Φ'_shi(δ) · S_σ(σ) · S_T(T) — consistent with compute_subgradient
        phi_prime = float(phi_shi_prime(delta, k3, k4))
        S_sigma   = float(s_soc(sigma, p))
        kernel    = phi_prime * S_sigma * S_T * count

        # --- Find ACTUAL peak and trough within the cycle span ---
        # i_start/i_end from the rainflow library mark the cycle's time
        # boundaries, but they may not be the actual local extrema — e.g.
        # if the battery is idle (flat SoC) for several hours at a turning
        # point, the library may return the start of the flat segment
        # rather than the extremum itself.  For full cycles (count=1.0)
        # the real turning point may be interior to the [i_start, i_end]
        # span entirely.
        #
        # Fix: search within the span for argmax / argmin.
        i0 = min(i_s, i_e)
        i1 = max(i_s, i_e)
        i0 = max(i0, 0)
        i1 = min(i1, n - 1)

        segment = storage_e[i0 : i1 + 1]
        peak_idx   = i0 + int(np.argmax(segment))
        trough_idx = i0 + int(np.argmin(segment))

        # Track how many cycles had their turning points shifted
        _n_total += 1
        if peak_idx not in (i_s, i_e) or trough_idx not in (i_s, i_e):
            _n_shifted += 1

        # Accumulate (+=), not overwrite — handles junctions correctly
        df_de[peak_idx]   += kernel     # raising peak deepens cycle
        df_de[trough_idx] -= kernel     # raising trough shallows cycle

    # Diagnostic: how many cycles had extrema shifted from i_start/i_end
    # (only printed on first call to avoid spamming during NLP iterations)
    if _n_total > 0 and _n_shifted > 0:
        if not hasattr(compute_df_de, "_diag_printed"):
            print(f"    compute_df_de: {_n_shifted}/{_n_total} cycles had "
                  f"peak/trough shifted from i_start/i_end")
            compute_df_de._diag_printed = True

    # Chain rule: x = e / E_cap → ∂f/∂e = (1/E_cap) · ∂f/∂x
    df_de /= e_cap

    return df_de


def compute_f_deg(
    storage_e: np.ndarray,
    e_cap: float,
    shi_fit: ShiPolynomialFit,
    T_C: float = 25.0,
    p: XuModelParams = XU_LMO,
) -> Tuple[float, List[Dict]]:
    """Compute total Shi degradation cost f_deg = Σ Φ_shi(δ_i)·count_i·S_σ·S_T.

    Returns both the scalar cost and the cycle list (for reuse in gradient).
    """
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


# ═══════════════════════════════════════════════════════════════════════════
# FD validation of compute_df_de
# ═══════════════════════════════════════════════════════════════════════════

def validate_df_de_finite_difference(
    storage_e: np.ndarray,
    e_cap: float,
    shi_fit: ShiPolynomialFit,
    eps_frac: float = 1e-4,
    n_test_points: int = 20,
    verbose: bool = True,
    export_csv: bool = True,
) -> Dict:
    """Validate compute_df_de against central finite differences.

    Exports two CSVs for diagnosis:
      - fd_validation_cycles.csv:  per-cycle data (i_start, i_end, peak, trough, delta, count)
      - fd_validation_gradient.csv: per-timestep FD vs analytical for ALL turning points
    """
    import csv

    f_base, cycles_base = compute_f_deg(storage_e, e_cap, shi_fit)
    df_de = compute_df_de(storage_e, cycles_base, e_cap, shi_fit)

    eps = eps_frac * e_cap
    n = len(storage_e)
    k3, k4 = shi_fit.k3, shi_fit.k4

    # ── Export 1: Cycle-level data ──────────────────────────────────────
    if export_csv:
        csv_path_cycles = Path(__file__).parent / "fd_validation_cycles.csv"
        with open(csv_path_cycles, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cycle_idx", "i_start", "i_end", "peak_idx", "trough_idx",
                         "delta", "count", "soc_mean",
                         "e_at_istart", "e_at_iend", "e_at_peak", "e_at_trough",
                         "peak_shifted", "trough_shifted"])
            for ci, c in enumerate(cycles_base):
                i_s, i_e = c["i_start"], c["i_end"]
                i0 = max(min(i_s, i_e), 0)
                i1 = min(max(i_s, i_e), n - 1)
                seg = storage_e[i0:i1+1]
                pk  = i0 + int(np.argmax(seg))
                tr  = i0 + int(np.argmin(seg))
                w.writerow([
                    ci, i_s, i_e, pk, tr,
                    f"{c['dod']:.6f}", c['count'], f"{c['soc_mean']:.4f}",
                    f"{storage_e[i_s]:.2f}", f"{storage_e[i_e]:.2f}",
                    f"{storage_e[pk]:.2f}", f"{storage_e[tr]:.2f}",
                    pk not in (i_s, i_e), tr not in (i_s, i_e),
                ])
        if verbose:
            print(f"  Exported: {csv_path_cycles.name}")

    # ── Identify turning points and interior points ────────────────────
    nonzero_idx = np.where(np.abs(df_de) > 1e-15)[0]
    zero_idx    = np.where(np.abs(df_de) < 1e-15)[0]

    if verbose:
        print(f"\n  FD Validation:  {len(nonzero_idx)} turning points, "
              f"{len(zero_idx)} interior points")
        print(f"  eps = {eps:.4f} MWh  ({eps_frac*100:.2f}% of E_cap)")

    # ── FD test at ALL turning points + sample of interior ─────────────
    test_indices = list(nonzero_idx)  # test ALL turning points
    # Add a sample of interior points
    if len(zero_idx) > 0:
        int_sample = zero_idx[np.linspace(0, len(zero_idx)-1,
                                           min(10, len(zero_idx)), dtype=int)]
        test_indices.extend(int_sample)

    gradient_rows = []
    for t in test_indices:
        e_plus  = storage_e.copy(); e_plus[t]  += eps
        e_minus = storage_e.copy(); e_minus[t] -= eps

        f_plus,  cyc_plus  = compute_f_deg(e_plus,  e_cap, shi_fit)
        f_minus, cyc_minus = compute_f_deg(e_minus, e_cap, shi_fit)

        fd_numerical  = (f_plus - f_minus) / (2 * eps)
        fd_analytical = df_de[t]
        is_turning    = abs(fd_analytical) > 1e-15

        ratio = (fd_numerical / fd_analytical
                 if abs(fd_analytical) > 1e-15 else float('nan'))

        # Detect cycle structure changes under perturbation
        n_cyc_base  = len(cycles_base)
        n_cyc_plus  = len(cyc_plus)
        n_cyc_minus = len(cyc_minus)
        structure_changed = (n_cyc_plus != n_cyc_base or
                             n_cyc_minus != n_cyc_base)

        # Which cycles touch this timestep?
        touching = []
        for ci, c in enumerate(cycles_base):
            i0 = min(c["i_start"], c["i_end"])
            i1 = max(c["i_start"], c["i_end"])
            if i0 <= t <= i1:
                seg = storage_e[i0:i1+1]
                pk  = i0 + int(np.argmax(seg))
                tr  = i0 + int(np.argmin(seg))
                role = "peak" if t == pk else ("trough" if t == tr else "interior")
                touching.append(f"c{ci}({role},d={c['dod']:.3f})")

        gradient_rows.append({
            "t": int(t),
            "e_t": storage_e[t],
            "analytical": fd_analytical,
            "fd_numerical": fd_numerical,
            "ratio": ratio,
            "is_turning_point": is_turning,
            "n_cycles_base": n_cyc_base,
            "n_cycles_plus": n_cyc_plus,
            "n_cycles_minus": n_cyc_minus,
            "structure_changed": structure_changed,
            "touching_cycles": "; ".join(touching),
        })

    # ── Export 2: Per-timestep gradient data ────────────────────────────
    if export_csv:
        csv_path_grad = Path(__file__).parent / "fd_validation_gradient.csv"
        with open(csv_path_grad, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "e_t", "analytical", "fd_numerical", "ratio",
                         "is_turning_point", "n_cyc_base", "n_cyc_plus",
                         "n_cyc_minus", "structure_changed", "touching_cycles"])
            for r in gradient_rows:
                w.writerow([
                    r["t"], f"{r['e_t']:.2f}",
                    f"{r['analytical']:.10e}", f"{r['fd_numerical']:.10e}",
                    f"{r['ratio']:.6f}" if not np.isnan(r['ratio']) else "NaN",
                    r["is_turning_point"],
                    r["n_cycles_base"], r["n_cycles_plus"], r["n_cycles_minus"],
                    r["structure_changed"], r["touching_cycles"],
                ])
        if verbose:
            print(f"  Exported: {csv_path_grad.name}")

    # ── Console summary ────────────────────────────────────────────────
    tp_rows  = [r for r in gradient_rows if r["is_turning_point"]]
    int_rows = [r for r in gradient_rows if not r["is_turning_point"]]

    # Split turning points by whether cycle structure changed
    tp_stable   = [r for r in tp_rows if not r["structure_changed"]]
    tp_unstable = [r for r in tp_rows if r["structure_changed"]]

    if verbose:
        print(f"\n  Turning points with STABLE cycle structure "
              f"({len(tp_stable)}/{len(tp_rows)}):")
        for r in tp_stable[:10]:
            print(f"    t={r['t']:5d}  analytical={r['analytical']:+.6e}  "
                  f"FD={r['fd_numerical']:+.6e}  ratio={r['ratio']:.4f}  "
                  f"cycles={r['touching_cycles']}")
        ratios_s = [r["ratio"] for r in tp_stable if not np.isnan(r["ratio"])]
        if ratios_s:
            print(f"    Mean ratio: {np.mean(ratios_s):.4f}  "
                  f"Std: {np.std(ratios_s):.4f}")

        print(f"\n  Turning points with CHANGED cycle structure "
              f"({len(tp_unstable)}/{len(tp_rows)}):")
        for r in tp_unstable[:10]:
            print(f"    t={r['t']:5d}  analytical={r['analytical']:+.6e}  "
                  f"FD={r['fd_numerical']:+.6e}  ratio={r['ratio']:.4f}  "
                  f"cyc: {r['n_cycles_base']}→{r['n_cycles_plus']}/{r['n_cycles_minus']}")
        ratios_u = [r["ratio"] for r in tp_unstable if not np.isnan(r["ratio"])]
        if ratios_u:
            print(f"    Mean ratio: {np.mean(ratios_u):.4f}  "
                  f"Std: {np.std(ratios_u):.4f}")

        print(f"\n  Interior points (expect FD ≈ 0, {len(int_rows)} tested):")
        for r in int_rows[:5]:
            cyc_info = f"  cycles={r['touching_cycles']}" if r['touching_cycles'] else ""
            print(f"    t={r['t']:5d}  FD={r['fd_numerical']:+.6e}{cyc_info}")

    results = {"turning_points": tp_rows, "interior_points": int_rows,
               "stable": tp_stable, "unstable": tp_unstable}
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Single-timestep cycle-by-cycle diagnostic
# ═══════════════════════════════════════════════════════════════════════════

def diagnose_single_timestep(
    storage_e: np.ndarray,
    e_cap: float,
    shi_fit: ShiPolynomialFit,
    t_probe: int,
    eps_frac: float = 1e-4,
) -> None:
    """Dump full per-cycle breakdown for base/+eps/-eps at one timestep.

    This shows exactly WHICH cycles change depth/mean/count under
    perturbation, and what the individual contribution to Δf is.
    """
    import csv

    eps = eps_frac * e_cap
    k3, k4 = shi_fit.k3, shi_fit.k4
    S_T = float(s_temp(25.0))

    e_base  = storage_e.copy()
    e_plus  = storage_e.copy(); e_plus[t_probe]  += eps
    e_minus = storage_e.copy(); e_minus[t_probe] -= eps

    f_base,  cyc_base  = compute_f_deg(e_base,  e_cap, shi_fit)
    f_plus,  cyc_plus  = compute_f_deg(e_plus,  e_cap, shi_fit)
    f_minus, cyc_minus = compute_f_deg(e_minus, e_cap, shi_fit)

    fd_numerical = (f_plus - f_minus) / (2 * eps)
    df_de = compute_df_de(e_base, cyc_base, e_cap, shi_fit)
    fd_analytical = df_de[t_probe]

    print(f"\n  {'═'*68}")
    print(f"  SINGLE-TIMESTEP DIAGNOSTIC: t = {t_probe}")
    print(f"  {'═'*68}")
    print(f"  e[{t_probe}] = {storage_e[t_probe]:.2f} MWh")
    print(f"  eps = {eps:.4f} MWh")
    print(f"  f_base  = {f_base:.12e}")
    print(f"  f_plus  = {f_plus:.12e}   Δ = {f_plus - f_base:+.6e}")
    print(f"  f_minus = {f_minus:.12e}   Δ = {f_minus - f_base:+.6e}")
    print(f"  FD numerical  = {fd_numerical:+.10e}")
    print(f"  Analytical    = {fd_analytical:+.10e}")
    ratio = fd_numerical / fd_analytical if abs(fd_analytical) > 1e-15 else float('nan')
    print(f"  Ratio         = {ratio:.6f}")
    print(f"  Cycles: base={len(cyc_base)}, plus={len(cyc_plus)}, "
          f"minus={len(cyc_minus)}")

    # Per-cycle contribution to f_deg
    def _cycle_f(c):
        d = c["dod"]
        if d < 1e-12:
            return 0.0
        return float(phi_shi(d, k3, k4)) * float(s_soc(c["soc_mean"])) * S_T * c["count"]

    # Build lookup: match cycles by (i_start, i_end) across the three cases
    print(f"\n  {'idx':>3s} {'i_s':>4s} {'i_e':>4s} {'dod':>8s} {'cnt':>4s} "
          f"{'sigma':>7s} {'f_base':>12s} {'f_plus':>12s} {'f_minus':>12s} "
          f"{'Δ(p-m)':>12s} {'note':s}")
    print(f"  {'─'*3} {'─'*4} {'─'*4} {'─'*8} {'─'*4} {'─'*7} "
          f"{'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*20}")

    # Index base cycles
    base_keys = {(c["i_start"], c["i_end"]): (i, c) for i, c in enumerate(cyc_base)}
    plus_keys = {(c["i_start"], c["i_end"]): c for c in cyc_plus}
    minus_keys = {(c["i_start"], c["i_end"]): c for c in cyc_minus}

    all_keys = set(base_keys.keys()) | set(plus_keys.keys()) | set(minus_keys.keys())
    total_delta = 0.0

    csv_path = Path(__file__).parent / f"fd_diagnostic_t{t_probe}.csv"
    with open(csv_path, "w", newline="") as csvf:
        w = csv.writer(csvf)
        w.writerow(["i_start", "i_end", "dod_base", "dod_plus", "dod_minus",
                     "sigma_base", "sigma_plus", "sigma_minus",
                     "count", "f_base", "f_plus", "f_minus", "delta_f_pm", "note"])

        for key in sorted(all_keys):
            cb = base_keys.get(key)
            cp = plus_keys.get(key)
            cm = minus_keys.get(key)

            i_s, i_e = key
            in_span = min(i_s, i_e) <= t_probe <= max(i_s, i_e)

            fb = _cycle_f(cb[1]) if cb else 0.0
            fp = _cycle_f(cp) if cp else 0.0
            fm = _cycle_f(cm) if cm else 0.0
            delta_pm = fp - fm
            total_delta += delta_pm

            note = ""
            if cb and not cp: note += "GONE_PLUS "
            if cb and not cm: note += "GONE_MINUS "
            if not cb and cp: note += "NEW_PLUS "
            if not cb and cm: note += "NEW_MINUS "
            if in_span: note += "IN_SPAN "
            if cb and cp and abs(cb[1]["dod"] - cp["dod"]) > 1e-8: note += "DOD_CHANGED "

            c_ref = cb[1] if cb else (cp if cp else cm)
            idx_str = f"{cb[0]:3d}" if cb else "  ?"

            print(f"  {idx_str} {i_s:4d} {i_e:4d} "
                  f"{cb[1]['dod'] if cb else 0:8.5f} {c_ref['count']:4.1f} "
                  f"{cb[1]['soc_mean'] if cb else 0:7.4f} "
                  f"{fb:12.6e} {fp:12.6e} {fm:12.6e} "
                  f"{delta_pm:+12.6e} {note}")

            w.writerow([
                i_s, i_e,
                f"{cb[1]['dod']:.6f}" if cb else "", f"{cp['dod']:.6f}" if cp else "",
                f"{cm['dod']:.6f}" if cm else "",
                f"{cb[1]['soc_mean']:.6f}" if cb else "", f"{cp['soc_mean']:.6f}" if cp else "",
                f"{cm['soc_mean']:.6f}" if cm else "",
                c_ref["count"],
                f"{fb:.10e}", f"{fp:.10e}", f"{fm:.10e}", f"{delta_pm:.10e}", note.strip(),
            ])

    print(f"\n  Σ Δ(f_plus − f_minus) = {total_delta:+.6e}")
    print(f"  Expected from FD:       {(f_plus - f_minus):+.6e}")
    print(f"  Match: {'YES' if abs(total_delta - (f_plus - f_minus)) < 1e-15 else 'CHECK'}")
    print(f"  Exported: {csv_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# CyIpopt NLP class
# ═══════════════════════════════════════════════════════════════════════════

class BatteryDispatchNLP:
    """Degradation-aware battery dispatch as a CyIpopt NLP.

    Decision variables (3T+1 total):
        x = [c_0, ..., c_{T-1},  d_0, ..., d_{T-1},  e_0, ..., e_T]
             ──── charge ────    ──── discharge ────   ──── energy ────
             indices 0..T-1      indices T..2T-1       indices 2T..3T

    Objective (MINIMISE negative utility):
        min  −Revenue + w · f_deg

    Constraints (T+1 equalities):
        g_t:  e[t+1] − e[t] − Δt·η_in·c[t] + (Δt/η_out)·d[t] = 0
        g_T:  e[0] − e[T] = 0   (periodicity)

    Variable bounds:
        c[t] ∈ [0, P_cap],  d[t] ∈ [0, P_cap],  e[t] ∈ [E_min, E_max]
    """

    def __init__(
        self,
        prices: np.ndarray,          # shape (T,)  [EUR/MWh]
        e_cap: float,                # nominal capacity [MWh]
        p_cap: float,                # power rating [MW]
        soc_min: float = 0.10,
        soc_max: float = 0.90,
        eff_in: float = 0.95,        # charging efficiency
        eff_out: float = 0.95,       # discharging efficiency
        dt: float = 1.0,             # timestep [h]
        replacement_cost: float = 150_000.0,  # EUR/MWh (150 EUR/kWh)
        shi_fit: ShiPolynomialFit = None,
        xu_params: XuModelParams = XU_LMO,
        T_C: float = 25.0,
        w_scale: float = 1.0,        # scale degradation weight (for testing)
    ):
        self.prices = np.asarray(prices, dtype=float)
        self.T = len(prices)
        self.e_cap = float(e_cap)
        self.p_cap = float(p_cap)
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.eff_in = float(eff_in)
        self.eff_out = float(eff_out)
        self.dt = float(dt)
        self.T_C = float(T_C)
        self.p = xu_params

        # Degradation weight: w = B · E_cap · w_scale
        # This converts fractional life loss into EUR
        self.w_deg = float(replacement_cost) * self.e_cap * float(w_scale)

        # Shi polynomial fit
        if shi_fit is None:
            self.shi_fit = fit_shi_polynomial(
                soc_min=soc_min, soc_max=soc_max,
                source="Path3 auto-fit", verbose=False,
            )
        else:
            self.shi_fit = shi_fit

        # Dimensions
        self.n_vars = 3 * self.T + 1       # c[T] + d[T] + e[T+1]
        self.n_cons = self.T + 1            # T energy balance + 1 periodicity

        # Precompute Jacobian sparsity (constant — all constraints are linear)
        self._build_jacobian_structure()

        # Iteration counter for diagnostics
        self._iter = 0

    # ------------------------------------------------------------------
    # Variable packing / unpacking
    # ------------------------------------------------------------------
    def _idx_c(self): return slice(0, self.T)
    def _idx_d(self): return slice(self.T, 2 * self.T)
    def _idx_e(self): return slice(2 * self.T, 3 * self.T + 1)

    def _unpack(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return x[self._idx_c()], x[self._idx_d()], x[self._idx_e()]

    def _pack(self, gc: np.ndarray, gd: np.ndarray, ge: np.ndarray) -> np.ndarray:
        return np.concatenate([gc, gd, ge])

    # ------------------------------------------------------------------
    # Objective:  min  −Revenue + w · f_deg
    # ------------------------------------------------------------------
    def objective(self, x: np.ndarray) -> float:
        c, d, e = self._unpack(x)
        revenue = np.sum(self.prices * (d - c) * self.dt)
        f_deg, _ = compute_f_deg(e, self.e_cap, self.shi_fit, self.T_C, self.p)
        return -revenue + self.w_deg * f_deg

    # ------------------------------------------------------------------
    # Gradient of objective
    # ------------------------------------------------------------------
    def gradient(self, x: np.ndarray) -> np.ndarray:
        c, d, e = self._unpack(x)

        # Revenue gradient (trivial, linear)
        grad_c_rev = -(-self.prices * self.dt)    # ∂(−Rev)/∂c = +price·dt
        grad_d_rev = -(+self.prices * self.dt)    # ∂(−Rev)/∂d = −price·dt

        # Degradation gradient (sparse, on e only)
        f_deg, cycles = compute_f_deg(e, self.e_cap, self.shi_fit, self.T_C, self.p)
        df_de = compute_df_de(e, cycles, self.e_cap, self.shi_fit, self.T_C, self.p)

        grad_c = grad_c_rev                       # no direct degradation on c
        grad_d = grad_d_rev                        # no direct degradation on d
        grad_e = self.w_deg * df_de                # [EUR/MWh]

        self._iter += 1
        return self._pack(grad_c, grad_d, grad_e)

    # ------------------------------------------------------------------
    # Constraints:  g(x) = 0
    # ------------------------------------------------------------------
    def constraints(self, x: np.ndarray) -> np.ndarray:
        c, d, e = self._unpack(x)
        g = np.empty(self.n_cons)

        # Energy balance: e[t+1] − e[t] − Δt·η_in·c[t] + (Δt/η_out)·d[t] = 0
        g[:self.T] = (e[1:] - e[:-1]
                      - self.dt * self.eff_in * c
                      + (self.dt / self.eff_out) * d)

        # Periodicity: e[0] − e[T] = 0
        g[self.T] = e[0] - e[-1]

        return g

    # ------------------------------------------------------------------
    # Jacobian of constraints (constant, precomputed)
    # ------------------------------------------------------------------
    def _build_jacobian_structure(self):
        """Precompute the sparse Jacobian structure (rows, cols, values).

        Each energy balance constraint t has 4 nonzeros:
            ∂g_t/∂c_t      = −Δt·η_in           col = t
            ∂g_t/∂d_t      = +Δt/η_out           col = T + t
            ∂g_t/∂e_t      = −1                   col = 2T + t
            ∂g_t/∂e_{t+1}  = +1                   col = 2T + t + 1

        Periodicity constraint T has 2 nonzeros:
            ∂g_T/∂e_0  = +1                       col = 2T
            ∂g_T/∂e_T  = −1                       col = 3T
        """
        T = self.T
        rows, cols, vals = [], [], []

        for t in range(T):
            # ∂g_t/∂c_t
            rows.append(t); cols.append(t);         vals.append(-self.dt * self.eff_in)
            # ∂g_t/∂d_t
            rows.append(t); cols.append(T + t);     vals.append(self.dt / self.eff_out)
            # ∂g_t/∂e_t
            rows.append(t); cols.append(2*T + t);   vals.append(-1.0)
            # ∂g_t/∂e_{t+1}
            rows.append(t); cols.append(2*T + t+1); vals.append(1.0)

        # Periodicity
        rows.append(T); cols.append(2*T);     vals.append(1.0)   # e[0]
        rows.append(T); cols.append(3*T);     vals.append(-1.0)  # e[T]

        self._jac_rows = np.array(rows, dtype=int)
        self._jac_cols = np.array(cols, dtype=int)
        self._jac_vals = np.array(vals, dtype=float)

    def jacobianstructure(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self._jac_rows, self._jac_cols)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        # Constant — doesn't depend on x
        return self._jac_vals

    # ------------------------------------------------------------------
    # Variable bounds
    # ------------------------------------------------------------------
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        T = self.T
        lb = np.empty(self.n_vars)
        ub = np.empty(self.n_vars)

        # c[t] ∈ [0, P_cap]
        lb[:T]       = 0.0
        ub[:T]       = self.p_cap

        # d[t] ∈ [0, P_cap]
        lb[T:2*T]    = 0.0
        ub[T:2*T]    = self.p_cap

        # e[t] ∈ [soc_min·E, soc_max·E]
        lb[2*T:]     = self.soc_min * self.e_cap
        ub[2*T:]     = self.soc_max * self.e_cap

        return lb, ub

    def get_constraint_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """All constraints are equalities → cl = cu = 0."""
        cl = np.zeros(self.n_cons)
        cu = np.zeros(self.n_cons)
        return cl, cu

    # ------------------------------------------------------------------
    # Initial point — midpoint SoC, zero dispatch
    # ------------------------------------------------------------------
    def get_initial_point(self) -> np.ndarray:
        x0 = np.zeros(self.n_vars)
        # Start with zero charge/discharge
        # Set energy at midpoint of SoC window
        e_mid = 0.5 * (self.soc_min + self.soc_max) * self.e_cap
        x0[2*self.T:] = e_mid
        return x0

    # ------------------------------------------------------------------
    # Warm start from LP solution
    # ------------------------------------------------------------------
    def warm_start_from_lp(
        self, p_vec: np.ndarray, e_vec: np.ndarray,
    ) -> np.ndarray:
        """Build initial point from LP (no-degradation) dispatch.

        p_vec uses SHIPP convention: positive = discharge, negative = charge.
        """
        x0 = np.zeros(self.n_vars)
        c = np.maximum(-p_vec, 0.0)   # charge power (≥0)
        d = np.maximum(+p_vec, 0.0)   # discharge power (≥0)
        x0[self._idx_c()] = c
        x0[self._idx_d()] = d
        x0[self._idx_e()] = e_vec
        return x0


# ═══════════════════════════════════════════════════════════════════════════
# LP baseline solver (Pyomo/Gurobi — for comparison)
# ═══════════════════════════════════════════════════════════════════════════

def solve_lp_baseline(
    prices: np.ndarray,
    e_cap: float,
    p_cap: float,
    soc_min: float = 0.10,
    soc_max: float = 0.90,
    eff_in: float = 0.95,
    eff_out: float = 0.95,
    dt: float = 1.0,
    solver_name: str = "gurobi",
    verbose: bool = False,
) -> Dict:
    """Solve the pure-revenue LP dispatch (no degradation).

    Uses separate c/d variables (no big-M binaries) matching the NLP structure.
    """
    import pyomo.environ as pyo

    T = len(prices)
    model = pyo.ConcreteModel("LP_baseline")

    # Sets
    model.time  = pyo.RangeSet(0, T - 1)
    model.etime = pyo.RangeSet(0, T)

    # Variables
    model.c = pyo.Var(model.time,  bounds=(0, p_cap))   # charge [MW]
    model.d = pyo.Var(model.time,  bounds=(0, p_cap))   # discharge [MW]
    model.e = pyo.Var(model.etime, bounds=(soc_min * e_cap, soc_max * e_cap))

    # Objective: maximise revenue
    model.obj = pyo.Objective(
        expr=sum(prices[t] * (model.d[t] - model.c[t]) * dt for t in model.time),
        sense=pyo.maximize,
    )

    # Energy balance
    def _energy_balance(m, t):
        return m.e[t+1] == m.e[t] + dt * eff_in * m.c[t] - (dt / eff_out) * m.d[t]
    model.energy_bal = pyo.Constraint(model.time, rule=_energy_balance)

    # Periodicity
    model.periodic = pyo.Constraint(expr=model.e[0] == model.e[T])

    # Duals suffix — declare BEFORE solving so duals are extracted in one pass
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Solve
    solver = pyo.SolverFactory(solver_name)
    if solver_name == "gurobi":
        solver.options["OutputFlag"] = int(verbose)
    result = solver.solve(model, tee=verbose)

    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"LP solver status: {result.solver.termination_condition}")

    # Extract
    c_val = np.array([pyo.value(model.c[t]) for t in model.time])
    d_val = np.array([pyo.value(model.d[t]) for t in model.time])
    e_val = np.array([pyo.value(model.e[t]) for t in model.etime])
    revenue = float(pyo.value(model.obj))

    # Extract duals
    duals = np.array([pyo.value(model.dual[model.energy_bal[t]])
                      for t in model.time])

    return {
        "c": c_val, "d": d_val, "e": e_val,
        "p_vec": d_val - c_val,   # SHIPP convention
        "revenue": revenue,
        "duals": duals,
    }


# ═══════════════════════════════════════════════════════════════════════════
# IPOPT solve wrapper
# ═══════════════════════════════════════════════════════════════════════════

def solve_nlp_ipopt(
    nlp: BatteryDispatchNLP,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    print_level: int = 5,
) -> Dict:
    """Solve the NLP using CyIpopt."""
    try:
        import cyipopt
    except ImportError:
        raise ImportError(
            "CyIpopt required: pip install cyipopt\n"
            "  Also needs IPOPT C library. On conda:\n"
            "    conda install -c conda-forge cyipopt"
        )

    if x0 is None:
        x0 = nlp.get_initial_point()

    lb, ub = nlp.get_bounds()
    cl, cu = nlp.get_constraint_bounds()

    # Build the problem
    problem = cyipopt.Problem(
        n=nlp.n_vars,
        m=nlp.n_cons,
        problem_obj=nlp,
        lb=lb, ub=ub,
        cl=cl, cu=cu,
    )

    # IPOPT options
    problem.add_option("max_iter", max_iter)
    problem.add_option("tol", tol)
    problem.add_option("print_level", print_level)
    problem.add_option("hessian_approximation", "limited-memory")  # L-BFGS
    problem.add_option("limited_memory_max_history", 20)
    problem.add_option("mu_strategy", "adaptive")
    problem.add_option("sb", "yes")   # suppress IPOPT banner

    # Solve
    t0 = time.perf_counter()
    x_opt, info = problem.solve(x0)
    solve_time = time.perf_counter() - t0

    c_opt, d_opt, e_opt = nlp._unpack(x_opt)

    revenue = float(np.sum(nlp.prices * (d_opt - c_opt) * nlp.dt))
    f_deg, cycles = compute_f_deg(e_opt, nlp.e_cap, nlp.shi_fit, nlp.T_C, nlp.p)
    objective = float(info["obj_val"])

    return {
        "c": c_opt, "d": d_opt, "e": e_opt,
        "p_vec": d_opt - c_opt,
        "revenue": revenue,
        "f_deg": f_deg,
        "objective": objective,
        "deg_cost_EUR": nlp.w_deg * f_deg,
        "n_cycles": len(cycles),
        "status": info["status"],
        "status_msg": info["status_msg"],
        "solve_time": solve_time,
        "n_iter": nlp._iter,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Scipy SLSQP fallback  (no IPOPT C library required)
# ═══════════════════════════════════════════════════════════════════════════

def solve_nlp_scipy(
    nlp: BatteryDispatchNLP,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-7,
    verbose: bool = True,
) -> Dict:
    """Solve the NLP using scipy.optimize.minimize (SLSQP).

    SLSQP handles equality/inequality constraints and bounds natively.
    Less powerful than IPOPT for large-scale problems, but requires
    zero additional installation and works fine for T ≤ 168.
    """
    from scipy.optimize import minimize

    if x0 is None:
        x0 = nlp.get_initial_point()

    lb, ub = nlp.get_bounds()

    # Variable bounds as list of (lo, hi) tuples
    bounds = list(zip(lb, ub))

    # Energy balance + periodicity as equality constraints
    constraints = {
        "type": "eq",
        "fun": nlp.constraints,
        "jac": lambda x: _dense_jacobian(nlp, x),
    }

    # Callback for iteration logging
    _state = {"iter": 0, "last_obj": None}

    def _callback(xk):
        _state["iter"] += 1
        if verbose and _state["iter"] % 25 == 0:
            obj = nlp.objective(xk)
            print(f"    iter {_state['iter']:4d}  obj = {obj:+.6e}")

    if verbose:
        print(f"  Solving with scipy SLSQP  (n={nlp.n_vars}, m={nlp.n_cons})")

    t0 = time.perf_counter()
    result = minimize(
        fun=nlp.objective,
        x0=x0,
        jac=nlp.gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        callback=_callback,
        options={
            "maxiter": max_iter,
            "ftol": tol,
            "disp": verbose,
        },
    )
    solve_time = time.perf_counter() - t0

    x_opt = result.x
    c_opt, d_opt, e_opt = nlp._unpack(x_opt)

    revenue = float(np.sum(nlp.prices * (d_opt - c_opt) * nlp.dt))
    f_deg, cycles = compute_f_deg(e_opt, nlp.e_cap, nlp.shi_fit, nlp.T_C, nlp.p)

    return {
        "c": c_opt, "d": d_opt, "e": e_opt,
        "p_vec": d_opt - c_opt,
        "revenue": revenue,
        "f_deg": f_deg,
        "objective": float(result.fun),
        "deg_cost_EUR": nlp.w_deg * f_deg,
        "n_cycles": len(cycles),
        "status": 0 if result.success else -1,
        "status_msg": result.message,
        "solve_time": solve_time,
        "n_iter": result.nit,
        "scipy_result": result,   # full scipy object for diagnostics
    }


def _dense_jacobian(nlp: BatteryDispatchNLP, x: np.ndarray) -> np.ndarray:
    """Convert sparse Jacobian to dense for scipy SLSQP."""
    J = np.zeros((nlp.n_cons, nlp.n_vars))
    J[nlp._jac_rows, nlp._jac_cols] = nlp._jac_vals
    return J


# ═══════════════════════════════════════════════════════════════════════════
# Unified solver: try IPOPT, fall back to scipy
# ═══════════════════════════════════════════════════════════════════════════

def solve_nlp(
    nlp: BatteryDispatchNLP,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 500,
    verbose: bool = True,
) -> Dict:
    """Solve the NLP — tries CyIpopt first, falls back to scipy SLSQP."""
    try:
        import cyipopt  # noqa: F401
        if verbose:
            print(f"  Using CyIpopt (IPOPT)")
        return solve_nlp_ipopt(nlp, x0=x0, max_iter=max_iter)
    except ImportError:
        if verbose:
            print(f"  CyIpopt not available — using scipy SLSQP fallback")
        return solve_nlp_scipy(nlp, x0=x0, max_iter=max_iter, verbose=verbose)


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic test data generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_test_data(
    T: int = 168,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """Generate synthetic price data mimicking DK1 day-ahead structure.

    Creates a price signal with:
      - Diurnal pattern (high morning/evening, low night/midday)
      - Day-to-day variation
      - A few price spikes (mimicking wind lulls)

    Args:
        T: Number of hourly timesteps (default 168 = one week).
        seed: Random seed.

    Returns:
        prices: shape (T,) in EUR/MWh
        metadata: dict with generation info
    """
    rng = np.random.RandomState(seed)

    hours = np.arange(T) % 24
    days  = np.arange(T) // 24

    # Base diurnal pattern (EUR/MWh)
    diurnal = 30 + 20 * np.sin(2 * np.pi * (hours - 6) / 24)

    # Day-to-day random shift
    day_shift = rng.normal(0, 10, size=int(np.ceil(T / 24)))
    daily_noise = np.repeat(day_shift, 24)[:T]

    # Hourly noise
    hourly_noise = rng.normal(0, 5, size=T)

    # A few spikes
    n_spikes = max(1, T // 48)
    spike_idx = rng.choice(T, n_spikes, replace=False)
    spikes = np.zeros(T)
    spikes[spike_idx] = rng.uniform(30, 80, size=n_spikes)

    prices = diurnal + daily_noise + hourly_noise + spikes
    prices = np.maximum(prices, 0)  # no negative prices in this test

    return prices, {"T": T, "mean": prices.mean(), "std": prices.std(),
                    "min": prices.min(), "max": prices.max()}


# ═══════════════════════════════════════════════════════════════════════════
# Main: run the prototype
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("Path 3 — CyIpopt Prototype: Degradation-Aware NLP Dispatch")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Configuration — WP2 battery parameters
    # ------------------------------------------------------------------
    E_CAP   = 300.0          # MWh  (WP2 reference)
    P_CAP   = 150.0          # MW
    SOC_MIN = 0.10
    SOC_MAX = 0.90
    EFF_IN  = 0.95           # η_in (√0.9025)
    EFF_OUT = 0.95           # η_out
    DT      = 1.0            # hourly
    B       = 150_000.0      # EUR/MWh replacement cost (150 EUR/kWh × 1000)

    # Small scale for prototype testing
    T = 168                  # one week (scale up once validated)

    print(f"\n  Battery: {E_CAP} MWh / {P_CAP} MW")
    print(f"  SoC window: [{SOC_MIN}, {SOC_MAX}]")
    print(f"  Efficiency: η_in={EFF_IN}, η_out={EFF_OUT}, "
          f"η_rt={EFF_IN*EFF_OUT:.4f}")
    print(f"  Replacement cost: {B:,.0f} EUR/MWh")
    print(f"  Test horizon: {T} hours ({T/24:.1f} days)")

    # ------------------------------------------------------------------
    # 2. Generate synthetic price data
    # ------------------------------------------------------------------
    prices, price_meta = generate_test_data(T=T)
    print(f"\n  Prices: mean={price_meta['mean']:.1f}  std={price_meta['std']:.1f}  "
          f"range=[{price_meta['min']:.1f}, {price_meta['max']:.1f}] EUR/MWh")

    # ------------------------------------------------------------------
    # 3. Fit Shi polynomial
    # ------------------------------------------------------------------
    shi_fit = fit_shi_polynomial(
        soc_min=SOC_MIN, soc_max=SOC_MAX,
        source="Path3 prototype", verbose=True,
    )
    print(f"\n  Shi fit: {shi_fit.summary()}")

    # ------------------------------------------------------------------
    # 4. Solve LP baseline (pure revenue, no degradation)
    # ------------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("Step 1: LP baseline (pure revenue, no degradation)")
    print(f"{'─'*72}")

    try:
        lp = solve_lp_baseline(
            prices, E_CAP, P_CAP, SOC_MIN, SOC_MAX,
            EFF_IN, EFF_OUT, DT, verbose=False,
        )
        print(f"  Revenue (LP):    {lp['revenue']:>12,.2f} EUR")

        # Compute degradation of LP dispatch (for comparison)
        f_lp, cycles_lp = compute_f_deg(lp["e"], E_CAP, shi_fit)
        deg_cost_lp = B * E_CAP * f_lp
        print(f"  Degradation fd:  {f_lp:.6e}")
        print(f"  Deg cost:        {deg_cost_lp:>12,.2f} EUR")
        print(f"  Net utility:     {lp['revenue'] - deg_cost_lp:>12,.2f} EUR")
        print(f"  Cycles:          {len(cycles_lp)}")

        lp_available = True
    except Exception as exc:
        print(f"  LP solver failed: {exc}")
        print(f"  Continuing without LP baseline...")
        lp_available = False

    # ------------------------------------------------------------------
    # 5. Validate ∂f/∂e_t against finite differences
    # ------------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("Step 2: Validate ∂f/∂e_t (finite differences)")
    print(f"{'─'*72}")

    if lp_available:
        test_e = lp["e"]
        print(f"  Using LP dispatch SoC profile for FD test")
    else:
        # Create a simple synthetic SoC profile
        test_e = (SOC_MIN + SOC_MAX) / 2 * E_CAP * np.ones(T + 1)
        for t in range(T):
            test_e[t+1] = test_e[t] + np.random.uniform(-10, 10)
            test_e[t+1] = np.clip(test_e[t+1], SOC_MIN * E_CAP, SOC_MAX * E_CAP)
        print(f"  Using synthetic SoC profile for FD test")

    fd_results = validate_df_de_finite_difference(test_e, E_CAP, shi_fit)

    # Deep-dive diagnostics on specific timesteps
    if lp_available:
        # t=12: stable, trough of small cycle, interior of large — ratio 0.44
        diagnose_single_timestep(test_e, E_CAP, shi_fit, t_probe=12)
        # t=13: stable, peak of same small cycle — ratio 1.11
        diagnose_single_timestep(test_e, E_CAP, shi_fit, t_probe=13)

    # ------------------------------------------------------------------
    # 6. Solve NLP with IPOPT
    # ------------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("Step 3: NLP with IPOPT (revenue − degradation cost)")
    print(f"{'─'*72}")

    nlp = BatteryDispatchNLP(
        prices=prices,
        e_cap=E_CAP, p_cap=P_CAP,
        soc_min=SOC_MIN, soc_max=SOC_MAX,
        eff_in=EFF_IN, eff_out=EFF_OUT,
        dt=DT, replacement_cost=B,
        shi_fit=shi_fit,
    )

    print(f"  Variables:    {nlp.n_vars}  (c: {T}, d: {T}, e: {T+1})")
    print(f"  Constraints:  {nlp.n_cons}  ({T} energy bal + 1 periodic)")
    print(f"  w_deg:        {nlp.w_deg:,.0f} EUR")

    # Warm start from LP if available
    if lp_available:
        x0 = nlp.warm_start_from_lp(lp["p_vec"], lp["e"])
        print(f"  Initial point: warm start from LP solution")
    else:
        x0 = nlp.get_initial_point()
        print(f"  Initial point: midpoint SoC, zero dispatch")

    try:
        nlp_result = solve_nlp(nlp, x0=x0, max_iter=500, verbose=True)

        print(f"\n  Solver status:   {nlp_result['status_msg']}")
        print(f"  Solve time:      {nlp_result['solve_time']:.2f} s")
        print(f"  Iterations:      {nlp_result['n_iter']}")
        print(f"  Revenue (NLP):   {nlp_result['revenue']:>12,.2f} EUR")
        print(f"  Degradation fd:  {nlp_result['f_deg']:.6e}")
        print(f"  Deg cost:        {nlp_result['deg_cost_EUR']:>12,.2f} EUR")
        net_nlp = nlp_result["revenue"] - nlp_result["deg_cost_EUR"]
        print(f"  Net utility:     {net_nlp:>12,.2f} EUR")

        nlp_available = True
    except Exception as exc:
        print(f"\n  NLP solve failed: {exc}")
        import traceback; traceback.print_exc()
        nlp_available = False

    # ------------------------------------------------------------------
    # 7. Compare LP vs NLP
    # ------------------------------------------------------------------
    if lp_available and nlp_available:
        print(f"\n{'─'*72}")
        print("Comparison: LP (pure revenue) vs NLP (revenue − degradation)")
        print(f"{'─'*72}")

        net_lp  = lp["revenue"] - deg_cost_lp
        net_nlp = nlp_result["revenue"] - nlp_result["deg_cost_EUR"]

        print(f"  {'Metric':<25s}  {'LP':>14s}  {'NLP':>14s}  {'Δ':>10s}")
        print(f"  {'─'*25}  {'─'*14}  {'─'*14}  {'─'*10}")
        print(f"  {'Revenue [EUR]':<25s}  {lp['revenue']:>14,.2f}  "
              f"{nlp_result['revenue']:>14,.2f}  "
              f"{nlp_result['revenue']-lp['revenue']:>+10,.2f}")
        print(f"  {'Deg cost [EUR]':<25s}  {deg_cost_lp:>14,.2f}  "
              f"{nlp_result['deg_cost_EUR']:>14,.2f}  "
              f"{nlp_result['deg_cost_EUR']-deg_cost_lp:>+10,.2f}")
        print(f"  {'Net utility [EUR]':<25s}  {net_lp:>14,.2f}  "
              f"{net_nlp:>14,.2f}  "
              f"{net_nlp-net_lp:>+10,.2f}")
        print(f"  {'Cycles':<25s}  {len(cycles_lp):>14d}  "
              f"{nlp_result['n_cycles']:>14d}")

        rev_sacrifice = lp["revenue"] - nlp_result["revenue"]
        deg_saving    = deg_cost_lp - nlp_result["deg_cost_EUR"]
        print(f"\n  Revenue sacrifice:   {rev_sacrifice:>+10,.2f} EUR")
        print(f"  Degradation saving:  {deg_saving:>+10,.2f} EUR")
        if deg_saving > 0:
            print(f"  Ratio (saved/sacrificed): {deg_saving/max(rev_sacrifice,1e-9):.2f}")

    print(f"\n{'═'*72}")
    print("Prototype complete.")
    print(f"{'═'*72}")


if __name__ == "__main__":
    main()
