"""
Verification of the dispatch sub-gradient against finite differences.
=====================================================================

Purpose
-------
Two separate questions, deliberately separated:

  TEST 1  Depth-only cost.  Set k_sigma = 0 and k_T = 0, so the per-cycle stress is Phi(delta) alone. This is exactly the cost function
          Shi et al. (2018) analyze. The analytic sub-gradient from compute_subgradient() must then reproduce a central finite
          difference of the cost to high accuracy. This VERIFIES the code against Shi Eqs. 17-18.

  TEST 2  Full modeled cost.  Restore k_sigma = 1.04, so the per-cycle stress is Phi(delta) * S_sigma(sigma) * S_T(T), which is what the
          production code actually differentiates. The analytic sub-gradient holds sigma fixed. The central finite difference does not. The gap
          between them is the neglected mean-SoC coupling term.

  TEST 3  Efficiency-ratio identity. The ratio of the discharge coefficient to the charge coefficient in Shi Eqs. 17-18 is 1 / (eta_in*eta_out),
          i.e. the inverse round-trip efficiency. This reduces to 1/eta_out only in the special case eta_in = 1.

Run from VS Code on Windows. Place this file in the same folder as degradation_xu.py, degradation_shi.py and degradation_subgradient.py.

Outputs
-------
    Results/Verification/subgradient_sigma_coupling.pdf   (300 dpi, vector)
    Results/Verification/subgradient_sigma_coupling.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from degradation_xu import XuModelParams, XU_LMO, fit_shi_polynomial
from degradation_shi import (
    rainflow_cycle_counting,
    phi_shi,
    s_soc,
    s_temp,
)
from degradation_subgradient import compute_subgradient, build_half_cycle_map


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "Results" / "Verification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# TU Delft palette
NAVY = "#0C2340"
DARKRED = "#A50034"
FILLBLUE = "#0076C2"
TEXTWIDTH_IN = 6.201

E_CAP = 300.0          # MWh
SOC_MIN, SOC_MAX = 0.10, 0.90
DT_HOURS = 1.0
B_REPL = 72.0 * 1000.0  # EUR/MWh (72 EUR/kWh)
T_CELL_C = 25.0

# Symmetric efficiency, as in run_battery_..._v5_6_RTE_test.py
RTE_AC = 0.910
ETA = float(np.sqrt(RTE_AC))
ETA_IN = ETA
ETA_OUT = ETA

# Finite-difference steps, expressed as a fractional SoC shift
H_SOC_LIST = [1e-4, 1e-5, 1e-6]


# =============================================================================
# Synthetic SoC trajectory with known nested cycles
# =============================================================================

def build_synthetic_soc(steps_per_leg: int = 8) -> np.ndarray:
    """Piecewise-linear SoC trajectory with nested cycles of varied depth.

    Deliberately includes deep cycles, mid cycles and shallow ripples so the coupling term can be measured across the full amplitude range.
    """
    turning_points = [
        0.50, 0.88, 0.14, 0.62, 0.44, 0.78, 0.20, 0.56,
        0.50, 0.60, 0.46, 0.90, 0.10, 0.55, 0.38, 0.70, 0.50,
    ]
    soc = []
    for a, b in zip(turning_points[:-1], turning_points[1:]):
        leg = np.linspace(a, b, steps_per_leg, endpoint=False)
        soc.append(leg)
    soc.append(np.array([turning_points[-1]]))
    return np.concatenate(soc)


def load_soc() -> tuple[np.ndarray, float, str]:
    """Use the real dispatch trace if it is on disk, otherwise the synthetic one."""
    real = SCRIPT_DIR / "Results" / "storage_e_fixed.npy"
    cap = SCRIPT_DIR / "Results" / "e_cap_fixed.npy"
    if real.exists() and cap.exists():
        e = np.load(real).astype(float)
        e_cap = float(np.load(cap)[0])
        return e, e_cap, f"real dispatch ({real.name}, {len(e)} steps)"
    soc = build_synthetic_soc()
    return soc * E_CAP, E_CAP, f"synthetic trajectory ({len(soc)} steps)"


# =============================================================================
# Modeled degradation cost, as a function of the energy trace
# =============================================================================

def deg_cost(e: np.ndarray, e_cap: float, k3: float, k4: float,
             p: XuModelParams) -> float:
    """f = E * B * sum_i count_i * Phi(delta_i) * S_sigma(sigma_i) * S_T(T).

    This is exactly what degradation_shi.analyze_degradation_shi accumulates, scaled to a cost. Passing p with k_sigma = 0 and k_T = 0 makes it the
    depth-only cost that Shi et al. prove convex.
    """
    cycles = rainflow_cycle_counting(e, e_cap)
    if not cycles:
        return 0.0
    dod = np.array([c["dod"] for c in cycles])
    cnt = np.array([c["count"] for c in cycles])
    sig = np.array([c["soc_mean"] for c in cycles])
    stress = s_soc(sig, p.k_sigma, p.sigma_ref) * s_temp(T_CELL_C, p.k_T, p.T_ref_C)
    return float(np.sum(cnt * phi_shi(dod, k3, k4) * stress) * e_cap * B_REPL)


def fd_charge(e: np.ndarray, e_cap: float, t: int, h_soc: float,
              k3: float, k4: float, p: XuModelParams) -> float:
    """Central difference of f with respect to the charging power c_t.

    Raising c_t by h raises the energy trace by dt*eta_in*h at every step after t. Expressing the step as a fractional SoC shift makes it
    dimensionless and comparable across capacities: de = h_soc * e_cap,  so  h = de / (dt * eta_in).
    """
    de = h_soc * e_cap
    h_power = de / (DT_HOURS * ETA_IN)

    e_plus = e.copy()
    e_plus[t + 1:] += de
    e_minus = e.copy()
    e_minus[t + 1:] -= de

    f_plus = deg_cost(e_plus, e_cap, k3, k4, p)
    f_minus = deg_cost(e_minus, e_cap, k3, k4, p)
    return (f_plus - f_minus) / (2.0 * h_power)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    e, e_cap, src = load_soc()
    n = len(e)

    fit = fit_shi_polynomial(SOC_MIN, SOC_MAX, source="verification", verbose=False)
    k3, k4 = fit.k3, fit.k4

    p_full = XU_LMO
    p_depth = XuModelParams(k_sigma=0.0, k_T=0.0)

    print("=" * 74)
    print("Sub-gradient verification: depth term and neglected mean-SoC coupling")
    print("=" * 74)
    print(f"  Source        : {src}")
    print(f"  e_cap         : {e_cap:.1f} MWh")
    print(f"  Phi fit       : k3={k3:.4e}  k4={k4:.4f}  R2={fit.r2:.4f}  "
          f"range=[{fit.fit_lo:.2f},{fit.fit_hi:.2f}]")
    print(f"  eta_in/out    : {ETA_IN:.4f} / {ETA_OUT:.4f}  (RTE_ac = {RTE_AC:.3f})")
    print(f"  k_sigma       : {XU_LMO.k_sigma}")

    cycles = rainflow_cycle_counting(e, e_cap)
    attr = build_half_cycle_map(cycles, e)
    charging = np.where((attr["direction"] == -1) & (attr["cycle_owner"] != -1))[0]
    charging = charging[(charging > 0) & (charging < n - 2)]

    # Sample charging timesteps spread across the horizon
    sample = charging if len(charging) <= 40 else charging[
        np.linspace(0, len(charging) - 1, 40).astype(int)]

    # -------------------------------------------------------------------
    # TEST 3 first: the efficiency-ratio identity (no FD needed)
    # -------------------------------------------------------------------
    print(f"\n{'-'*74}\nTEST 3  Efficiency-ratio identity\n{'-'*74}")
    sg_full = compute_subgradient(
        e, cycles, dt_hours=DT_HOURS,
        battery_replacement_cost_per_MWh=B_REPL,
        eff_in=ETA_IN, eff_out=ETA_OUT, T_C=T_CELL_C, p=p_full, shi_fit=fit,
    )
    act_d = sg_full["subgrad_discharge"][sg_full["subgrad_discharge"] > 0]
    act_c = sg_full["subgrad_charge"][sg_full["subgrad_charge"] > 0]
    ratio = float(act_d.mean() / act_c.mean())
    correct = 1.0 / (ETA_IN * ETA_OUT)
    legacy = ETA_IN / ETA_OUT
    print(f"  measured discharge/charge ratio : {ratio:.4f}")
    print(f"  1 / (eta_in * eta_out)          : {correct:.4f}   <- correct identity")
    print(f"  eta_in / eta_out  (code line 461): {legacy:.4f}   <- WRONG unless eta_in = 1")
    print(f"  1 / eta_out                     : {1.0/ETA_OUT:.4f}   <- also only valid if eta_in = 1")

    # -------------------------------------------------------------------
    # TEST 1: depth-only cost. Analytic must match FD.
    # -------------------------------------------------------------------
    print(f"\n{'-'*74}\nTEST 1  Depth-only cost (k_sigma = 0): code vs finite difference\n{'-'*74}")
    sg_depth = compute_subgradient(
        e, cycles, dt_hours=DT_HOURS,
        battery_replacement_cost_per_MWh=B_REPL,
        eff_in=ETA_IN, eff_out=ETA_OUT, T_C=T_CELL_C, p=p_depth, shi_fit=fit,
    )
    an_depth = sg_depth["subgrad_charge"][sample]

    best_h = None
    for h_soc in H_SOC_LIST:
        fd = np.array([fd_charge(e, e_cap, int(t), h_soc, k3, k4, p_depth)
                       for t in sample])
        ok = np.abs(an_depth) > 0
        rel = np.abs(fd[ok] - an_depth[ok]) / np.abs(an_depth[ok])
        print(f"  h_soc = {h_soc:.0e}   median |rel error| = {np.median(rel)*100:7.3f} %   "
              f"max = {np.max(rel)*100:7.3f} %")
        if best_h is None:
            best_h, fd_depth_best = h_soc, fd

    # -------------------------------------------------------------------
    # TEST 2: full cost. Analytic freezes sigma; FD does not.
    # -------------------------------------------------------------------
    print(f"\n{'-'*74}\nTEST 2  Full cost (k_sigma = 1.04): neglected mean-SoC coupling\n{'-'*74}")
    an_full = sg_full["subgrad_charge"][sample]
    fd_full = np.array([fd_charge(e, e_cap, int(t), best_h, k3, k4, p_full)
                        for t in sample])

    dod_owner = attr["dod_at_t"][sample]
    ok = np.abs(fd_full) > 0
    rel_gap = (fd_full[ok] - an_full[ok]) / np.abs(fd_full[ok])
    predicted = XU_LMO.k_sigma * dod_owner[ok] / (2.0 * k4)

    print(f"  analytic sub-gradient omits  dPsi/dSigma * dSigma/dc_t")
    print(f"  median relative gap (FD - analytic)/|FD| : {np.median(rel_gap)*100:+7.2f} %")
    print(f"  mean   relative gap                     : {np.mean(rel_gap)*100:+7.2f} %")
    print(f"  max |relative gap|                      : {np.max(np.abs(rel_gap))*100:7.2f} %")
    print(f"  owning-cycle-only prediction k_s*d/(2*k4): median {np.median(predicted)*100:7.2f} %")
    print(f"\n  Note: the prediction accounts only for the owning cycle. Raising c_t")
    print(f"  also lifts every downstream cycle's mean SoC, so the measured gap can")
    print(f"  exceed it. That trajectory-shift effect is precisely what breaks Shi's")
    print(f"  parameterization: half-cycle DEPTHS depend only on the powers in their")
    print(f"  own index set, but half-cycle MEANS do not.")

    # -------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.42))

    ax = axes[0]
    lo = min(np.min(an_depth), np.min(fd_depth_best))
    hi = max(np.max(an_depth), np.max(fd_depth_best))
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, zorder=1)
    ax.scatter(an_depth, fd_depth_best, s=18, color=NAVY, zorder=2,
               label="depth-only cost")
    ax.scatter(an_full, fd_full, s=18, color=DARKRED, marker="^", zorder=3,
               label="full cost (with $S_\\sigma$)")
    ax.set_xlabel(r"analytic $\partial f / \partial c_t$  [EUR/MW]")
    ax.set_ylabel(r"central difference  [EUR/MW]")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)

    ax = axes[1]
    order = np.argsort(dod_owner[ok])
    ax.scatter(dod_owner[ok][order], 100 * rel_gap[order], s=18,
               color=DARKRED, zorder=3, label="measured gap")
    d_line = np.linspace(max(1e-3, np.min(dod_owner[ok])), np.max(dod_owner[ok]), 100)
    ax.plot(d_line, 100 * XU_LMO.k_sigma * d_line / (2.0 * k4),
            color=FILLBLUE, lw=1.4, zorder=2,
            label=r"$k_\sigma \delta / (2 k_4)$, owning cycle only")
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=1)
    ax.set_xlabel(r"cycle amplitude $\delta$ of the owning half-cycle  [-]")
    ax.set_ylabel("relative gap  [%]")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.tight_layout()
    pdf = OUT_DIR / "subgradient_sigma_coupling.pdf"
    png = OUT_DIR / "subgradient_sigma_coupling.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    csv = OUT_DIR / "subgradient_sigma_coupling.csv"
    np.savetxt(
        csv,
        np.column_stack([sample[ok], dod_owner[ok], an_full[ok], fd_full[ok],
                         rel_gap, predicted]),
        delimiter=",",
        header="t,dod_owner,analytic,central_difference,rel_gap,predicted_owning_cycle",
        comments="",
        fmt="%.8e",
    )

    print(f"\n  Saved: {pdf}")
    print(f"  Saved: {png}")
    print(f"  Saved: {csv}")


if __name__ == "__main__":
    main()
