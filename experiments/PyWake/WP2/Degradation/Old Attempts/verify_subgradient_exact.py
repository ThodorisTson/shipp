"""
Verification of the exact rainflow sub-gradient.
===============================================

Reads the .npy pair from  Results/RTE Tests/ , which is where
run_battery_xu_shi_degradation_v5_6_RTE_test.py writes them (RESULTS_DIR, line 134).
Do NOT point this at Results/ : the pair sitting there is left over from an older
routing of the script, and its two files came from different runs.

Three checks, each testing one thing:

  GUARD    Are storage_e and e_cap from the same solve? If the trace never reaches
           soc_max, the two files are out of sync and every normalized depth below
           is wrong. Checked first and loudly, because a stale e_cap silently
           rescales everything.

  TEST A   Exactness, split into two populations:
             smooth : forward and backward differences agree, so the cost is
                      differentiable. The analytic value must match to machine
                      precision.
             kink   : they disagree, so the rainflow map changes combinatorial
                      structure under the perturbation and the cost is not
                      differentiable. A central difference returns the AVERAGE of
                      the two one-sided derivatives and cannot match any
                      sub-gradient. The analytic value must instead coincide with
                      ONE of them, which is what makes it a valid element of the
                      sub-differential.

           Reporting one median against a central difference conflates the two and
           hides failures. That is how the attribution bug survived: median
           0.000 percent, max 1512 percent.

  TEST B   Efficiency identity, pointwise. |df/dd_t| / |df/dc_t| = 1/(eta_in*eta_out)
           at EVERY timestep, exactly, because G(t) is common to both. A ratio of
           MEANS over the horizon is not a valid test: it recovers the coefficient
           ratio only when the charging and discharging depth distributions coincide,
           which is why it read exactly 1/eta_out while the attribution map was
           collapsed onto a single cycle.

  TEST C   Why S_sigma is excluded from the gradient path. Depth-only analytic
           against a central difference of the S_sigma-weighted cost. The gap is the
           non-local mean-SoC term, and it grows with the number of cycles downstream
           of t.

Run from VS Code on Windows, in the folder holding degradation_xu.py,
degradation_shi.py and degradation_subgradient.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from degradation_xu import XuModelParams, XU_LMO, fit_shi_polynomial
from degradation_shi import rainflow_cycle_counting, phi_shi, s_soc, s_temp
from degradation_subgradient import compute_subgradient


SCRIPT_DIR = Path(__file__).parent

# Must match RESULTS_DIR in run_battery_xu_shi_degradation_v5_6_RTE_test.py (line 134)
RESULTS_DIR = SCRIPT_DIR / "Results" / "RTE Tests"
OUT_DIR = RESULTS_DIR / "Verification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#0C2340"
DARKRED = "#A50034"
FILLBLUE = "#0076C2"
TEXTWIDTH_IN = 6.201

SOC_MIN, SOC_MAX = 0.10, 0.90
DT_HOURS = 1.0
B_REPL = 72.0 * 1000.0        # EUR/MWh  (72 EUR/kWh)
T_CELL_C = 25.0
RTE_AC = 0.910
ETA = float(np.sqrt(RTE_AC))
ETA_IN = ETA_OUT = ETA

H_SOC = 1e-6
N_SAMPLE = 200


# =============================================================================
# Data
# =============================================================================

def load_trace() -> tuple[np.ndarray, float, str]:
    e_path = RESULTS_DIR / "storage_e_fixed.npy"
    c_path = RESULTS_DIR / "e_cap_fixed.npy"

    if not (e_path.exists() and c_path.exists()):
        raise FileNotFoundError(
            f"Expected the .npy pair in {RESULTS_DIR}\n"
            f"Run run_battery_xu_shi_degradation_v5_6_RTE_test.py first.\n"
            f"Do not fall back to Results/ : the pair there is stale."
        )

    e = np.load(e_path).astype(float)
    e_cap = float(np.load(c_path)[0])
    return e, e_cap, f"{e_path.parent.name}/{e_path.name}  ({len(e)} steps)"


def guard(e: np.ndarray, e_cap: float) -> bool:
    """Refuse to trust the run if the trace and the capacity disagree."""
    lo, hi = e.min() / e_cap, e.max() / e_cap
    print(f"\n{'-'*74}\nGUARD  Are storage_e and e_cap from the same solve?\n{'-'*74}")
    print(f"  e_cap             : {e_cap:.1f} MWh")
    print(f"  SoC range implied : {lo:.4f} to {hi:.4f}")
    print(f"  SoC window (YAML) : {SOC_MIN:.2f} to {SOC_MAX:.2f}")

    ok = (hi > SOC_MAX - 0.02) and (lo < SOC_MIN + 0.02)
    if not ok:
        implied = e.max() / SOC_MAX
        print(f"\n  *** INCONSISTENT ***")
        print(f"  The trace never reaches soc_max. An e_cap of {implied:.1f} MWh")
        print(f"  would put the range at [{e.min()/implied:.3f}, {e.max()/implied:.3f}],")
        print(f"  which matches the window. The two files are from different runs.")
        print(f"  Every normalized depth below is scaled by {implied/e_cap:.3f}.\n")
    else:
        print(f"  consistent\n")
    return ok


# =============================================================================
# Cost and one-sided differences
# =============================================================================

def deg_cost(e, e_cap, k3, k4, p: XuModelParams) -> float:
    """f = E * B * sum_i n_i * Phi(delta_i) * S_sigma(sigma_i) * S_T(T).

    Passing p with k_sigma = 0 and k_T = 0 gives the depth-only cost that Shi
    et al. prove convex and that the gradient path differentiates.
    """
    c = rainflow_cycle_counting(e, e_cap)
    if not c:
        return 0.0
    d = np.array([x["dod"] for x in c])
    q = np.array([x["count"] for x in c])
    s = np.array([x["soc_mean"] for x in c])
    stress = s_soc(s, p.k_sigma, p.sigma_ref) * s_temp(T_CELL_C, p.k_T, p.T_ref_C)
    return float(np.sum(q * phi_shi(d, k3, k4) * stress) * e_cap * B_REPL)


def one_sided(e, e_cap, t, k3, k4, p, f0) -> tuple[float, float]:
    """Forward and backward differences of f with respect to c_t.

    Raising c_t by h raises the energy trace by dt*eta_in*h at every step after t.
    The step is expressed as a fractional SoC shift so it is dimensionless.
    """
    de = H_SOC * e_cap
    h = de / (DT_HOURS * ETA_IN)
    ep = e.copy(); ep[t + 1:] += de
    em = e.copy(); em[t + 1:] -= de
    fwd = (deg_cost(ep, e_cap, k3, k4, p) - f0) / h
    bwd = (f0 - deg_cost(em, e_cap, k3, k4, p)) / h
    return fwd, bwd


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    e, e_cap, src = load_trace()
    n = len(e)
    fit = fit_shi_polynomial(SOC_MIN, SOC_MAX, source="verification")
    k3, k4 = fit.k3, fit.k4
    p_depth = XuModelParams(k_sigma=0.0, k_T=0.0)

    print("=" * 74)
    print("Exact rainflow sub-gradient: verification")
    print("=" * 74)
    print(f"  Source : {src}")
    print(f"  Phi    : k3={k3:.4e}  k4={k4:.4f}  R2={fit.r2:.4f}  "
          f"fit=[{fit.fit_lo:.2f},{fit.fit_hi:.2f}]")
    print(f"  eta    : {ETA_IN:.4f} (symmetric, RTE_ac = {RTE_AC:.3f})")

    guard(e, e_cap)

    cycles = rainflow_cycle_counting(e, e_cap)
    sg = compute_subgradient(
        e, cycles, dt_hours=DT_HOURS,
        battery_replacement_cost_per_MWh=B_REPL,
        eff_in=ETA_IN, eff_out=ETA_OUT, shi_fit=fit,
    )

    rising = np.diff(e, append=e[-1]) > 0
    idx = np.where(rising)[0]
    idx = idx[(idx > 10) & (idx < n - 10)]
    sample = idx[np.linspace(0, len(idx) - 1, min(N_SAMPLE, len(idx))).astype(int)]

    # ---------------- TEST A ----------------
    print(f"{'-'*74}\nTEST A  Exactness: smooth points vs rainflow topology changes\n{'-'*74}")
    f0_d = deg_cost(e, e_cap, k3, k4, p_depth)
    fwd = np.zeros(len(sample)); bwd = np.zeros(len(sample))
    for i, t in enumerate(sample):
        fwd[i], bwd[i] = one_sided(e, e_cap, int(t), k3, k4, p_depth, f0_d)
    ctr = 0.5 * (fwd + bwd)
    an = sg["dfdc"][sample]

    scale = np.maximum(np.abs(ctr), 1e-9)
    kink = np.abs(fwd - bwd) / scale > 1e-6
    err_ctr = np.abs(an - ctr) / scale
    err_side = np.minimum(np.abs(an - fwd), np.abs(an - bwd)) / scale

    print(f"  smooth timesteps          : {int(np.sum(~kink)):4d}/{len(sample)}")
    if np.any(~kink):
        print(f"    vs central difference   : median {np.median(err_ctr[~kink])*100:.2e} %"
              f"   max {np.max(err_ctr[~kink])*100:.2e} %   <- must be machine precision")
    print(f"  rainflow topology changes : {int(np.sum(kink)):4d}/{len(sample)}")
    if np.any(kink):
        print(f"    vs central difference   : median {np.median(err_ctr[kink])*100:8.3f} %"
              f"   max {np.max(err_ctr[kink])*100:8.3f} %   <- meaningless at a kink")
        print(f"    vs nearest one-sided    : median {np.median(err_side[kink])*100:.2e} %"
              f"   max {np.max(err_side[kink])*100:.2e} %   <- valid sub-differential element")
    print(f"\n  {100*np.mean(kink):.0f} % of charging timesteps sit at a topology change under a")
    print(f"  {H_SOC:.0e} SoC perturbation. This is the same non-smoothness that stops a")
    print(f"  smooth NLP solver converging on the monolithic form.")

    # ---------------- TEST B ----------------
    print(f"\n{'-'*74}\nTEST B  Efficiency identity, pointwise\n{'-'*74}")
    nz = np.abs(sg["dfdc"]) > 1e-12
    r = np.abs(sg["dfdd"][nz] / sg["dfdc"][nz])
    tgt = 1.0 / (ETA_IN * ETA_OUT)
    print(f"  |df/dd| / |df/dc|  : min {r.min():.10f}   max {r.max():.10f}")
    print(f"  1/(eta_in*eta_out) : {tgt:.10f}")
    print(f"  max deviation      : {np.max(np.abs(r - tgt)):.2e}")
    assert np.max(np.abs(r - tgt)) < 1e-9, "efficiency wiring is wrong"
    print(f"  PASS")

    ns = sg["n_straddled"]
    print(f"\n  cycles straddling each timestep : min {ns.min()}  median {int(np.median(ns))}"
          f"  max {ns.max()}")
    print(f"  Shi Eqs. 17-18 assume exactly 1. That holds at "
          f"{100*np.mean(ns == 1):.1f} % of timesteps.")

    # ---------------- TEST C ----------------
    print(f"\n{'-'*74}\nTEST C  Why S_sigma is excluded from the gradient path\n{'-'*74}")
    f0_f = deg_cost(e, e_cap, k3, k4, XU_LMO)
    fd_full = np.zeros(len(sample))
    for i, t in enumerate(sample):
        a_, b_ = one_sided(e, e_cap, int(t), k3, k4, XU_LMO, f0_f)
        fd_full[i] = 0.5 * (a_ + b_)
    gap = (fd_full - an) / np.maximum(np.abs(fd_full), 1e-9)
    print(f"  depth-only analytic vs central difference of the S_sigma-weighted cost")
    print(f"    median relative gap : {np.median(gap)*100:+7.2f} %")
    print(f"    max                 : {np.max(np.abs(gap))*100:7.2f} %")
    print(f"  Cycle depths are invariant to a uniform shift of the SoC trajectory.")
    print(f"  Cycle means are not. Raising c_t lifts the mean SoC of every cycle")
    print(f"  downstream of t, so the derivative of the weighted cost is not local.")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.42))

    ax = axes[0]
    lim = [min(an.min(), ctr.min()), max(an.max(), ctr.max())]
    ax.plot(lim, lim, color="0.6", lw=0.8, zorder=1)
    ax.scatter(an[~kink], ctr[~kink], s=16, color=NAVY, zorder=3,
               label=f"smooth ({int(np.sum(~kink))})")
    ax.scatter(an[kink], ctr[kink], s=16, color=DARKRED, marker="^", zorder=2,
               label=f"topology change ({int(np.sum(kink))})")
    ax.set_xlabel(r"analytic $\partial f / \partial c_t$  [EUR/MW]")
    ax.set_ylabel("central difference  [EUR/MW]")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    ax = axes[1]
    bins = np.logspace(-16, 1, 40)
    ax.hist(np.clip(err_side, 1e-16, None), bins=bins, color=FILLBLUE,
            label="vs nearest one-sided")
    ax.hist(np.clip(err_ctr, 1e-16, None), bins=bins, color=DARKRED, alpha=0.55,
            label="vs central difference")
    ax.set_xscale("log")
    ax.set_xlabel("relative error  [-]")
    ax.set_ylabel("timesteps  [-]")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    for a in axes:
        a.tick_params(labelsize=7)
        a.xaxis.label.set_size(8)
        a.yaxis.label.set_size(8)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "subgradient_exact.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "subgradient_exact.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Saved: {OUT_DIR / 'subgradient_exact.pdf'}")
    print(f"  Saved: {OUT_DIR / 'subgradient_exact.png'}")


if __name__ == "__main__":
    main()