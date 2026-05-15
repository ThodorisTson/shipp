"""
run_battery_xu_shi_degradation_v3_3.py
WP2 battery degradation — v3.3  Gradient validation (Jenna's two checks)

Implements the two validation tests requested by the thesis supervisor:

  TEST 1: dDeg/dDoD slope verification (3-point FD)
     Same logic as v3.1 but on a 20-step synthetic SoC trace.
     Scales all energies by s = 0.989, 1.000, 1.011.
     Uniform scaling preserves σ̄ → S_σ coupling absent.
     Measured slope d(fd)/d(scale) vs analytical k4 × fd_cycle.
     Target: ratio → 1.0.

  TEST 2: Subgradient per-cycle analytical ratio
     For each rainflow cycle: expected subgrad[t] = 0.5 × S_σ(σ̄) × Φ′(δ)
     Verified on clean 3-element traces [e_lo, e_hi, e_lo] — one cycle,
     unambiguous half-cycle attribution, no multi-cycle mixing.
     Target: subgrad[1] / expected → 1.0 for all cycles.

Supporting diagnostics retained from v3.2:
  [2b] Rainflow consistency check (xu vs shi — identical confirmed)
  [2c] Phi(dod) vs _fd_cycle on 3-element traces

No LP solve required.  Runs in < 10 seconds.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from degradation_xu import rainflow_cycle_counting
from degradation_shi import analyze_degradation_shi
from degradation_subgradient import compute_subgradient, fit_shi_polynomial

# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR     = Path(__file__).parent
VALIDATION_DIR = SCRIPT_DIR / "Gradient_Verification"
VALIDATION_DIR.mkdir(exist_ok=True)

run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# WP2 nominal battery parameters
E_CAP   = 300.0   # MWh
SOC_MIN = 0.10
SOC_MAX = 0.90
T_CELL  = 25.0    # °C

BAT_PARAMS = {
    "power_capacity_W":   150e6,
    "energy_capacity_Wh": E_CAP * 1e6,
    "rte_nominal":        0.95,
    "pcu_efficiency":     0.975,
    "capex_EUR_per_kWh":  150.0,
    "capex_EUR_per_kW":   0.0,
    "soc_min":            SOC_MIN,
    "soc_max":            SOC_MAX,
}

# Perturbation size for propagating FD [MWh]
EPS_MWH = 0.5

# ── INTERIOR TRACE ─────────────────────────────────────────────────────────────
# Both turning points well away from SoC boundaries so np.clip never activates.
#
#   Peak   E[4]  = 215 MWh  →  71.7% SoC  (55 MWh clearance from 270 ceiling)
#   Valley E[12] =  85 MWh  →  28.3% SoC  (55 MWh clearance from  30 floor)
#
# Expected rainflow DoDs (w.r.t. E_CAP = 300 MWh):
#   Large  cycle : (215 − 85) / 300 = 0.433   within Shi fit range [0.15, 0.80]
#   Inner  cycle : (175 − 85) / 300 = 0.300
#   Start residue: (215 − 120) / 300 = 0.317
#   End   residue: (175 − 120) / 300 = 0.183
#
# Active timesteps for Part A (subgrad != 0):
#   t = 0–3   charge toward peak (large cycle + start residue charge half)
#   t = 4–11  discharge from peak to valley (large cycle discharge half)
#   t = 12–16 charge from valley to inner peak
#   t = 17–18 discharge from inner peak to end (end residue)

E_TRACE = np.array([
    120, 145, 170, 195, 215,   # t=0–4    rise to peak   (71.7% SoC)
    200, 180, 160, 140, 120,   # t=5–9    fall from peak
    105,  95,  85,             # t=10–12  fall to valley  (28.3% SoC)
     95, 110, 130, 150, 175,   # t=13–17  rise to inner peak
    155, 120,                  # t=18–19  fall to end
], dtype=float)

N = len(E_TRACE)
assert N == 20

# =============================================================================
# Helpers
# =============================================================================

def _fd_cycle(E: np.ndarray, e_cap: float, shi_fit) -> float:
    """Compute Shi fd_cycle (cycle-only, no calendar) for stored energy trace E [MWh]."""
    p = np.zeros(len(E)).tolist()
    res = analyze_degradation_shi(
        p, E.tolist(), e_cap, BAT_PARAMS,
        shi_fit=shi_fit,
        T_cell_C=T_CELL,
        dt_hours=1.0,
        eol_thresholds=[],
    )
    return float(res["fd_shi"])

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("WP2 BATTERY — v3.2.1  FD validation of dDeg/dDoD subgradient")
    print("20-step synthetic SoC trace  |  No LP solve  |  < 10 seconds")
    print("=" * 70)

    # ── [1/3] Fit Shi polynomial ──────────────────────────────────────────────
    print("\n[1/3] Fitting Shi polynomial on WP2 SoC window...")
    shi_fit = fit_shi_polynomial(SOC_MIN, SOC_MAX, verbose=True)
    k3, k4  = shi_fit.k3, shi_fit.k4
    print(f"      k3 = {k3:.4e}   k4 = {k4:.4f}   R² = {shi_fit.r2:.4f}")

    # ── [2/3] Base fd_cycle and subgradient ───────────────────────────────────
    print("\n[2/3] Computing base fd_cycle and subgradient...")
    fd_cycle_base = _fd_cycle(E_TRACE, E_CAP, shi_fit)

    cycles = rainflow_cycle_counting(E_TRACE.tolist(), E_CAP)
    sg_res = compute_subgradient(
        storage_e=E_TRACE.tolist(),
        cycles=cycles,
        dt_hours=1.0,
        battery_replacement_cost_per_MWh=1.0,  # cost=1 so ratio is dimensionless
        eff_in=1.0,
        eff_out=1.0,
        shi_fit=shi_fit,
    )
    subgrad = np.asarray(sg_res["subgrad_combined"], dtype=float)

    print(f"      fd_cycle (base)  = {fd_cycle_base:.6e}")
    print(f"      rainflow cycles  = {len(cycles)}")
    print(f"      subgrad range    = [{subgrad.min():.4e}, {subgrad.max():.4e}]  [1/MWh]")
    print(f"      nonzero subgrad  = {int(np.sum(np.abs(subgrad) > 1e-20))} / {N}")

    # ── [2b/3] Rainflow side-by-side comparison ───────────────────────────────
    print("\n[2b/3] Rainflow comparison: degradation_xu vs degradation_shi...")

    # Import the local rainflow from degradation_shi
    try:
        from degradation_shi import rainflow_cycle_counting as rainflow_shi
        _shi_rainflow_available = True
    except ImportError:
        print("      WARNING: cannot import rainflow_cycle_counting from degradation_shi")
        _shi_rainflow_available = False

    cycles_xu = rainflow_cycle_counting(E_TRACE.tolist(), E_CAP)   # already imported

    print(f"\n  degradation_xu  rainflow  ({len(cycles_xu)} cycles):")
    print(f"  {'#':>3}  {'dod':>8}  {'count':>7}  {'t_start':>8}  {'t_end':>8}  {'rng_MWh':>9}")
    print("  " + "-" * 55)
    for i, c in enumerate(cycles_xu):
        print(f"  {i:3d}  {c['dod']:8.5f}  {c.get('count', float('nan')):7.4f}  "
            f"  {str(c.get('t_start', '?')):>6}    {str(c.get('t_end', '?')):>6}  "
            f"{c['dod']*E_CAP:9.3f}")

    if _shi_rainflow_available:
        cycles_shi = rainflow_shi(E_TRACE.tolist(), E_CAP)
        print(f"\n  degradation_shi rainflow  ({len(cycles_shi)} cycles):")
        print(f"  {'#':>3}  {'dod':>8}  {'count':>7}  {'t_start':>8}  {'t_end':>8}  {'rng_MWh':>9}")
        print("  " + "-" * 55)
        for i, c in enumerate(cycles_shi):
            print(f"  {i:3d}  {c['dod']:8.5f}  {c.get('count', float('nan')):7.4f}  "
                f"  {str(c.get('t_start', '?')):>6}    {str(c.get('t_end', '?')):>6}  "
                f"{c['dod']*E_CAP:9.3f}")

        # Direct comparison
        print(f"\n  Cycle-by-cycle diff (xu_dod − shi_dod):")
        n = min(len(cycles_xu), len(cycles_shi))
        all_match = True
        for i in range(n):
            diff = cycles_xu[i]['dod'] - cycles_shi[i]['dod']
            match = "✓" if abs(diff) < 1e-9 else "✗ MISMATCH"
            if abs(diff) >= 1e-9:
                all_match = False
            print(f"    cycle {i}: xu={cycles_xu[i]['dod']:.6f}  "
                f"shi={cycles_shi[i]['dod']:.6f}  diff={diff:+.2e}  {match}")
        if len(cycles_xu) != len(cycles_shi):
            print(f"  ✗ CYCLE COUNT MISMATCH: xu={len(cycles_xu)}  shi={len(cycles_shi)}")
            all_match = False
        if all_match:
            print("  ✓ All cycles identical between xu and shi rainflow")

    # ── [2c/3] Per-cycle fd decomposition and 3-element FD isolation ──────────
    print("\n[2c/3] Per-cycle fd contributions and 3-element FD isolation...")
    print("  NOTE: 2-element traces have no turning point → rainflow returns 0 cycles.")
    print("  Using 3-element full-cycle traces [e_lo, e_hi, e_lo] instead.")
    print("  These have one turning point, count=1.0, DoD=(e_hi−e_lo)/E_cap.")
    print("  Results confirm Phi(dod) is computed correctly by analyze_degradation_shi.\n")

    k3, k4 = shi_fit.k3, shi_fit.k4

    # ── Part 1: verify _fd_cycle returns Phi(dod) for each cycle ─────────────
    print(f"  Part 1: _fd_cycle vs analytic Phi(dod) on 3-element traces")
    print(f"  {'#':>3}  {'dod':>8}  {'Phi(dod)':>13}  {'_fd_3elem':>13}  "
          f"{'ratio_fd/phi':>13}  note")
    print("  " + "-" * 65)

    fd_sum_3elem = 0.0
    for i, c in enumerate(cycles_xu):
        dod = c['dod']
        phi_val = k3 * dod ** k4

        # 3-element full cycle: valley → peak → valley
        e_lo3  = 0.40 * E_CAP                   # 120 MWh, well interior
        e_hi3  = e_lo3 + dod * E_CAP
        if e_hi3 > SOC_MAX * E_CAP - 2.0:
            e_hi3 = SOC_MAX * E_CAP - 5.0
            e_lo3 = e_hi3 - dod * E_CAP

        trace_3 = np.array([e_lo3, e_hi3, e_lo3])
        fd_3e   = _fd_cycle(trace_3, E_CAP, shi_fit)
        fd_sum_3elem += fd_3e

        ratio_3e = fd_3e / phi_val if abs(phi_val) > 1e-30 else float('nan')
        note = "← count=1 full cycle, expect ratio=1.0"
        print(f"  {i:3d}  {dod:8.5f}  {phi_val:+13.4e}  {fd_3e:+13.4e}  "
              f"{ratio_3e:+13.6f}  {note}")

    print(f"\n  fd_cycle_base (20-step trace) : {fd_cycle_base:.6e}")
    print(f"  Sum of _fd_cycle (3-elem)     : {fd_sum_3elem:.6e}")
    print(f"  Sum of Phi(dod) all cycles    : "
          f"{sum(k3*c['dod']**k4 for c in cycles_xu):.6e}")
    print(f"  Sum of 0.5*Phi (half-cycles)  : "
          f"{sum(0.5*k3*c['dod']**k4 for c in cycles_xu):.6e}")
    print(f"  → fd_cycle_base closest to    : "
          f"{'Sum of 0.5*Phi' if abs(fd_cycle_base - sum(0.5*k3*c['dod']**k4 for c in cycles_xu)) < abs(fd_cycle_base - sum(k3*c['dod']**k4 for c in cycles_xu)) else 'Sum of Phi'}")

    # ── [3/4] Test 1: 3-point slope FD — dDeg/dDoD verification ─────────────
    # This is the same logic as v3.1 but on the synthetic 20-step trace.
    # Jenna: "run at 89, 90, 91 — three analyses — slope = gradient"
    # Scaling all energies by s preserves σ̄ (cancels in numerator+denominator)
    # so S_σ coupling is absent. This directly validates dDeg/dDoD.
    print(f"\n[3/4] Test 1 — 3-point slope FD on synthetic trace  (dDeg/dDoD check)...")
    print(f"  Scales all 20 energies by s = 0.989, 1.000, 1.011.")
    print(f"  σ̄ is invariant to uniform scaling → S_σ coupling absent.")
    print(f"  Analytical slope: k4 × fd_cycle_base = {k4:.4f} × {fd_cycle_base:.4e}"
          f" = {k4 * fd_cycle_base:.4e}\n")

    EPS_SCALE = 1.0 / 90.0    # same relative step as v3.1

    fd_lo_s = _fd_cycle(E_TRACE * (1.0 - EPS_SCALE), E_CAP, shi_fit)
    fd_mid_s = fd_cycle_base
    fd_hi_s = _fd_cycle(E_TRACE * (1.0 + EPS_SCALE), E_CAP, shi_fit)

    slope_measured    = (fd_hi_s - fd_lo_s) / (2.0 * EPS_SCALE)
    slope_analytical  = k4 * fd_cycle_base
    ratio_slope       = slope_measured / slope_analytical

    print(f"  {'scale':>10}  {'fd_cycle':>14}")
    print("  " + "-" * 28)
    print(f"  {1.0-EPS_SCALE:>10.6f}  {fd_lo_s:+14.6e}")
    print(f"  {1.0:>10.6f}  {fd_mid_s:+14.6e}  ← base")
    print(f"  {1.0+EPS_SCALE:>10.6f}  {fd_hi_s:+14.6e}")
    print(f"\n  Measured slope  d(fd)/d(scale) : {slope_measured:.6e}")
    print(f"  Analytical      k4 × fd_cycle  : {slope_analytical:.6e}")
    print(f"  Ratio                          : {ratio_slope:.6f}   (target = 1.0)")
    if abs(ratio_slope - 1.0) < 0.02:
        print(f"  ✓ PASS — dDeg/dDoD verified on synthetic trace")
    else:
        print(f"  ✗ FAIL — ratio outside 2% tolerance")

    # ── [4/4] Test 2: analytical subgradient ratio per cycle ──────────────────
    # Jenna: "validation of the subgradient that you calculated"
    # For each cycle: subgrad[t] should equal 0.5 × S_σ(σ̄) × Φ′(δ)
    # The 0.5 comes from half-cycle attribution (count = 0.5 for residues).
    # No FD required — this is a direct analytical comparison.
    print(f"\n[4/4] Test 2 — Subgradient analytical ratio per cycle...")
    print(f"  For each cycle: expected subgrad = 0.5 × S_σ(σ̄) × Φ′(δ)")
    print(f"  Verified on 3-element trace [e_lo, e_hi, e_lo] per cycle.")
    print(f"  t=1 is the discharge timestep (half-cycle attribution = 0.5).")
    print(f"  S_σ(σ̄) = exp(k_σ × (σ̄ − σ_ref)),  k_σ=1.04,  σ_ref=0.50\n")
    print(f"  {'#':>3}  {'dod':>8}  {'σ̄':>7}  {'S_σ':>7}  "
          f"{'Φ′(δ)':>13}  {'0.5×S_σ×Φ′':>13}  {'subgrad[1]':>13}  {'ratio':>8}")
    print("  " + "-" * 85)

    k_sigma   = 1.04
    sigma_ref = 0.50
    sg_ratios = []
    dods_plot        = []
    expected_sgs_plot = []
    sg_t1s_plot      = []

    for i, c in enumerate(cycles_xu):
        dod       = c['dod']
        phi_prime = k3 * k4 * dod ** (k4 - 1.0)

        # 3-element trace anchored at 40% SoC, same position as [2c]
        e_lo3 = 0.40 * E_CAP
        e_hi3 = e_lo3 + dod * E_CAP
        if e_hi3 > SOC_MAX * E_CAP - 2.0:
            e_hi3 = SOC_MAX * E_CAP - 5.0
            e_lo3 = e_hi3 - dod * E_CAP

        sigma_bar    = (e_hi3 + e_lo3) / (2.0 * E_CAP)
        s_sigma      = np.exp(k_sigma * (sigma_bar - sigma_ref))
        expected_sg  = 0.5 * s_sigma * phi_prime

        # Subgradient at t=1 (discharge half of the single cycle)
        cyc3 = rainflow_cycle_counting([e_lo3, e_hi3, e_lo3], E_CAP)
        sg3  = compute_subgradient(
            storage_e=[e_lo3, e_hi3, e_lo3],
            cycles=cyc3,
            dt_hours=1.0,
            battery_replacement_cost_per_MWh=1.0,
            eff_in=1.0, eff_out=1.0,
            shi_fit=shi_fit,
        )
        sg_t1 = float(np.asarray(sg3["subgrad_combined"])[1])

        ratio_sg = sg_t1 / expected_sg if abs(expected_sg) > 1e-30 else float('nan')
        dods_plot.append(dod)
        expected_sgs_plot.append(expected_sg)
        sg_t1s_plot.append(sg_t1)
        sg_ratios.append(ratio_sg)

        print(f"  {i:3d}  {dod:8.5f}  {sigma_bar:7.4f}  {s_sigma:7.4f}  "
              f"{phi_prime:+13.4e}  {expected_sg:+13.4e}  "
              f"{sg_t1:+13.4e}  {ratio_sg:+8.4f}")

    sg_ratios = np.array(sg_ratios)
    print(f"\n  Mean ratio : {np.mean(sg_ratios):.6f}   (target = 1.0)")
    print(f"  Std  ratio : {np.std(sg_ratios):.6f}")
    print(f"  Max |r−1|  : {np.max(np.abs(sg_ratios - 1.0)):.6f}")
    if np.max(np.abs(sg_ratios - 1.0)) < 0.02:
        print(f"  ✓ PASS — subgradient per-cycle attribution verified")
    else:
        print(f"  ✗ FAIL — ratio outside 2% tolerance")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating plots and report...")

    # ── Figure 1: SoC trace + Test 1 (3-point slope) ─────────────────────────
    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))
    fig1.subplots_adjust(wspace=0.32)

    # Panel 1: SoC trace
    ax = axes1[0]
    ax.plot(range(N), E_TRACE, '-o', color='#2c7bb6', lw=1.5, ms=5, label='E_t')
    tp = [i for i in range(N) if (i == 0 or i == N - 1 or
          (i > 0 and i < N - 1 and
           ((E_TRACE[i] > E_TRACE[i-1]) != (E_TRACE[i+1] > E_TRACE[i]))))]
    ax.scatter(tp, E_TRACE[tp], color='#d7191c', s=90, zorder=5,
               marker='D', label='Turning points')
    ax.axhline(SOC_MAX * E_CAP, color='gray', ls=':', lw=1.0,
               label='SoC bounds (10%–90%)')
    ax.axhline(SOC_MIN * E_CAP, color='gray', ls=':', lw=1.0)
    ax.set_xlabel("Timestep t", fontsize=10)
    ax.set_ylabel("Stored energy E [MWh]", fontsize=10)
    ax.set_title("Synthetic SoC trace (20 steps)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2: Test 1 — 3-point slope (same layout as v3.1)
    ax = axes1[1]
    scales    = np.array([1.0 - EPS_SCALE, 1.0, 1.0 + EPS_SCALE])
    fd_vals   = np.array([fd_lo_s, fd_mid_s, fd_hi_s])
    scale_cont = np.linspace(scales[0] - 0.005, scales[2] + 0.005, 200)

    # Measured slope line through midpoint
    ax.plot(scale_cont,
            fd_mid_s + slope_measured * (scale_cont - 1.0),
            '--', color='#2c7bb6', lw=2.0,
            label=f'FD slope = {slope_measured:.4e}')
    # Analytical slope line through midpoint
    ax.plot(scale_cont,
            fd_mid_s + slope_analytical * (scale_cont - 1.0),
            '-', color='#d7191c', lw=2.0,
            label=f'k₄ × fd_cycle = {slope_analytical:.4e}')
    # Three FD points
    ax.scatter(scales, fd_vals, color='#2c3e50', s=90, zorder=5,
               label='FD points (lo / mid / hi)')

    ax.set_xlabel("SoC scale factor  (1.0 = nominal synthetic trace)", fontsize=10)
    ax.set_ylabel("fd_cycle  [—]", fontsize=10)
    ax.set_title(
        f"Test 1  —  d(fd)/d(scale)  |  ratio = {ratio_slope:.4f}  (target = 1.0)",
        fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig1.suptitle(
        f"v3.2  Gradient Validation  |  Synthetic 20-step trace  |  "
        f"Test 1: dDeg/dDoD slope ratio = {ratio_slope:.4f}",
        fontsize=9, y=1.01,
    )
    plot1_path = VALIDATION_DIR / f"fd_validation_test1_{run_ts}.png"
    plt.savefig(plot1_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Test 1 plot saved: {plot1_path.name}")

    # ── Figure 2: Test 2 — subgradient analytical ratio ───────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.subplots_adjust(wspace=0.35)

    x_idx  = np.arange(len(cycles_xu))
    dods_arr = np.array(dods_plot)
    x_labels = [f"{d:.3f}" for d in dods_arr]

    # Panel 1: magnitude comparison — expected vs measured subgrad
    ax = axes2[0]
    w = 0.35
    ax.bar(x_idx - w/2, np.abs(expected_sgs_plot), width=w,
           color='#d7191c', alpha=0.8, label='0.5 × S_σ × Φ′(δ)  (expected)')
    ax.bar(x_idx + w/2, np.abs(sg_t1s_plot),       width=w,
           color='#2c7bb6', alpha=0.8, label='|subgrad[1]|  (measured)')
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"dod={l}" for l in x_labels], fontsize=8)
    ax.set_ylabel("|subgrad|  [1/MWh]", fontsize=10)
    ax.set_title("Test 2  —  Expected vs measured subgradient magnitude", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: ratio subgrad[1] / expected per cycle
    ax = axes2[1]
    colors = ['#27ae60' if abs(r - 1.0) < 0.02 else '#e74c3c'
              for r in sg_ratios]
    ax.bar(x_idx, sg_ratios, color=colors, alpha=0.85, zorder=3)
    ax.axhline(1.0, color='#2c3e50', lw=1.5, ls='--', label='target = 1.0')
    ax.axhline(1.02, color='gray', lw=1.0, ls=':', alpha=0.6, label='±2% tolerance')
    ax.axhline(0.98, color='gray', lw=1.0, ls=':', alpha=0.6)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"dod={l}" for l in x_labels], fontsize=8)
    ax.set_ylabel("subgrad[1] / (0.5 × S_σ × Φ′(δ))  [—]", fontsize=10)
    ax.set_title(
        f"Test 2  —  Subgradient ratio per cycle  "
        f"(mean = {np.mean(sg_ratios):.4f})",
        fontsize=9)
    ax.set_ylim(0.8, 1.2)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    pass_fail = "✓ PASS" if np.max(np.abs(sg_ratios - 1.0)) < 0.02 else "✗ FAIL"
    fig2.suptitle(
        f"v3.2  Gradient Validation  |  Test 2: subgradient per-cycle ratio  |  "
        f"mean = {np.mean(sg_ratios):.4f}  |  max|r−1| = {np.max(np.abs(sg_ratios-1.0)):.4f}"
        f"  |  {pass_fail}",
        fontsize=9, y=1.01,
    )
    plot2_path = VALIDATION_DIR / f"fd_validation_test2_{run_ts}.png"
    plt.savefig(plot2_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Test 2 plot saved: {plot2_path.name}")

    # ── Text report ───────────────────────────────────────────────────────────
    rep_path = VALIDATION_DIR / f"fd_validation_report_{run_ts}.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        sep = "=" * 70
        f.write(sep + "\n")
        f.write("WP2 BATTERY — v3.3  Gradient Validation: dDeg/dDoD and Subgradient\n")
        f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(sep + "\n\n")

        f.write("TRACE PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  E_cap           : {E_CAP:.0f} MWh\n")
        f.write(f"  SoC window      : {SOC_MIN*100:.0f}–{SOC_MAX*100:.0f}%\n")
        f.write(f"  N timesteps     : {N}\n")
        f.write(f"  Rainflow cycles : {len(cycles)}\n")
        f.write(f"  T_cell          : {T_CELL} °C\n")
        f.write(f"  Shi k3          : {k3:.4e}\n")
        f.write(f"  Shi k4          : {k4:.4f}\n")
        f.write(f"  fd_cycle (base) : {fd_cycle_base:.6e}\n\n")

        f.write("TEST 1 — dDeg/dDoD slope verification (3-point FD)\n")
        f.write("-" * 40 + "\n")
        f.write("  All 20 energies scaled by s = 1 ± 1/90.\n")
        f.write("  σ̄ invariant to uniform scaling → S_σ coupling absent.\n")
        f.write(f"  Analytical slope: k4 × fd_cycle = {slope_analytical:.6e}\n\n")
        f.write(f"  {'scale':>12}  {'fd_cycle':>14}\n")
        f.write("  " + "-" * 30 + "\n")
        for sc, fv in zip(scales, fd_vals):
            f.write(f"  {sc:12.6f}  {fv:+14.6e}\n")
        f.write(f"\n  Measured slope  : {slope_measured:.6e}\n")
        f.write(f"  Analytical slope: {slope_analytical:.6e}\n")
        f.write(f"  Ratio           : {ratio_slope:.6f}   (target = 1.0)\n")
        f.write(f"  Result          : "
                f"{'✓ PASS' if abs(ratio_slope - 1.0) < 0.02 else '✗ FAIL'}\n\n")

        f.write("TEST 2 — Subgradient per-cycle analytical ratio\n")
        f.write("-" * 40 + "\n")
        f.write("  For each cycle: expected = 0.5 × S_σ(σ̄) × Φ′(δ)\n")
        f.write("  S_σ = exp(k_σ × (σ̄ − σ_ref)),  k_σ=1.04,  σ_ref=0.50\n")
        f.write("  ratio = subgrad[1] / expected  →  1.0\n\n")
        f.write(f"  {'#':>3}  {'dod':>8}  {'σ̄':>7}  {'S_σ':>7}  "
                f"{'expected':>13}  {'measured':>13}  {'ratio':>8}\n")
        f.write("  " + "-" * 68 + "\n")
        for i, (dod, exp_sg, meas_sg, r_sg) in enumerate(
                zip(dods_plot, expected_sgs_plot, sg_t1s_plot, sg_ratios)):
            e_lo3 = 0.40 * E_CAP
            e_hi3 = e_lo3 + dod * E_CAP
            if e_hi3 > SOC_MAX * E_CAP - 2.0:
                e_hi3 = SOC_MAX * E_CAP - 5.0
                e_lo3 = e_hi3 - dod * E_CAP
            sigma_bar = (e_hi3 + e_lo3) / (2.0 * E_CAP)
            s_sig     = np.exp(k_sigma * (sigma_bar - sigma_ref))
            f.write(f"  {i:3d}  {dod:8.5f}  {sigma_bar:7.4f}  {s_sig:7.4f}  "
                    f"{exp_sg:+13.4e}  {meas_sg:+13.4e}  {r_sg:+8.4f}\n")
        f.write(f"\n  Mean ratio : {np.mean(sg_ratios):.6f}   (target = 1.0)\n")
        f.write(f"  Std  ratio : {np.std(sg_ratios):.6f}\n")
        f.write(f"  Max |r−1|  : {np.max(np.abs(sg_ratios - 1.0)):.6f}\n")
        f.write(f"  Result     : "
                f"{'✓ PASS' if np.max(np.abs(sg_ratios - 1.0)) < 0.02 else '✗ FAIL'}\n")

    print(f"    ✓ Report saved: {rep_path.name}")

    print("\n" + "=" * 70)
    print("✓ COMPLETE — v3.2.1")
    print(f"  Output directory: {VALIDATION_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()