"""
run_battery_xu_shi_degradation_v3_2.py
WP2 battery degradation — v3.2  FD validation of dDeg/dDoD subgradient

Validates that the per-timestep subgradient returned by compute_subgradient()
equals d(fd_cycle)/d(delta_E[t]) up to the expected sign and E_cap factor.

Strategy — propagating central-difference FD on a 20-step synthetic SoC trace:

  At each t, shift E[t+1:] by ±ε.
  Because E[s] and E[s+1] both shift by ε for all s ≥ t+1, only
  delta_E[t] = E[t+1] − E[t] changes; all later delta_E values cancel.
  Therefore FD[t] = d(fd_cycle)/d(delta_E[t]).

  compute_subgradient uses a DISCHARGE-POSITIVE sign convention and
  multiplies through by cost × E_cap:

      subgrad[t] = d(cost × E_cap × fd_cycle) / d(power_out[t])
                 = cost × E_cap × d(fd_cycle) / d(power_out[t])
                 = −cost × E_cap × d(fd_cycle) / d(delta_E[t])
                 = −cost × E_cap × FD[t]

  With cost = 1.0, the correct validation ratio is:

      ratio[t] = −FD[t] × E_cap / subgrad[t]  →  1.0

TRACE DESIGN — interior placement to avoid SoC boundary clamping:
  Peak  E[4]  = 215 MWh  →  71.7% SoC  (55 MWh below ceiling 270 MWh)
  Valley E[12] =  85 MWh  →  28.3% SoC  (55 MWh above floor 30 MWh)
  At ε = 0.5 MWh: boundary clearance = 55 MWh >> 0.5 MWh → zero clipping.

No LP solve required.  Runs in < 5 seconds.
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


def _propagating_fd(E: np.ndarray, e_cap: float, shi_fit, eps: float) -> np.ndarray:
    """Per-timestep central-difference propagating FD [1/MWh].

    Shifts E[t+1:] by ±eps; only delta_E[t] changes.
    The trace is designed so no clipping occurs (see E_TRACE comment above).
    Returns array of length N; FD[N-1] = 0 (no future energy to shift).
    """
    lo = SOC_MIN * e_cap
    hi = SOC_MAX * e_cap
    FD = np.zeros(N)
    for t in range(N - 1):
        E_hi = E.copy(); E_lo = E.copy()
        E_hi[t + 1:] = np.clip(E_hi[t + 1:] + eps, lo, hi)
        E_lo[t + 1:] = np.clip(E_lo[t + 1:] - eps, lo, hi)
        fd_hi = _fd_cycle(E_hi, e_cap, shi_fit)
        fd_lo = _fd_cycle(E_lo, e_cap, shi_fit)
        FD[t] = (fd_hi - fd_lo) / (2.0 * eps)
    return FD


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("WP2 BATTERY — v3.2  FD validation of dDeg/dDoD subgradient")
    print("20-step synthetic SoC trace  |  No LP solve  |  < 5 seconds")
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
    print("  FD is tested at t=1 (only E[2]=e_lo shifts → only DoD changes).\n")

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

    # ── Part 2: FD at t=1 vs subgradient at t=1 for each cycle ──────────────
    print(f"\n  Part 2: FD at t=1 vs subgradient at t=1 on 3-element traces")
    print(f"  Trace: [e_lo, e_hi, e_lo]. Shift E[2] by ±ε at t=1.")
    print(f"  Expected: FD[1] ≈ −Phi'(dod)/E_cap,  subgrad[1] ≈ Phi'(dod)")
    print(f"  Target ratio: −FD[1] × E_cap / subgrad[1] → 1.0\n")
    print(f"  {'#':>3}  {'dod':>8}  {'Phi_prime':>13}  {'FD[1]':>13}  "
          f"{'subgrad[1]':>13}  {'ratio':>8}  {'FD/expect':>10}  {'sg/expect':>10}")
    print("  " + "-" * 90)

    for i, c in enumerate(cycles_xu):
        dod = c['dod']
        phi_prime = k3 * k4 * dod ** (k4 - 1.0)

        e_lo3 = 0.40 * E_CAP
        e_hi3 = e_lo3 + dod * E_CAP
        if e_hi3 > SOC_MAX * E_CAP - 2.0:
            e_hi3 = SOC_MAX * E_CAP - 5.0
            e_lo3 = e_hi3 - dod * E_CAP

        # FD at t=1: only E[2]=e_lo shifts → only the valley changes → DoD changes
        fd_hi3 = _fd_cycle(np.array([e_lo3, e_hi3, e_lo3 + EPS_MWH]), E_CAP, shi_fit)
        fd_lo3 = _fd_cycle(np.array([e_lo3, e_hi3, e_lo3 - EPS_MWH]), E_CAP, shi_fit)
        fd_t1  = (fd_hi3 - fd_lo3) / (2.0 * EPS_MWH)

        # subgradient at t=1
        cyc3   = rainflow_cycle_counting([e_lo3, e_hi3, e_lo3], E_CAP)
        sg3    = compute_subgradient(
            storage_e=[e_lo3, e_hi3, e_lo3],
            cycles=cyc3,
            dt_hours=1.0,
            battery_replacement_cost_per_MWh=1.0,
            eff_in=1.0, eff_out=1.0,
            shi_fit=shi_fit,
        )
        sg_t1 = float(np.asarray(sg3["subgrad_combined"])[1])

        ratio = (-fd_t1 * E_CAP / sg_t1) if abs(sg_t1) > 1e-30 else float('nan')
        fd_over_expect  = fd_t1  / (-phi_prime / E_CAP) if abs(phi_prime) > 1e-30 else float('nan')
        sg_over_expect  = sg_t1  /  phi_prime            if abs(phi_prime) > 1e-30 else float('nan')

        print(f"  {i:3d}  {dod:8.5f}  {phi_prime:+13.4e}  {fd_t1:+13.4e}  "
              f"{sg_t1:+13.4e}  {ratio:+8.4f}  {fd_over_expect:+10.4f}  {sg_over_expect:+10.4f}")

    print(f"\n  Interpretation:")
    print(f"    ratio ≈ 1.0        → FD and subgradient agree ✓")
    print(f"    FD/expect ≈ 1.0   → _fd_cycle DoD convention correct")
    print(f"    sg/expect ≈ 1.0   → compute_subgradient DoD convention correct")
    print(f"    If ratio≈1 but both ≠1.0 → consistent internal convention, wrong against Shi (2018)")
    print(f"    If FD/expect ≠ sg/expect → normalization mismatch between the two functions")

    # ── [3/3] Propagating FD and ratio ───────────────────────────────────────
    print(f"\n[3/3] Propagating FD  (ε = {EPS_MWH} MWh, {N-1} pairs)...")

    FD = _propagating_fd(E_TRACE, E_CAP, shi_fit, EPS_MWH)

    # Correct ratio: subgrad[t] = −cost × E_cap × FD[t]  (discharge-positive convention)
    # With cost=1: ratio = −FD[t] × E_cap / subgrad[t]  →  1.0
    active = np.abs(subgrad) > 1e-20
    ratio  = np.where(active, -FD * E_CAP / subgrad, np.nan)
    valid_r = ratio[active]

    print(f"\n    {'t':>3}  {'E_t':>7}  {'dE_t':>7}  "
          f"{'FD_t [1/MWh]':>16}  {'subgrad [1/MWh]':>16}  {'−FD×Ecap/sg':>13}")
    print("    " + "-" * 75)
    for t in range(N):
        dE    = f"{E_TRACE[t+1]-E_TRACE[t]:+6.1f}" if t < N - 1 else "    —"
        r_str = f"{ratio[t]:+13.4f}" if active[t] else "          N/A"
        note  = " ★" if active[t] else ""
        print(f"    {t:3d}  {E_TRACE[t]:7.1f}  {dE}  "
              f"{FD[t]:+16.6e}  {subgrad[t]:+16.6e}  {r_str}{note}")

    print(f"\n    Active timesteps      : {int(active.sum())} / {N}")
    if len(valid_r):
        print(f"    Mean ratio            : {float(np.mean(valid_r)):.6f}   (target = 1.0)")
        print(f"    Std  ratio            : {float(np.std(valid_r)):.6f}")
        print(f"    Max  |ratio − 1|      : {float(np.max(np.abs(valid_r - 1.0))):.6f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating plots and report...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.subplots_adjust(wspace=0.32)

    # Panel 1: SoC trace
    ax = axes[0]
    ax.plot(range(N), E_TRACE, '-o', color='#2c7bb6', lw=1.5, ms=5, label='E_t')
    tp = [i for i in range(N) if (i == 0 or i == N-1 or
          (i > 0 and i < N-1 and
           ((E_TRACE[i] > E_TRACE[i-1]) != (E_TRACE[i+1] > E_TRACE[i]))))]
    ax.scatter(tp, E_TRACE[tp], color='#d7191c', s=90, zorder=5,
               marker='D', label='Turning points')
    ax.axhline(SOC_MAX * E_CAP, color='gray', ls=':', lw=1.0, label='SoC bounds (10%–90%)')
    ax.axhline(SOC_MIN * E_CAP, color='gray', ls=':', lw=1.0)
    ax.set_xlabel("Timestep t", fontsize=10)
    ax.set_ylabel("Stored energy E [MWh]", fontsize=10)
    ax.set_title("Synthetic SoC trace (20 steps)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2: FD vs subgrad scatter
    ax = axes[1]
    sg_a = subgrad[active]; fd_a = FD[active]
    sg_z = subgrad[~active]; fd_z = FD[~active]
    lim  = max(np.abs(np.concatenate([sg_a, fd_a])).max() * 1.1, 1e-20)
    ax.scatter(sg_a, fd_a, color='#2c3e50', s=60, zorder=5, label='Active timesteps')
    ax.scatter(sg_z, fd_z, color='gray',    s=40, zorder=4, label='Zero (expected)')
    ax.plot([-lim, lim], [lim/E_CAP, -lim/E_CAP],
            '--', color='#d7191c', lw=1.5, label='slope = −1/E_cap  (ratio=1)')
    ax.set_xlabel("subgrad_combined[t]  [1/MWh]", fontsize=10)
    ax.set_ylabel("FD_t  [1/MWh]", fontsize=10)
    ax.set_title("Part A: FD_t vs subgrad_combined[t]", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 3: ratio per timestep
    ax = axes[2]
    t_active = np.where(active)[0]
    ax.scatter(t_active, ratio[active], color='#2c3e50', s=50, zorder=5,
               label=f'ratio (mean={float(np.mean(valid_r)):.4f})')
    ax.axhline(+1.0, color='#d7191c', lw=1.5, ls='--', label='target = 1.0')
    ax.set_xlabel("Timestep t", fontsize=10)
    ax.set_ylabel("−FD_t × E_cap / subgrad_combined[t]  [—]", fontsize=10)
    ax.set_title("Per-timestep ratio  (target = 1.0)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    mean_r = float(np.mean(valid_r)) if len(valid_r) else float("nan")
    max_e  = float(np.max(np.abs(valid_r - 1.0))) if len(valid_r) else float("nan")
    fig.suptitle(
        f"Part A  —  Propagating FD vs Shi subgradient  |  ε = {EPS_MWH} MWh  |  "
        f"active = {int(active.sum())}/{N}  |  "
        f"mean ratio = {mean_r:.4f}  |  max |r−1| = {max_e:.4f}  (target ≈ 1.0)",
        fontsize=9, y=1.01,
    )

    plot_path = VALIDATION_DIR / f"fd_validation_partA_{run_ts}.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Plot saved: {plot_path.name}")

    # ── Text report ───────────────────────────────────────────────────────────
    rep_path = VALIDATION_DIR / f"fd_validation_report_{run_ts}.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        sep = "=" * 70
        f.write(sep + "\n")
        f.write("WP2 BATTERY — v3.2  FD Validation of dDeg/dDoD Subgradient\n")
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

        f.write("MATHEMATICAL BASIS\n")
        f.write("-" * 40 + "\n")
        f.write("  Propagating FD: shift E[t+1:] by ±ε.\n")
        f.write("  For s >= t+1: both E[s] and E[s+1] shift by ε → delta_E[s] unchanged.\n")
        f.write("  Only delta_E[t] = E[t+1]−E[t] changes by ε.\n")
        f.write("  → FD_t = d(fd_cycle)/d(delta_E[t]).\n\n")
        f.write("  compute_subgradient uses discharge-positive convention:\n")
        f.write("    subgrad[t] = d(cost × E_cap × fd) / d(power_out[t])\n")
        f.write("               = −cost × E_cap × d(fd) / d(delta_E[t])\n")
        f.write("               = −cost × E_cap × FD[t]\n\n")
        f.write("  With cost=1.0, E_cap=300: FD[t] = −subgrad[t] / 300.\n")
        f.write("  Validation ratio: −FD[t] × E_cap / subgrad[t]  →  1.0.\n\n")

        f.write("PART A — Per-timestep propagating FD\n")
        f.write(f"  ε = {EPS_MWH} MWh  |  eff_in = eff_out = 1.0  |  cost = 1.0\n")
        f.write(f"  ratio = −FD[t] × E_cap / subgrad[t]  (target = 1.0)\n\n")
        hdr = (f"  {'t':>3}  {'E_t':>7}  {'dE_t':>7}  "
               f"{'FD_t [1/MWh]':>16}  {'subgrad [1/MWh]':>16}  {'−FD×Ecap/sg':>13}  note")
        f.write(hdr + "\n")
        f.write("  " + "-" * 80 + "\n")
        for t in range(N):
            dE    = f"{E_TRACE[t+1]-E_TRACE[t]:+6.1f}" if t < N - 1 else "    —"
            r_str = f"{ratio[t]:+13.4f}" if active[t] else "          N/A"
            note  = " ★ active" if active[t] else ""
            f.write(f"  {t:3d}  {E_TRACE[t]:7.1f}  {dE}  "
                    f"{FD[t]:+16.6e}  {subgrad[t]:+16.6e}  {r_str}{note}\n")
        f.write("\n")
        if len(valid_r):
            f.write(f"  Active timesteps         : {int(active.sum())} / {N}\n")
            f.write(f"  Mean ratio               : {float(np.mean(valid_r)):.6f}"
                    f"   (target = 1.000000)\n")
            f.write(f"  Std  ratio               : {float(np.std(valid_r)):.6f}\n")
            f.write(f"  Max  |ratio − 1|         : "
                    f"{float(np.max(np.abs(valid_r - 1.0))):.6f}\n")
        else:
            f.write("  No active timesteps found — check subgradient output.\n")

    print(f"    ✓ Report saved: {rep_path.name}")

    print("\n" + "=" * 70)
    print("✓ COMPLETE — v3.2")
    print(f"  Output directory: {VALIDATION_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()