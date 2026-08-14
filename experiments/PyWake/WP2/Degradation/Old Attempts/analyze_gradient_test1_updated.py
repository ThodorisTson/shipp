r"""
analyze_gradient_test1_updated.py
=================================
Aggregate-slope verification of the degradation subgradient (Test 1), reading
the year-1 dispatch from the most-recent per-priceset multiyear npy.

WHAT THIS VERIFIES
------------------
Scaling the year-1 SoC trajectory by (1 + eps) scales every cycle depth by
(1 + eps); since the Shi cycle term is Phi = k3 * delta^k4,
    d(fd_cycle)/d(eps) = k4 * fd_cycle,
and the calendar term is scale-independent in cycling, so the finite-difference
slope of total fd against the scale factor should equal
    k4 * fd_cycle + fd_calendar.
Ratio FD / prediction near 1.0 is the pass condition. Richardson extrapolation
over three perturbation sizes confirms O(eps^2) convergence.

DISPATCH SOURCE (changed)
-------------------------
The year-1 SoC trajectory is read from the most-recent multiyear summary npy for
each priceset (annual_soc[0], in MWh), NOT re-solved. This validates the exact
dispatch the v56 run produced and removes the SHIPP / PyWake / Gurobi solve.
The npy is located in RESULTS_DIR by the YYYYMMDD_HHMMSS timestamp in its name.

POWER-TRACE CAVEAT
------------------
The multiyear npy stores SoC (annual_soc) but not the power trace. The
degradation call takes a storage_p argument, so it is reconstructed from the SoC
differences (power is the rate of energy change). A consistency check recomputes
year-1 fd at scale 1.0 and compares it to the npy's stored annual_fd; a warning
is printed if they diverge, which would indicate the reconstruction matters.

OUTPUTS (per priceset)
----------------------
  Gradient_Verification/test1_updated_<ts>/subgrad_validation_yr1_<priceset>.{pdf,png}
  Gradient_Verification/test1_updated_<ts>/richardson_convergence_yr1_<priceset>.{pdf,png}
  Gradient_Verification/test1_updated_<ts>/verification_summary_<ts>.csv

Reproducible from VS Code on Windows. Needs the degradation modules and
wp2_common (for the battery YAML); no SHIPP / PyWake solve.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wp2_common import load_yaml, load_battery

from degradation_xu import ft_calendar, sei_capacity_loss, XU_LMO
from degradation_shi import analyze_degradation_shi, phi_shi_cycle, ShiModelParams
from degradation_subgradient import fit_shi_polynomial

# Mean-SoC stress coefficients. The Shi cycle term and the Xu calendar term both
# carry S_sigma(sigma) = exp(k_sigma*(sigma - sigma_ref)); these are the single
# source of those two constants, so the complete-slope terms below use exactly
# the values the degradation model uses.
K_SIGMA   = float(XU_LMO.k_sigma)     # 1.04
SIGMA_REF = float(XU_LMO.sigma_ref)   # 0.50

# -- Thesis style ------------------------------------------------------------
import sys as _sys
for _d in Path(__file__).resolve().parents:
    if (_d / "thesis_style.py").exists():
        _sys.path.insert(0, str(_d))
        break
from thesis_style import (apply_thesis_style, figsize, TUDELFT,
                          FS_BASE, FS_LABEL, FS_LEGEND, FS_ANNOT)

PALETTE = apply_thesis_style(palette="brand", usetex=False)

# ============================================================================
# CONFIG
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
HPP_YAML   = SCRIPT_DIR / "WP2_HPP.yaml"          # battery params only (no solve)

RESULTS_DIR    = SCRIPT_DIR / "Results" / "RTE Tests"
NPY_TAG        = "rte910"

VALIDATION_DIR = SCRIPT_DIR / "Gradient_Verification"
VALIDATION_DIR.mkdir(exist_ok=True)

PRICESETS      = ["dk2019", "dk2022"]
dt             = 1.0                               # timestep [h]
eol_thresholds = [0.80, 0.70, 0.60]

# Finite-difference perturbation (fractional scale of the SoC trajectory)
ECAP_EPS       = 30.0                              # MWh; eps_frac = ECAP_EPS / e_cap
RICH_EPS_FRACS = [0.10, 0.05, 0.025]

# Consistency-check tolerance: recomputed year-1 fd vs npy annual_fd
FD_MATCH_RTOL  = 0.02                              # 2% relative

run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
_TS_RE  = re.compile(r"(\d{8}_\d{6})")


# ============================================================================
# NPY SELECTION + LOADING
# ============================================================================

def find_multiyear_npy(priceset: str) -> Path:
    """Most recent multiyear summary npy for *priceset* in RESULTS_DIR."""
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results folder not found: {RESULTS_DIR}")
    cands = [p for p in RESULTS_DIR.glob("multiyear_*.npy")
             if priceset in p.name and (NPY_TAG in p.name if NPY_TAG else True)]
    if not cands:
        raise FileNotFoundError(
            f"No multiyear_*.npy for priceset '{priceset}' with tag "
            f"'{NPY_TAG}' in {RESULTS_DIR}")

    def _key(p: Path) -> str:
        m = _TS_RE.search(p.name)
        return m.group(1) if m else p.name

    return max(cands, key=_key)


def load_year1_dispatch(npy_path: Path) -> Dict:
    """Pull the year-1 dispatch and reference numbers from a multiyear npy."""
    d = np.load(npy_path, allow_pickle=True).item()

    storage_e = np.asarray(d["annual_soc"][0], dtype=float)   # year-1 SoC [MWh]
    if storage_e.shape[0] not in (8760, 8784):
        raise ValueError(
            f"annual_soc[0] has length {storage_e.shape[0]}, expected ~8760.")

    e_cap = float(d["e_cap_nominal"])
    fd_annual, fd_cycle, fd_cal = (float(x) for x in d["annual_fd"][0])
    dDeg_dDoD = float(d["annual_gradient"][0][0])             # deprecated metric

    return {
        "storage_e":  storage_e,
        "e_cap":      e_cap,
        "npy_fd_annual":   fd_annual,
        "npy_fd_cycle":    fd_cycle,
        "npy_fd_calendar": fd_cal,
        "dDeg_dDoD":  dDeg_dDoD,
    }


def reconstruct_power(storage_e: np.ndarray, dt_hours: float) -> np.ndarray:
    """Net power proxy [MW] from the SoC trace (discharge-positive, matching
    SHIPP's storage_p sign where positive power adds to grid export).

    Power is the rate of energy change. Degradation is driven by the SoC cycle
    depths, not this trace; the consistency check confirms it does not move fd.
    """
    de = np.diff(storage_e, prepend=storage_e[0])
    return -de / dt_hours


# ============================================================================
# DEGRADATION: Shi cycle term + Xu calendar correction (reporting path only)
# ============================================================================

def _shi_with_calendar_correction(storage_p, storage_e, e_cap_eff, bat_params,
                                  shi_fit, T_cell_C, dt_hours, eol_thresholds) -> Dict:
    """Shi cycle degradation with the Xu calendar term added to the reporting
    path only. The gradient path is never touched (dual-Phi separation).
    """
    n_steps         = len(storage_e)
    t_total_seconds = n_steps * dt_hours * 3600.0
    t_total_hours   = t_total_seconds / 3600.0
    e_arr           = np.asarray(storage_e, dtype=float)
    sigma_mean      = float(np.mean(e_arr)) / max(e_cap_eff, 1e-9)

    degr = analyze_degradation_shi(
        storage_p, storage_e, e_cap_eff, bat_params,
        shi_fit=shi_fit, T_cell_C=T_cell_C, dt_hours=dt_hours,
        eol_thresholds=eol_thresholds,
    )

    fd_cal       = ft_calendar(t_total_seconds, sigma_mean, T_cell_C)
    fd_corrected = degr["fd_shi"] + fd_cal
    cal_frac_pct = 100.0 * fd_cal / max(fd_corrected, 1e-30)

    L_corr       = sei_capacity_loss(fd_corrected)
    cap_ret_corr = 1.0 - L_corr
    soh_corr     = cap_ret_corr * 100.0

    fd_per_yr = fd_corrected / max(t_total_hours / 8760.0, 1e-9)
    eol_corr: Dict[float, Optional[float]] = {}
    for thr in eol_thresholds:
        if (1.0 - sei_capacity_loss(0.0)) < thr:
            eol_corr[thr] = 0.0
            continue
        lo, hi = 0.0, 200.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if 1.0 - sei_capacity_loss(fd_per_yr * mid) > thr:
                lo = mid
            else:
                hi = mid
        val = (lo + hi) / 2.0
        eol_corr[thr] = round(val, 2) if val < 190.0 else None

    degr["fd_calendar"]           = float(fd_cal)
    degr["fd"]                    = float(fd_corrected)
    degr["capacity_retention"]    = float(cap_ret_corr)
    degr["capacity_loss"]         = float(L_corr)
    degr["soh"]                   = float(soh_corr)
    degr["capacity_fade_percent"] = float(L_corr * 100.0)
    degr["eol_years"]             = eol_corr
    degr["e_cap_degraded"]        = e_cap_eff * cap_ret_corr
    degr["meta"]["calendar_correction"] = (
        "Xu ft_calendar added to Shi reporting path (dual-Phi: gradient unchanged)"
    )
    degr["meta"]["fd_calendar_note"] = (
        f"fd_cal={fd_cal:.4e}  ({cal_frac_pct:.1f}% of corrected total)  "
        f"sigma_mean={sigma_mean:.3f}"
    )
    return degr


# ============================================================================
# THESIS-STYLED FIGURES (titleless)
# ============================================================================

def _save_fig(fig, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf")           # PDF first (vector)
    fig.savefig(outdir / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  saved {stem}.pdf")


def plot_fd_validation(fd_lo, fd_mid, fd_hi, eps_frac, fd_slope,
                       total_pred, total_complete, ratio_full,
                       outdir: Path, tag: str) -> None:
    pad = eps_frac * 0.4
    x   = np.linspace(1.0 - eps_frac - pad, 1.0 + eps_frac + pad, 80)
    y_full  = fd_mid + total_complete * (x - 1.0)
    y_naive = fd_mid + total_pred     * (x - 1.0)

    fig, ax = plt.subplots(figsize=figsize(0.74, aspect=0.74))
    ax.plot(x, y_full, "-", color=PALETTE["secondary"], lw=2.0, zorder=3)
    ax.plot(x, y_naive, ":", color=PALETTE["neutral"], lw=1.6, alpha=0.8, zorder=2)
    ax.plot([1.0 - eps_frac, 1.0, 1.0 + eps_frac], [fd_lo, fd_mid, fd_hi],
            "o", color=PALETTE["fill_a"], ms=8, zorder=5)

    ax.set_xlabel("SoC scale factor  (1.0 = nominal dispatch)")
    ax.set_ylabel(r"Annual degradation fraction  $f_d$  (-)")
    _save_fig(fig, outdir, f"subgrad_validation_yr1_{tag}_{run_ts}")


def plot_richardson(eps_vals, fd_errors, resid_cyc, resid_cal,
                    outdir: Path, tag: str) -> None:
    fig, ax = plt.subplots(figsize=figsize(0.74, aspect=0.74))
    ax.loglog(eps_vals, fd_errors, "o--", color=PALETTE["fill_a"], lw=1.8, ms=7)
    ax.axhline(resid_cyc, color=PALETTE["primary"], lw=1.6, ls="--")
    ax.axhline(resid_cal, color=PALETTE["secondary"], lw=1.6, ls="--")
    ax.set_xlabel(r"Perturbation size  $\epsilon$  (fraction of nominal dispatch)")
    ax.set_ylabel(r"$|\,$slope $-$ reference$\,|$")
    _save_fig(fig, outdir, f"richardson_convergence_yr1_{tag}_{run_ts}")


# ============================================================================
# PER-PRICESET VERIFICATION
# ============================================================================

def verify_priceset(priceset: str, bat_params: dict, soc_min: float,
                    soc_max: float, T_cell: float, outdir: Path) -> dict:
    npy_path = find_multiyear_npy(priceset)
    print(f"  Multiyear npy : {npy_path.name}")

    disp = load_year1_dispatch(npy_path)
    storage_e = disp["storage_e"]
    e_cap     = disp["e_cap"]
    storage_p = reconstruct_power(storage_e, dt)
    print(f"  Year-1 SoC    : [{storage_e.min():.1f}, {storage_e.max():.1f}] MWh "
          f"| e_cap {e_cap:.0f} MWh | dDeg/dDoD(npy, deprecated) {disp['dDeg_dDoD']:.3e}")

    shi_fit = fit_shi_polynomial(soc_min, soc_max, verbose=True)

    # Unperturbed (scale 1.0) degradation -> prediction terms + consistency check.
    degr_mid = _shi_with_calendar_correction(
        storage_p.tolist(), storage_e.tolist(), e_cap,
        bat_params, shi_fit, T_cell, dt, eol_thresholds,
    )
    fd_mid   = degr_mid["fd"]
    fd_cycle = degr_mid["fd_cycle"]
    fd_cal   = degr_mid["fd_calendar"]

    # Consistency check vs the npy's own stored numbers.
    rel_fd  = abs(fd_mid - disp["npy_fd_annual"]) / max(abs(disp["npy_fd_annual"]), 1e-30)
    rel_cyc = abs(fd_cycle - disp["npy_fd_cycle"]) / max(abs(disp["npy_fd_cycle"]), 1e-30)
    status  = "OK" if rel_fd <= FD_MATCH_RTOL else "WARNING"
    print(f"  Consistency   : recomputed fd {fd_mid:.6f} vs npy {disp['npy_fd_annual']:.6f} "
          f"(rel {rel_fd:.2%}); fd_cycle rel {rel_cyc:.2%}  -> {status}")
    if status == "WARNING":
        print("    [WARNING] recomputed year-1 fd differs from the npy by more than "
              f"{FD_MATCH_RTOL:.0%}. The power reconstruction or a parameter mismatch "
              "may be affecting fd; do not trust the slope until this is resolved.")

    # Finite-difference verification.
    # The total fd is decomposed into its cycle and calendar parts at every
    # scale point so the FD slope can be checked term-by-term against the two
    # prediction pieces (k4*fd_cycle and fd_cal). This uses the model's own
    # fd_cycle / fd_calendar outputs, so no quantity is reconstructed.
    eps_frac = ECAP_EPS / e_cap
    fd_vals  = {"mid": fd_mid}
    fd_cyc_v = {"mid": fd_cycle}
    fd_cal_v = {"mid": fd_cal}
    for tag, scale in [("lo", 1.0 - eps_frac), ("hi", 1.0 + eps_frac)]:
        degr = _shi_with_calendar_correction(
            (storage_p * scale).tolist(), (storage_e * scale).tolist(), e_cap,
            bat_params, shi_fit, T_cell, dt, eol_thresholds,
        )
        fd_vals[tag]  = degr["fd"]
        fd_cyc_v[tag] = degr["fd_cycle"]
        fd_cal_v[tag] = degr["fd_calendar"]

    fd_slope      = (fd_vals["hi"]  - fd_vals["lo"])  / (2.0 * eps_frac)
    cyc_slope     = (fd_cyc_v["hi"] - fd_cyc_v["lo"]) / (2.0 * eps_frac)
    cal_slope     = (fd_cal_v["hi"] - fd_cal_v["lo"]) / (2.0 * eps_frac)
    cycle_pred    = shi_fit.k4 * fd_cycle
    calendar_pred = fd_cal
    total_pred    = cycle_pred + calendar_pred
    ratio_total   = fd_slope  / total_pred    if abs(total_pred)    > 1e-30 else float("nan")
    ratio_cyc     = cyc_slope / cycle_pred    if abs(cycle_pred)    > 1e-30 else float("nan")
    ratio_cal     = cal_slope / calendar_pred if abs(calendar_pred) > 1e-30 else float("nan")

    print(f"  fd(lo,mid,hi) : {fd_vals['lo']:.6f}  {fd_vals['mid']:.6f}  {fd_vals['hi']:.6f}")
    print(f"  FD slope      : {fd_slope:.6e}  | prediction {total_pred:.6e}  "
          f"| ratio {ratio_total:.4f}")
    print(f"    cycle term  : FD {cyc_slope:.6e} | pred k4*fd_cyc {cycle_pred:.6e} "
          f"| ratio {ratio_cyc:.4f}")
    print(f"    calendar    : FD {cal_slope:.6e} | pred fd_cal    {calendar_pred:.6e} "
          f"| ratio {ratio_cal:.4f}")

    # Complete analytical slope (mean-SoC stress coupling included).
    # The prediction above holds each cycle's mean SoC fixed and treats the
    # calendar term as linear in mean SoC. Scaling the SoC trajectory also scales
    # the mean SoC, so the shared stress factor
    # S_sigma(sigma) = exp(k_sigma*(sigma - sigma_ref)) responds. Differentiating
    # the Shi cycle accumulation and the Xu calendar term at unit scale gives the
    # exact slope. The cycle part is rebuilt from the model's own rainflow cycles.
    shi_params = ShiModelParams.from_fit(shi_fit)
    cycles_mid = degr_mid["cycle_depth_distribution"]
    sig_i = np.array([c["soc_mean"] for c in cycles_mid], dtype=float)
    fd_i  = np.array(
        [float(c["count"]) * float(phi_shi_cycle(c["dod"], c["soc_mean"], T_cell, shi_params))
         for c in cycles_mid], dtype=float)
    assert abs(float(fd_i.sum()) - fd_cycle) <= 1e-9 * max(abs(fd_cycle), 1e-30), (
        "per-cycle reconstruction does not reproduce fd_cycle")
    sigma_bar         = float(np.mean(storage_e)) / e_cap
    cycle_complete    = float(np.sum(fd_i * (shi_fit.k4 + K_SIGMA * sig_i)))
    calendar_complete = K_SIGMA * sigma_bar * fd_cal
    total_complete    = cycle_complete + calendar_complete
    ratio_full_cyc = cyc_slope / cycle_complete    if abs(cycle_complete)    > 1e-30 else float("nan")
    ratio_full_cal = cal_slope / calendar_complete if abs(calendar_complete) > 1e-30 else float("nan")
    ratio_full_tot = fd_slope  / total_complete    if abs(total_complete)    > 1e-30 else float("nan")
    print(f"  Complete pred : total {total_complete:.6e} | ratio {ratio_full_tot:.4f}")
    print(f"    cycle term  : complete {cycle_complete:.6e} (ratio {ratio_full_cyc:.4f}) | "
          f"calendar {calendar_complete:.6e} (ratio {ratio_full_cal:.4f})")
    print(f"    mean SoC    : sigma_bar {sigma_bar:.4f}  "
          f"(calendar slope = k_sigma*sigma_bar*fd_cal)")

    plot_fd_validation(fd_vals["lo"], fd_vals["mid"], fd_vals["hi"], eps_frac,
                       fd_slope, total_pred, total_complete, ratio_full_tot,
                       outdir, priceset)

    # Richardson convergence (total slope + per-term decomposition).
    # Each evaluation already returns fd_cycle and fd_calendar, so the cycle and
    # calendar slopes are extrapolated alongside the total at no extra cost. The
    # converged per-term slopes are compared against k4*fd_cycle and fd_cal to
    # attribute the prediction gap to the correct term rather than assuming it.
    rich_tot, rich_cyc, rich_cal = [], [], []
    for eps_f in RICH_EPS_FRACS:
        dlo = _shi_with_calendar_correction(
            (storage_p * (1.0 - eps_f)).tolist(), (storage_e * (1.0 - eps_f)).tolist(),
            e_cap, bat_params, shi_fit, T_cell, dt, eol_thresholds)
        dhi = _shi_with_calendar_correction(
            (storage_p * (1.0 + eps_f)).tolist(), (storage_e * (1.0 + eps_f)).tolist(),
            e_cap, bat_params, shi_fit, T_cell, dt, eol_thresholds)
        rich_tot.append((eps_f, (dhi["fd"]          - dlo["fd"])          / (2.0 * eps_f)))
        rich_cyc.append((eps_f, (dhi["fd_cycle"]    - dlo["fd_cycle"])    / (2.0 * eps_f)))
        rich_cal.append((eps_f, (dhi["fd_calendar"] - dlo["fd_calendar"]) / (2.0 * eps_f)))

    # Richardson R2 (4-point ratio) on each series.
    _r2 = lambda s: (4.0 * s[2][1] - s[1][1]) / 3.0
    _r1 = lambda s: (4.0 * s[1][1] - s[0][1]) / 3.0
    r1, r2         = _r1(rich_tot), _r2(rich_tot)
    r2_cyc, r2_cal = _r2(rich_cyc), _r2(rich_cal)
    eps_vals       = [r[0] for r in rich_tot]
    fd_errors      = [abs(r[1] - r2) for r in rich_tot]
    # Residuals are measured against the complete (stress-coupled) prediction.
    # In the eps -> 0 limit the converged per-term slopes equal the complete
    # terms, so these residuals fall to the extrapolation floor. The gap to the
    # stress-frozen prediction is reported separately as ratio_cycle / ratio_calendar.
    resid_cyc      = abs(r2_cyc - cycle_complete)
    resid_cal      = abs(r2_cal - calendar_complete)
    print(f"  Richardson    : R1 {r1:.6e} | R2 {r2:.6e} | ratio R2/complete {r2/total_complete:.4f}")
    print(f"    converged   : cycle {r2_cyc:.6e} (resid vs complete {resid_cyc:.2e}) | "
          f"calendar {r2_cal:.6e} (resid vs complete {resid_cal:.2e})")

    plot_richardson(eps_vals, fd_errors, resid_cyc, resid_cal, outdir, priceset)

    return {
        "priceset": priceset, "npy": npy_path.name,
        "e_cap_MWh": e_cap, "k3": shi_fit.k3, "k4": shi_fit.k4, "shi_r2": shi_fit.r2,
        "T_cell_C": T_cell, "eps_frac": eps_frac,
        "fd_lo": fd_vals["lo"], "fd_mid": fd_vals["mid"], "fd_hi": fd_vals["hi"],
        "npy_fd_annual": disp["npy_fd_annual"], "fd_consistency_rel": rel_fd,
        "fd_consistency_status": status,
        "fd_slope": fd_slope, "cycle_pred": cycle_pred, "calendar_pred": calendar_pred,
        "total_pred": total_pred, "ratio_FD_pred": ratio_total,
        "cyc_slope": cyc_slope, "cal_slope": cal_slope,
        "ratio_cycle": ratio_cyc, "ratio_calendar": ratio_cal,
        "sigma_bar": sigma_bar,
        "cycle_complete": cycle_complete, "calendar_complete": calendar_complete,
        "total_complete": total_complete,
        "ratio_full_cyc": ratio_full_cyc, "ratio_full_cal": ratio_full_cal,
        "ratio_full_tot": ratio_full_tot,
        "richardson_R1": r1, "richardson_R2": r2,
        "ratio_R2_pred": r2 / total_pred, "ratio_R2_complete": r2 / total_complete,
        "r2_cycle": r2_cyc, "r2_calendar": r2_cal,
        "resid_cycle": resid_cyc, "resid_calendar": resid_cal,
        "dDeg_dDoD_npy_deprecated": disp["dDeg_dDoD"],
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    outdir = VALIDATION_DIR / f"test1_updated_{run_ts}"
    print("=" * 80)
    print("AGGREGATE-SLOPE VERIFICATION (Test 1, updated)  |  reads multiyear npy")
    print("=" * 80)

    # Battery parameters from the YAML (no solve).
    hpp = load_yaml(HPP_YAML)
    bat = load_battery(hpp, verbose=False)
    soc_min = float(bat.get("soc_min", 0.10))
    soc_max = float(bat.get("soc_max", 0.90))
    T_cell  = float(bat.get("temperature_C", 25.0))
    print(f"Battery YAML  : SoC {soc_min*100:.0f}-{soc_max*100:.0f}%  T_cell {T_cell:.0f} C")

    rows = []
    for ps in PRICESETS:
        print("\n" + "#" * 70)
        print(f"# PRICESET: {ps}")
        print("#" * 70)
        try:
            rows.append(verify_priceset(ps, bat, soc_min, soc_max, T_cell, outdir))
        except Exception as e:
            print(f"  [FAILED] {ps}: {type(e).__name__}: {e}")
            rows.append({"priceset": ps, "fd_consistency_status": "FAILED",
                         "error": str(e)})

    # Combined CSV.
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"verification_summary_{run_ts}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("SUMMARY")
    for r in rows:
        if r.get("fd_consistency_status") == "FAILED":
            print(f"  {r['priceset']:8} FAILED  {r.get('error','')}")
        else:
            print(f"  {r['priceset']:8} full-slope ratio {r['ratio_full_tot']:.4f}  "
                  f"(stress-frozen {r['ratio_FD_pred']:.4f})  "
                  f"consistency {r['fd_consistency_status']} ({r['fd_consistency_rel']:.2%})")
    print(f"\nFigures and CSV in: {outdir}")
    print("=" * 80)


if __name__ == "__main__":
    main()