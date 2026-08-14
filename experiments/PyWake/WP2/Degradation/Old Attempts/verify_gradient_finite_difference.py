"""
verify_gradient_finite_difference.py

Finite-difference verification of the degradation-cost design gradient, in the
style of Fig. 6.7 of Martins & Ning (the classic step-size "checkmark").

What it checks
--------------
Raising E_cap with the dispatch held fixed divides the entire normalised SoC
trajectory by the scale factor, so every rainflow cycle's depth (delta) and mean
(sigma) scale together. This is the uniform-scale perturbation swept below. Its
analytic derivative is

    d f_cyc / ds = sum_i count_i * Phi_i * (k4 + k_sigma * sigma_i)          (complete)

where Phi_i = phi_shi_cycle(delta_i, sigma_i) already carries S_sigma and S_T.
The finite difference converges to this complete derivative, which is the design
gradient the outer loop now uses (with the mean-SoC coupling term included).

An optional reference line (SHOW_REFERENCE_LINE) marks a gradient that omits the
mean-SoC coupling, sum_i count_i * Phi'(delta_i) * S_sigma * delta_i = k4 * f_cyc,
which for this site is ~31% below the complete derivative. It is off by default;
turn it on only to illustrate why the coupling term is needed.

Reproducibility
---------------
Reads the v5.6 RTE-test outputs only (no run-script changes):
  - multiyear_*rte910*.npy  -> annual_soc[0], e_cap_nominal
  - npv_summary_*.csv        -> soc_min, soc_max  (the SoC window; single drift source removed)
Re-fits the Shi polynomial deterministically from the window, so k3/k4 are
recomputed rather than trusted. Runs on Windows/VS Code; uses the real `rainflow`
package (pulled in transitively by degradation_shi).

Paths default to <script>/Results/RTE Tests and <script>/Gradient_Verification,
and can be overridden with the WP2_RESULTS_DIR / WP2_OUT_DIR environment variables.
"""

from __future__ import annotations
import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- locate thesis_style.py (lives in a parent of the working dir) ------------
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
    if (_p / "thesis_style.py").exists():
        sys.path.insert(0, str(_p))
        break
from thesis_style import apply_thesis_style, figsize, FS_ANNOT, FS_LEGEND  # noqa: E402

# degradation_shi.py is co-located with the run scripts (script dir is on sys.path)
from degradation_shi import (                                             # noqa: E402
    fit_shi_polynomial,
    rainflow_cycle_counting,
    compute_fd_shi,
    phi_shi_cycle,
    ShiModelParams,
)

# ------------------------------- configuration --------------------------------
RESULTS_DIR = Path(os.environ.get("WP2_RESULTS_DIR", SCRIPT_DIR / "Results" / "RTE Tests"))
OUT_DIR     = Path(os.environ.get("WP2_OUT_DIR",     SCRIPT_DIR / "Gradient_Verification"))

DIFF_MODES          = ("central", "forward")  # Jenna suggested one is enough; both shown for contrast
H_SWEEP             = np.logspace(-1, -13, 40)  # step sizes, large -> tiny
SHOW_REFERENCE_LINE = False                     # draw the no-coupling reference line (teaching aid only)
SHOW_SLOPE_GUIDES   = True                      # thin O(h) / O(h^2) guides on the truncation branch
T_CELL_C            = 25.0
K_SIGMA             = 1.04                       # k_sigma in S_sigma = exp(k_sigma*(sigma - sigma_ref))
DPI                 = 300

# --------------------------------- helpers ------------------------------------
def discover_cases(results_dir: Path):
    """Find the newest multiyear rte910 .npy per priceset, paired with its npv_summary CSV.

    Prints the resolved folder and the exact files chosen, so you can confirm the
    right run is being read before the sweep begins. Only names that both start
    with 'multiyear_' and contain 'rte910' are considered, so e_cap_fixed.npy and
    storage_e_fixed.npy are ignored.
    """
    print(f"[discover] results folder : {results_dir}")
    if not results_dir.is_dir():
        raise SystemExit(f"[discover] folder does not exist: {results_dir}")

    all_npy = sorted(results_dir.glob("multiyear_*rte910*.npy"))
    if not all_npy:
        raise SystemExit(f"[discover] no 'multiyear_*rte910*.npy' files in {results_dir}")

    # group by priceset; the filename embeds YYYYMMDD_HHMMSS after 'multiyear_',
    # so the alphabetically largest name is the most recent run
    by_label = {}
    for npy in all_npy:
        stem  = npy.name[len("multiyear_"):-len(".npy")]
        label = "dk2019" if "dk2019" in stem else "dk2022" if "dk2022" in stem else stem
        by_label.setdefault(label, []).append(npy)

    cases = []
    for label in sorted(by_label):
        group  = by_label[label]
        chosen = max(group, key=lambda p: p.name)          # newest run for this priceset
        stem   = chosen.name[len("multiyear_"):-len(".npy")]
        csv_hits = list(results_dir.glob(f"npv_summary_{stem}.csv"))
        if not csv_hits:
            print(f"[discover] {label}: SKIP (no npv_summary matching {chosen.name})")
            continue
        print(f"[discover] {label}: {chosen.name}")
        print(f"[discover]         + {csv_hits[0].name}")
        for other in group:
            if other is not chosen:
                print(f"[discover]         (ignoring older {other.name})")
        cases.append((label, chosen, csv_hits[0]))
    return cases


def load_case(npy_path: Path, csv_path: Path):
    d = np.load(npy_path, allow_pickle=True).item()
    soc   = np.asarray(d["annual_soc"][0], dtype=float)
    e_cap = float(d["e_cap_nominal"])
    par   = pd.read_csv(csv_path)
    soc_min = float(par["soc_min"].iloc[0])
    soc_max = float(par["soc_max"].iloc[0])
    fit = fit_shi_polynomial(soc_min=soc_min, soc_max=soc_max, verbose=False)
    p   = ShiModelParams.from_fit(fit)
    return soc, e_cap, p, fit


def f_cyc_of_scale(soc, e_cap, p, s):
    """Full pipeline: re-run rainflow on the scaled trajectory, return f_cyc."""
    return compute_fd_shi(rainflow_cycle_counting(soc * s, e_cap), p, T_C=T_CELL_C)[0]


def analytic_references(soc, e_cap, p):
    """Return (complete derivative, no-coupling reference)."""
    cyc = rainflow_cycle_counting(soc, e_cap)
    cnt = np.array([c["count"]    for c in cyc])
    sig = np.array([c["soc_mean"] for c in cyc])
    phi = np.array([float(phi_shi_cycle(c["dod"], c["soc_mean"], T_CELL_C, p)) for c in cyc])
    complete    = float(np.sum(cnt * phi * (p.k4 + K_SIGMA * sig)))
    no_coupling = float(p.k4 * float((cnt * phi).sum()))
    return complete, no_coupling


def sweep(soc, e_cap, p, reference):
    f0 = f_cyc_of_scale(soc, e_cap, p, 1.0)
    err = {m: [] for m in DIFF_MODES}
    for h in H_SWEEP:
        fp = f_cyc_of_scale(soc, e_cap, p, 1.0 + h)
        if "central" in DIFF_MODES:
            fm = f_cyc_of_scale(soc, e_cap, p, 1.0 - h)
            err["central"].append(abs((fp - fm) / (2 * h) - reference) / abs(reference))
        if "forward" in DIFF_MODES:
            err["forward"].append(abs((fp - f0) / h - reference) / abs(reference))
    return {m: np.array(v) for m, v in err.items()}


# ---------------------------------- plotting ----------------------------------
def make_figure(label, err, complete, no_coupling, pal, out_dir: Path):
    style = {
        "central": dict(color=pal["primary"],   marker="o", ms=3.2, name="central difference"),
        "forward": dict(color=pal["secondary"], marker="s", ms=3.0, name="forward difference"),
    }
    fig, ax = plt.subplots(figsize=figsize(0.74, aspect=0.80))

    for m in DIFF_MODES:
        s = style[m]
        ax.loglog(H_SWEEP, err[m], s["marker"] + "-", color=s["color"],
                  ms=s["ms"], lw=1.2, label=s["name"])
        i = int(np.argmin(err[m]))
        ax.plot(H_SWEEP[i], err[m][i], s["marker"], mfc="none",
                mec=s["color"], ms=9, mew=1.2)  # ring the minimum; values go in the caption

    if SHOW_SLOPE_GUIDES:
        ia    = 2                       # anchor in the truncation regime (h ~ 2e-2)
        hg    = np.array([H_SWEEP[0], H_SWEEP[9]])
        h_lab = H_SWEEP[0] * 0.80       # left-hand anchor (large h) for the labels
        if "central" in DIFF_MODES:
            g2 = err["central"][ia] * 0.5 * (hg / H_SWEEP[ia]) ** 2
            ax.loglog(hg, g2, ":", color=pal["primary"], lw=0.9)
            y2 = err["central"][ia] * 0.5 * (h_lab / H_SWEEP[ia]) ** 2
            ax.text(h_lab, y2 * 0.12, r"$\propto h^{2}$", fontsize=FS_ANNOT,
                    color=pal["primary"], ha="center", va="top")
        if "forward" in DIFF_MODES:
            g1 = err["forward"][ia] * 0.5 * (hg / H_SWEEP[ia]) ** 1
            ax.loglog(hg, g1, ":", color=pal["secondary"], lw=0.9)
            y1 = err["forward"][ia] * 0.5 * (h_lab / H_SWEEP[ia]) ** 1
            ax.text(h_lab, y1 * 0.35, r"$\propto h$", fontsize=FS_ANNOT,
                    color=pal["secondary"], ha="center", va="top")

    if SHOW_REFERENCE_LINE:
        gap = abs(complete - no_coupling) / abs(complete)
        ax.axhline(gap, color=pal["neutral"], lw=1.0, ls="--")
        ax.text(H_SWEEP[1], gap * 1.25,
                f"gradient without mean-SoC coupling ({gap*100:.0f}%)",
                fontsize=FS_ANNOT, color=pal["neutral"], va="bottom", ha="left")

    ax.invert_xaxis()  # Fig. 6.7 orientation: h decreases to the right
    ax.grid(True, which="both", color=pal["grid"], lw=0.5, alpha=0.6)
    ax.set_xlabel(r"finite-difference step size $h$ (fractional SoC scale)")
    ax.set_ylabel(r"relative error $|\,D_{h}-\mathrm{d}f_{\mathrm{cyc}}/\mathrm{d}s\,|\,/\,"
                  r"|\mathrm{d}f_{\mathrm{cyc}}/\mathrm{d}s|$")
    ax.legend(loc="upper right", fontsize=FS_LEGEND, frameon=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"fd_verification_{label}.pdf")
    fig.savefig(out_dir / f"fd_verification_{label}.png", dpi=DPI)
    plt.close(fig)


# ------------------------------------ main ------------------------------------
def main():
    pal = apply_thesis_style(palette="brand", usetex=False)
    cases = discover_cases(RESULTS_DIR)
    if not cases:
        raise SystemExit(f"No rte910 cases found in {RESULTS_DIR}")

    rows = []
    for label, npy_path, csv_path in cases:
        soc, e_cap, p, fit = load_case(npy_path, csv_path)
        complete, no_coupling = analytic_references(soc, e_cap, p)
        err = sweep(soc, e_cap, p, reference=complete)
        make_figure(label, err, complete, no_coupling, pal, OUT_DIR)

        line = {"case": label, "k3": fit.k3, "k4": fit.k4,
                "complete_deriv": complete, "no_coupling_deriv": no_coupling,
                "no_coupling_error_pct": 100 * abs(complete - no_coupling) / abs(complete)}
        for m in DIFF_MODES:
            i = int(np.argmin(err[m]))
            line[f"{m}_min_relerr"] = err[m][i]
            line[f"{m}_h_at_min"]   = H_SWEEP[i]
        rows.append(line)
        print(f"[{label}] k4={fit.k4:.4f}  complete={complete:.6e}  "
              f"no-coupling={no_coupling:.6e}  gap={line['no_coupling_error_pct']:.1f}%  "
              + "  ".join(f"{m}:{err[m].min():.1e}@{H_SWEEP[int(np.argmin(err[m]))]:.0e}"
                          for m in DIFF_MODES))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "fd_verification_summary.csv", index=False)
    print(f"saved figures + summary to {OUT_DIR}")


if __name__ == "__main__":
    main()