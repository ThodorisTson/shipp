r"""
analyze_gradient_investigation.py  (v56-compatible, multi-priceset)
===================================================================
Standalone diagnostic script for the inner-loop gradient analysis.
Reads the multiyear-trajectory and gradient-timeseries CSVs produced by the
v56 run and produces five thesis-ready figures for Appendix B, FOR EACH of the
configured pricesets (dk2019 and dk2022).

WHAT CHANGED vs the previous version
------------------------------------
1. Column rename handled. The v56 run renamed dDeg_dDoD -> dDeg_dDoD_DEPRECATED.
   load_run() aliases it back to dDeg_dDoD so the figure code is unchanged, and
   prints a warning because the metric is flagged deprecated in the run output.
2. Both pricesets. find_run() selects the most recent multiyear CSV per priceset
   and pairs it with the gradient CSV FROM THE SAME RUN (prefix swap), so a
   dk2019 trajectory can never be paired with a dk2022 gradient.
3. Per-priceset, timestamped outputs. Figures are written to
   <PLOTS_DIR>/investigation_<run_ts>/figB_<name>_<priceset>.pdf so the two
   pricesets do not overwrite each other.

SCOPE NOTE
----------
This is a multi-year analysis: it studies how the annual gradient varies across
the FIRST BATTERY LIFETIME (year 1 up to the first replacement year). A single
year cannot be analysed this way (the cross-year correlations need >1 point).

FILE SELECTION
  Most recent run per priceset is auto-detected from RESULTS_DIR by the
  YYYYMMDD_HHMMSS timestamp embedded in the filename.
"""
from __future__ import annotations
import re
import sys
import datetime
from pathlib import Path

# -- Path guard --------------------------------------------------------------
for _d in Path(__file__).resolve().parents:
    if (_d / "thesis_style.py").exists():
        sys.path.insert(0, str(_d))
        break
else:
    raise FileNotFoundError("thesis_style.py not found in any parent folder")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from thesis_style import (apply_thesis_style, figsize, TUDELFT,
                          FS_BASE, FS_LABEL, FS_ANNOT)

P = apply_thesis_style(palette="brand", usetex=False)

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "Results"
PLOTS_DIR   = SCRIPT_DIR / "Degradation Plots" / "thesis_figs"

# Pricesets to process. Each is matched as a substring of the CSV filename.
PRICESETS = ["dk2019", "dk2022"]

# Regime colour assignment -- consistent with thesis Figs 4.4-4.6
C_HIGH = TUDELFT["darkred"]   # high-alignment years
C_LOW  = TUDELFT["navy"]      # low-alignment years

ALIGNMENT_THRESHOLD = 3000    # boundary between the two regimes

RUN_TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Gradient column names (canonical vs the deprecated v56 alias)
GRAD_COL            = "dDeg_dDoD"
DEPRECATED_GRAD_COL = "dDeg_dDoD_DEPRECATED"

# Columns each CSV must contain after aliasing
REQUIRED_MT = ["year", "replacement_this_year", "e_cap_eff",
               "mean_abs_subgrad", "mean_abs_dual", "mean_dod", GRAD_COL]
REQUIRED_GT = ["year", "subgrad_combined"]

# Per-run output state -- set in main() for each priceset, read by _save()
_TAG: str = ""
_OUTDIR: Path = PLOTS_DIR

_TS_RE = re.compile(r"(\d{8}_\d{6})")

# ============================================================================
# FILE SELECTION AND LOADING
# ============================================================================

def find_run(priceset: str) -> tuple[Path, Path, str]:
    """Most recent multiyear+gradient CSV pair for *priceset*.

    Picks the multiyear_trajectory CSV with the latest embedded
    YYYYMMDD_HHMMSS timestamp, then derives the matching gradient_timeseries
    file by swapping the filename prefix, so the pair is from one run.
    """
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results folder not found: {RESULTS_DIR}")

    cands = [p for p in RESULTS_DIR.glob("multiyear_trajectory_*.csv")
             if priceset in p.name]
    if not cands:
        raise FileNotFoundError(
            f"No multiyear_trajectory CSV for priceset '{priceset}' in {RESULTS_DIR}")

    def _key(p: Path) -> str:
        m = _TS_RE.search(p.name)
        return m.group(1) if m else p.name

    mt_path = max(cands, key=_key)
    gt_path = mt_path.with_name(
        mt_path.name.replace("multiyear_trajectory_", "gradient_timeseries_", 1))
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Matching gradient CSV not found for run:\n"
            f"  multiyear: {mt_path.name}\n"
            f"  expected : {gt_path.name}")

    run_stem = mt_path.stem.replace("multiyear_trajectory_", "", 1)
    return mt_path, gt_path, run_stem


def load_run(mt_path: Path, gt_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the CSV pair, alias the deprecated gradient column, validate schema."""
    mt = pd.read_csv(mt_path)
    gt = pd.read_csv(gt_path)

    # Alias deprecated gradient column so the figure code stays unchanged.
    if GRAD_COL not in mt.columns:
        if DEPRECATED_GRAD_COL in mt.columns:
            mt[GRAD_COL] = mt[DEPRECATED_GRAD_COL]
            print(f"  [warn] '{GRAD_COL}' absent; using '{DEPRECATED_GRAD_COL}'.")
            print(f"         This metric is deprecated in the v56 output. Confirm")
            print(f"         Appendix B still reports dDeg/dDoD; the current sizing")
            print(f"         gradient is in the dNPV_dEcap / dDegCost_dEcap columns.")
        else:
            raise KeyError(
                f"Neither '{GRAD_COL}' nor '{DEPRECATED_GRAD_COL}' in "
                f"{mt_path.name}. Columns: {list(mt.columns)}")

    missing_mt = [c for c in REQUIRED_MT if c not in mt.columns]
    missing_gt = [c for c in REQUIRED_GT if c not in gt.columns]
    if missing_mt:
        raise KeyError(f"{mt_path.name} missing columns: {missing_mt}")
    if missing_gt:
        raise KeyError(f"{gt_path.name} missing columns: {missing_gt}")

    return mt, gt


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare(mt: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Build the analysis dataframe -- first battery lifetime only."""
    repl_years = mt.loc[mt["replacement_this_year"] == True, "year"].tolist()
    cutoff = repl_years[0] if repl_years else mt["year"].max()

    d = mt[mt["year"] <= cutoff].copy()
    d["product_3"] = d["e_cap_eff"] * d["mean_abs_subgrad"] * d["mean_abs_dual"]
    d["alignment"] = d[GRAD_COL] / d["product_3"]
    d["regime"]    = d["alignment"].apply(
        lambda a: "High-alignment" if a >= ALIGNMENT_THRESHOLD else "Low-alignment"
    )
    d["color"] = d["regime"].map({"High-alignment": C_HIGH, "Low-alignment": C_LOW})

    gt_cut = gt[gt["year"] <= cutoff]
    nz = (
        gt_cut.groupby("year")["subgrad_combined"]
        .apply(lambda x: int(np.sum(x != 0)))
        .reset_index(name="n_nonzero")
    )
    return d.merge(nz, on="year", how="left")


# ============================================================================
# CONSOLE DIAGNOSTICS
# ============================================================================

def print_correlations(d: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print(f"PEARSON CORRELATION  (n = {len(d)} years, first battery lifetime)")
    print("=" * 70)
    components = [
        ("e_cap_eff",        "e_cap_eff"),
        ("mean_abs_subgrad", "mean |subgrad|"),
        ("mean_abs_dual",    "mean |dual price|"),
        ("mean_dod",         "mean DoD"),
        ("product_3",        "e_cap x |subgrad| x |dual|"),
        ("alignment",        "alignment factor"),
    ]
    for col, label in components:
        if col not in d.columns:
            continue
        r, p_val = stats.pearsonr(d[col], d[GRAD_COL])
        stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"  {label:<30}  r={r:+.3f}  p={p_val:.4f}  {stars}")
    print("=" * 70 + "\n")

    high = d[d["regime"] == "High-alignment"]
    low  = d[d["regime"] == "Low-alignment"]
    print("REGIME SUMMARY")
    print(f"  High years: {high['year'].tolist()}  (mean alignment = {high['alignment'].mean():.0f})")
    print(f"  Low  years: {low['year'].tolist()}   (mean alignment = {low['alignment'].mean():.0f})")
    if len(low) and low["alignment"].mean() != 0:
        print(f"  Ratio: {high['alignment'].mean()/low['alignment'].mean():.2f}x\n")
    else:
        print("  Ratio: n/a (one regime empty)\n")


# ============================================================================
# SAVE HELPER
# ============================================================================

def _save(fig: plt.Figure, stem: str) -> None:
    _OUTDIR.mkdir(parents=True, exist_ok=True)
    name = f"{stem}_{_TAG}" if _TAG else stem
    fig.savefig(_OUTDIR / f"{name}.pdf")            # PDF first (vector)
    fig.savefig(_OUTDIR / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  done {name}.pdf")


# ============================================================================
# FIG B1 -- dDeg/dDoD per year, regime-coloured
# ============================================================================

def fig1_gradient_per_year(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.55))

    ax.bar(d["year"], d[GRAD_COL] * 1e-6,
           color=d["color"], alpha=0.85, width=0.7)

    mean_g = d[GRAD_COL].mean() * 1e-6
    ax.axhline(mean_g, ls="--", lw=1.1, color=P["neutral"], alpha=0.8)
    ax.text(d["year"].max() + 0.1, mean_g,
            f"Mean {mean_g:.2e}", fontsize=FS_ANNOT,
            color=P["neutral"], va="center")

    handles = [
        Patch(facecolor=C_HIGH, alpha=0.85, label="High-alignment"),
        Patch(facecolor=C_LOW,  alpha=0.85, label="Low-alignment"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=FS_ANNOT)
    ax.set_xlabel("Project year  (-)")
    ax.set_ylabel(r"$\partial\mathrm{Deg}/\partial\mathrm{DoD}$  (EUR / MWh)")
    ax.set_xticks(d["year"])
    ax.yaxis.get_major_formatter().set_useMathText(True)

    _save(fig, "figB_gradient_per_year")

    high = d[d["regime"] == "High-alignment"]
    low  = d[d["regime"] == "Low-alignment"]
    if len(low) and low[GRAD_COL].mean() != 0:
        ratio = high[GRAD_COL].mean() / low[GRAD_COL].mean()
        print(f"  [figB1] mean gradient = {mean_g:.3e}  |  high/low ratio = {ratio:.2f}x")
    else:
        print(f"  [figB1] mean gradient = {mean_g:.3e}  |  high/low ratio = n/a")


# ============================================================================
# FIG B2 -- Three-component normalised decomposition
# ============================================================================

def fig2_component_decomposition(d: pd.DataFrame) -> None:
    d = d.copy()
    d["dot_product"] = d[GRAD_COL] / (-d["e_cap_eff"])

    yr1    = d[d["year"] == d["year"].min()].iloc[0]
    n_dot  = d["dot_product"]      / (yr1[GRAD_COL] / (-yr1["e_cap_eff"]))
    n_ecap = d["e_cap_eff"]        / yr1["e_cap_eff"]
    n_mas  = d["mean_abs_subgrad"] / yr1["mean_abs_subgrad"]

    bar_cols = d["color"].tolist()

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.55))

    g_arr = d[GRAD_COL].values.astype(float)
    g_norm = g_arr / g_arr.mean()
    ax.bar(d["year"], g_norm, width=0.82,
           color=bar_cols, alpha=0.15, zorder=0, align="center")

    ax.plot(d["year"], n_dot,  "o-",  color=TUDELFT["navy"],    lw=1.8,
            markersize=4, zorder=4,
            label=r"$\langle s,\,\lambda \rangle$  (drives oscillation)")
    ax.plot(d["year"], n_ecap, "s--", color=TUDELFT["blue"],    lw=1.1,
            markersize=3.5, alpha=0.85,
            label=r"$\bar{e}_\mathrm{cap}$  (smooth decline)")
    ax.plot(d["year"], n_mas,  "^--", color=TUDELFT["orange"],  lw=1.1,
            markersize=3.5, alpha=0.85,
            label=r"$|\bar{s}|$  (nearly flat)")

    ax.axhline(1.0, color=P["neutral"], lw=0.7, ls=":", alpha=0.5)
    ax.set_xlabel("Project year  (-)")
    ax.set_ylabel("Value / year-1 value  (-)")
    ax.set_xticks(d["year"])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="upper right")

    _save(fig, "figB_component_decomp")


# ============================================================================
# FIG B3 -- Mean |subgradient| per year + active timesteps
# ============================================================================

def fig3_mean_abs_subgrad(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.55))

    ax.bar(d["year"], d["mean_abs_subgrad"],
           color=TUDELFT["blue"], alpha=0.82, width=0.7)
    ax.set_xlabel("Project year  (-)")
    ax.set_ylabel(r"Mean $|\bar{s}|$  (EUR / MWh)")
    ax.set_xticks(d["year"])

    if "n_nonzero" in d.columns:
        axr = ax.twinx()
        axr.plot(d["year"], d["n_nonzero"], "o--",
                 color=TUDELFT["orange"], lw=1.3, markersize=4,
                 label="Active timesteps")
        axr.set_ylabel("Active timesteps  (n)", color=TUDELFT["orange"])
        axr.tick_params(axis="y", colors=TUDELFT["orange"])
        axr.legend(frameon=False, fontsize=FS_ANNOT, loc="lower right")
        axr.set_ylim(bottom=0)

    _save(fig, "figB_mean_subgrad")


# ============================================================================
# FIG B4 -- Pearson r correlation bar chart
# ============================================================================

def fig4_correlation_bar(d: pd.DataFrame) -> None:
    components = [
        ("mean_dod",         r"Mean DoD"),
        ("e_cap_eff",        r"$\bar{e}_\mathrm{cap}$"),
        ("mean_abs_subgrad", r"Mean $|\bar{s}|$"),
        ("mean_abs_dual",    r"Mean $|\bar{\lambda}|$"),
        ("product_3",        r"$\bar{e}_\mathrm{cap} \times |\bar{s}| \times |\bar{\lambda}|$"),
        ("alignment",        "Alignment factor  (primary)"),
    ]

    labels, rvals, pvals, bar_cols = [], [], [], []
    for col, lbl in components:
        if col not in d.columns:
            continue
        r, p_val = stats.pearsonr(d[col], d[GRAD_COL])
        labels.append(lbl)
        rvals.append(r)
        pvals.append(p_val)
        if col == "alignment":
            bar_cols.append(C_HIGH)
        elif abs(r) > 0.6:
            bar_cols.append(TUDELFT["blue"])
        else:
            bar_cols.append(P["neutral"])

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.55))

    ax.barh(labels, rvals, color=bar_cols, alpha=0.85)
    ax.axvline(0,    color=P["neutral"], lw=0.9, alpha=0.5)
    ax.axvline( 0.5, color=P["neutral"], lw=0.7, alpha=0.35, ls="--")
    ax.axvline(-0.5, color=P["neutral"], lw=0.7, alpha=0.35, ls="--")

    for i, (r, p_val) in enumerate(zip(rvals, pvals)):
        stars  = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        offset = 0.025 if r >= 0 else -0.025
        ha     = "left" if r >= 0 else "right"
        ax.text(r + offset, i, f"{r:+.3f}  {stars}",
                va="center", ha=ha, fontsize=FS_ANNOT)

    ax.set_xlim(-0.65, 1.25)
    ax.set_xlabel(r"Pearson $r$  with  $\partial\mathrm{Deg}/\partial\mathrm{DoD}$")

    ax.legend(handles=[Patch(facecolor=C_HIGH, alpha=0.85,
                             label="Alignment factor")],
              frameon=False, fontsize=FS_ANNOT, loc="lower right")

    print(f"  [figB4] n={len(d)} years")
    _save(fig, "figB_correlation_bar")


# ============================================================================
# FIG B5 -- Alignment factor vs actual gradient scatter
# ============================================================================

def fig5_alignment_scatter(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=figsize(0.72, aspect=0.88))

    for _, row in d.iterrows():
        col = C_HIGH if row["regime"] == "High-alignment" else C_LOW
        ax.scatter(row["alignment"], row[GRAD_COL] * 1e-6,
                   c=col, s=85, zorder=4, edgecolors="white", linewidths=0.5)
        ax.annotate(str(int(row["year"])),
                    (row["alignment"], row[GRAD_COL] * 1e-6),
                    fontsize=FS_ANNOT, color=P["text"],
                    xytext=(5, 3), textcoords="offset points")

    rval, pval = stats.pearsonr(d["alignment"], d[GRAD_COL])
    m, b = np.polyfit(d["alignment"], d[GRAD_COL] * 1e-6, 1)
    xs = np.linspace(d["alignment"].min(), d["alignment"].max(), 60)
    ax.plot(xs, m * xs + b, "--", color=P["neutral"], lw=1.2, alpha=0.7,
            label=f"r = {rval:+.3f}  (p = {pval:.4f})")

    ax.axvline(ALIGNMENT_THRESHOLD, color=P["neutral"], lw=0.8, ls=":",
               alpha=0.5)

    handles = [
        Patch(facecolor=C_HIGH, alpha=0.85, label="High-alignment"),
        Patch(facecolor=C_LOW,  alpha=0.85, label="Low-alignment"),
        Line2D([0],[0], ls="--", color=P["neutral"], alpha=0.7,
               label=f"r = {rval:+.3f}"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=FS_ANNOT)
    ax.set_xlabel("Alignment factor  (-)")
    ax.set_ylabel(r"$\partial\mathrm{Deg}/\partial\mathrm{DoD}$  "
                  r"($\times 10^6$ EUR / MWh)")

    print(f"  [figB5] r = {rval:+.3f}  p = {pval:.4f}")
    _save(fig, "figB_alignment_scatter")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    global _TAG, _OUTDIR
    print("=" * 65)
    print("Gradient investigation -- Appendix B figures (per priceset)")
    print("=" * 65)

    out_root = PLOTS_DIR / f"investigation_{RUN_TS}"
    summary = []

    for ps in PRICESETS:
        print("\n" + "#" * 65)
        print(f"# PRICESET: {ps}")
        print("#" * 65)
        try:
            mt_path, gt_path, stem = find_run(ps)
            print(f"  Multiyear CSV : {mt_path.name}")
            print(f"  Gradient CSV  : {gt_path.name}")

            _TAG    = ps
            _OUTDIR = out_root

            mt, gt = load_run(mt_path, gt_path)
            d = prepare(mt, gt)
            print(f"  Years in analysis : {d['year'].tolist()}")
            print(f"  Replacement years : "
                  f"{mt.loc[mt['replacement_this_year']==True, 'year'].tolist()}")
            print_correlations(d)

            fig1_gradient_per_year(d)
            fig2_component_decomposition(d)
            fig3_mean_abs_subgrad(d)
            fig4_correlation_bar(d)
            fig5_alignment_scatter(d)
            summary.append((ps, "OK", stem, len(d)))
        except Exception as e:
            print(f"  [FAILED] {ps}: {type(e).__name__}: {e}")
            summary.append((ps, "FAILED", str(e), 0))

    print("\n" + "=" * 65)
    print("SUMMARY")
    for ps, status, info, n in summary:
        if status == "OK":
            print(f"  {ps:8} OK      ({n} years, first lifetime)  run={info}")
        else:
            print(f"  {ps:8} FAILED  {info}")
    print(f"\nFigures saved to: {out_root}")
    print("=" * 65)


if __name__ == "__main__":
    main()