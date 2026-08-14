r"""
plot_soh_figs_dk.py
===================
Regenerates the three SoH result figures for EITHER price year, in the thesis
style of thesis_plots_results.py. Replaces the two per-year scripts so figure
style and EoL logic live in one place.

Set YEAR_TAG to "dk2019" or "dk2022" and run. Figures written to ./thesis_figs/:
    dk2019 -> fig42_soh_dk1_2019 , fig43_fd_dk1_2019 , fig_soh_loss_dk1_2019
    dk2022 -> fig41_soh_dk1_2022 , fig43_fd_dk1_2022 , fig_soh_loss_dk1_2022

EoL BASIS (important)
  The EoL threshold crossings are read off the MULTI-YEAR TRAJECTORY (the same
  curve that is plotted and that drives the replacement decision). This keeps
  the figure, the table, the replacement year, and the economics on one basis.
  The degradation report's EoL values use a constant year-1 f_d rate and run
  ~0.3 yr later; they are printed at runtime for comparison but are NOT the
  default annotation source. Set USE_REPORT_EOL = True to switch (not advised:
  for 2022 the report 70% value lands after the year-12 replacement jump).

Reproducible on Windows / VS Code:
  - depends only on numpy + pandas + matplotlib
  - all paths anchored to this file's folder (Path(__file__).parent)
  - reads the plain-text trajectory CSV (no pickle / no .npy)
"""
from __future__ import annotations
import sys, glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # comment out to view interactively in VS Code
import matplotlib.pyplot as plt
from matplotlib.lines   import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------------- #
YEAR_TAG       = "dk2022"       # "dk2019" or "dk2022"
USE_REPORT_EOL = False          # False = trajectory basis (recommended)
SHOW_GEN2_EOL  = False          # True = also annotate 2nd-battery 80% crossing
N_YEARS        = 20

HERE    = Path(__file__).parent
OUT_DIR = HERE / "thesis_figs"

# thesis_style.py must sit next to this file (or a parent folder)
for _d in [HERE, *HERE.parents]:
    if (_d / "thesis_style.py").exists():
        sys.path.insert(0, str(_d))
        break
else:
    raise FileNotFoundError("thesis_style.py not found next to this script.")
from thesis_style import apply_thesis_style, figsize, TUDELFT, FS_ANNOT
P = apply_thesis_style(palette="brand", usetex=False)

# per-year figure labels
_YEAR   = YEAR_TAG[-4:]
_LABEL  = f"DK1 {_YEAR}"
_SOHFIG = "fig41" if YEAR_TAG == "dk2022" else "fig42"

# ── Colour conventions (identical to thesis_plots_results.py) ──────────────
C_SOH        = TUDELFT["navy"]
C_THR_80     = TUDELFT["darkred"]
C_THR_70     = TUDELFT["orange"]
C_REPL       = TUDELFT["red"]
C_FD_CYCLE   = TUDELFT["blue"]
C_FD_CAL     = TUDELFT["orange"]
C_NORMAL_BAR = TUDELFT["navy"]
C_FRESH_BAR  = TUDELFT["dgreen"]
C_EOL_BAR    = TUDELFT["red"]


# --------------------------------------------------------------------------- #
# DATA LOADING
# --------------------------------------------------------------------------- #
def _one(pattern: str) -> Path:
    hits = sorted(glob.glob(str(HERE / pattern)))
    if not hits:
        raise FileNotFoundError(f"no file matching {pattern} in {HERE}")
    return Path(hits[-1])


def load_run(tag: str):
    df = pd.read_csv(_one(f"multiyear_trajectory_*{tag}*.csv")).sort_values("year")
    df = df.reset_index(drop=True)
    deg = pd.read_csv(_one(f"battery_degradation_results_*{tag}*.csv")).iloc[0]

    multiyear = {
        "soh_trajectory":    [(int(r.year), float(r.soh_pct)) for r in df.itertuples()],
        "annual_fd":         [(float(r.fd_annual), float(r.fd_cycle), float(r.fd_calendar))
                              for r in df.itertuples()],
        "replacement_years": [int(r.year) for r in df.itertuples()
                              if bool(r.replacement_this_year)],
    }
    report_eol = {0.80: float(deg["eol_80_yr"]), 0.70: float(deg["eol_70_yr"])}
    return multiyear, report_eol


def _save(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  saved {stem}.pdf / .png")


def _split_segments(years_full, soh_full, replacement_years):
    repl_set = set(replacement_years)
    segments, cur_x, cur_y = [], [], []
    for yr, soh in zip(years_full, soh_full):
        cur_x.append(yr); cur_y.append(soh)
        if yr in repl_set:
            segments.append((list(cur_x), list(cur_y)))
            cur_x, cur_y = [yr], [100.0]
    if cur_x:
        segments.append((cur_x, cur_y))
    return segments


def _seg_crossings(seg_x, seg_soh, thr_pct):
    """All downward crossings of thr_pct within one battery-generation segment."""
    out = []
    for i in range(1, len(seg_soh)):
        if seg_soh[i - 1] >= thr_pct >= seg_soh[i]:
            f = (seg_soh[i - 1] - thr_pct) / max(seg_soh[i - 1] - seg_soh[i], 1e-9)
            out.append(seg_x[i - 1] + f * (seg_x[i] - seg_x[i - 1]))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# SoH TRAJECTORY  (fig41 / fig42)
# ═══════════════════════════════════════════════════════════════════════════
def plot_soh_trajectory(multiyear, report_eol):
    traj    = multiyear["soh_trajectory"]
    years   = [r[0] for r in traj]
    soh_pct = [r[1] for r in traj]
    repl    = multiyear["replacement_years"]

    years_full = [0] + years
    soh_full   = [100.0] + soh_pct
    thr_style  = {0.80: (C_THR_80, "80 %  [IEC / EV]"),
                  0.70: (C_THR_70, "70 %  [warranty]")}

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.55))

    segments = _split_segments(years_full, soh_full, repl)
    for sx, sy in segments:
        ax.plot(sx, sy, color=C_SOH, lw=1.8, marker="o", markersize=3.0, zorder=3)
    for i in range(len(segments) - 1):
        x = segments[i][0][-1]
        ax.plot([x, x], [segments[i][1][-1], segments[i + 1][1][0]],
                color=C_SOH, lw=1.8, zorder=4)

    for thr in sorted(thr_style, reverse=True):
        col, _ = thr_style[thr]
        ax.axhline(thr * 100, color=col, lw=1.0, ls="--", alpha=0.85)

        for gi, (sx, sy) in enumerate(segments):
            if gi > 0 and not SHOW_GEN2_EOL:
                continue                       # gen-1 only unless toggled on
            crossings = _seg_crossings(sx, sy, thr * 100)
            # gen-1 override to the report's constant-rate EoL, if requested
            if gi == 0 and USE_REPORT_EOL and crossings:
                crossings = [report_eol[thr]]
            for cross in crossings:
                if cross > N_YEARS:
                    continue
                ax.axvline(cross, color=col, lw=0.8, ls=":", alpha=0.4)
                ha = "right" if cross > N_YEARS * 0.55 else "left"
                ox = -0.3 if ha == "right" else 0.3
                ax.annotate(
                    f"{thr*100:.0f} %  yr {cross:.1f}",
                    xy=(cross, thr * 100),
                    xytext=(cross + ox, thr * 100 + 2.5),
                    fontsize=FS_ANNOT, color=col, fontweight="bold", ha=ha,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.5),
                )

    for yr in repl:
        ax.axvline(yr, color=C_REPL, lw=1.2, ls="-.", alpha=0.85, zorder=2)
        ax.text(yr - 0.25, min(soh_pct) - 4, f"Replace\nyr {yr}",
                fontsize=FS_ANNOT, color=C_REPL, ha="right", va="top")

    handles = [Line2D([0], [0], color=C_SOH, lw=1.8, marker="o", markersize=4,
                      label="Simulated SoH")]
    for thr in sorted(thr_style, reverse=True):
        col, lbl = thr_style[thr]
        handles.append(Line2D([0], [0], color=col, lw=1.0, ls="--", label=lbl))
    if repl:
        handles.append(Line2D([0], [0], color=C_REPL, lw=1.2, ls="-.",
                              label="Battery replacement"))
    ax.legend(handles=handles, frameon=False, fontsize=FS_ANNOT, loc="lower left")

    ax.set_xlim(0, N_YEARS)
    ax.set_xticks(range(0, N_YEARS + 1, 2))
    ax.set_ylim(max(min(soh_pct) - 8, 50), 103)
    ax.set_xlabel("Project year  (–)")
    ax.set_ylabel("State of Health  (%)")

    _save(fig, f"{_SOHFIG}_soh_dk1_{_YEAR}")


# ═══════════════════════════════════════════════════════════════════════════
# f_d DECOMPOSITION  (fig43)   — copied verbatim from thesis_plots_results.py
# ═══════════════════════════════════════════════════════════════════════════
def plot_fd_decomposition(multiyear):
    traj   = multiyear["soh_trajectory"]
    years  = [r[0] for r in traj]
    ann_fd = multiyear["annual_fd"]
    repl   = multiyear["replacement_years"]

    fd_t = [t[0] for t in ann_fd]
    fd_c = [t[1] for t in ann_fd]
    fd_k = [t[2] for t in ann_fd]

    mean_cyc = 100.0 * np.mean(fd_c) / max(np.mean(fd_t), 1e-12)
    print(f"  [fig43] cycle {mean_cyc:.0f}% / calendar {100-mean_cyc:.0f}% -> caption")

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.50))
    ax.bar(years, fd_c, width=0.7, color=C_FD_CYCLE, alpha=0.88,
           label=r"Cycle $f_d$  (Shi $\Phi$ accumulation)")
    ax.bar(years, fd_k, width=0.7, color=C_FD_CAL, alpha=0.88, bottom=fd_c,
           label=r"Calendar $f_d$  (Xu $f_{t,\mathrm{cal}}$)")
    for yr in repl:
        ax.axvline(yr + 0.5, color=C_REPL, lw=1.2, ls="-.", alpha=0.8)
    ax.set_xlim(0.5, N_YEARS + 0.5)
    ax.set_xticks(range(2, N_YEARS + 1, 2))
    ax.set_ylim(0, max(fd_t) * 1.12)
    ax.set_xlabel("Project year  (–)")
    ax.set_ylabel(r"Annual $f_d$ contribution  (–)")
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="upper left")
    _save(fig, f"fig43_fd_dk1_{_YEAR}")


# ═══════════════════════════════════════════════════════════════════════════
# SoH LOSS RATE  — copied verbatim from thesis_plots_results.py
# ═══════════════════════════════════════════════════════════════════════════
def plot_soh_loss_rate(multiyear):
    traj    = multiyear["soh_trajectory"]
    years   = [r[0] for r in traj]
    soh_pct = [r[1] for r in traj]
    repl    = multiyear["replacement_years"]

    years_full = [0] + years
    soh_full   = [100.0] + soh_pct
    post_repl  = {yr + 1 for yr in repl}

    delta = []
    for i, yr in enumerate(years):
        delta.append(100.0 - soh_full[i + 1] if yr in post_repl
                     else soh_full[i] - soh_full[i + 1])

    bar_colors = [C_FRESH_BAR if yr in post_repl else
                  C_EOL_BAR   if yr in set(repl) else
                  C_NORMAL_BAR for yr in years]
    normal = [d for yr, d in zip(years, delta) if yr not in set(repl)]
    mean_loss = np.mean(normal) if normal else 0
    print(f"  [soh_loss] mean loss rate (incl. break-in yrs) = {mean_loss:.2f} %/yr")

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.50))
    ax.bar(years, delta, width=0.7, color=bar_colors, alpha=0.85)
    ax.axhline(mean_loss, color=P["neutral"], lw=1.1, ls="--",
               label=f"Mean loss rate: {mean_loss:.2f} %/yr")
    for yr in repl:
        ax.axvline(yr + 0.5, color=C_REPL, lw=1.2, ls="-.", alpha=0.8)
        ax.text(yr, delta[years.index(yr)] + 0.08, f"EoL yr {yr}",
                fontsize=FS_ANNOT, color=C_REPL, ha="center")
    extra = [Patch(facecolor=C_FRESH_BAR, alpha=0.85,
                   label="Post-replacement yr  (ref = 100% fresh)"),
             Patch(facecolor=C_EOL_BAR, alpha=0.85, label="EoL year"),
             Patch(facecolor=C_NORMAL_BAR, alpha=0.85, label="Normal operating year")]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + extra, frameon=False, fontsize=FS_ANNOT,
              loc="upper right")
    ax.set_xlim(0.5, N_YEARS + 0.5)
    ax.set_xticks(range(2, N_YEARS + 1, 2))
    ax.set_ylim(0, max(delta) * 1.35)
    ax.set_xlabel("Project year  (–)")
    ax.set_ylabel("Annual SoH loss  (% / yr)")
    _save(fig, f"fig_soh_loss_dk1_{_YEAR}")


# --------------------------------------------------------------------------- #
def main():
    multiyear, report_eol = load_run(YEAR_TAG)
    repl = multiyear["replacement_years"]
    print(f"{_LABEL}: {len(multiyear['soh_trajectory'])} years | replacement yr {repl}")

    # transparency: print both EoL bases so figure vs report is explicit
    segs = _split_segments([0] + [r[0] for r in multiyear["soh_trajectory"]],
                           [100.0] + [r[1] for r in multiyear["soh_trajectory"]], repl)
    for thr in (80.0, 70.0):
        tc = _seg_crossings(segs[0][0], segs[0][1], thr)
        tc = tc[0] if tc else float("nan")
        print(f"  EoL {thr:.0f}%: trajectory {tc:.2f} yr | report "
              f"{report_eol[thr/100]:.2f} yr")
    print(f"  EoL basis in figure: {'REPORT' if USE_REPORT_EOL else 'TRAJECTORY'}")
    print("-" * 60)

    plot_soh_trajectory(multiyear, report_eol)
    plot_fd_decomposition(multiyear)
    plot_soh_loss_rate(multiyear)
    print("-" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
