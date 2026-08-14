r"""
plot_dk2019_rte910.py
=====================
Regenerates the three DK1-2019 result figures after the DEA efficiency
correction (rte_ac 0.877 -> 0.910, eta_oneway 0.9539), in the exact thesis
style of thesis_plots_results.py.

Figures produced (PDF + PNG, 300 dpi) into ./thesis_figs/:
    fig42_soh_dk1_2019      SoH trajectory + EoL thresholds + replacement
    fig43_fd_dk1_2019       cycle / calendar f_d stacked bars
    fig_soh_loss_dk1_2019   annual SoH loss rate bars

Only the plotting DATA changed; the three functions below are copied from
thesis_plots_results.py so the style is identical. The single intentional
deviation is marked  # >>> DEVIATION  in plot_soh_trajectory (EoL annotations
restricted to the first battery generation; see USE_REPORT_EOL toggle).

Reproducible on Windows / VS Code:
  - depends only on numpy + pandas + matplotlib
  - all paths anchored to this file's folder (Path(__file__).parent)
  - reads the plain-text multiyear_trajectory CSV (no pickle / no .npy needed)

USAGE
  Put this file, thesis_style.py, and the CSV in the same folder, then:
      python plot_dk2019_rte910.py
  Or set CSV_PATH below to an absolute path.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # comment out to view interactively in VS Code
import matplotlib.pyplot as plt
from matplotlib.lines   import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
# PATHS  (Windows-safe: everything relative to this file)
# --------------------------------------------------------------------------- #
HERE = Path(__file__).parent

# thesis_style.py must sit next to this file (or in a parent folder)
for _d in [HERE, *HERE.parents]:
    if (_d / "thesis_style.py").exists():
        sys.path.insert(0, str(_d))
        break
else:
    raise FileNotFoundError("thesis_style.py not found next to this script.")

from thesis_style import apply_thesis_style, figsize, TUDELFT, FS_ANNOT

P = apply_thesis_style(palette="brand", usetex=False)

# --------------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------------- #
CSV_PATH = HERE / ("multiyear_trajectory_20260703_161234_dk2019_150mw_"
                   "300mwh_soc10_90_v56_rtetest_rte910.csv")
OUT_DIR  = HERE / "thesis_figs"
PRICE_LABEL = "DK1 2019"
N_YEARS  = 20

# If True, EoL annotations show the authoritative values from the degradation
# report (finer within-year resolution). If False, they show the coarse
# linear interpolation between annual points (what the master script does).
# Set this to whatever your thesis TEXT cites, so figure and text agree.
USE_REPORT_EOL  = True   # figure now agrees with tab:eol_comparison (6.9 / 12.6)
REPORT_EOL_80   = 6.94   # from degradation_report_...txt  (EoL 80%)
REPORT_EOL_70   = 12.6   # from degradation_report_...txt  (EoL 70%)

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
# DATA LOADING  (CSV -> the same dict the master functions consume)
# --------------------------------------------------------------------------- #
def load_multiyear_from_csv(csv_path: Path) -> dict:
    """Build the multiyear dict used by the plotting functions from the
    plain-text trajectory CSV. Transparent and portable (no pickle)."""
    df = pd.read_csv(csv_path)
    df = df.sort_values("year").reset_index(drop=True)

    soh_trajectory = [(int(r.year), float(r.soh_pct)) for r in df.itertuples()]
    annual_fd = [(float(r.fd_annual), float(r.fd_cycle), float(r.fd_calendar))
                 for r in df.itertuples()]
    repl_years = [int(r.year) for r in df.itertuples()
                  if bool(r.replacement_this_year)]

    return {
        "soh_trajectory":    soh_trajectory,
        "annual_fd":         annual_fd,
        "replacement_years": repl_years,
    }


def _save(fig, stem, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  saved {stem}.pdf / .png")


def _split_battery_segments(years_full, soh_full, replacement_years):
    """Split SoH list into one segment per battery generation."""
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


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4.2 — SoH TRAJECTORY  (copied from thesis_plots_results.py)
# ═══════════════════════════════════════════════════════════════════════════
def plot_soh_trajectory(multiyear, price_label, out_dir,
                        n_years=N_YEARS, eol_thresholds=(0.80, 0.70),
                        fig_code="fig42"):
    traj    = multiyear["soh_trajectory"]
    years   = [r[0] for r in traj]
    soh_pct = [r[1] for r in traj]
    repl    = multiyear["replacement_years"]

    years_full = [0] + years
    soh_full   = [100.0] + soh_pct

    thr_style = {0.80: (C_THR_80, "80 %  [IEC / EV]"),
                 0.70: (C_THR_70, "70 %  [warranty]")}

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.55))

    # Battery generation segments
    segments = _split_battery_segments(years_full, soh_full, repl)
    for sx, sy in segments:
        ax.plot(sx, sy, color=C_SOH, lw=1.8, marker="o", markersize=3.0, zorder=3)
    for i in range(len(segments) - 1):
        x = segments[i][0][-1]
        ax.plot([x, x], [segments[i][1][-1], segments[i + 1][1][0]],
                color=C_SOH, lw=1.8, zorder=4)

    # >>> DEVIATION from master script -------------------------------------- #
    # Restrict EoL-crossing search to the FIRST battery generation (years up to
    # the first replacement). Rationale: under the DEA efficiency the gen-2
    # battery ends at 79.70 %, so it now dips below the 80 % marker near yr 20.
    # The verbatim master loop would print a second "80 % yr ~19.8" label jammed
    # against the right axis. The design-relevant EoL is gen-1's; gen-2 needs no
    # replacement within the horizon. The fact that gen-2 ends below 80 % is
    # real -> mention in text if wanted, but do not clutter the figure.
    repl_yr   = repl[0] if repl else n_years
    gen1_x    = years_full[:repl_yr + 1]          # 0 .. repl_yr
    gen1_soh  = soh_full[:repl_yr + 1]
    # ----------------------------------------------------------------------- #

    for thr in sorted(eol_thresholds, reverse=True):
        col, lbl = thr_style.get(thr, ("#888888", f"{thr*100:.0f}%"))
        ax.axhline(thr * 100, color=col, lw=1.0, ls="--", alpha=0.85)

        for i in range(1, len(gen1_soh)):
            if gen1_soh[i - 1] >= thr * 100 >= gen1_soh[i]:
                f = (gen1_soh[i - 1] - thr * 100) / max(gen1_soh[i - 1] - gen1_soh[i], 1e-9)
                cross = gen1_x[i - 1] + f * (gen1_x[i] - gen1_x[i - 1])

                # optional override: annotate the report's finer EoL value
                if USE_REPORT_EOL:
                    cross = REPORT_EOL_80 if thr == 0.80 else REPORT_EOL_70

                if cross <= n_years:
                    ax.axvline(cross, color=col, lw=0.8, ls=":", alpha=0.4)
                    ha = "right" if cross > n_years * 0.55 else "left"
                    ox = -0.3 if ha == "right" else 0.3
                    ax.annotate(
                        f"{thr*100:.0f} %  yr {cross:.1f}",
                        xy=(cross, thr * 100),
                        xytext=(cross + ox, thr * 100 + 2.5),
                        fontsize=FS_ANNOT, color=col, fontweight="bold", ha=ha,
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.5),
                    )
                break   # one crossing per threshold (gen-1)

    # Replacement lines
    for yr in repl:
        ax.axvline(yr, color=C_REPL, lw=1.2, ls="-.", alpha=0.85, zorder=2)
        ax.text(yr - 0.25, min(soh_pct) - 4, f"Replace\nyr {yr}",
                fontsize=FS_ANNOT, color=C_REPL, ha="right", va="top")

    # Legend
    handles = [Line2D([0], [0], color=C_SOH, lw=1.8, marker="o", markersize=4,
                      label="Simulated SoH")]
    for thr in sorted(eol_thresholds, reverse=True):
        col, lbl = thr_style[thr]
        handles.append(Line2D([0], [0], color=col, lw=1.0, ls="--", label=lbl))
    if repl:
        handles.append(Line2D([0], [0], color=C_REPL, lw=1.2, ls="-.",
                              label="Battery replacement"))
    ax.legend(handles=handles, frameon=False, fontsize=FS_ANNOT, loc="lower left")

    ax.set_xlim(0, n_years)
    ax.set_xticks(range(0, n_years + 1, 2))
    ax.set_ylim(max(min(soh_pct) - 8, 50), 103)
    ax.set_xlabel("Project year  (–)")
    ax.set_ylabel("State of Health  (%)")

    _save(fig, f"{fig_code}_soh_dk1_2019", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4.3 — f_d DECOMPOSITION  (copied verbatim from thesis_plots_results.py)
# ═══════════════════════════════════════════════════════════════════════════
def plot_fd_decomposition(multiyear, price_label, out_dir, n_years=N_YEARS):
    traj   = multiyear["soh_trajectory"]
    years  = [r[0] for r in traj]
    ann_fd = multiyear["annual_fd"]
    repl   = multiyear["replacement_years"]

    fd_t = [t[0] for t in ann_fd]
    fd_c = [t[1] for t in ann_fd]
    fd_k = [t[2] for t in ann_fd]

    mean_cyc_pct = 100.0 * np.mean(fd_c) / max(np.mean(fd_t), 1e-12)
    mean_cal_pct = 100.0 - mean_cyc_pct
    print(f"  [Fig 4.3] Cycle {mean_cyc_pct:.0f}% / Calendar {mean_cal_pct:.0f}% "
          f"-> put in caption")

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.50))

    ax.bar(years, fd_c, width=0.7, color=C_FD_CYCLE, alpha=0.88,
           label=r"Cycle $f_d$  (Shi $\Phi$ accumulation)")
    ax.bar(years, fd_k, width=0.7, color=C_FD_CAL, alpha=0.88,
           bottom=fd_c, label=r"Calendar $f_d$  (Xu $f_{t,\mathrm{cal}}$)")

    for yr in repl:
        ax.axvline(yr + 0.5, color=C_REPL, lw=1.2, ls="-.", alpha=0.8)

    ax.set_xlim(0.5, n_years + 0.5)
    ax.set_xticks(range(2, n_years + 1, 2))
    ax.set_ylim(0, max(fd_t) * 1.12)
    ax.set_xlabel("Project year  (–)")
    ax.set_ylabel(r"Annual $f_d$ contribution  (–)")
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="upper left")

    _save(fig, "fig43_fd_dk1_2019", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# SUPPORT — SoH LOSS RATE BARS  (copied verbatim from thesis_plots_results.py)
# ═══════════════════════════════════════════════════════════════════════════
def plot_soh_loss_rate(multiyear, price_label, out_dir, n_years=N_YEARS):
    traj    = multiyear["soh_trajectory"]
    years   = [r[0] for r in traj]
    soh_pct = [r[1] for r in traj]
    repl    = multiyear["replacement_years"]

    years_full = [0] + years
    soh_full   = [100.0] + soh_pct
    post_repl  = {yr + 1 for yr in repl}

    delta_soh = []
    for i, yr in enumerate(years):
        if yr in post_repl:
            delta_soh.append(100.0 - soh_full[i + 1])
        else:
            delta_soh.append(soh_full[i] - soh_full[i + 1])

    bar_colors = [
        C_FRESH_BAR if yr in post_repl else
        C_EOL_BAR   if yr in set(repl) else
        C_NORMAL_BAR
        for yr in years
    ]
    normal_losses = [d for yr, d in zip(years, delta_soh) if yr not in set(repl)]
    mean_loss = np.mean(normal_losses) if normal_losses else 0
    print(f"  [SoH loss] mean loss rate (incl. break-in yrs) = {mean_loss:.2f} %/yr")

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.50))

    ax.bar(years, delta_soh, width=0.7, color=bar_colors, alpha=0.85)
    ax.axhline(mean_loss, color=P["neutral"], lw=1.1, ls="--",
               label=f"Mean loss rate: {mean_loss:.2f} %/yr")

    for yr in repl:
        ax.axvline(yr + 0.5, color=C_REPL, lw=1.2, ls="-.", alpha=0.8)
        idx = years.index(yr)
        ax.text(yr, delta_soh[idx] + 0.08, f"EoL yr {yr}",
                fontsize=FS_ANNOT, color=C_REPL, ha="center")

    extra = [
        Patch(facecolor=C_FRESH_BAR, alpha=0.85,
              label="Post-replacement yr  (ref = 100% fresh)"),
        Patch(facecolor=C_EOL_BAR,   alpha=0.85, label="EoL year"),
        Patch(facecolor=C_NORMAL_BAR, alpha=0.85, label="Normal operating year"),
    ]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + extra, frameon=False, fontsize=FS_ANNOT,
              loc="upper right")

    ax.set_xlim(0.5, n_years + 0.5)
    ax.set_xticks(range(2, n_years + 1, 2))
    ax.set_ylim(0, max(delta_soh) * 1.35)
    ax.set_xlabel("Project year  (–)")
    ax.set_ylabel("Annual SoH loss  (% / yr)")

    _save(fig, "fig_soh_loss_dk1_2019", out_dir)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    my = load_multiyear_from_csv(CSV_PATH)
    print(f"Loaded {len(my['soh_trajectory'])} years | "
          f"replacement yr {my['replacement_years']}")
    print(f"Writing figures to: {OUT_DIR}")
    print("-" * 60)
    plot_soh_trajectory(my, PRICE_LABEL, OUT_DIR)
    plot_fd_decomposition(my, PRICE_LABEL, OUT_DIR)
    plot_soh_loss_rate(my, PRICE_LABEL, OUT_DIR)
    print("-" * 60)
    print("Done. Copy the printed caption values into the .tex captions.")


if __name__ == "__main__":
    main()
