r"""
thesis_plots_results.py
=======================
Produces 8 individual thesis-ready figures from saved multiyear result dicts.
One function per panel — replaces manual cropping from 2×2 diagnostic grids.

STEP 1 — Add to v5.6 main(), immediately after the multiyear computation (right before the MAKE_PLOT block):

    # ── Save slim multiyear dict for thesis_plots_results.py ──────────────
    _slim = {k: v for k, v in multiyear.items() if k != 'annual_gradient'}
    _slim['annual_gradient'] = [
        tuple(g[:6]) if (g is not None and len(g) >= 6) else g
        for g in multiyear.get('annual_gradient', [])
    ]
    np.save(RESULTS_DIR / f'multiyear_{run_label}.npy', _slim, allow_pickle=True)
    print(f'  ✓ multiyear_{run_label}.npy saved')

    Index [6] (per-timestep subgrad array, ~66 kB/year) is dropped to keep
    files small. All 8 figures use only indices [0]–[5].

STEP 2 — Run v5.6 twice:
    PRICE_CSV = "dk1_prices_2022.csv"  →  multiyear_*dk2022*.npy
    PRICE_CSV = "dk1_prices_2019.csv"  →  multiyear_*dk2019*.npy

STEP 3 — Run this file. Auto-discovers both .npy files in Results/ and
         produces all 8 figures in Results/thesis_figs/.

FIGURE MAP
  Fig 4.1  fig41_soh_dk2022.pdf      SoH trajectory, DK1 2022
  Fig 4.2  fig42_soh_dk2019.pdf      SoH trajectory, DK1 2019
  Fig 4.3  fig43_fd_dk2022.pdf       fd decomposition, DK1 2022
  Fig 4.4  fig44_grad_dk2022.pdf     gradient bars (years 1–repl_yr)
  Fig 4.5  fig45_norm_dk2022.pdf     normalised components (years 1–repl_yr)
  support  fig_fd_total_*.pdf        annual fd total bars (both years)
  support  fig_soh_loss_*.pdf        SoH loss rate bars (both years)
  support  fig_align_scatter_*.pdf   alignment scatter (years 1–repl_yr)

CAPTION VALUES printed at runtime:
  Fig 4.3: cycle % and calendar % (put in caption, not title)
  Fig 4.4: r value (correlation of alignment vs gradient)
  Fig 4.5: r value (dot-product vs gradient alignment)
"""
import sys
from pathlib import Path

# ── Path guard ────────────────────────────────────────────────────────────
for _d in Path(__file__).resolve().parents:
    if (_d / "thesis_style.py").exists():
        sys.path.insert(0, str(_d))
        break
else:
    raise FileNotFoundError("thesis_style.py not found in any parent folder")

import numpy as np
import matplotlib
# matplotlib.use('Agg')   # uncomment if running headless
import matplotlib.pyplot as plt
from matplotlib.lines  import Line2D
from matplotlib.patches import Patch
from thesis_style import (apply_thesis_style, figsize, TUDELFT,
                          FS_BASE, FS_LABEL, FS_ANNOT)

P = apply_thesis_style(palette="brand", usetex=False)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
FORCE_SYNTHETIC = False   # True = test layout without .npy files
ALIGN_THRESHOLD = 3000.0  # high / low alignment regime boundary
N_YEARS         = 20

# ── Colour conventions for this file ────────────────────────────────────
C_SOH         = TUDELFT["navy"]      # SoH trajectory line
C_THR_80      = TUDELFT["darkred"]   # 80% EoL threshold
C_THR_70      = TUDELFT["orange"]    # 70% warranty threshold
C_REPL        = TUDELFT["red"]       # replacement event
C_HIGH_ALIGN  = TUDELFT["darkred"]   # high-alignment regime bar
C_LOW_ALIGN   = TUDELFT["navy"]      # low-alignment regime bar
C_FD_CYCLE    = TUDELFT["blue"]      # cycle fd (Shi Φ accumulation)
C_FD_CAL      = TUDELFT["orange"]    # calendar fd (Xu ft_calendar)
C_FD_TOTAL    = TUDELFT["dgreen"]    # total fd bars
C_NORM_GRAD   = TUDELFT["navy"]      # dDeg/dDoD line (key result)
C_NORM_PROD   = TUDELFT["blue"]      # dot product line
C_NORM_SUB    = TUDELFT["orange"]    # |subgrad| line
C_NORM_ECAP   = "#888888"            # e_cap line (neutral, smooth)
C_NORMAL_BAR  = TUDELFT["navy"]      # SoH loss — normal years
C_FRESH_BAR   = TUDELFT["dgreen"]    # SoH loss — post-replacement year
C_EOL_BAR     = TUDELFT["red"]       # SoH loss — EoL year


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

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


def _unpack_gradient(multiyear):
    """Extract scalar columns from annual_gradient list (indices 0–4)."""
    ag = multiyear.get("annual_gradient", [])
    traj = multiyear["soh_trajectory"]
    years = [r[0] for r in traj]

    def _get(idx):
        return [g[idx] if (g is not None and len(g) > idx) else None for g in ag]

    dDeg   = _get(0)   # dDeg/dDoD   EUR/MWh
    mas    = _get(1)   # mean |subgrad|
    m_dod  = _get(2)   # mean DoD (fraction)
    m_dual = _get(3)   # mean |dual|
    e_cap  = _get(4)   # e_cap_eff   MWh

    # alignment factor = dDeg/dDoD / (e_cap × |subgrad| × |dual|)
    alignment = []
    for d, s, du, ec in zip(dDeg, mas, m_dual, e_cap):
        if all(v is not None for v in (d, s, du, ec)):
            den = ec * s * du
            alignment.append(d / den if abs(den) > 1e-9 else None)
        else:
            alignment.append(None)

    regime = [
        C_HIGH_ALIGN if (al is not None and al >= ALIGN_THRESHOLD) else C_LOW_ALIGN
        for al in alignment
    ]
    return years, dDeg, mas, m_dod, m_dual, e_cap, alignment, regime


def _save(fig, stem, out_dir):
    """Save PDF + PNG with consistent settings."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  ✓ {stem}.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4.1 / 4.2 — SoH TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════════

def plot_soh_trajectory(multiyear, price_label, out_dir,
                        n_years=N_YEARS, eol_thresholds=(0.80, 0.70),
                        fig_code="fig4x"):
    """
    SoH trajectory with EoL threshold lines, replacement events, and annotations.
    Use fig_code='fig41' for DK1 2022, 'fig42' for DK1 2019.

    CAPTION: No changes needed — existing caption already describes the content.
    """
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
        ax.plot(sx, sy, color=C_SOH, lw=1.8, marker="o", markersize=3.0,
                zorder=3)
    # Vertical connector at replacement
    for i in range(len(segments) - 1):
        x = segments[i][0][-1]
        ax.plot([x, x], [segments[i][1][-1], segments[i+1][1][0]],
                color=C_SOH, lw=1.8, zorder=4)

    # EoL threshold lines and annotations
    for thr in sorted(eol_thresholds, reverse=True):
        col, lbl = thr_style.get(thr, ("#888888", f"{thr*100:.0f}%"))
        ax.axhline(thr * 100, color=col, lw=1.0, ls="--", alpha=0.85)

        # EoL crossing interpolation
        for i in range(1, len(soh_full)):
            if soh_full[i-1] >= thr*100 >= soh_full[i]:
                f = (soh_full[i-1] - thr*100) / max(soh_full[i-1]-soh_full[i], 1e-9)
                cross = years_full[i-1] + f * (years_full[i] - years_full[i-1])
                if cross <= n_years:
                    ax.axvline(cross, color=col, lw=0.8, ls=":", alpha=0.4)
                    ha = "right" if cross > n_years * 0.55 else "left"
                    ox = -0.3 if ha == "right" else 0.3
                    ax.annotate(
                        f"{thr*100:.0f} %  yr {cross:.1f}",
                        xy=(cross, thr*100),
                        xytext=(cross + ox, thr*100 + 2.5),
                        fontsize=FS_ANNOT, color=col, fontweight="bold", ha=ha,
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.5),
                    )

    # Replacement lines
    for yr in repl:
        ax.axvline(yr, color=C_REPL, lw=1.2, ls="-.", alpha=0.85, zorder=2)
        ax.text(yr - 0.25, min(soh_pct) - 4, f"Replace\nyr {yr}",
                fontsize=FS_ANNOT, color=C_REPL, ha="right", va="top")

    # Legend
    handles = [
        Line2D([0], [0], color=C_SOH, lw=1.8, marker="o", markersize=4,
               label="Simulated SoH"),
    ]
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

    _save(fig, f"{fig_code}_soh_{price_label.lower().replace(' ', '_')}", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4.3 — fd DECOMPOSITION (stacked bars)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fd_decomposition(multiyear, price_label, out_dir, n_years=N_YEARS):
    """
    Stacked bars: cycle fd (blue, Shi Φ) + calendar fd (orange, Xu calendar).

    CAPTION UPDATE: title carried the cycle/calendar percentages.
    This function prints them — copy the printed values into the caption:
      "Cycle aging contributes approximately XX% and calendar aging YY%
       of total annual fd across both battery generations."
    """
    traj   = multiyear["soh_trajectory"]
    years  = [r[0] for r in traj]
    ann_fd = multiyear["annual_fd"]
    repl   = multiyear["replacement_years"]

    fd_t = [t[0] for t in ann_fd]
    fd_c = [t[1] for t in ann_fd]
    fd_k = [t[2] for t in ann_fd]

    mean_cyc_pct = 100.0 * np.mean(fd_c) / max(np.mean(fd_t), 1e-12)
    mean_cal_pct = 100.0 - mean_cyc_pct
    print(f"  [Fig 4.3] Cycle {mean_cyc_pct:.0f}% / Calendar {mean_cal_pct:.0f}%  "
          f"→ put in caption")

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

    _save(fig, f"fig43_fd_{price_label.lower().replace(' ', '_')}", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# SUPPORT — ANNUAL fd TOTAL BARS
# ═══════════════════════════════════════════════════════════════════════════

def plot_annual_fd_total(multiyear, price_label, out_dir, n_years=N_YEARS):
    """Annual fd total (Shi cycles + Xu calendar). Y-axis zoomed to show trend."""
    traj   = multiyear["soh_trajectory"]
    years  = [r[0] for r in traj]
    ann_fd = multiyear["annual_fd"]
    repl   = multiyear["replacement_years"]
    fd_t   = [t[0] for t in ann_fd]

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.48))

    ax.bar(years, fd_t, width=0.7, color=C_FD_TOTAL, alpha=0.85,
           label=r"Annual $f_d$  (Shi cycles + Xu calendar)")
    for yr in repl:
        ax.axvline(yr + 0.5, color=C_REPL, lw=1.2, ls="-.", alpha=0.8)

    fd_min = min(fd_t) * 0.996
    fd_max = max(fd_t) * 1.008
    ax.set_xlim(0.5, n_years + 0.5)
    ax.set_xticks(range(2, n_years + 1, 2))
    ax.set_ylim(fd_min, fd_max)
    ax.set_xlabel("Project year  (–)")
    ax.set_ylabel(r"Annual $f_d$  (–)")
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="upper left")

    _save(fig, f"fig_fd_total_{price_label.lower().replace(' ', '_')}", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# SUPPORT — SoH LOSS RATE BARS
# ═══════════════════════════════════════════════════════════════════════════

def plot_soh_loss_rate(multiyear, price_label, out_dir, n_years=N_YEARS):
    """Year-over-year SoH loss rate, coloured by year type."""
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

    _save(fig, f"fig_soh_loss_{price_label.lower().replace(' ', '_')}", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4.4 — GRADIENT BARS (first battery generation)
# ═══════════════════════════════════════════════════════════════════════════

def plot_gradient_bars(multiyear, price_label, out_dir, max_year=None):
    """
    dDeg/dDoD per year, coloured by alignment regime (high = dark red, low = navy).
    Clipped to years 1–max_year (default: first replacement year).

    CAPTION UPDATE: add
      "The gradient alternates between high- and low-alignment regimes;
       neither the SoH trajectory nor the mean DoD explains the oscillation."
    """
    years, dDeg, *_, alignment, regime = _unpack_gradient(multiyear)
    repl = multiyear["replacement_years"]
    if max_year is None:
        max_year = repl[0] if repl else max(years)

    mask    = [yr <= max_year for yr in years]
    y_clip  = [yr for yr, m in zip(years, mask) if m]
    g_clip  = [g  for g,  m in zip(dDeg,  mask) if m and g is not None]
    rc_clip = [c  for c,  m in zip(regime, mask) if m]
    al_clip = [a  for a,  m in zip(alignment, mask) if m]

    mean_g = np.mean(g_clip) if g_clip else 0
    print(f"  [Fig 4.4] mean dDeg/dDoD = {mean_g:.3e} EUR/MWh  (years 1–{max_year})")

    # r value: alignment vs dDeg/dDoD
    valid = [(al, g) for al, g in zip(al_clip, g_clip) if al is not None]
    if len(valid) >= 2:
        al_v, g_v = zip(*valid)
        r = float(np.corrcoef(al_v, g_v)[0, 1])
        print(f"  [Fig 4.4] r(alignment, gradient) = {r:+.3f}  → put in caption")

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.50))

    ax.bar(y_clip, g_clip, width=0.7, color=rc_clip, alpha=0.85)
    ax.axhline(mean_g, color=P["neutral"], lw=1.1, ls="--",
               label=f"Mean: {mean_g:.2e} EUR / MWh")
    for yr in repl:
        if yr <= max_year:
            ax.axvline(yr + 0.5, color=C_REPL, lw=1.2, ls="-.", alpha=0.8)

    ax.legend(handles=[
        Patch(facecolor=C_HIGH_ALIGN, alpha=0.85, label="High-alignment"),
        Patch(facecolor=C_LOW_ALIGN,  alpha=0.85, label="Low-alignment"),
        Line2D([0], [0], color=P["neutral"], lw=1.1, ls="--",
               label=f"Mean: {mean_g:.2e}"),
    ], frameon=False, fontsize=FS_ANNOT)

    ax.set_xlim(0.5, max_year + 0.5)
    ax.set_xticks(range(2, max_year + 1, 2))
    ax.set_xlabel("Project year  (–)")
    ax.set_ylabel(r"$\partial \mathrm{Deg} / \partial \mathrm{DoD}$  (EUR / MWh)")

    _save(fig, f"fig44_grad_{price_label.lower().replace(' ', '_')}", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# SUPPORT — ALIGNMENT SCATTER
# ═══════════════════════════════════════════════════════════════════════════

def plot_alignment_scatter(multiyear, price_label, out_dir, max_year=None):
    """Alignment factor vs dDeg/dDoD scatter with r annotation."""
    years, dDeg, *_, alignment, regime = _unpack_gradient(multiyear)
    repl = multiyear["replacement_years"]
    if max_year is None:
        max_year = repl[0] if repl else max(years)

    mask    = [yr <= max_year for yr in years]
    valid   = [(yr, al, g, c)
               for yr, al, g, c, m in zip(years, alignment, dDeg, regime, mask)
               if m and al is not None and g is not None]

    fig, ax = plt.subplots(figsize=figsize(0.7, aspect=0.85))

    if valid:
        v_yr, v_al, v_g, v_c = zip(*valid)
        ax.scatter(v_al, v_g, color=v_c, s=55, zorder=3,
                   edgecolors="white", linewidths=0.5, alpha=0.9)
        for yr, al, g in zip(v_yr, v_al, v_g):
            ax.annotate(str(yr), (al, g), fontsize=FS_ANNOT - 0.5,
                        xytext=(3, 3), textcoords="offset points", alpha=0.8)
        if len(v_al) >= 2:
            m, b = np.polyfit(v_al, v_g, 1)
            xs = np.linspace(min(v_al), max(v_al), 60)
            r  = float(np.corrcoef(v_al, v_g)[0, 1])
            ax.plot(xs, m * np.array(xs) + b, color=P["neutral"],
                    lw=1.2, ls="--", alpha=0.7, label=f"r = {r:+.3f}")
            ax.legend(frameon=False, fontsize=FS_ANNOT)

    ax.axvline(ALIGN_THRESHOLD, color=P["neutral"], lw=0.8, ls=":",
               alpha=0.5, label=f"Regime boundary = {ALIGN_THRESHOLD:.0f}")
    ax.set_xlabel(r"Alignment factor  =  $\partial \mathrm{Deg}/\partial \mathrm{DoD}$"
                  r"  /  ($\bar{e} \cdot |\bar{s}| \cdot |\bar{\lambda}|$)  (–)")
    ax.set_ylabel(r"$\partial \mathrm{Deg} / \partial \mathrm{DoD}$  (EUR / MWh)")

    _save(fig, f"fig_align_{price_label.lower().replace(' ', '_')}", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# SUPPORT — MEAN |SUBGRAD| + DoD
# ═══════════════════════════════════════════════════════════════════════════

def plot_subgrad_dod(multiyear, price_label, out_dir, n_years=N_YEARS):
    """Mean |subgradient| bars (primary) + mean DoD overlay (secondary axis)."""
    years, dDeg, mas, m_dod, *_ = _unpack_gradient(multiyear)
    repl = multiyear["replacement_years"]

    valid_mas  = [(yr, m) for yr, m in zip(years, mas) if m is not None]
    valid_dod  = [(yr, d) for yr, d in zip(years, m_dod) if d is not None]

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.48))

    if valid_mas:
        v_yr, v_m = zip(*valid_mas)
        ax.bar(v_yr, v_m, width=0.7, color=C_FD_CYCLE, alpha=0.82)

    for yr in repl:
        ax.axvline(yr + 0.5, color=C_REPL, lw=1.2, ls="-.", alpha=0.8)

    ax.set_xlim(0.5, n_years + 0.5)
    ax.set_xticks(range(2, n_years + 1, 2))
    ax.set_xlabel("Project year  (–)")
    ax.set_ylabel(r"Mean $|\bar{s}|$  (EUR / MWh)")

    if valid_dod:
        ax2 = ax.twinx()
        v_yr2, v_d = zip(*valid_dod)
        ax2.plot(v_yr2, [d * 100 for d in v_d], "s--",
                 color=C_FD_CAL, lw=1.1, markersize=3.5, alpha=0.85,
                 label="Mean DoD (%)")
        ax2.set_ylabel("Mean DoD  (%)", color=C_FD_CAL)
        ax2.tick_params(axis="y", colors=C_FD_CAL)
        ax2.legend(frameon=False, fontsize=FS_ANNOT, loc="lower right")
        max_dod = max([d * 100 for d in v_d]) if v_d else 50
        ax2.set_ylim(0, max_dod * 2.2)

    _save(fig, f"fig_subgrad_dod_{price_label.lower().replace(' ', '_')}", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4.5 — NORMALISED SCALAR COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_normalised_components(multiyear, price_label, out_dir, max_year=None):
    """
    dDeg/dDoD vs its scalar decomposition, all normalised to mean = 1.
    Clips to years 1–max_year (default: first replacement year).

    CAPTION UPDATE: add
      "The effective capacity declines smoothly and the mean sub-gradient
       magnitude is nearly flat; the dot product ⟨s, λ⟩ oscillates in
       near-perfect fidelity with the gradient, identifying temporal alignment
       as the sole driver (r = +X.XXX — printed at runtime)."
    """
    years, dDeg, mas, m_dod, m_dual, e_cap, alignment, regime = _unpack_gradient(multiyear)
    repl = multiyear["replacement_years"]
    if max_year is None:
        max_year = repl[0] if repl else max(years)

    mask = [yr <= max_year for yr in years]

    def _clip(lst):
        return [v for v, m in zip(lst, mask) if m and v is not None]
    def _norm(lst):
        mu = np.mean(lst)
        return [v / mu for v in lst] if mu != 0 else lst

    y_c    = [yr for yr, m in zip(years, mask) if m]
    g_c    = _clip(dDeg)
    ec_c   = _clip(e_cap)
    mas_c  = _clip(mas)
    dual_c = _clip(m_dual)
    rc_clip = [c for c, m in zip(regime, mask) if m]   # regime colour per year

    if not (g_c and ec_c and mas_c and dual_c):
        print("  [Fig 4.5] insufficient data — skipping")
        return

    product = [ec * s * d for ec, s, d in zip(ec_c, mas_c, dual_c)]

    # Correlation: product vs gradient  (the thesis finding)
    r = float(np.corrcoef(product, g_c)[0, 1]) if len(product) >= 2 else float("nan")
    print(f"  [Fig 4.5] r(dot-product, gradient) = {r:+.3f}  → put in caption")

    ng = _norm(g_c)
    np_ = _norm(product)
    ns = _norm(mas_c)
    ne = _norm(ec_c)

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.55))

    ax.plot(y_c, ng,  "o-",  color=C_NORM_GRAD, lw=1.8, markersize=4,
            zorder=4, label=r"$\partial\mathrm{Deg}/\partial\mathrm{DoD}$  (normalised)")
    ax.plot(y_c, np_, "s--", color=C_NORM_PROD, lw=1.1, markersize=3.5,
            alpha=0.85, label=r"$\langle s,\, \lambda \rangle$  (normalised)")
    ax.plot(y_c, ns,  "^--", color=C_NORM_SUB,  lw=1.1, markersize=3.5,
            alpha=0.80, label=r"$|\bar{s}|$  (normalised)")
    ax.plot(y_c, ne,  "D--", color=C_NORM_ECAP, lw=1.1, markersize=3.5,
            alpha=0.75, label=r"$\bar{e}_\mathrm{cap}$  (normalised)")

    ax.axhline(1.0, color=P["neutral"], lw=0.8, ls=":", alpha=0.5)
    for yr in repl:
        if yr <= max_year:
            ax.axvline(yr + 0.5, color=C_REPL, lw=1.2, ls="-.", alpha=0.8)

    ax.set_xlim(0.5, max_year + 0.5)
    ax.set_xticks(range(2, max_year + 1, 2))
    ax.set_xlabel("Project year  (–)")
    ax.set_ylabel("Value / mean  (normalised to 1.0)")
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="upper right")

    _save(fig, f"fig45_norm_{price_label.lower().replace(' ', '_')}", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4.6 — GRADIENT BARS (Fig 4.4) OVERLAID AS BACKGROUND ON FIG 4.5
# ═══════════════════════════════════════════════════════════════════════════

def plot_gradient_overlay(multiyear, price_label, out_dir, max_year=None):
    """
    Fig 4.6 — Combined figure: Fig 4.4 gradient bars drawn at low opacity as
    background, Fig 4.5 normalised component lines drawn on top.

    The gradient line (navy, solid) traces the tops of the background bars,
    making the oscillation pattern and its independence from the flat/smooth
    components visually immediate.

    CAPTION: Same finding as Figs 4.4 + 4.5 combined:
      Background bars show dDeg/dDoD per year (dark red = high-alignment,
      navy = low-alignment). The solid navy line traces the same quantity
      normalised to the mean; the other components (dot product, |subgrad|,
      e_cap) are flat or smoothly declining, confirming temporal alignment
      as the sole driver of gradient oscillation.
    """
    years, dDeg, mas, m_dod, m_dual, e_cap, alignment, regime = _unpack_gradient(multiyear)
    repl = multiyear["replacement_years"]
    if max_year is None:
        max_year = repl[0] if repl else max(years)

    mask   = [yr <= max_year for yr in years]

    def _clip(lst):
        return [v for v, m in zip(lst, mask) if m and v is not None]
    def _norm(lst):
        mu = np.mean(lst)
        return [v / mu for v in lst] if mu != 0 else lst

    y_c    = [yr for yr, m in zip(years, mask) if m]
    g_c    = _clip(dDeg)
    ec_c   = _clip(e_cap)
    mas_c  = _clip(mas)
    dual_c = _clip(m_dual)
    rc_clip = [c for c, m in zip(regime, mask) if m]

    if not (g_c and ec_c and mas_c and dual_c):
        print("  [Fig 4.6] insufficient data — skipping")
        return

    product = [ec * s * d for ec, s, d in zip(ec_c, mas_c, dual_c)]

    ng  = _norm(g_c)
    np_ = _norm(product)
    ns  = _norm(mas_c)
    ne  = _norm(ec_c)

    r = float(np.corrcoef(product, g_c)[0, 1]) if len(product) >= 2 else float("nan")
    print(f"  [Fig 4.6] r(dot-product, gradient) = {r:+.3f}")

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.58))

    # ── Background: Fig 4.4 bars at reduced opacity ───────────────────────
    # Bars normalised to mean=1 so they share the y-axis with the lines.
    # The gradient line will trace the bar tops exactly — that IS the point.
    ax.bar(y_c, ng, width=0.82,
           color=rc_clip, alpha=0.18, zorder=0, align="center")

    # ── Foreground: normalised component lines ────────────────────────────
    ax.plot(y_c, ng,  "o-",  color=C_NORM_GRAD, lw=1.8, markersize=4,
            zorder=4, label=r"$\partial\mathrm{Deg}/\partial\mathrm{DoD}$  (normalised)")
    ax.plot(y_c, np_, "s--", color=C_NORM_PROD, lw=1.1, markersize=3.5,
            alpha=0.85, label=r"$\langle s,\, \lambda \rangle$  (normalised)")
    ax.plot(y_c, ns,  "^--", color=C_NORM_SUB,  lw=1.1, markersize=3.5,
            alpha=0.80, label=r"$|\bar{s}|$  (normalised)")
    ax.plot(y_c, ne,  "D--", color=C_NORM_ECAP, lw=1.1, markersize=3.5,
            alpha=0.75, label=r"$\bar{e}_\mathrm{cap}$  (normalised)")

    ax.axhline(1.0, color=P["neutral"], lw=0.8, ls=":", alpha=0.5)
    for yr in repl:
        if yr <= max_year:
            ax.axvline(yr + 0.5, color=C_REPL, lw=1.2, ls="-.", alpha=0.8)

    ax.set_xlim(0.5, max_year + 0.5)
    ax.set_ylim(0, max(ng) * 1.15)
    ax.set_xticks(range(2, max_year + 1, 2))
    ax.set_xlabel("Project year  (–)")
    ax.set_ylabel("Value / mean  (normalised to 1.0)")
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="upper right")

    _save(fig, f"fig46_grad_overlay_{price_label.lower().replace(' ', '_')}", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA (layout testing without .npy files)
# ═══════════════════════════════════════════════════════════════════════════

def _make_synthetic(price_year="dk2022"):
    """Approximate values matching 2022 / 2019 DK1 runs from thesis images."""
    repl_yr = 12 if price_year == "dk2022" else 13

    # SoH (years 1 to N_YEARS-1 = 19)
    n1 = repl_yr          # generation 1 length
    n2 = N_YEARS - 1 - n1 # generation 2 length
    # Gen 1: steep first year (~7.7%), then ~2.1%/yr
    soh1 = [100 - 7.7 - max(0, i-1) * 2.25 - max(0, i-4) * 0.07
            for i in range(1, n1 + 1)]
    soh1[-1] = 70.0  # force EoL at repl_yr
    # Gen 2: same pattern
    soh2 = [100 - 7.2 - max(0, i-1) * 2.20
            for i in range(1, n2 + 1)]
    soh_all = soh1 + soh2
    years_all = list(range(1, N_YEARS))

    traj = [(y, s) for y, s in zip(years_all, soh_all)]

    # Annual fd
    rng = np.random.default_rng(42)
    fd_base1 = np.linspace(0.02421, 0.02527, n1) + rng.normal(0, 3e-5, n1)
    fd_base2 = np.linspace(0.02421, 0.02491, n2) + rng.normal(0, 3e-5, n2)
    fd_base  = np.clip(np.concatenate([fd_base1, fd_base2]), 0.0240, 0.0260)
    ann_fd   = [(float(t), float(t * 0.48), float(t * 0.52)) for t in fd_base]

    # annual_gradient — alternating high/low pattern
    H_YEARS = {1, 3, 7, 8, 9, 11, 13, 15, 18}   # high-alignment years
    e_cap_start = 300.0

    e_caps = [e_cap_start]
    for s in soh_all[:-1]:
        e_caps.append(e_cap_start * s / 100.0)
    # After replacement, e_cap resets
    for i, yr in enumerate(years_all):
        if yr == repl_yr + 1:
            e_caps[i] = e_cap_start

    ag = []
    for i, yr in enumerate(years_all):
        ec  = float(e_caps[i])
        mas = 0.70 + rng.normal(0, 0.005)       # nearly flat
        dod = 0.33 + rng.normal(0, 0.003)        # nearly flat
        g   = (2.95e6 if yr in H_YEARS else 1.30e6) + rng.normal(0, 0.04e6)
        # back-compute dual to give correct alignment
        al_target = 3700.0 if yr in H_YEARS else 1850.0
        dual = g / (ec * mas * al_target)
        ag.append((float(g), float(mas), float(dod), float(dual), float(ec)))

    return {
        "soh_trajectory":   traj,
        "annual_fd":        ann_fd,
        "replacement_years": [repl_yr],
        "annual_gradient":  ag,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _load_multiyear(results_dir):
    """Auto-discover saved multiyear .npy files, picking the most recent
    by file modification time for each price year.  Prints every candidate
    so you can confirm the right file was selected.
    Returns {yr_tag: dict}.
    """
    found = {}
    for yr_tag in ["dk2022", "dk2019"]:
        candidates = list(results_dir.glob(f"multiyear_*{yr_tag}*.npy"))
        if not candidates:
            print(f"  [{yr_tag}] no .npy found in {results_dir}")
            continue
        # Sort by file modification time — newest last
        candidates.sort(key=lambda p: p.stat().st_mtime)
        chosen = candidates[-1]
        print(f"  [{yr_tag}] {len(candidates)} file(s) found:")
        for c in candidates:
            tag = "  <- SELECTED (most recent)" if c is chosen else ""
            print(f"    {c.name}{tag}")
        d = np.load(chosen, allow_pickle=True).item()
        n_yrs = len(d.get("soh_trajectory", []))
        repl  = d.get("replacement_years", [])
        print(f"    => {n_yrs} revenue years | replacement yr {repl}")
        found[yr_tag] = d
    return found


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    RESULTS_DIR  = Path(__file__).parent / "Results"
    OUT_DIR      = RESULTS_DIR / "thesis_figs"

    if FORCE_SYNTHETIC:
        print("FORCE_SYNTHETIC=True — using synthetic data for layout testing")
        datasets = {
            "dk2022": (_make_synthetic("dk2022"), "DK1 2022"),
            "dk2019": (_make_synthetic("dk2019"), "DK1 2019"),
        }
    else:
        raw = _load_multiyear(RESULTS_DIR)
        if not raw:
            raise FileNotFoundError(
                f"No multiyear_*.npy files found in {RESULTS_DIR}.\n"
                "Run v5.6 with the save block (see STEP 1 in this file),\n"
                "or set FORCE_SYNTHETIC=True to test layout."
            )
        datasets = {k: (v, f"DK1 {k[-4:]}") for k, v in raw.items()}

    print(f"\nProducing figures in: {OUT_DIR}")
    print("─" * 60)

    for tag, (my, label) in datasets.items():
        repl_yr = my["replacement_years"][0] if my["replacement_years"] else N_YEARS

        # Fig 4.1 / Fig 4.2 — SoH trajectory
        fig_code = "fig41" if tag == "dk2022" else "fig42"
        plot_soh_trajectory(my, label, OUT_DIR, fig_code=fig_code)

        # Fig 4.3 — fd decomposition (2022 only for thesis, but generate for both)
        plot_fd_decomposition(my, label, OUT_DIR)

        # Support — annual fd total + SoH loss rate
        plot_annual_fd_total(my, label, OUT_DIR)
        plot_soh_loss_rate(my, label, OUT_DIR)

        if tag == "dk2022":   # gradient figures: 2022 only (primary analysis)
            # Fig 4.4 — gradient bars (years 1–repl_yr)
            plot_gradient_bars(my, label, OUT_DIR, max_year=repl_yr)

            # Fig 4.5 — normalised components (years 1–repl_yr)
            plot_normalised_components(my, label, OUT_DIR, max_year=repl_yr)
            plot_gradient_overlay(my, label, OUT_DIR, max_year=repl_yr)

            # Support — alignment scatter + subgrad/DoD
            plot_alignment_scatter(my, label, OUT_DIR, max_year=repl_yr)
            plot_subgrad_dod(my, label, OUT_DIR)

    print("─" * 60)
    print("Done. Caption values printed above — copy into .tex before submission.")


if __name__ == "__main__":
    main()