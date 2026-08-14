r"""
sigma_effect_real.py  (patched to thesis_style)

Fig 2.3 -- Two real rainflow cycles with identical cycle amplitude δ = 0.30
but different mean SoC σ, illustrating the isolated effect of Sσ on battery
degradation.

Requires: storage_e_fixed.npy in ../Results/ relative to this script.
If the file is not found, a synthetic SoC trace is used so you can check
the layout locally before swapping in real data. The synthetic trace is NOT
real dispatch data -- shapes are illustrative only.

CAPTION UPDATE (update in .tex after removing the figure banner):
  Current: "Two real cycles with the same depth (δ = 0.30) at different
            mean SoC levels, showing the isolated effect of Sσ."
  Replace with: "Two real cycles from the 2022 IEA Wind Task 50 dispatch,
                 both with cycle amplitude δ = 0.30 identified by rainflow
                 counting. Cycle A operates in the 0.10–0.40 SoC window
                 (σ = 0.25); Cycle B operates in the 0.60–0.90 window
                 (σ = 0.75). Since Sδ is identical for both, the mean SoC
                 stress Sσ is the sole driver of the degradation difference."

WHAT CHANGED vs the original:
  - Path guard added (works from Midterm\, Degradation\, Outer Loop Tests\)
  - apply_thesis_style() replaces scattered rcParams
  - figsize: (14, 5.8) -> figsize(1.0, 0.52); constrained_layout=False
    + manual subplots_adjust (wspace kept tight for two-panel continuity)
  - Colors: #2166ac -> TUDELFT["blue"], #b5351b -> TUDELFT["darkred"]
  - Bracket/σ offsets scaled to fractions of window width (were hard-coded
    data-coordinate values calibrated for 14 in; break at 6.3 in)
  - ax.set_title() x2 REMOVED -- replaced by ax.text() in axes coords
  - fig.text() banner + subtitle x2 REMOVED -- key message -> caption
  - ax.grid(True, color='white') REMOVED (white-on-white = invisible)
  - axes facecolor: kept as P["bg"] (deliberate exception -- the light
    background makes context/highlight contrast clearer for this figure)
  - Font sizes: normalized to FS_BASE / FS_LABEL / FS_ANNOT
  - savefig: PDF first, no bbox_inches='tight'
  - matplotlib.use('Agg') commented out (VS Code on Windows uses interactive)
  - Synthetic data fallback added for layout testing without .npy file
"""
import sys
from pathlib import Path

# --- path guard ----------------------------------------------------------- #
for _d in Path(__file__).resolve().parents:
    if (_d / "thesis_style.py").exists():
        sys.path.insert(0, str(_d))
        break
else:
    raise FileNotFoundError("thesis_style.py not found in any parent folder")

import numpy as np
import matplotlib
# matplotlib.use('Agg')  # uncomment if running headless / on a server
import matplotlib.pyplot as plt
from thesis_style import (apply_thesis_style, figsize, TUDELFT,
                          FS_BASE, FS_LABEL, FS_ANNOT)

P = apply_thesis_style(palette="brand", usetex=False)

# ── Data loading (with synthetic fallback) ────────────────────────────────
def _synthetic_soc():
    """
    Layout-test fallback. NOT real dispatch data.
    Creates plausible SoC traces in the two windows only.
    """
    rng  = np.random.default_rng(42)
    data = np.full(2300, 0.50 * 300.0)   # baseline: SoC=0.50

    # Left window (hrs 2200-2272): low-σ cycle, SoC mostly 0.25-0.45
    pre   = np.linspace(0.40, 0.36, 28) + rng.normal(0, 0.008, 28)
    cyc_l = np.concatenate([np.linspace(0.36, 0.10, 9),
                             np.linspace(0.10, 0.40, 17)])
    post  = np.linspace(0.40, 0.43, 18) + rng.normal(0, 0.006, 18)
    left  = np.concatenate([pre, cyc_l, post])[:72]
    data[2200:2272] = np.clip(left, 0.10, 0.90) * 300.0

    # Right window (hrs 288-360): high-σ cycle, SoC mostly 0.65-0.90
    pre2  = np.linspace(0.82, 0.86, 23) + rng.normal(0, 0.008, 23)
    cyc_r = np.concatenate([np.linspace(0.86, 0.60, 13),
                             np.linspace(0.60, 0.90, 32)])
    post2 = np.linspace(0.90, 0.88, 4)
    right = np.concatenate([pre2, cyc_r, post2])[:72]
    data[288:360] = np.clip(right, 0.10, 0.90) * 300.0

    return data


data_path = Path(__file__).parent.parent / 'Results' / 'storage_e_fixed.npy'
if data_path.exists():
    soc_full = np.load(data_path) / 300.0
    print("Loaded real dispatch data.")
else:
    print("WARNING: storage_e_fixed.npy not found -- synthetic data for layout test")
    soc_full = _synthetic_soc() / 300.0

# ── Panel configuration ───────────────────────────────────────────────────
DELTA = 0.30

panels = [
    dict(win_s=2200, win_e=2272,
         cyc_lo=0.10, cyc_hi=0.40, sigma=0.25,
         hi_s=2228,   hi_e=2254,
         col=TUDELFT["blue"],                       # CHANGED from #2166ac
         label='Cycle A  \u2192  Low \u03c3',      # → Low σ
         day_start=91),
    dict(win_s=288,  win_e=360,
         cyc_lo=0.60, cyc_hi=0.90, sigma=0.75,
         hi_s=311,    hi_e=356,
         col=TUDELFT["darkred"],                    # CHANGED from #b5351b
         label='Cycle B  \u2192  High \u03c3',     # → High σ
         day_start=12),
]

# ── Figure: tight two-panel layout (constrained_layout=False for wspace) ─
fig, axes = plt.subplots(1, 2, figsize=figsize(1.0, aspect=0.52),
                          constrained_layout=False)           # CHANGED figsize
# left/right give room for ylabel and right margin
# bottom/top give room for day labels and a small top margin
# wspace=0.06 keeps the panels visually joined (same as original intent)
fig.subplots_adjust(left=0.13, right=0.99, top=0.97,
                    bottom=0.22, wspace=0.06)                 # CHANGED margins


def plot_panel(ax, p, show_ylabel=True):
    seg = soc_full[p['win_s'] : p['win_e']]
    n   = len(seg)
    t   = np.arange(n)
    col = p['col']

    hl_s = p['hi_s'] - p['win_s']
    hl_e = p['hi_e'] - p['win_s']

    # Deliberate facecolor exception: light background makes context vs
    # highlight contrast clearer for this specific figure type.
    ax.set_facecolor(P["bg"])

    # REMOVED: ax.grid(True, color='white') -- white on white = invisible

    # ── SoC_min / SoC_max reference lines ─────────────────────────────────
    for y, lbl in [(0.10, r'SoC$_\mathrm{min}$'), (0.90, r'SoC$_\mathrm{max}$')]:
        ax.axhline(y, color='#c0c0c0', lw=0.7, ls=':', alpha=0.8, zorder=1)
        ax.text(n - 0.5, y, lbl, va='center', ha='right',
                fontsize=FS_ANNOT, color='#b8b8b8', zorder=6)

    # ── Highlight band ────────────────────────────────────────────────────
    ax.axvspan(hl_s, hl_e, color=col, alpha=0.07, zorder=1, lw=0)

    # ── Context trace (neutral grey) ──────────────────────────────────────
    ax.plot(t[:hl_s+1], seg[:hl_s+1],
            color=P["neutral"], lw=0.9, alpha=0.55,
            zorder=2, solid_capstyle='round')
    ax.plot(t[hl_e-1:], seg[hl_e-1:],
            color=P["neutral"], lw=0.9, alpha=0.55,
            zorder=2, solid_capstyle='round')

    # ── Highlighted trace (coloured) ─────────────────────────────────────
    t_hl = t[hl_s : hl_e + 1]
    s_hl = seg[hl_s : hl_e + 1]
    ax.fill_between(t_hl, p['cyc_lo'], s_hl,
                    where=s_hl >= p['cyc_lo'],
                    color=col, alpha=0.15, zorder=3)
    ax.plot(t_hl, s_hl, color=col, lw=1.5, zorder=4, solid_capstyle='round')

    # ── Cycle bounds (dashed) ─────────────────────────────────────────────
    for y in [p['cyc_lo'], p['cyc_hi']]:
        ax.plot([hl_s, hl_e], [y, y],
                color=col, lw=0.8, ls='--', alpha=0.50, zorder=3)

    # ── δ bracket ─────────────────────────────────────────────────────────
    x_brk = hl_s + 0.5 * (hl_e - hl_s)
    lo, hi = p['cyc_lo'], p['cyc_hi']
    # CHANGED: offsets scaled to window width (original was hardcoded for 14 in)
    sw = n * 0.025   # serif half-width in data coords (≈4.5 pt at thesis size)
    ax.annotate('', xy=(x_brk, hi), xytext=(x_brk, lo),
                arrowprops=dict(arrowstyle='<->', color=col, lw=1.4,
                                mutation_scale=8, shrinkA=0, shrinkB=0))
    for y_s in [lo, hi]:
        ax.plot([x_brk - sw, x_brk + sw], [y_s, y_s], color=col, lw=1.1)
    # Both δ and σ sit on the LEFT of the bracket (same side), δ below σ.
    # This unifies the annotation column: σ describes the midpoint (top),
    # δ describes the range width (bottom). Both use ha='right' so they
    # share a common right edge at x_brk - n*0.06.
    ax.text(x_brk - n * 0.06, lo + (hi - lo) * 0.15,
            f'$\\delta$ = {DELTA:.2f}',
            va='bottom', ha='right',
            fontsize=FS_BASE, color=col, fontweight='bold')

    # ── σ midline ─────────────────────────────────────────────────────────
    sigma = p['sigma']
    ax.plot([hl_s, hl_e], [sigma, sigma],
            color=col, lw=0.9, ls=(0, (5, 3)), alpha=0.80, zorder=5)
    ax.text(x_brk - n * 0.06, sigma + 0.030,
            f'$\\sigma$ = {sigma:.2f}',
            va='bottom', ha='right',
            fontsize=FS_BASE, color=col, style='italic', zorder=6)  # CHANGED

    # ── Panel label (replaces removed ax.set_title) ───────────────────────
    # CHANGED: was set_title(); now an in-axes text annotation
    ax.text(0.03, 0.96, p['label'],
            transform=ax.transAxes, va='top', ha='left',
            fontsize=FS_BASE, fontweight='bold', color=col)

    # ── "target cycle" label ──────────────────────────────────────────────
    ax.text((hl_s + hl_e) / 2, 0.965,
            '\u2190 target cycle \u2192',    # ← target cycle →
            va='top', ha='center',
            fontsize=FS_ANNOT, color=col, style='italic', alpha=0.7, zorder=7)

    # ── Axes setup ────────────────────────────────────────────────────────
    ax.set_xlim(0, n - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('Hour  (–)')

    day0   = p['day_start']
    xticks = list(range(0, n, 24))
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [f'Day {day0 + i}\n00:00' for i in range(len(xticks))],
        fontsize=FS_ANNOT)                                  # CHANGED size

    if show_ylabel:
        ax.set_yticklabels(
            [f'{v:.1f}' for v in np.arange(0, 1.1, 0.1)],
            fontsize=FS_ANNOT)
        ax.set_ylabel('State of Charge  (–)')
        # Colour the lo/hi tick labels (requires canvas flush to populate objects)
        fig.canvas.draw()
        for tick, val in zip(ax.yaxis.get_ticklabels(), np.arange(0, 1.01, 0.1)):
            if abs(round(val, 1) - lo) < 0.01 or abs(round(val, 1) - hi) < 0.01:
                tick.set_color(col)
                tick.set_fontweight('bold')
    else:
        ax.set_yticklabels([])
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)
        for y in [lo, hi]:
            ax.text(-0.015, y, f'{y:.2f}',
                    va='center', ha='right',
                    fontsize=FS_ANNOT, color=col, fontweight='bold',
                    transform=ax.get_yaxis_transform(), clip_on=False)


# REMOVED: ax.set_title() in loop -- replaced by ax.text() in plot_panel()
# REMOVED: fig.text() banner "Same δ, Different σ..." -- see CAPTION UPDATE
# REMOVED: fig.text() italic subtitle -- redundant with caption

plot_panel(axes[0], panels[0], show_ylabel=True)
plot_panel(axes[1], panels[1], show_ylabel=False)

# ── Save ──────────────────────────────────────────────────────────────────
OUT = Path(__file__).parent
fig.savefig(OUT / 'fig23_sigma_effect.pdf')                  # CHANGED: PDF first
fig.savefig(OUT / 'fig23_sigma_effect.png', dpi=300)         # CHANGED: no bbox='tight'
print("Fig 2.3 saved.")