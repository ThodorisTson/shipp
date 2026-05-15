"""
alignment_schematic_midterm.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.15, wspace=0.08)

TEAL       = '#1D9E75'
TEAL_DARK  = '#0F6E56'
CORAL      = '#D85A30'
CORAL_DARK = '#993C1D'
DGRAY      = '#333333'
LGRAY      = '#AAAAAA'
SEP        = '#DDDDDD'

def setup(ax):
    ax.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 10); ax.set_ylim(-0.10, 1.10)

def draw_panel(ax, color, color_dark, title, dod_xs, dod_ys, highlight_xs, subtitle):
    setup(ax)

    # highlight bands (left panel only when highlight_xs given)
    for x in highlight_xs:
        ax.axvspan(x - 0.5, x + 0.5, ymin=0.0/1.2, ymax=1.0/1.2,
                   color=color, alpha=0.07, lw=0)

    # price signal — connected line profile, same in both panels
    x_price = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10]
    y_price = [0.55, 0.52, 0.58, 0.90, 0.60, 0.50, 0.48, 0.52, 0.55, 0.88, 0.58, 0.50, 0.48, 0.52, 0.55, 0.92, 0.58, 0.52, 0.50]
    ax.plot(x_price, y_price, color=color, lw=2, zorder=3)

    # DoD bars — deep bars full color, shallow bars light
    for x, y in zip(dod_xs, dod_ys):
        if y > 0.25:
            ax.bar(x, y, width=0.25, bottom=0.0, color=color, alpha=0.85, zorder=3)
        else:
            ax.bar(x, y, width=0.25, bottom=0.0, color=color, alpha=0.25, zorder=3)

    # separator line
    ax.axhline(0.45, color=SEP, lw=1.0, ls='--')

    # row labels
    ax.text(-0.3, 0.75, 'Price', color=DGRAY, fontsize=11, va='center', ha='right')
    ax.text(-0.3, 0.20, 'DoD',   color=DGRAY, fontsize=11, va='center', ha='right')

    # hour arrow
    ax.annotate('', xy=(9.9, -0.06), xytext=(0.1, -0.06),
                arrowprops=dict(arrowstyle='->', color=DGRAY, lw=1.0))
    ax.text(10.0, -0.06, ' hour', color=DGRAY, fontsize=9, va='center', ha='left')

    # subtitle
    ax.text(5.0, -0.145, subtitle, color=DGRAY, fontsize=11,
            ha='center', va='bottom', style='italic')

    # title
    ax.set_title(title, color=color_dark, fontsize=13, fontweight='bold', pad=10)

draw_panel(ax1, TEAL,  TEAL_DARK,
           'High-alignment year',
           dod_xs=[1.8, 2.1, 2.6, 4.8, 5.2, 5.6, 7.8, 8.1, 8.5, 3.2, 6.3],
           dod_ys=[0.38, 0.32, 0.28, 0.35, 0.30, 0.25, 0.37, 0.33, 0.27, 0.15, 0.12],
           highlight_xs=[2, 5, 8],
           subtitle='deep cycles align with price peaks')

draw_panel(ax2, CORAL, CORAL_DARK,
           'Low-alignment year',
           dod_xs=[0.8, 1.5, 2.9, 3.6, 4.2, 5.8, 6.4, 7.1, 8.8, 9.3, 2.2],
           dod_ys=[0.36, 0.30, 0.35, 0.28, 0.32, 0.34, 0.30, 0.37, 0.15, 0.12, 0.14],
           highlight_xs=[],
           subtitle='deep cycles fall in low-price hours')

# vertical divider in figure coords
fig.add_artist(plt.Line2D([0.51, 0.51], [0.10, 0.96],
    transform=fig.transFigure, color=LGRAY, lw=1.0, ls='--'))

out = Path(__file__).parent / 'alignment_schematic_midterm.png'
fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
print(f"Saved → {out}")
