import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── data ───────────────────────────────────────────────────────────────────
soc_mwh = np.load(Path(__file__).parent.parent / 'Results' / 'storage_e_fixed.npy')
seg = (soc_mwh / 300.0)[32 * 168 : 32 * 168 + 72]
t   = np.arange(72)

# ── cycle definitions ──────────────────────────────────────────────────────
# A: shallow / low-σ    hi=seg[12]=0.544  lo2=0.10
# B: deep / mid-σ       hi=seg[40]=0.90   lo2=seg[44]=0.10
# C: partial / high-σ   hi=seg[51]=0.90   lo2=seg[55]=0.33
cycles = [
    dict(t_lo1=0,  t_hi=12, t_lo2=19, soc_lo2=0.10,     color='#b5351b', lbl='A'),
    dict(t_lo1=24, t_hi=40, t_lo2=44, soc_lo2=seg[44],  color='#2166ac', lbl='B'),
    dict(t_lo1=49, t_hi=51, t_lo2=55, soc_lo2=seg[55],  color='#1a7a4a', lbl='C'),
]
for cy in cycles:
    hi = seg[cy['t_hi']]
    lo = cy['soc_lo2']
    cy['soc_hi'] = hi
    cy['lo']     = lo
    cy['delta']  = round(hi - lo, 2)
    cy['sigma']  = round((hi + lo) / 2, 2)

# ── style — matches rainflow_explainer_v3 ──────────────────────────────────
BG = '#f7f9fc'
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.facecolor':   BG,
    'figure.facecolor': 'white',
    'text.color':       '#0d1b2a',
    'axes.edgecolor':   '#cccccc',
    'axes.labelcolor':  '#0d1b2a',
    'xtick.color':      '#444444',
    'ytick.color':      '#444444',
    'grid.color':       'white',
    'grid.linewidth':   0.8,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

fig, ax = plt.subplots(figsize=(12, 5.2))
fig.subplots_adjust(left=0.08, right=0.90, top=0.82, bottom=0.14)

# ── SoC trace ────────────────────────────────────────────────────────────
SOC_COL = '#2166ac'
ax.fill_between(t, 0.10, seg, where=seg >= 0.10,
                color=SOC_COL, alpha=0.10, zorder=1)
ax.plot(t, seg, color=SOC_COL, lw=1.4, zorder=3, solid_capstyle='round')

ax.set_facecolor(BG)
ax.grid(True, color='white', lw=0.8, zorder=0)

ax.set_xlim(0, 71)
ax.set_ylim(0.02, 0.975)
ax.set_xlabel('Hour  (–)', fontsize=10)
ax.set_ylabel('State of Charge  (–)', fontsize=10)
ax.set_title('SoC Rainflow Cycles for a Representative Week  (August 2022, IEA Task Force 50 site)',
             fontsize=13, fontweight='bold', color='#0d1b2a', pad=18)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}'))
ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
ax.set_xticks(range(0, 72, 12))
ax.set_xticklabels([f'Day {d}\n00:00' for d in range(6)], fontsize=9)
ax.tick_params(labelsize=9)

# SoC bounds
for y, lbl in [(0.10, 'SoC$_{min}$'), (0.90, 'SoC$_{max}$')]:
    ax.axhline(y, color=SOC_COL, lw=0.7, ls='--', alpha=0.45, zorder=1)
    ax.text(71.8, y, lbl, va='center', ha='left', fontsize=8,
            color=SOC_COL, alpha=0.75)

# ── cycle annotation helper ────────────────────────────────────────────────
def annotate_cycle(ax, cy):
    lo    = cy['lo']
    hi    = cy['soc_hi']
    t_hi  = cy['t_hi']
    t_lo2 = cy['t_lo2']
    c     = cy['color']
    δ     = cy['delta']
    σ     = cy['sigma']
    lbl   = cy['lbl']

    t_brk = (t_hi + t_lo2) / 2 + 2.0   # bracket x-position

    # Double-headed arrow spanning δ
    ax.annotate('', xy=(t_brk, hi), xytext=(t_brk, lo),
                arrowprops=dict(arrowstyle='<->', color=c, lw=1.6,
                                mutation_scale=9,
                                shrinkA=0, shrinkB=0))

    # Serif ticks
    sw = 0.55
    for y_s in [lo, hi]:
        ax.plot([t_brk - sw, t_brk + sw], [y_s, y_s], color=c, lw=1.3)

    # σ midpoint dashed line
    ax.plot([t_brk - sw*1.6, t_brk + sw*1.6], [σ, σ],
            color=c, lw=0.9, ls='--', alpha=0.70)

    # δ label — right
    ax.text(t_brk + 1.2, (hi + lo) / 2,
            f'δ = {δ:.2f}',
            va='center', ha='left', fontsize=9.5,
            color=c, fontweight='bold')

    # σ label — left at midpoint
    ax.text(t_brk - 1.3, σ + 0.025,
            f'σ = {σ:.2f}',
            va='bottom', ha='right', fontsize=9,
            color=c, style='italic')

    # Cycle letter badge at peak
    ax.text(t_hi, hi + 0.030, lbl,
            va='bottom', ha='center', fontsize=10,
            color=c, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor=c, linewidth=1.1))

for cy in cycles:
    annotate_cycle(ax, cy)

OUT = Path(__file__).parent
fig.savefig(OUT / 'soc_cycle_annotation_v3.png', dpi=200,
            bbox_inches='tight', facecolor='white')
fig.savefig(OUT / 'soc_cycle_annotation_v3.pdf', dpi=300,
            bbox_inches='tight', facecolor='white')
print(f"Saved to {OUT}")
