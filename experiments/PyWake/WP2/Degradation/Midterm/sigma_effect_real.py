import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

# ── real data ──────────────────────────────────────────────────────────────
soc_full = np.load(Path(__file__).parent.parent / 'Results' / 'storage_e_fixed.npy') / 300.0

# Low-σ  cycle: δ=0.30, σ=0.25 → SoC 0.10–0.40  at hrs 2228–2245 (day 92-93)
# High-σ cycle: δ=0.30, σ=0.75 → SoC 0.60–0.90  at hrs  311– 338 (day 12-14)
panels = [
    dict(win_s=2200, win_e=2272,
         cyc_lo=0.10, cyc_hi=0.40, sigma=0.25,
         hi_s=2228, hi_e=2254,          # highlight window (covers the 0.10–0.40 segment + plateau)
         col='#2166ac', label='Cycle A  →  Low σ',
         day_start=91,
         subtitle='σ = 0.25   window  0.10 - 0.40'),
    dict(win_s=288,  win_e=360,
         cyc_lo=0.60, cyc_hi=0.90, sigma=0.75,
         hi_s=311,   hi_e=356,           # highlight window (covers the 0.60–0.90 segment + plateau)
         col='#b5351b', label='Cycle B  →  High σ',
         day_start=12,
         subtitle='σ = 0.75   window  0.60 - 0.90'),
]

DELTA = 0.30
BG    = '#f7f9fc'

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
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

fig = plt.figure(figsize=(14, 5.8), facecolor='white')
gs  = gridspec.GridSpec(1, 2, figure=fig,
                        wspace=0.08,
                        left=0.07, right=0.96,
                        top=0.82, bottom=0.14)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

def plot_panel(ax, p, show_ylabel=True):
    seg   = soc_full[p['win_s'] : p['win_e']]
    n     = len(seg)
    t     = np.arange(n)
    col   = p['col']

    # Relative positions of highlight region within window
    hl_s = p['hi_s'] - p['win_s']
    hl_e = p['hi_e'] - p['win_s']

    ax.set_facecolor(BG)
    ax.grid(True, color='white', lw=0.8, zorder=0)

    # ── highlight band (background) ───────────────────────────────────────
    ax.axvspan(hl_s, hl_e, color=col, alpha=0.08, zorder=1, lw=0)

    # ── SoC_min / SoC_max reference lines ─────────────────────────────────
    for y, lbl in [(0.10, 'SoC$_{min}$'), (0.90, 'SoC$_{max}$')]:
        ax.axhline(y, color='#c0c0c0', lw=0.7, ls=':', alpha=0.8, zorder=1)
        ax.text(n - 0.5, y, lbl, va='center', ha='right',
                fontsize=7.5, color='#b0b0b0', zorder=6)

    # ── context trace (grey) ──────────────────────────────────────────────
    # Left context
    ax.plot(t[:hl_s+1], seg[:hl_s+1],
            color='#aaaaaa', lw=1.4, zorder=2, solid_capstyle='round')
    # Right context
    ax.plot(t[hl_e-1:], seg[hl_e-1:],
            color='#aaaaaa', lw=1.4, zorder=2, solid_capstyle='round')

    # ── highlighted trace (coloured) ─────────────────────────────────────
    t_hl  = t[hl_s : hl_e + 1]
    s_hl  = seg[hl_s : hl_e + 1]
    ax.fill_between(t_hl, p['cyc_lo'], s_hl,
                    where=s_hl >= p['cyc_lo'],
                    color=col, alpha=0.18, zorder=3)
    ax.plot(t_hl, s_hl, color=col, lw=2.2, zorder=4, solid_capstyle='round')

    # ── target window bounds (dashed) ─────────────────────────────────────
    for y in [p['cyc_lo'], p['cyc_hi']]:
        ax.plot([hl_s, hl_e], [y, y],
                color=col, lw=0.9, ls='--', alpha=0.55, zorder=3)

    # ── δ bracket ─────────────────────────────────────────────────────────
    # Place at 80% through the highlight window
    x_brk = hl_s + 0.5 * (hl_e - hl_s)
    lo, hi = p['cyc_lo'], p['cyc_hi']
    sigma  = p['sigma']

    ax.annotate('', xy=(x_brk, hi), xytext=(x_brk, lo),
                arrowprops=dict(arrowstyle='<->', color=col, lw=1.7,
                                mutation_scale=9, shrinkA=0, shrinkB=0))
    for y_s in [lo, hi]:
        ax.plot([x_brk - 0.6, x_brk + 0.6], [y_s, y_s],
                color=col, lw=1.3)
    ax.text(x_brk + 1.0, (lo + hi) / 2,
            f'δ = {DELTA:.2f}',
            va='center', ha='left', fontsize=10.5,
            color=col, fontweight='bold')

    # ── σ midline ─────────────────────────────────────────────────────────
    ax.plot([hl_s, hl_e], [sigma, sigma],
            color=col, lw=1.1, ls=(0, (5, 3)), alpha=0.85, zorder=5)
    ax.text(x_brk - 1.2, sigma + 0.030,
            f'σ = {sigma:.2f}',
            va='bottom', ha='right', fontsize=10,
            color=col, style='italic', zorder=6)

    # ── axes setup ────────────────────────────────────────────────────────
    ax.set_xlim(0, n - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('Hour  (–)', fontsize=10)

    # x-ticks at 24h intervals, labelled as Day N
    day0 = p['day_start']
    xticks = list(range(0, n, 24))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'Day {day0 + i}\n00:00' for i in range(len(xticks))],
                        fontsize=8.5)

    if show_ylabel:
        ax.set_yticklabels([f'{v:.1f}' for v in np.arange(0, 1.1, 0.1)], fontsize=9)
        ax.set_ylabel('State of Charge  (–)', fontsize=10)
        # Colour the lo/hi tick labels
        fig.canvas.draw()
        for tick, val in zip(ax.yaxis.get_ticklabels(), np.arange(0, 1.01, 0.1)):
            if abs(round(val, 1) - lo) < 0.01 or abs(round(val, 1) - hi) < 0.01:
                tick.set_color(col)
                tick.set_fontweight('bold')
    else:
        ax.set_yticklabels([])
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)
        # Label the bounds in axes coords
        for y in [lo, hi]:
            ax.text(-0.015, y, f'{y:.2f}',
                    va='center', ha='right', fontsize=8.5,
                    color=col, fontweight='bold',
                    transform=ax.get_yaxis_transform(), clip_on=False)

    # ── "highlighted cycle" label ─────────────────────────────────────────
    ax.text((hl_s + hl_e) / 2, 0.965,
            '← target cycle →',
            va='top', ha='center', fontsize=8,
            color=col, style='italic', alpha=0.7, zorder=7)


plot_panel(axes[0], panels[0], show_ylabel=True)
plot_panel(axes[1], panels[1], show_ylabel=False)

# ── Panel titles ──────────────────────────────────────────────────────────
for ax, p in zip(axes, panels):
    ax.set_title(f'{p["label"]}\n{p["subtitle"]}',
                 fontsize=12, fontweight='bold',
                 color=p['col'], pad=10)

# ── Figure-level text ─────────────────────────────────────────────────────
fig.text(0.515, 0.965,
         'Same δ, Different σ : Same Energy Moved, Different Degradation',
         ha='center', fontsize=13, fontweight='bold', color='#0d1b2a')
fig.text(0.515, 0.920,
         f'Both cycles: δ = {DELTA:.2f}  (exact rainflow match, 2022 IEA Task Force 50 wind farm dispatch)   '
         r'$S_\delta$ identical  →  $S_\sigma$ is the sole difference',
         ha='center', fontsize=10, color='#555', style='italic')

OUT = Path(__file__).parent
fig.savefig(OUT / 'sigma_effect_real.png', dpi=200,
            bbox_inches='tight', facecolor='white')
fig.savefig(OUT / 'sigma_effect_real.pdf', dpi=300,
            bbox_inches='tight', facecolor='white')
print("Saved.")
