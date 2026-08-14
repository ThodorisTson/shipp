r"""
rainflow_cycle_illustration.py  (styled to thesis_style.py)

Same illustration as before (Fig. 2.1), now driven by the shared style module.
Lines marked  # CHANGED  are the only edits relative to your v7 script.

Requires thesis_style.py in the same folder (or on PYTHONPATH).
Runs in VS Code on Windows: matplotlib + numpy only, bundled font, no LaTeX.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- path guard: locate the shared style module --------------------------- #
for _d in Path(__file__).resolve().parents:
    if (_d / "thesis_style.py").exists():
        sys.path.insert(0, str(_d))
        break
else:
    raise FileNotFoundError("thesis_style.py not found in any parent folder")
 
import numpy as np
import matplotlib.pyplot as plt
from thesis_style import (apply_thesis_style, figsize, TUDELFT,
                          FS_BASE, FS_LABEL, FS_ANNOT)
 
P = apply_thesis_style(palette="brand", usetex=False)   # CHANGED

# ── Synthetic SoC trajectory (condensed) ─────────────────────────────────
t_pts = [0,  2,  6, 10,   12, 19, 23,  25]
s_pts = [0.30, 0.30, 0.70, 0.25,  0.15, 0.85, 0.25, 0.25]

t_full = np.linspace(0, 25, 500)
s_full = np.interp(t_full, t_pts, s_pts)

# ── Figure ───────────────────────────────────────────────────────────────
# CHANGED: draw at the width it is included (full \textwidth), so matplotlib
# points == on-page points. Aspect lifted slightly (0.32 -> 0.40) so the
# point-sized fonts and brackets are not cramped on the smaller canvas.
fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.40))
# CHANGED: removed fig.subplots_adjust(...) -- constrained_layout handles margins

for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
ax.spines['left'].set_color(P["grid"])      # CHANGED: #CCCCCC -> palette grid
ax.spines['bottom'].set_color(P["grid"])    # CHANGED

# CHANGED: trajectory line uses palette neutral; lw from style (1.2), not 2.2
ax.plot(t_full, s_full, color=P["neutral"], zorder=3, solid_capstyle='round')

ax.set_xlim(-0.5, 27.5)
ax.set_ylim(0.0, 1.0)
ax.set_xticks([])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}'))
ax.tick_params(axis='y', color=P["grid"])   # CHANGED: labelsize now inherits FS_BASE
ax.tick_params(axis='x', length=0)
# CHANGED: no bold, no size 12 -> inherit FS_LABEL (9); black is fine for labels
ax.set_ylabel('State of Charge  (–)')
ax.set_xlabel('Time  (hours)')

# ── Semantic colours mapped to TU Delft brand ────────────────────────────
# CHANGED: caption language "red / green / blue" is preserved, brand hexes used
c_lo  = TUDELFT["red"]      # red   — lower turning points (minima)
c_hi  = TUDELFT["dgreen"]   # green — upper turning points (maxima)
c_ann = TUDELFT["blue"]     # blue  — annotations

# ── Turning point markers ────────────────────────────────────────────────
# CHANGED: marker sizes 11/4 -> 9/3, mew 1.8 -> 1.5 to suit the smaller canvas
for t_tp, s_tp in [(2, 0.30), (10, 0.25), (12, 0.15), (23, 0.25)]:
    ax.plot(t_tp, s_tp, 'o', ms=9, mfc='none', mec=c_lo, mew=1.5, zorder=5)
    ax.plot(t_tp, s_tp, 'o', ms=3, mfc=c_lo, mec=c_lo, zorder=6)

for t_tp, s_tp in [(6, 0.70), (19, 0.85)]:
    ax.plot(t_tp, s_tp, 'o', ms=9, mfc='none', mec=c_hi, mew=1.5, zorder=5)
    ax.plot(t_tp, s_tp, 'o', ms=3, mfc=c_hi, mec=c_hi, zorder=6)

# ── Left cycle: two half-cycle brackets (raised) ─────────────────────────
y_brk = 0.08

ax.annotate('', xy=(6, y_brk), xytext=(2, y_brk),
            arrowprops=dict(arrowstyle='<->', color=c_ann, lw=1.2,
                            mutation_scale=8, shrinkA=0, shrinkB=0))
for t_s in [2, 6]:
    ax.plot([t_s, t_s], [y_brk - 0.016, y_brk + 0.016], color=c_ann, lw=1.1)
ax.text(4, y_brk - 0.040, 'half-cycle', ha='center', va='top',
        fontsize=FS_ANNOT, color=c_ann)   # CHANGED: 9 -> FS_ANNOT, no bold

ax.annotate('', xy=(10, y_brk), xytext=(6, y_brk),
            arrowprops=dict(arrowstyle='<->', color=c_ann, lw=1.2,
                            mutation_scale=8, shrinkA=0, shrinkB=0, linestyle='--'))
for t_s in [6, 10]:
    ax.plot([t_s, t_s], [y_brk - 0.016, y_brk + 0.016], color=c_ann, lw=1.1, ls='--')
ax.text(8, y_brk - 0.040, 'half-cycle', ha='center', va='top',
        fontsize=FS_ANNOT, color=c_ann, style='italic')   # CHANGED

# ── Right cycle: δ and σ ─────────────────────────────────────────────────
lo_B, hi_B = 0.15, 0.85
delta_B = hi_B - lo_B
sigma_B = (hi_B + lo_B) / 2
t_brk = 21.5
sw = 0.35

ax.annotate('', xy=(t_brk, hi_B), xytext=(t_brk, lo_B),
            arrowprops=dict(arrowstyle='<->', color=c_ann, lw=1.4,
                            mutation_scale=8, shrinkA=0, shrinkB=0))
for y in [lo_B, hi_B]:
    ax.plot([t_brk - sw, t_brk + sw], [y, y], color=c_ann, lw=1.1)

ax.plot([t_brk - sw*1.6, t_brk + sw*1.6], [sigma_B, sigma_B],
        color=c_ann, lw=0.9, ls='--', alpha=0.6)

# CHANGED: sizes normalised to the style (delta=FS_LABEL, sigma=FS_BASE)
ax.text(t_brk + 1.0, (hi_B + lo_B) / 2, r'$\delta$ = %.2f' % delta_B,
        va='center', ha='left', fontsize=FS_LABEL, color=c_ann)
ax.text(t_brk - 1.0, sigma_B + 0.02, r'$\sigma$ = %.2f' % sigma_B,
        va='bottom', ha='right', fontsize=FS_BASE, color=c_ann, style='italic')

# ── Save (vector first; PNG only for quick preview) ──────────────────────
out = Path(__file__).parent
fig.savefig(out / 'rainflow_cycle_illustration.pdf')          # CHANGED: vector
fig.savefig(out / 'rainflow_cycle_illustration.png', dpi=300) # CHANGED: no bbox='tight'
print(f"Saved -> {out / 'rainflow_cycle_illustration.pdf'} and .png")