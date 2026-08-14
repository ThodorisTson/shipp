"""
xu_sdelta_d2.py
===============
Standalone slide figure: second derivative of Xu S_delta only.
Shows the non-convex region (S_delta'' < 0) left of delta ~ 0.144.

Run:  python xu_sdelta_d2.py
Output: xu_sdelta_d2.png  (same directory as this script)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Xu model parameters ───────────────────────────────────────────────────────
K1, K2_EXP, K3C = 1.40e5, -0.501, -1.23e5
BOUNDARY = 0.1437          # sign-change point (S_delta'' = 0)

def s_dod_d2(d):
    d   = np.clip(np.asarray(d, float), 1e-6, 1.0)
    D   = K1 * d**K2_EXP + K3C
    Dp  = K1 * K2_EXP * d**(K2_EXP - 1)
    Dpp = K1 * K2_EXP * (K2_EXP - 1) * d**(K2_EXP - 2)
    return 2 * Dp**2 / D**3 - Dpp / D**2

# ── Style (matches sigma_effect_real palette) ─────────────────────────────────
BG       = '#f7f9fc'
C_XU     = '#C94C2A'       # red  — Xu curve
C_SHADE  = '#E8873A'       # orange — non-convex shaded region
C_ZERO   = '#0d1b2a'       # near-black — zero line
TXT      = '#0d1b2a'
ANNOT    = '#555555'
GRID     = 'white'
EDGE     = '#cccccc'

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.facecolor':    BG,
    'figure.facecolor':  'white',
    'text.color':        TXT,
    'axes.edgecolor':    EDGE,
    'axes.labelcolor':   TXT,
    'xtick.color':       '#444444',
    'ytick.color':       '#444444',
    'grid.color':        GRID,
    'grid.linewidth':    0.8,
    'axes.grid':         True,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# ── Data ──────────────────────────────────────────────────────────────────────
d_all = np.linspace(0.02, 0.80, 800)
xu_d2 = np.clip(s_dod_d2(d_all), -3e-5, 1e-4) * 1e5   # scale to ×10⁵

d_nc  = d_all[d_all <= BOUNDARY]                         # non-convex region
xu_nc = np.clip(s_dod_d2(d_nc), -3e-5, 1e-4) * 1e5

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.2), facecolor='white')
fig.subplots_adjust(left=0.11, right=0.95, top=0.88, bottom=0.16)

# Orange shaded non-convex region (fill between curve and zero)
ax.fill_between(d_nc, xu_nc, 0,
                color=C_SHADE, alpha=0.25, zorder=1,
                label='non-convex region  (δ < 0.144)')

# Vertical boundary line
ax.axvline(BOUNDARY, color=C_SHADE, lw=1.4, ls='--', alpha=0.85, zorder=2)

# Zero line
ax.axhline(0, color=C_ZERO, lw=1.6, alpha=0.85, zorder=3)

# Xu S_delta'' curve
ax.plot(d_all, xu_d2, color=C_XU, lw=2.5, zorder=4,
        label=r"$S_\delta''(\delta)$  from Xu et al. (2016)")


# ── Axes labels & limits ──────────────────────────────────────────────────────
ax.set_xlim(0.02, 0.80)
ax.set_ylim(-2.5, 12.0)
ax.set_xlabel(r'Cycle depth of discharge  $\delta$  (–)', fontsize=11)
ax.set_ylabel(r"$S_\delta''(\delta) \times 10^5$", fontsize=11)

ax.set_title(
    r'Second derivative of Xu $S_\delta$ → non-convex region below $\delta \approx 0.144$',
    fontsize=11, fontweight='bold', pad=10, color=TXT)

ax.legend(fontsize=9, framealpha=0.6, loc='upper left')

# ── Save ─────────────────────────────────────────────────────────────────────
out = Path(__file__).parent / 'xu_sdelta_d2.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved → {out}")
