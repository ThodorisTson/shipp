"""
phi_extrapolation_minimal.py
============================
Minimal two-panel slide figure: Xu S_delta vs Phi_shi (top),
second derivatives (bottom). Compact, clean, slide-ready.

Run:  python phi_extrapolation_minimal.py
Output: phi_extrapolation_minimal.png  (same directory as this script)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Model parameters ──────────────────────────────────────────────────────────
K1, K2_EXP, K3C   = 1.40e5, -0.501, -1.23e5
K3_SHI, K4_SHI    = 3.2418e-5, 1.1785
BOUNDARY, FIT_FLOOR = 0.1437, 0.15

def s_dod(d):
    d = np.clip(np.asarray(d, float), 1e-6, 1.0)
    return 1.0 / (K1 * d**K2_EXP + K3C)

def phi_shi(d):
    d = np.clip(np.asarray(d, float), 1e-9, 1.0)
    return K3_SHI * d**K4_SHI

def s_dod_d2(d):
    d   = np.clip(np.asarray(d, float), 1e-6, 1.0)
    D   = K1 * d**K2_EXP + K3C
    Dp  = K1 * K2_EXP * d**(K2_EXP - 1)
    Dpp = K1 * K2_EXP * (K2_EXP - 1) * d**(K2_EXP - 2)
    return 2 * Dp**2 / D**3 - Dpp / D**2

def phi_shi_d2(d):
    d = np.clip(np.asarray(d, float), 1e-9, 1.0)
    return K3_SHI * K4_SHI * (K4_SHI - 1) * d**(K4_SHI - 2)

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = '#f7f9fc'
C_XU    = '#C94C2A'
C_SHI   = '#2166ac'
C_SHADE = '#E8873A'
TXT     = '#0d1b2a'
GRID    = 'white'
EDGE    = '#cccccc'

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        9,
    'axes.facecolor':   BG,
    'figure.facecolor': 'white',
    'text.color':       TXT,
    'axes.edgecolor':   EDGE,
    'axes.labelcolor':  TXT,
    'xtick.color':      '#444444',
    'ytick.color':      '#444444',
    'grid.color':       GRID,
    'grid.linewidth':   0.8,
    'axes.grid':        True,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

# ── Data ──────────────────────────────────────────────────────────────────────
d = np.linspace(0.02, 0.80, 800)

xu_val  = s_dod(d)   * 1e5
shi_val = phi_shi(d) * 1e5
xu_d2   = np.clip(s_dod_d2(d),   -3e-5, 1e-4) * 1e5
shi_d2  = np.clip(phi_shi_d2(d), -3e-5, 1e-4) * 1e5

d_nc = d[d <= BOUNDARY]

# ── Figure: compact two-panel ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(6.5, 5.2), facecolor='white',
    gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.10}
)
fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.10)

# ── Shared region shading ─────────────────────────────────────────────────────
for ax in (ax1, ax2):
    ax.axvspan(0.02, BOUNDARY,   color=C_SHADE, alpha=0.12, lw=0)
    ax.axvline(BOUNDARY,         color=C_SHADE, lw=1.1, ls='--', alpha=0.75)
    ax.axvline(FIT_FLOOR,        color=EDGE,    lw=0.8, ls='--', alpha=0.6)
    ax.set_xlim(0.02, 0.80)

# ── Top panel: function values ────────────────────────────────────────────────
ax1.plot(d, xu_val,  color=C_XU,  lw=2.0, label=r'Xu  $S_\delta(\delta)$  → reporting')
ax1.plot(d, shi_val, color=C_SHI, lw=2.0, ls='--',
         label=r'$\Phi_{shi}(\delta) = k_3\cdot\delta^{k_4}$  → gradient')

ax1.set_ylabel(r'$\Phi(\delta) \times 10^5$', fontsize=9)
ax1.set_ylim(bottom=0)
ax1.tick_params(labelbottom=False)
ax1.legend(fontsize=8, framealpha=0.0, loc='upper center',
           handlelength=1.8, borderpad=0.3, labelspacing=0.3)

# Region label (top panel only)
ax1.text(0.074, ax1.get_ylim()[1] * 0.55 if ax1.get_ylim()[1] > 0 else 1.5,
         'non-\nconvex', ha='center', va='center', fontsize=7.5,
         color=C_SHADE, style='italic')

# ── Bottom panel: second derivatives ─────────────────────────────────────────
# Orange fill for Xu non-convex region
xu_nc_d2 = np.clip(s_dod_d2(d_nc), -3e-5, 1e-4) * 1e5
ax2.fill_between(d_nc, xu_nc_d2, 0, color=C_SHADE, alpha=0.20, zorder=1)

ax2.axhline(0, color=TXT, lw=1.3, alpha=0.7, zorder=2)
ax2.plot(d, xu_d2,  color=C_XU,  lw=2.0, zorder=3, label=r"$S_\delta''(\delta)$")
ax2.plot(d, shi_d2, color=C_SHI, lw=2.0, ls='--', zorder=3,
         label=r"$\Phi_{shi}''(\delta)$")

ax2.set_ylabel(r"$\Phi''(\delta) \times 10^5$", fontsize=9)
ax2.set_xlabel(r'Cycle depth of discharge  $\delta$', fontsize=9)
ax2.set_ylim(-2.5, 12.0)
ax2.legend(fontsize=8, framealpha=0.0, loc='center right',
           handlelength=1.8, borderpad=0.3, labelspacing=0.3)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(r'$\Phi_{shi}$ fitted to Xu $S_\delta$  →  extrapolation below $\delta = 0.15$',
             fontsize=10, fontweight='bold', color=TXT)

# ── Save ──────────────────────────────────────────────────────────────────────
out = Path(__file__).parent / 'phi_extrapolation_minimal.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved → {out}")
