r"""
slide_schematics_v2.py  (patched to thesis_style -- v2)

Changes vs previous patch:
  Fig 2.2: C_SD navy -> TU Delft blue; C_SS blue -> TU Delft orange (contrast)
            sigma_ref annotation moved into a box (curve-colour border + text,
            consistent with k3/k4 convention in Shi figure)
  Fig 3.6: k3/k4 box -> upper left, fontsize FS_BASE, larger padding
            Phi'' box -> checkmark removed

All other patches (titles, path guard, savefig) unchanged from previous version.
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

# ── Model parameters ──────────────────────────────────────────────────────
K1, K2_EXP, K3C    = 1.40e5, -0.501, -1.23e5
K_SIGMA, SIG_REF    = 1.04, 0.50
K3_SHI, K4_SHI      = 3.2418e-5, 1.1785

def s_dod(d):
    return 1.0 / (K1 * np.clip(d, 1e-6, 1.0)**K2_EXP + K3C)

def s_soc(sigma):
    return np.exp(K_SIGMA * (np.asarray(sigma, float) - SIG_REF))

def phi_shi(d):
    return K3_SHI * np.clip(d, 1e-9, 1.0)**K4_SHI

# ── Semantic colours -- TU Delft brand ───────────────────────────────────
# CHANGED: two distinct hues (blue vs orange) instead of two blues
C_SD  = TUDELFT["blue"]    # Sδ  -- TU Delft blue  #0076C2
C_SS  = TUDELFT["orange"]  # Sσ  -- TU Delft orange #EC6842  (clear contrast)
C_PHI = TUDELFT["blue"]    # Φshi -- blue (unchanged)
C_CV  = TUDELFT["dgreen"]  # Φ''  -- green (unchanged)

OUT = Path(__file__).parent

# ══════════════════════════════════════════════════════════════════════════
# Fig 2.2  --  Xu (2016) stress factor models
# ══════════════════════════════════════════════════════════════════════════
fig22, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize(1.0, aspect=0.48))

d_arr   = np.linspace(0.02, 0.80, 400)
sig_arr = np.linspace(0.10, 0.90, 400)

# ── Left: Sδ(δ) ──────────────────────────────────────────────────────────
sd = s_dod(d_arr)
ax1.fill_between(d_arr, 0, sd * 1e5, color=C_SD, alpha=0.08)
ax1.plot(d_arr, sd * 1e5, color=C_SD, zorder=3)
ax1.set_xlabel(r'Cycle depth of discharge $\delta$  (–)')
ax1.set_ylabel(r'$S_\delta(\delta) \times 10^5$')
ax1.set_xlim(0.02, 0.80)
ax1.set_ylim(bottom=0)

# ── Right: Sσ(σ) ─────────────────────────────────────────────────────────
ss = s_soc(sig_arr)
ax2.fill_between(sig_arr, 1.0, ss, where=(ss >= 1.0), color=C_SS, alpha=0.10)
ax2.fill_between(sig_arr, ss,  1.0, where=(ss < 1.0),  color=C_SS, alpha=0.05)
ax2.plot(sig_arr, ss, color=C_SS, zorder=3)
ax2.axvline(SIG_REF, color=P["neutral"], lw=0.8, ls=':', alpha=0.5, zorder=1)
ax2.axhline(1.0,     color=P["neutral"], lw=0.7, ls=':', alpha=0.4, zorder=1)
# CHANGED: annotation now in a box, curve colour (orange) for text + border
#          consistent with the k3/k4 box convention in the Shi figure
ax2.text(SIG_REF + 0.04, 0.70,
         r'$\sigma_\mathrm{ref} = 0.50$' + '\n' + r'$S_\sigma = 1.0$',
         fontsize=FS_ANNOT, color=C_SS, va='bottom', ha='left',
         bbox=dict(boxstyle='round,pad=0.40', facecolor='white',
                   edgecolor=C_SS, linewidth=0.8, alpha=0.92))
ax2.set_xlabel(r'Cycle mean SoC $\sigma$  (–)')
ax2.set_ylabel(r'$S_\sigma(\sigma)$  (–)')
ax2.set_xlim(0.10, 0.90)

fig22.savefig(OUT / 'fig22_xu_stress.pdf')
fig22.savefig(OUT / 'fig22_xu_stress.png', dpi=300)
plt.close(fig22)
print("Fig 2.2 saved.")

# ══════════════════════════════════════════════════════════════════════════
# Fig 3.6  --  Φshi and its second derivative
# ══════════════════════════════════════════════════════════════════════════
fig36, (ax3, ax4) = plt.subplots(1, 2, figsize=figsize(1.0, aspect=0.48))

d_all   = np.linspace(0.02, 0.80, 400)
phi_all = phi_shi(d_all)

# ── Left: Φshi(δ) ────────────────────────────────────────────────────────
ax3.fill_between(d_all, 0, phi_all * 1e5, color=C_PHI, alpha=0.08)
ax3.plot(d_all, phi_all * 1e5, color=C_PHI, zorder=3)
# CHANGED: box moved to upper left empty space; fontsize FS_BASE (was FS_ANNOT);
#          padding 0.50 (was 0.40); anchor va='top' so it sits near the top
ax3.text(0.06, 2.20,
         f'$k_3 = {K3_SHI:.2e}$\n$k_4 = {K4_SHI:.4f}$',
         fontsize=FS_BASE, color=C_PHI,
         va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.50', facecolor='white',
                   edgecolor=C_PHI, linewidth=0.9, alpha=0.92))
ax3.set_xlabel(r'Cycle depth of discharge $\delta$  (–)')
ax3.set_ylabel(r'$\Phi_\mathrm{shi}(\delta) \times 10^5$')
ax3.set_xlim(0.02, 0.80)
ax3.set_ylim(bottom=0)

# ── Right: Φ''shi(δ) ─────────────────────────────────────────────────────
phi_d2 = K3_SHI * K4_SHI * (K4_SHI - 1) * np.clip(d_all, 1e-9, 1.0)**(K4_SHI - 2)
ax4.fill_between(d_all, 0, phi_d2 * 1e5, color=C_CV, alpha=0.10)
ax4.plot(d_all, phi_d2 * 1e5, color=C_CV, zorder=3)
# FIXED: axhline(0) removed -- the spine at ylim bottom IS y=0, matching ax3.
#        The floating line existed only because ylim(bottom=0) was missing below.
# CHANGED: checkmark removed
ax4.text(0.40, (phi_d2 * 1e5).max() * 0.50,
         r"$\Phi''_\mathrm{shi}(\delta) > 0$" + "\neverywhere",
         fontsize=FS_BASE, color=C_CV,
         bbox=dict(boxstyle='round,pad=0.42', facecolor='white',
                   edgecolor=C_CV, linewidth=0.9, alpha=0.95))
ax4.set_xlabel(r'Cycle depth of discharge $\delta$  (–)')
ax4.set_ylabel(r"$\Phi''_\mathrm{shi}(\delta) \times 10^5$")
ax4.set_xlim(0.02, 0.80)
ax4.set_ylim(bottom=0)   # FIXED: was missing -- caused spine to sit at ~-0.4

fig36.savefig(OUT / 'fig36_shi_convex.pdf')
fig36.savefig(OUT / 'fig36_shi_convex.png', dpi=300)
plt.close(fig36)
print("Fig 3.6 saved.")