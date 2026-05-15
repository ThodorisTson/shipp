import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BG   = "white"
FG   = "#2c2c2c"
GRID = "#e8e8e8"
C_XU  = "#C94C2A"
C_SIG = "#6A3D9A"
C_SHI = "#2878BD"
C_GRN = "#3B6D11"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": "#f7f9fc",
    "axes.edgecolor": "#cccccc", "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.grid": True,
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

K1, K2_EXP, K3C  = 1.40e5, -0.501, -1.23e5
K_SIGMA, SIG_REF = 1.04, 0.50
K3_SHI, K4_SHI   = 3.2418e-5, 1.1785

def s_dod(d):
    d = np.clip(d, 1e-6, 1.0)
    return 1.0 / (K1 * d**K2_EXP + K3C)

def s_soc(sigma):
    return np.exp(K_SIGMA * (np.asarray(sigma, float) - SIG_REF))

def phi_shi(d):
    return K3_SHI * np.clip(d, 1e-9, 1.0)**K4_SHI

OUT = Path(__file__).parent

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Xu (2016)
# ══════════════════════════════════════════════════════════════════════════
fig10, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
fig10.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.16, wspace=0.38)

d_arr   = np.linspace(0.02, 0.80, 400)
sig_arr = np.linspace(0.10, 0.90, 400)

# Left: S_delta
sd = s_dod(d_arr)
ax1.plot(d_arr, sd * 1e5, color=C_XU, lw=2.5, zorder=3)
ax1.fill_between(d_arr, 0, sd * 1e5, color=C_XU, alpha=0.10, zorder=2)
ax1.set_xlabel('Cycle depth of discharge  δ  (–)', fontsize=10)
ax1.set_ylabel(r'$S_\delta(\delta) \times 10^5$', fontsize=10)
ax1.set_title(r'$S_\delta(\delta)$ : depth of discharge stress',
              fontsize=11, fontweight='bold', color=C_XU, pad=8)
ax1.set_xlim(0.02, 0.80)
ax1.set_ylim(bottom=0)
ax1.tick_params(labelsize=9)

# Right: S_sigma
ss = s_soc(sig_arr)
ax2.plot(sig_arr, ss, color=C_SIG, lw=2.5, zorder=3)
ax2.fill_between(sig_arr, 1.0, ss, where=ss >= 1.0, color=C_SIG, alpha=0.12, zorder=2)
ax2.fill_between(sig_arr, ss, 1.0, where=ss < 1.0,  color=C_SIG, alpha=0.06, zorder=2)
ax2.axvline(SIG_REF, color=FG, lw=1.0, ls=':', alpha=0.5, zorder=1)
ax2.axhline(1.0,     color=FG, lw=0.8, ls=':', alpha=0.4, zorder=1)
ax2.text(SIG_REF + 0.015, 0.68,
         r'$\sigma_{ref} = 0.50$' + '\n' + r'$S_\sigma = 1.0$',
         fontsize=8, color=FG, alpha=0.65)
ax2.set_xlabel('Cycle mean SoC  σ  (–)', fontsize=10)
ax2.set_ylabel(r'$S_\sigma(\sigma)$  (–)', fontsize=10)
ax2.set_title(r'$S_\sigma(\sigma)$ : mean SoC stress',
              fontsize=11, fontweight='bold', color=C_SIG, pad=8)
ax2.set_xlim(0.10, 0.90)
ax2.tick_params(labelsize=9)

# Title
fig10.text(0.50, 0.97,
           'Xu (2016) : Stress-Based Degradation Reporting',
           ha='center', fontsize=13, fontweight='bold', color=FG)

# Formula subtitle — split into two fig.text calls to avoid LaTeX concat issues
fig10.text(0.50, 0.895,
           r'$f_d = S_t(t)\cdot S_\sigma(\bar{\sigma})\cdot S_T(T_c)'
           r'\ +\ \sum_i\, n_i\cdot S_\delta(\delta_i)'
           r'\cdot S_\sigma(\sigma_i)\cdot S_T(T_c)$',
           ha='center', fontsize=11, color='#444')

fig10.savefig(OUT / 'slide10_xu_schematic.png', dpi=200,
              bbox_inches='tight', facecolor='white')
fig10.savefig(OUT / 'slide10_xu_schematic.pdf', dpi=300,
              bbox_inches='tight', facecolor='white')
plt.close(fig10)
print("Slide 10 saved.")

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 11 — Shi (2018)
# ══════════════════════════════════════════════════════════════════════════
fig11, (ax3, ax4) = plt.subplots(1, 2, figsize=(11, 4.6))
fig11.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.16, wspace=0.38)

d_all = np.linspace(0.02, 0.80, 400)
phi_all = phi_shi(d_all)

# Left: Phi_shi
ax3.fill_between(d_all, 0, phi_all * 1e5, color=C_SHI, alpha=0.10, zorder=1)
ax3.plot(d_all, phi_all * 1e5, color=C_SHI, lw=2.8, zorder=3)
ax3.text(0.40, phi_shi(0.36) * 1e5,
         f'$k_3 = {K3_SHI:.2e}$\n$k_4 = {K4_SHI:.4f}$',
         fontsize=9.5, color=C_SHI,
         bbox=dict(boxstyle='round,pad=0.40', facecolor='white',
                   edgecolor=C_SHI, linewidth=0.9, alpha=0.92))
ax3.set_xlabel('Cycle depth of discharge  δ  (–)', fontsize=10)
ax3.set_ylabel(r'$\Phi_{shi}(\delta) \times 10^5$', fontsize=10)
ax3.set_title(r'$\Phi_{shi}(\delta) = k_3 \cdot \delta^{k_4}$',
              fontsize=12, fontweight='bold', color=C_SHI, pad=8)
ax3.set_xlim(0.02, 0.80)
ax3.set_ylim(bottom=0)
ax3.tick_params(labelsize=9)

# Right: Phi_shi second derivative
phi_d2 = K3_SHI * K4_SHI * (K4_SHI - 1) * np.clip(d_all, 1e-9, 1)**(K4_SHI - 2)
ax4.plot(d_all, phi_d2 * 1e5, color=C_SHI, lw=2.5, zorder=3)
ax4.fill_between(d_all, 0, phi_d2 * 1e5, color=C_SHI, alpha=0.10, zorder=2)
ax4.axhline(0, color=FG, lw=1.0, ls='-', alpha=0.4, zorder=1)
ax4.text(0.40, (phi_d2 * 1e5).max() * 0.50,
         r"$\Phi''_{shi}(\delta) > 0$" + "\neverywhere  ✓",
         fontsize=10.5, color=C_GRN, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.42', facecolor='white',
                   edgecolor=C_GRN, linewidth=1.2, alpha=0.95))
ax4.set_xlabel('Cycle depth of discharge  δ  (–)', fontsize=10)
ax4.set_ylabel(r"$\Phi''_{shi}(\delta) \times 10^5$", fontsize=10)
ax4.set_title(r"$\Phi''_{shi}(\delta)$ → second derivative (convexity guarantee)",
              fontsize=11, fontweight='bold', color=C_GRN, pad=8)
ax4.set_xlim(0.02, 0.80)
ax4.tick_params(labelsize=9)

# Title
fig11.text(0.50, 0.97,
           'Shi (2018) : Convex Polynomial Degradation',
           ha='center', fontsize=13, fontweight='bold', color=FG)

# Formula subtitle
fig11.text(0.50, 0.895,
           r'$f_{d,shi} = \sum_i\, n_i \cdot \Phi(\delta_i)'
           r'\cdot S_\sigma(\sigma_i) \cdot S_T(T_c)$'
           r'$\qquad \Phi(\delta) = k_3 \cdot \delta^{k_4},\quad k_4 > 1$',
           ha='center', fontsize=11, color='#444')

fig11.savefig(OUT / 'slide11_shi_schematic.png', dpi=200,
              bbox_inches='tight', facecolor='white')
fig11.savefig(OUT / 'slide11_shi_schematic.pdf', dpi=300,
              bbox_inches='tight', facecolor='white')
plt.close(fig11)
print("Slide 11 saved.")
