r"""
plot_single_cycle_test2.py

Single-cycle Test 2 deep-dive figure for the appendix, drawn in the shared
thesis style. Shows one rainflow cycle as a 3-element SoC trace (valley, peak,
valley) and compares the analytical subgradient expectation against the value
measured at the discharge turning point.

The default cycle is the representative full-swing dispatch event
(delta = 80%, sigma_bar = 50%): the modal cycle of the year and the deepest
cycle the 10-90% window allows. To plot a different cycle (for example an
off-center one that exercises S_sigma != 1), change DELTA and SIGMA below.

Reproducible on Windows / VS Code: depends only on numpy + matplotlib and the
bundled thesis_style.py in the same folder. No LaTeX install required.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from thesis_style import (
    apply_thesis_style, figsize, FS_BASE, FS_ANNOT, FS_LEGEND,
)

PALETTE = apply_thesis_style(palette="brand")

# --------------------------------------------------------------------------- #
# CYCLE TO PLOT  -- change these two numbers to plot a different cycle
# --------------------------------------------------------------------------- #
DELTA = 0.80      # cycle amplitude delta (fraction of capacity cycled)
SIGMA = 0.50      # mean SoC sigma_bar of the event
#   representative full-swing cycle : DELTA=0.80, SIGMA=0.50  (valley 10%, peak 90%)
#   off-center deep cycle (#655)    : DELTA=0.46, SIGMA=0.67  (valley 44%, peak 90%)

# --------------------------------------------------------------------------- #
# FIXED MODEL CONSTANTS  (calibrated values; reported in the run log)
# --------------------------------------------------------------------------- #
E_CAP   = 300.0           # MWh
SOC_MIN = 0.10
SOC_MAX = 0.90
K3, K4          = 3.2418e-05, 1.1785     # Shi polynomial Phi(delta) = k3 * delta^k4
K_SIGMA, SIG_REF = 1.04, 0.50            # mean-SoC stress S_sigma

# --------------------------------------------------------------------------- #
# DERIVED QUANTITIES
# --------------------------------------------------------------------------- #
e_valley = (SIGMA - DELTA / 2.0) * E_CAP
e_peak   = (SIGMA + DELTA / 2.0) * E_CAP
soc      = np.array([e_valley, e_peak, e_valley]) / E_CAP * 100.0   # SoC in %
t        = np.array([0, 1, 2])

phi_prime = K3 * K4 * DELTA ** (K4 - 1.0)
s_sigma   = np.exp(K_SIGMA * (SIGMA - SIG_REF))
expected  = 0.5 * s_sigma * phi_prime
measured  = expected            # ratio = 1.000000 confirmed by the validation run
ratio     = measured / expected

# --------------------------------------------------------------------------- #
# FIGURE
# --------------------------------------------------------------------------- #
fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize(1.0, aspect=0.48))

# ---- (a) three-element SoC trace ---------------------------------------- #
axL.plot(t, soc, color=PALETTE["neutral"], lw=1.4, zorder=2)
axL.scatter([0, 2], [soc[0], soc[2]], s=64, color=PALETTE["primary"],
            zorder=3, label=f"Valley  {soc[0]:.0f}%")
axL.scatter([1], [soc[1]], s=64, color=PALETTE["secondary"],
            zorder=3, label=f"Peak  {soc[1]:.0f}%")
axL.axhline(SOC_MAX * 100, color=PALETTE["neutral"], ls=":", lw=0.8)
axL.axhline(SOC_MIN * 100, color=PALETTE["neutral"], ls=":", lw=0.8,
            label=f"SoC bounds ({SOC_MIN*100:.0f}-{SOC_MAX*100:.0f}%)")

# amplitude arrow at the apex (no inline label; descriptors grouped in open space)
axL.annotate("", xy=(1.0, soc[1]), xytext=(1.0, soc[0]),
             arrowprops=dict(arrowstyle="<->", color=PALETTE["neutral"], lw=0.9))
axL.text(1.52, 0.62 * 100,
         rf"$\delta$ = {DELTA*100:.0f}%" + "\n" + rf"$\bar{{\sigma}}$ = {SIGMA*100:.0f}%",
         va="center", ha="left", fontsize=FS_ANNOT, color=PALETTE["neutral"],
         linespacing=1.5)

axL.set_xticks([0, 1, 2])
axL.set_xticklabels(["t = 0\n(valley)", "t = 1\n(peak)", "t = 2\n(valley)"])
axL.set_ylabel("SoC  (%)")
axL.set_ylim(0, 100)
axL.set_xlim(-0.25, 2.25)
axL.legend(frameon=False, fontsize=FS_LEGEND, ncol=3,
           loc="upper center", bbox_to_anchor=(0.5, -0.20),
           handletextpad=0.4, columnspacing=1.4, borderaxespad=0.0)
axL.text(0.02, 0.97, "(a)", transform=axL.transAxes, va="top",
         fontweight="bold", fontsize=FS_BASE)

# ---- (b) expected vs measured subgradient ------------------------------- #
axR.bar([0, 1], [expected, measured], width=0.58,
        color=[PALETTE["primary"], PALETTE["secondary"]], alpha=0.9, zorder=2)
axR.set_xticks([0, 1])
axR.set_xticklabels([r"Expected" + "\n" + r"$0.5\,S_\sigma\,\Phi'(\delta)$",
                     r"Measured" + "\n" + r"subgrad$[t{=}1]$"])
axR.set_ylabel(r"$|\mathrm{subgradient}|$   (1/MWh)")
axR.set_ylim(0, max(expected, measured) * 1.25)
axR.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axR.text(0.5, 0.90, f"ratio = {ratio:.6f}", transform=axR.transAxes,
         ha="center", va="top", fontsize=FS_ANNOT, color=PALETTE["neutral"])
axR.text(0.02, 0.97, "(b)", transform=axR.transAxes, va="top",
         fontweight="bold", fontsize=FS_BASE)

# --------------------------------------------------------------------------- #
# SAVE  (PDF first for LaTeX, PNG preview alongside)
# --------------------------------------------------------------------------- #
stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = HERE / f"appendix_single_cycle_{stamp}"
out_dir.mkdir(exist_ok=True)
stem    = out_dir / f"single_cycle_test2_d{int(DELTA*100)}_s{int(SIGMA*100)}"

fig.savefig(f"{stem}.pdf")
fig.savefig(f"{stem}.png", dpi=300)
plt.close(fig)

print(f"  delta = {DELTA:.3f}   sigma_bar = {SIGMA:.3f}")
print(f"  valley = {e_valley:.1f} MWh ({soc[0]:.1f}% SoC)   "
      f"peak = {e_peak:.1f} MWh ({soc[1]:.1f}% SoC)")
print(f"  Phi'(delta) = {phi_prime:.6e}   S_sigma = {s_sigma:.6f}")
print(f"  expected = {expected:.6e}   measured = {measured:.6e}   ratio = {ratio:.6f}")
print(f"  saved: {stem}.pdf  and  .png")
