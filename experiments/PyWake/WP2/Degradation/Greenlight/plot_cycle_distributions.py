r"""
plot_cycle_distributions.py

Coverage figure for the Test 2 appendix, drawn in the shared thesis style.
Two panels: (a) the cycle depth distribution and (b) the cycle mean-SoC
distribution, over the cycles that Test 2 actually tested. These show that the
per-cycle subgradient check spans the depth and mean-SoC ranges the battery
reaches over the year, rather than a hand-picked set.

Data source: the per-cycle CSV written by the Test 2 run
(analyze_rainflow_real_*_cycles.csv). The 250 zero-amplitude cycles carry the
note "skip_zero_dod" and are excluded, so the medians match the run report
(80% depth, 50% mean SoC) computed over the 533 tested cycles.

Reproducible on Windows / VS Code: depends only on numpy + matplotlib and the
bundled thesis_style.py in the same folder. No LaTeX install required.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from thesis_style import apply_thesis_style, figsize, FS_BASE, FS_LEGEND

PALETTE = apply_thesis_style(palette="brand")

# --------------------------------------------------------------------------- #
# DATA  -- most-recent per-cycle CSV under this folder; override if needed
# --------------------------------------------------------------------------- #
CSV_PATH = None   # e.g. Path("Results/analyze_rainflow_real_..._cycles.csv")
SIG_REF  = 50.0   # stress-factor reference SoC, in percent


def find_cycles_csv() -> Path:
    if CSV_PATH is not None:
        return Path(CSV_PATH)
    cands = list(HERE.rglob("analyze_rainflow_real_*_cycles.csv"))
    if not cands:
        raise FileNotFoundError(
            f"No analyze_rainflow_real_*_cycles.csv found under {HERE}. "
            "Set CSV_PATH explicitly.")
    return max(cands, key=lambda p: p.name)


csv_path = find_cycles_csv()
with open(csv_path, newline="") as fh:
    rows = list(csv.DictReader(fh))

dod  = np.array([float(r["dod"]) for r in rows]) * 100.0          # percent
sb   = np.array([float(r["sigma_bar"]) for r in rows]) * 100.0    # percent
skip = np.array([r["note"].strip() == "skip_zero_dod" for r in rows])
tested = ~skip

dod_t, sb_t = dod[tested], sb[tested]
dod_med, sb_med = np.median(dod_t), np.median(sb_t)

# --------------------------------------------------------------------------- #
# FIGURE
# --------------------------------------------------------------------------- #
fig, (axA, axB) = plt.subplots(1, 2, figsize=figsize(1.0, aspect=0.45))

# ---- (a) cycle depth distribution --------------------------------------- #
axA.hist(dod_t, bins=np.arange(0, 82, 2), color=PALETTE["fill_a"], zorder=2)
axA.axvline(dod_med, color=PALETTE["secondary"], ls="--", lw=1.4, zorder=3)
axA.set_xlabel("DoD  (%)")
axA.set_ylabel("Cycle count")
axA.set_xlim(0, 82)
axA.text(0.02, 0.97, "(a)", transform=axA.transAxes, va="top",
         fontweight="bold", fontsize=FS_BASE)

# ---- (b) cycle mean-SoC distribution ------------------------------------ #
axB.hist(sb_t, bins=np.arange(0, 102, 2), color=PALETTE["fill_a"], zorder=2)
axB.axvline(SIG_REF, color=PALETTE["neutral"], ls=":", lw=1.0, zorder=3)
axB.axvline(sb_med, color=PALETTE["secondary"], ls="--", lw=1.4, zorder=4)
axB.set_xlabel(r"Mean SoC  $\bar{\sigma}$  (%)")
axB.set_ylabel("Cycle count")
axB.set_xlim(0, 100)
axB.text(0.02, 0.97, "(b)", transform=axB.transAxes, va="top",
         fontweight="bold", fontsize=FS_BASE)

# --------------------------------------------------------------------------- #
# SAVE  (PDF first for LaTeX, PNG preview alongside)
# --------------------------------------------------------------------------- #
stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = HERE / f"cycle_distributions_{stamp}"
out_dir.mkdir(exist_ok=True)
stem    = out_dir / "cycle_distributions"

fig.savefig(f"{stem}.pdf")
fig.savefig(f"{stem}.png", dpi=300)
plt.close(fig)

print(f"  source CSV : {csv_path.name}")
print(f"  tested     : {int(tested.sum())} cycles (excluded {int(skip.sum())} zero-amplitude)")
print(f"  DoD   median {dod_med:.1f}%   range {dod_t.min():.1f}-{dod_t.max():.1f}%")
print(f"  sigma median {sb_med:.1f}%   range {sb_t.min():.1f}-{sb_t.max():.1f}%")
print(f"  saved: {stem}.pdf  and  .png")
