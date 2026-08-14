r"""
plot_fd_split_slide.py
======================
Slide version of the cycle / calendar split of annual f_d.

One stacked bar per price year, using the lifetime mean of the annual
components over the full project horizon. Companion to
plot_fd_components.py, which keeps the year-by-year detail for the thesis
and the appendix.

Why a separate figure: the thesis figure carries 40 bars so that the
year-to-year trend is available. On a slide the claim is a single share,
so the figure carries two bars and states that share directly.

Reproducible on Windows / VS Code: matplotlib + numpy + pandas, bundled
DejaVu font, no LaTeX toolchain. Anchored on Path(__file__).parent.

Design (from thesis_style):
  - Stacked bar, so components use the fill_a / fill_b slots
    (blue = cycle, orange = calendar), matching plot_fd_components.py.
  - Price years are separated by position and axis label, not by colour,
    so colour continues to mean "component".
  - Zero-based y-axis.
  - A short dashed rule at half of each bar total makes the "larger half"
    claim readable without arithmetic.
"""
from __future__ import annotations
import sys
from pathlib import Path

for _d in Path(__file__).resolve().parents:
    if (_d / "thesis_style.py").exists():
        sys.path.insert(0, str(_d))
        break
else:
    raise FileNotFoundError("thesis_style.py not found in any parent folder")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from thesis_style import apply_thesis_style, figsize, FS_ANNOT
P = apply_thesis_style(palette="brand", usetex=False)

# ===========================================================================
# CONFIGURATION
# ===========================================================================
HERE     = Path(__file__).parent
DATA_DIR = HERE / "Results" / "RTE Tests"
OUT_DIR  = HERE / "Results" / "RTE Tests" / "Degradation Plots"
STEM     = "fig_fd_split_slide"

C_CYCLE, C_CAL, C_NEU = P["fill_a"], P["fill_b"], P["neutral"]

YEARS = [("dk2019", "DK1 2019"), ("dk2022", "DK1 2022")]
BAR_W = 0.46
FS_BIG = 11          # in-bar share labels
FS_TOT = 9           # total above bar


def find_traj(year_tag: str) -> Path:
    hits = sorted(DATA_DIR.glob(f"multiyear_trajectory_*{year_tag}*.csv"))
    if not hits:
        hits = sorted(HERE.rglob(f"multiyear_trajectory_*{year_tag}*.csv"))
    if not hits:
        raise FileNotFoundError(
            f"No 'multiyear_trajectory_*{year_tag}*.csv' in {DATA_DIR} "
            f"or under {HERE}. Set DATA_DIR to your results folder.")
    return hits[-1]


def load_means(path: Path):
    """Lifetime mean of the annual cycle and calendar components."""
    df = pd.read_csv(path).sort_values("year")
    c = float(df["fd_cycle"].mean())
    k = float(df["fd_calendar"].mean())
    return c, k, len(df)


def main():
    data = []
    for tag, label in YEARS:
        c, k, n = load_means(find_traj(tag))
        data.append((label, c, k, n))

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.52))

    x = np.arange(len(data), dtype=float)
    cyc = np.array([d[1] for d in data])
    cal = np.array([d[2] for d in data])
    tot = cyc + cal

    ax.bar(x, cyc, BAR_W, color=C_CYCLE, edgecolor="white",
           linewidth=0.6, zorder=2, label="Cycle")
    ax.bar(x, cal, BAR_W, bottom=cyc, color=C_CAL, edgecolor="white",
           linewidth=0.6, zorder=2, label="Calendar")

    # in-bar share labels
    for xi, c, k, t in zip(x, cyc, cal, tot):
        ax.text(xi, c / 2, f"Cycle\n{100*c/t:.0f}%", ha="center",
                va="center", color="white", fontsize=FS_BIG,
                fontweight="bold", zorder=4)
        ax.text(xi, c + k / 2, f"Calendar\n{100*k/t:.0f}%", ha="center",
                va="center", color="white", fontsize=FS_BIG,
                fontweight="bold", zorder=4)
        # half-of-total rule
        ax.plot([xi - BAR_W / 2 - 0.04, xi + BAR_W / 2 + 0.04], [t / 2, t / 2],
                ls=(0, (3, 2)), lw=1.1, color=C_NEU, zorder=5)
        ax.text(xi + BAR_W / 2 + 0.07, t / 2, "half", ha="left", va="center",
                fontsize=FS_ANNOT, color=C_NEU, zorder=5)
        ax.text(xi, t * 1.03, f"{t:.4f} / yr", ha="center", va="bottom",
                fontsize=FS_TOT, color=C_NEU, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in data], fontsize=FS_BIG)
    ax.set_xlim(-0.62, len(data) - 0.38)
    ax.set_ylim(0, tot.max() * 1.16)
    ax.set_ylabel("Lifetime mean annual $f_d$  (-)")
    ax.tick_params(axis="x", length=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{STEM}.pdf")
    fig.savefig(OUT_DIR / f"{STEM}.png", dpi=300)
    plt.close(fig)

    print("-- caption values --")
    for (label, c, k, n) in data:
        t = c + k
        print(f"  {label}: {n} yr mean | cycle {c:.5f} ({100*c/t:.1f}%) | "
              f"calendar {k:.5f} ({100*k/t:.1f}%) | total {t:.5f}")
    print(f"saved: {OUT_DIR / (STEM + '.pdf')}  (+ .png)")


if __name__ == "__main__":
    main()
