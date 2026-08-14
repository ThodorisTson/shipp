r"""
plot_window_levers.py
=====================
The two SoC window sweeps on one set of axes, so the two levers can be
compared rather than only inspected one at a time.

  x   annual degradation rate f_d in year 1, Xu branch, percent per year
  y   lifetime NPV over the 20-year horizon, MEUR

Both series share the 10--90% window, which is the single point where the
two lines meet. The direction each line travels is the finding:

  width series    up and to the right, a trade of wear for revenue
  center series   up and to the left, less wear at no revenue cost

The shared y axis is the point of the figure. The width series spans
73.1 MEUR and the center series 3.55 MEUR; on independent axes the two
look comparable, which they are not. The inset magnifies the center
series so its ordering stays readable.

Style is taken from thesis_style if it is importable, otherwise from the
inline TU Delft constants below, so the script runs unchanged on Windows
in VS Code with only matplotlib, numpy and pandas.

Reads v54_dodsweep_*_E*_P*.csv from the folder containing this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

# --- style ----------------------------------------------------------------
_styled = False
for _d in [HERE, *HERE.resolve().parents]:
    if (_d / "thesis_style.py").exists():
        sys.path.insert(0, str(_d))
        from thesis_style import (apply_thesis_style, TUDELFT,
                                  FS_LABEL, FS_LEGEND, FS_ANNOT)
        pal = apply_thesis_style(palette="brand", usetex=False)
        NAVY, DARKRED = TUDELFT["navy"], TUDELFT["darkred"]
        BLUE = TUDELFT.get("blue", "#0076C2")
        GRID, NEUTRAL = pal["grid"], pal["neutral"]
        _styled = True
        break

if not _styled:
    NAVY, DARKRED, BLUE = "#0C2340", "#A50034", "#0076C2"
    GRID, NEUTRAL = "#D9D9D9", "#6E6E6E"
    FS_LABEL, FS_LEGEND, FS_ANNOT = 12, 10, 9
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Carlito", "Calibri", "DejaVu Sans"],
        "font.size": 11,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.9, "savefig.dpi": 300, "figure.dpi": 120,
    })

# --- configuration --------------------------------------------------------
CSV_GLOB = "v54_dodsweep_*_E*_P*.csv"
OUT_STEM = "fig_window_levers"
BASE_WIDTH, BASE_CENTER = 0.80, 0.50
SHOW_INSET = False
X_TICK_STEP = 0.1


def find_csv() -> Path:
    hits = sorted(HERE.glob(CSV_GLOB)) or sorted(HERE.rglob(CSV_GLOB))
    if not hits:
        raise FileNotFoundError(f"No {CSV_GLOB} under {HERE}")
    return hits[-1]


def window_label(center: float, width: float) -> str:
    lo = int(round((center - width / 2) * 100))
    hi = int(round((center + width / 2) * 100))
    return f"{lo}-{hi}\\%" if plt.rcParams.get("text.usetex") else f"{lo}-{hi}%"


def series(df: pd.DataFrame, kind: str):
    if kind == "width":
        d = df[df["series"] == "width"].sort_values("width")
        lab = [window_label(BASE_CENTER, v) for v in d["width"]]
    else:
        d = df[df["series"] == "center"].sort_values("center")
        lab = [window_label(v, BASE_WIDTH) for v in d["center"]]
    fd = 100.0 * d["fd_yr1_xu"].to_numpy()
    npv = d["npv_bat_multiyear_xu_MEUR"].to_numpy()
    return fd, npv, lab


def draw_series(ax, fd, npv, color, marker, label, lw=1.4, ms=5):
    ax.plot(fd, npv, lw=lw, marker=marker, ms=ms, color=color,
            label=label, zorder=3)


def main() -> None:
    csv = find_csv()
    df = pd.read_csv(csv)
    print(f"reading {csv.name}")

    fd_w, npv_w, lab_w = series(df, "width")
    fd_c, npv_c, lab_c = series(df, "center")

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)

    draw_series(ax, fd_w, npv_w, NAVY, "o", "Width series, center 0.50")
    draw_series(ax, fd_c, npv_c, BLUE, "s", "Center series, width 0.80")

    # shared point: the 10-90% window belongs to both series
    i_base = int(np.argmin(np.abs(npv_w - npv_c[len(npv_c) // 2])))
    ax.plot([fd_w[i_base]], [npv_w[i_base]], lw=0, marker="o", ms=11,
            markerfacecolor="none", markeredgecolor=DARKRED,
            markeredgewidth=1.3, zorder=5)

    for xx, yy, lb in zip(fd_w, npv_w, lab_w):
        off = (0, -18) if yy < npv_w.mean() else (0, 12)
        ax.annotate(lb, (xx, yy), textcoords="offset points", xytext=off,
                    ha="center", fontsize=FS_LEGEND, color=NAVY)

    for j, off in ((0, (-14, 11)), (len(fd_c) - 1, (6, -18))):
        ax.annotate(lab_c[j], (fd_c[j], npv_c[j]), textcoords="offset points",
                    xytext=off, ha="center", fontsize=FS_LEGEND, color=BLUE)

    ax.set_xlabel(r"Annual degradation rate $f_d$   (% per year)",
                  fontsize=FS_LABEL)
    ax.set_ylabel("Lifetime NPV   (MEUR)", fontsize=FS_LABEL)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.margins(x=0.14, y=0.16)
    lo = np.floor(min(fd_w.min(), fd_c.min()) / X_TICK_STEP) * X_TICK_STEP
    hi = np.ceil(max(fd_w.max(), fd_c.max()) / X_TICK_STEP) * X_TICK_STEP
    ax.set_xticks(np.arange(lo, hi + 1e-9, X_TICK_STEP))
    ax.legend(loc="upper left", frameon=False, fontsize=FS_LEGEND)

    span_w = npv_w.max() - npv_w.min()
    span_c = npv_c.max() - npv_c.min()
    # ax.annotate(f"NPV span: width {span_w:.1f} MEUR,\ncenter "
    #             f"{span_c:.2f} MEUR",
    #             xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
    #             va="bottom", fontsize=FS_ANNOT, color=NEUTRAL)

    if SHOW_INSET:
        axi = ax.inset_axes([0.55, 0.08, 0.42, 0.36], zorder=6)
        axi.set_facecolor("white")
        axi.patch.set_alpha(1.0)
        draw_series(axi, fd_c, npv_c, BLUE, "s", None, lw=1.2, ms=4)
        axi.plot([fd_c[2]], [npv_c[2]], lw=0, marker="s", ms=9,
                 markerfacecolor="none", markeredgecolor=DARKRED,
                 markeredgewidth=1.1, zorder=5)
        for xx, yy, lb in zip(fd_c, npv_c, lab_c):
            axi.annotate(lb, (xx, yy), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=FS_ANNOT - 1,
                         color=BLUE)
        axi.grid(True, color=GRID, lw=0.5, alpha=0.7)
        axi.set_axisbelow(True)
        axi.margins(x=0.20, y=0.34)
        axi.tick_params(labelsize=FS_ANNOT - 1)
        for s in ("top", "right"):
            axi.spines[s].set_visible(True)
        for s in axi.spines.values():
            s.set_linewidth(0.7)
            s.set_color(NEUTRAL)

    for ext in ("pdf", "png"):
        out = HERE / f"{OUT_STEM}.{ext}"
        fig.savefig(out)
        print(f"written: {out}")
    plt.close(fig)

    print(f"\nwidth  series: NPV {npv_w.min():.2f} to {npv_w.max():.2f} MEUR"
          f"  (span {span_w:.2f}), f_d {fd_w.min():.2f} to {fd_w.max():.2f} %/yr")
    print(f"center series: NPV {npv_c.min():.2f} to {npv_c.max():.2f} MEUR"
          f"  (span {span_c:.2f}), f_d {fd_c.min():.2f} to {fd_c.max():.2f} %/yr")
    print(f"NPV span ratio {span_w / span_c:.1f}x, "
          f"f_d span ratio "
          f"{(fd_w.max()-fd_w.min())/(fd_c.max()-fd_c.min()):.1f}x")


if __name__ == "__main__":
    main()
