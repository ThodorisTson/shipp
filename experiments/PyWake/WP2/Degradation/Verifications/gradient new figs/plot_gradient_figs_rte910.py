r"""
plot_gradient_figs_rte910.py

Regenerates the three gradient-subsection figures from the DEA-corrected 2022
run (rte_ac = 0.91), in the style of the original plot_gradient_figs.py:

    fig44_gradient_bars     normalized g_k as bars, coloured by alignment regime
    fig45_gradient_lines    the two decomposition terms as lines
    fig46_gradient_overlay  fig44 bars (faded) behind fig45 lines

CAUTION -- read before using in the thesis
------------------------------------------
g_k is taken from annual_gradient[0], which is the CSV column
`dDeg_dDoD_DEPRECATED`. In this run that column oscillates ~2.9x between
adjacent years, while the gradient columns that actually enter the NPV
(`dNPV_dEcap`, `dDegCost_dEcap`) are smooth and monotonic. Confirm that
annual_gradient[0] is the intended g_k = -E_k <s_k, lambda_k> before relying
on the oscillation these figures show. This script does NOT change the plotted
quantity; it reproduces exactly what the original script plotted.

Decomposition (definitional):
    g_k = magnitude x alignment
    magnitude = E_k |s_bar_k| |lambda_bar_k|          (annual_gradient[4,1,3])
    alignment = g_k / magnitude

Reproducible on Windows / VS Code: numpy + matplotlib only, DejaVu Sans via
thesis_style, all paths anchored to this file's folder.
"""
from __future__ import annotations
import sys, glob
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).parent
for _d in [HERE, *HERE.parents]:
    if (_d / "thesis_style.py").exists():
        sys.path.insert(0, str(_d)); break
else:
    raise FileNotFoundError("thesis_style.py not found next to this script.")
from thesis_style import apply_thesis_style, figsize, TUDELFT

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
YEAR_TAG = "dk2022"
PRICE_TAG = "dk1_2022"
NPY_GLOB  = f"multiyear_*{YEAR_TAG}*.npy"        # newest match is used

C_HIGH  = TUDELFT["darkred"]   # above-mean alignment
C_LOW   = TUDELFT["blue"]      # below-mean alignment
C_ALIGN = TUDELFT["navy"]      # alignment term line
C_MAG   = TUDELFT["orange"]    # magnitude term line
C_REF   = "#404040"            # 1.0 / mean reference

OVERLAY_BAR_ALPHA = 0.18
OVERLAY_BAR_WIDTH = 0.82
BARS_BAR_WIDTH    = 0.72

PALETTE = apply_thesis_style(palette="brand")


def _find_npy() -> Path:
    hits = sorted(glob.glob(str(HERE / NPY_GLOB)))
    if not hits:
        raise FileNotFoundError(f"no file matching {NPY_GLOB} in {HERE}")
    return Path(hits[-1])


def load_decomposition(npy_file: Path) -> dict:
    d  = np.load(npy_file, allow_pickle=True).item()
    # first battery generation = years up to (and including) the first replacement
    repl = d.get("replacement_years", [])
    n_gen = int(repl[0]) if repl else len(d["annual_gradient"])
    ag = d["annual_gradient"][:n_gen]
    if any(g is None for g in ag):
        raise ValueError("Gradient tuple missing for a year in the first generation.")

    g0   = np.array([g[0] for g in ag])   # dDeg_dDoD_DEPRECATED  (see CAUTION)
    s_ab = np.array([g[1] for g in ag])   # mean |subgradient|
    l_ab = np.array([g[3] for g in ag])   # mean |dual|
    ecap = np.array([g[4] for g in ag])   # effective capacity

    magnitude = ecap * s_ab * l_ab
    alignment = g0 / magnitude
    nrm = lambda x: x / np.mean(x)
    out = {
        "years":   np.arange(1, n_gen + 1),
        "g0_n":    nrm(g0),
        "mag_n":   nrm(magnitude),
        "align_n": nrm(alignment),
        "r_align": float(np.corrcoef(nrm(g0), nrm(alignment))[0, 1]),
        "r_mag":   float(np.corrcoef(nrm(g0), nrm(magnitude))[0, 1]),
        "mean_g0": float(np.mean(g0)),
        "swing":   float(g0.max() / g0.min()),
    }
    out["above_mean"] = out["align_n"] >= 1.0
    return out


def _bar_colours(above_mean):
    return [C_HIGH if hi else C_LOW for hi in above_mean]


def _finish(ax, n_gen, ymax):
    ax.set_xlabel("Project year")
    ax.set_ylabel("Value / annual mean")
    ax.set_xticks(np.arange(1, n_gen + 1))
    ax.set_xlim(0.4, n_gen + 0.6)
    ax.set_ylim(0.0, ymax)


def _save(fig, out_dir, stem):
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"saved: {stem}.pdf / .png")


def plot_gradient_bars(dec, out_dir):
    y, ng = dec["years"], dec["g0_n"]
    fig, ax = plt.subplots(figsize=figsize(0.95, 0.56))
    ax.bar(y, ng, width=BARS_BAR_WIDTH, color=_bar_colours(dec["above_mean"]),
           edgecolor="none", zorder=1)
    ax.axhline(1.0, color=C_REF, linestyle=":", linewidth=0.8, zorder=2)
    _finish(ax, len(y), ng.max() * 1.15)
    handles = [
        Patch(facecolor=C_HIGH, edgecolor="none", label="Above-mean alignment"),
        Patch(facecolor=C_LOW,  edgecolor="none", label="Below-mean alignment"),
        Line2D([0], [0], color=C_REF, linestyle=":", label="Annual mean"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7)
    _save(fig, out_dir, f"fig44_gradient_bars_{PRICE_TAG}")


def plot_gradient_lines(dec, out_dir):
    y = dec["years"]
    fig, ax = plt.subplots(figsize=figsize(0.95, 0.56))
    ax.axhline(1.0, color=C_REF, linestyle=":", linewidth=0.8, zorder=1)
    ax.plot(y, dec["mag_n"], marker="s", linestyle="--", color=C_MAG,
            linewidth=1.3, markersize=4.0, zorder=2,
            label=f"Magnitude term  (r = {dec['r_mag']:+.2f})")
    ax.plot(y, dec["align_n"], marker="o", linestyle="-", color=C_ALIGN,
            linewidth=1.7, markersize=4.5, zorder=3,
            label=f"Alignment term  (r = {dec['r_align']:+.2f})")
    _finish(ax, len(y), max(dec["align_n"].max(), dec["mag_n"].max()) * 1.15)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _save(fig, out_dir, f"fig45_gradient_lines_{PRICE_TAG}")


def plot_gradient_overlay(dec, out_dir):
    y, ng = dec["years"], dec["g0_n"]
    fig, ax = plt.subplots(figsize=figsize(0.95, 0.56))
    ax.bar(y, ng, width=OVERLAY_BAR_WIDTH, color=_bar_colours(dec["above_mean"]),
           alpha=OVERLAY_BAR_ALPHA, edgecolor="none", zorder=1)
    ax.axhline(1.0, color=C_REF, linestyle=":", linewidth=0.8, zorder=2)
    ax.plot(y, dec["mag_n"], marker="s", linestyle="--", color=C_MAG,
            linewidth=1.3, markersize=4.0, zorder=3,
            label=f"Magnitude term  (r = {dec['r_mag']:+.2f})")
    ax.plot(y, dec["align_n"], marker="o", linestyle="-", color=C_ALIGN,
            linewidth=1.7, markersize=4.5, zorder=4,
            label=f"Alignment term  (r = {dec['r_align']:+.2f})")
    _finish(ax, len(y), max(ng.max(), dec["align_n"].max()) * 1.15)
    handles = [
        Line2D([0], [0], color=C_ALIGN, marker="o", linestyle="-", linewidth=1.7,
               markersize=4.5, label=f"Alignment term  (r = {dec['r_align']:+.2f})"),
        Line2D([0], [0], color=C_MAG, marker="s", linestyle="--", linewidth=1.3,
               markersize=4.0, label=f"Magnitude term  (r = {dec['r_mag']:+.2f})"),
        Patch(facecolor=C_HIGH, alpha=OVERLAY_BAR_ALPHA, edgecolor="none",
              label="$g_k$, above-mean alignment"),
        Patch(facecolor=C_LOW, alpha=OVERLAY_BAR_ALPHA, edgecolor="none",
              label="$g_k$, below-mean alignment"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7,
              handlelength=1.8)
    _save(fig, out_dir, f"fig46_gradient_overlay_{PRICE_TAG}")


def main():
    npy = _find_npy()
    print(f"loaded: {npy.name}")
    dec = load_decomposition(npy)
    print(f"first generation years : 1..{len(dec['years'])}")
    print(f"mean g_k               : {dec['mean_g0']:.4e}")
    print(f"g_k swing (max/min)    : {dec['swing']:.2f}x")
    print(f"corr(g_k, alignment)   : {dec['r_align']:+.3f}")
    print(f"corr(g_k, magnitude)   : {dec['r_mag']:+.3f}")
    below = [int(y) for y, hi in zip(dec["years"], dec["above_mean"]) if not hi]
    print(f"below-mean-alignment yr: {below}")

    out_dir = HERE / "gradient_figs"
    out_dir.mkdir(exist_ok=True)
    plot_gradient_bars(dec, out_dir)
    plot_gradient_lines(dec, out_dir)
    plot_gradient_overlay(dec, out_dir)


if __name__ == "__main__":
    main()
