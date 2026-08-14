"""
plot_sweep_revenue_cost.py

Figure: discounted revenue, discounted cost, and the marginal quantities that
set the optimum, as functions of battery energy capacity at fixed power
capacity. DK1 2022 prices.

Reads the two Plan B lifetime sweep CSVs (coarse pass and refined pass) that
sit next to this script and reconstructs the revenue and cost components from
the NPV identity used in planB_2d_parameter_sweep_a_npv.py:

    NPV = -capex + sum_k rev_k * (1+r)^-k - sum_repl repl_cost * (1+r)^-k_repl

so that

    discounted revenue (no degradation) = npv_no_deg  + capex
    discounted revenue (Xu)             = npv_with_xu + capex + discounted replacement
    discounted replacement (Xu)         = n_repl * (repl_e * E + repl_p * P) * (1+r)^-eol_year

Cost conventions follow wp2_econ.capex and wp2_econ.replacement_cost
(energy component plus power component).

Outputs, written next to this script:
    fig_sweep_revenue_cost_p175.pdf / .png

Run from VS Code on Windows with the two CSVs in the same folder as this file.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE = Path(__file__).parent
RESULTS_DIR = HERE / "Plan B Results"

# Output folder for thesis figures (creates it if it doesn't exist)
FIGURES_DIR = HERE / "Thesis Figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CSV_COARSE = HERE / "Plan B Results" / "planB_lifetime_sweep_8760h_20yr_20260703_221934.csv"
CSV_REFINED = HERE / "Plan B Results" / "planB_lifetime_sweep_8760h_20yr_20260704_014612.csv"

P_FIXED_MW = 175.0          # power capacity of the section
E_OPT_MWH = 550.0           # Xu grid optimum, marked with a dotted line

DISCOUNT_RATE = 0.03
REPL_E_EUR_PER_MWH = 72.0e3     # DEA energy expansion cost, 72 EUR/kWh
REPL_P_EUR_PER_MW = 96.0e3      # DEA power expansion cost, 96 EUR/kW

# Panel (b) starts here. The 300 to 450 MWh interval is a 150 MWh step on the
# coarse grid, so its difference quotient is not comparable with the 50 MWh
# steps above 450 MWh and is left out of the slope panel.
SLOPE_E_MIN = 450.0

OUT_STEM = "fig_sweep_revenue_cost_p175"

# Fraction of \\textwidth the figure is included at in the .tex. MUST match the
# width= argument of \\includegraphics, otherwise LaTeX rescales the figure and
# every font size on the page changes with it.
INCLUDE_WIDTH_FRAC = 0.9
ASPECT = 1.00               # height / width, two stacked panels

# ----------------------------------------------------------------------------
# Style: thesis_style if importable, otherwise the inline equivalent
# ----------------------------------------------------------------------------
_styled = False
for _d in [HERE, *HERE.resolve().parents]:
    if (_d / "thesis_style.py").exists():
        sys.path.insert(0, str(_d))
        from thesis_style import (apply_thesis_style, figsize, TUDELFT,
                                  FS_LABEL, FS_ANNOT)
        pal = apply_thesis_style(palette="brand", usetex=False)
        NAVY, DARKRED = TUDELFT["navy"], TUDELFT["darkred"]
        GREY = pal["neutral"]
        _styled = True
        break

if not _styled:
    NAVY, DARKRED, GREY = "#0C2340", "#A50034", "#404040"
    FS_LABEL, FS_ANNOT = 9, 7

    def figsize(width_frac: float = 1.0, aspect: float = 0.60):
        return (6.201 * width_frac, 6.201 * width_frac * aspect)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "savefig.bbox": "standard", "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })

mpl.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.3,
    "lines.markersize": 4,
})


# ----------------------------------------------------------------------------
# Data assembly
# ----------------------------------------------------------------------------
def load_pooled(csv_coarse: Path, csv_refined: Path) -> pd.DataFrame:
    """Pool the two passes, with the refined pass taking precedence."""
    coarse = pd.read_csv(csv_coarse)
    refined = pd.read_csv(csv_refined)

    key = ["e_cap", "p_cap"]
    refined_index = refined.set_index(key).index
    coarse_only = coarse[~coarse.set_index(key).index.isin(refined_index)]

    pooled = pd.concat([refined, coarse_only], ignore_index=True)
    pooled = pooled.dropna(subset=["npv_with_xu"])
    return pooled.sort_values(["p_cap", "e_cap"]).reset_index(drop=True)


def add_components(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct discounted revenue and cost components, in MEUR."""
    out = df.copy()

    repl_cost_eur = (REPL_E_EUR_PER_MWH * out["e_cap"]
                     + REPL_P_EUR_PER_MW * out["p_cap"])
    discount = (1.0 + DISCOUNT_RATE) ** (-out["eol_year_xu"].astype(float))
    out["repl_disc"] = out["n_repl_xu"] * repl_cost_eur * discount / 1e6

    out["capex"] = out["capex_eur"] / 1e6
    out["npv_nodeg"] = out["npv_no_deg"] / 1e6
    out["npv_xu"] = out["npv_with_xu"] / 1e6

    out["rev_nodeg"] = out["npv_nodeg"] + out["capex"]
    out["rev_xu"] = out["npv_xu"] + out["capex"] + out["repl_disc"]
    out["cost_xu"] = out["capex"] + out["repl_disc"]
    out["deg_cost"] = out["npv_nodeg"] - out["npv_xu"]
    return out


def midpoint_slopes(x: np.ndarray, y: np.ndarray):
    """Forward-difference slopes and the midpoints they apply to."""
    xm = 0.5 * (x[:-1] + x[1:])
    return xm, np.diff(y) / np.diff(x)


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def make_figure(sec: pd.DataFrame) -> plt.Figure:
    e = sec["e_cap"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1,
                             figsize=figsize(INCLUDE_WIDTH_FRAC, ASPECT),
                             sharex=True, constrained_layout=True)

    # -- (a) revenue and cost levels ------------------------------------------
    ax = axes[0]
    ax.plot(e, sec["rev_nodeg"], color=NAVY, marker="o",
            label="Battery revenue, no degradation")
    ax.plot(e, sec["rev_xu"], color=NAVY, marker="o", linestyle="--",
            markerfacecolor="white", label="Battery revenue, Xu")
    ax.plot(e, sec["cost_xu"], color=DARKRED, marker="s",
            label="Capital and replacement cost, Xu")
    ax.plot(e, sec["capex"], color=DARKRED, marker="s", linestyle="--",
            markerfacecolor="white", label="Capital cost")
    ax.fill_between(e, sec["capex"], sec["cost_xu"], color=DARKRED,
                    alpha=0.12, linewidth=0)
    ax.set_ylabel("Present value  (MEUR)")
    ax.legend(loc="upper left", frameon=False)

    # -- (b) marginal quantities ----------------------------------------------
    ax = axes[1]
    mask = e >= SLOPE_E_MIN
    xm, s_nodeg = midpoint_slopes(e[mask], sec["npv_nodeg"].to_numpy(float)[mask])
    _, s_deg = midpoint_slopes(e[mask], sec["deg_cost"].to_numpy(float)[mask])

    ax.axhline(0.0, color=GREY, linewidth=0.7, linestyle="--")
    ax.plot(xm, s_nodeg, color=NAVY, marker="o",
            label="Lifetime NPV, no degradation")
    ax.plot(xm, s_deg, color=DARKRED, marker="s",
            label="Lifetime cost of degradation")
    ax.set_ylabel("Change per added MWh  (MEUR/MWh)")
    ax.set_xlabel("Energy capacity  E  (MWh)")
    ax.legend(loc="upper right", frameon=False)

    for ax, lab in zip(axes, ["(a)", "(b)"]):
        ax.axvline(E_OPT_MWH, color=DARKRED, linestyle=":", linewidth=1.0)
        ax.margins(x=0.02)
        ax.text(0.0, 1.015, lab, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=FS_LABEL)

    ax_b = axes[1]
    ax_b.text(E_OPT_MWH - 16,
              ax_b.get_ylim()[0] + 0.12 * np.ptp(ax_b.get_ylim()),
              f"E* = {E_OPT_MWH:.0f} MWh", color=DARKRED, fontsize=FS_ANNOT,
              ha="right", va="center")
    return fig


# ----------------------------------------------------------------------------
def main() -> None:
    pooled = add_components(load_pooled(CSV_COARSE, CSV_REFINED))
    sec = pooled[np.isclose(pooled["p_cap"], P_FIXED_MW)].sort_values("e_cap")
    if sec.empty:
        raise SystemExit(f"No rows at P = {P_FIXED_MW} MW")

    cols = ["e_cap", "eol_year_xu", "rev_nodeg", "rev_xu", "capex",
            "repl_disc", "cost_xu", "npv_nodeg", "npv_xu", "deg_cost"]
    print(sec[cols].round(2).to_string(index=False))

    fig = make_figure(sec)
    for ext in ("pdf", "png"):
        # Output directly to the Thesis Figures folder
        path = FIGURES_DIR / f"{OUT_STEM}.{ext}"
        fig.savefig(path)
        print(f"written: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
