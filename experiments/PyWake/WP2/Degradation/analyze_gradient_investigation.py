"""
analyze_gradient_investigation.py
==================================
Standalone diagnostic script for the inner-loop gradient analysis.

Reads the two key CSVs produced by run_battery_xu_shi_degradation_v2.py:
  - multiyear_trajectory_*.csv   (one row per year, scalar metrics)
  - gradient_timeseries_*.csv    (one row per timestep per year, subgrad arrays)

Produces five individual figures plus a Pearson correlation table printed
to the console.

FILE SELECTION
--------------
Two modes, set via USE_SPECIFIC_FILES below:

  USE_SPECIFIC_FILES = False  (default)
      Reads the most recently modified pair of CSVs from RESULTS_DIR.

  USE_SPECIFIC_FILES = True
      Reads the exact files named in MULTIYEAR_CSV and GRADIENT_CSV.

OUTPUTS
-------
Five PNG files saved to PLOTS_DIR:
  fig1_gradient_per_year.png
  fig2_component_decomposition.png
  fig3_mean_abs_subgrad.png
  fig4_correlation_bar.png
  fig5_alignment_vs_gradient.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats  # for Pearson r + p-value
import datetime
_TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================================================================
# CONFIG,  edit here only
# =============================================================================

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "Results"
PLOTS_DIR   = SCRIPT_DIR / "Degradation Plots"

# ── File selection ────────────────────────────────────────────────────────────
USE_SPECIFIC_FILES = False     # True → use the two paths below; False → auto-detect latest

MULTIYEAR_CSV = RESULTS_DIR / "multiyear_trajectory_20260331_222010_dk2022_150mw_300mwh_v2.csv"
GRADIENT_CSV  = RESULTS_DIR / "gradient_timeseries_20260331_222010_dk2022_150mw_300mwh_v2.csv"

# ── Plot style ────────────────────────────────────────────────────────────────
C_HIGH      = '#e74c3c'   # high-alignment regime colour
C_LOW       = '#4C72B0'   # low-alignment regime colour
ALIGNMENT_THRESHOLD = 3000   # boundary between the two regimes

DPI = 200

# =============================================================================
# File loading helpers
# =============================================================================

def _latest(pattern: str) -> Path:
    """Return the most recently modified file matching pattern in RESULTS_DIR."""
    candidates = sorted(RESULTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No file matching '{pattern}' found in {RESULTS_DIR}.\n"
            f"Run run_battery_xu_shi_degradation_v2.py first, or set USE_SPECIFIC_FILES=True."
        )
    return candidates[-1]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and return (multiyear_df, gradient_timeseries_df)."""
    if USE_SPECIFIC_FILES:
        mt_path = MULTIYEAR_CSV
        gt_path = GRADIENT_CSV
    else:
        mt_path = _latest("multiyear_trajectory_*.csv")
        gt_path = _latest("gradient_timeseries_*.csv")

    print(f"  Multiyear CSV : {mt_path.name}")
    print(f"  Gradient CSV  : {gt_path.name}")

    mt = pd.read_csv(mt_path)
    gt = pd.read_csv(gt_path)
    return mt, gt


# =============================================================================
# Data preparation
# =============================================================================

def prepare(mt: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Build the analysis dataframe,  first battery lifetime only."""
    # Identify replacement years so we can isolate the first lifetime
    repl_years = mt.loc[mt['replacement_this_year'] == True, 'year'].tolist()
    if repl_years:
        cutoff = repl_years[0]          # year of first replacement
    else:
        cutoff = mt['year'].max()       # no replacement,  use all years

    d = mt[mt['year'] <= cutoff].copy()

    # Derived columns
    d['product_3'] = d['e_cap_eff'] * d['mean_abs_subgrad'] * d['mean_abs_dual']
    d['alignment'] = d['dDeg_dDoD'] / d['product_3']
    d['regime']    = d['alignment'].apply(
        lambda a: 'High-alignment' if a >= ALIGNMENT_THRESHOLD else 'Low-alignment'
    )
    d['color']     = d['regime'].apply(
        lambda r: C_HIGH if r == 'High-alignment' else C_LOW
    )

    # Non-zero active timesteps per year from the timeseries CSV
    gt_cut = gt[gt['year'] <= cutoff]
    nz = (
        gt_cut.groupby('year')['subgrad_combined']
        .apply(lambda x: int(np.sum(x != 0)))
        .reset_index(name='n_nonzero')
    )
    d = d.merge(nz, on='year', how='left')

    return d


# =============================================================================
# Pearson correlation table
# =============================================================================

def print_correlations(d: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PEARSON CORRELATION  (n = {:d} years, first battery lifetime)".format(len(d)))
    print("=" * 70)
    print(f"  {'Component':<30}  {'r':>7}  {'p-value':>10}  {'interpretation'}")
    print("  " + "-" * 64)

    components = [
        ('e_cap_eff',         'e_cap_eff'),
        ('mean_abs_subgrad',  'mean |subgrad|'),
        ('mean_abs_dual',     'mean |dual price|'),
        ('mean_dod',          'mean DoD'),
        ('product_3',         'e_cap × |subgrad| × |dual|'),
        ('alignment',         'alignment factor (dot residual)'),
    ]

    for col, label in components:
        if col not in d.columns:
            print(f"  {label:<30}  (column not found,  skipping)")
            continue
        r, p = stats.pearsonr(d[col], d['dDeg_dDoD'])
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        print(f"  {label:<30}  {r:>+7.3f}  {p:>10.4f}  {stars}")

    print("  " + "-" * 64)
    print("  Significance: *** p<0.001  ** p<0.01  * p<0.05  n.s. not significant")
    print("=" * 70 + "\n")

    # Two-regime summary
    high = d[d['regime'] == 'High-alignment']
    low  = d[d['regime'] == 'Low-alignment']
    print("REGIME SUMMARY")
    print(f"  High-alignment years : {high['year'].tolist()}  "
          f"(mean alignment = {high['alignment'].mean():.0f})")
    print(f"  Low-alignment  years : {low['year'].tolist()}   "
          f"(mean alignment = {low['alignment'].mean():.0f})")
    print(f"  Regime ratio         : {high['alignment'].mean()/low['alignment'].mean():.2f}×\n")

    print("COMPONENT MEANS BY REGIME")
    for col, label in [('e_cap_eff','e_cap_eff'), ('mean_abs_subgrad','mean |subgrad|'),
                       ('mean_abs_dual','mean |dual|'), ('mean_dod','mean DoD')]:
        if col not in d.columns: continue
        print(f"  {label:<22}  high={high[col].mean():.4f}   low={low[col].mean():.4f}   "
              f"ratio={high[col].mean()/low[col].mean():.3f}")
    print()


# =============================================================================
# Plot helpers
# =============================================================================

def _style_ax(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25)


def _save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / f"{_TS}_{name}"
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓  {path.name}")


def _regime_legend(ax: plt.Axes) -> None:
    ax.legend(
        handles=[
            Patch(color=C_HIGH, alpha=0.85, label='High-alignment regime'),
            Patch(color=C_LOW,  alpha=0.85, label='Low-alignment regime'),
        ],
        fontsize=8,
    )


# =============================================================================
# Figure 1,  dDeg/dDoD per year, regime-coloured
# =============================================================================

def fig1_gradient_per_year(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _style_ax(ax)

    ax.bar(d['year'], d['dDeg_dDoD'] * 1e-6, color=d['color'],
           alpha=0.85, width=0.7)

    mean_g = d['dDeg_dDoD'].mean() * 1e-6
    ax.axhline(mean_g, linestyle='--', linewidth=1.2, color='#2c3e50',
               alpha=0.8, label=f'Mean: {mean_g:.3f} ×10⁶')

    handles = [
        Patch(color=C_HIGH, alpha=0.85, label='High-alignment regime'),
        Patch(color=C_LOW,  alpha=0.85, label='Low-alignment regime'),
        plt.Line2D([0],[0], linestyle='--', color='#2c3e50',
                   label=f'Mean: {mean_g:.3f} ×10⁶'),
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel('Project year', fontsize=10)
    ax.set_ylabel('dDeg/dDoD  [×10⁶ USD/MWh]', fontsize=10)
    ax.set_title('Inner-Loop Gradient dDeg/dDoD per Year\n'
                 '(oscillates between two regimes,  neither SoH nor DoD explains the pattern)',
                 fontsize=10, pad=10)
    ax.set_xticks(d['year'])
    ax.grid(True, alpha=0.25, axis='y')

    _save(fig, 'fig1_gradient_per_year.png')

# =============================================================================
# Figure 2,  Three-component decomposition
# =============================================================================

def fig2_component_decomposition(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _style_ax(ax)

    d = d.copy()
    d['dot_product'] = d['dDeg_dDoD'] / (-d['e_cap_eff'])

    yr1     = d[d['year'] == d['year'].min()].iloc[0]
    n_ecap  = d['e_cap_eff']      / yr1['e_cap_eff']
    n_mas   = d['mean_abs_subgrad'] / yr1['mean_abs_subgrad']
    n_dot   = d['dot_product']    / (yr1['dDeg_dDoD'] / (-yr1['e_cap_eff']))

    ax.plot(d['year'], n_dot,  'o-',  color='#2c3e50', linewidth=2.0,
            markersize=6, zorder=4,
            label='dot(subgrad, dual)  [drives oscillation]')
    ax.plot(d['year'], n_ecap, 's--', color='#3498db', linewidth=1.3,
            markersize=4, alpha=0.85,
            label='e_cap_eff  [smooth monotone decline]')
    ax.plot(d['year'], n_mas,  '^--', color='#8e44ad', linewidth=1.3,
            markersize=4, alpha=0.85,
            label='mean |subgrad|  [nearly flat]')

    ax.axhline(1.0, linestyle=':', linewidth=1.0, color='grey', alpha=0.5)

    # remove the axvspan loop and replace with:
    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    bar_colors = [C_HIGH if r == 'High-alignment' else C_LOW for r in d['regime']]
    ax2.bar(d['year'], d['dDeg_dDoD'] * 1e-6, color=bar_colors,
            alpha=0.15, width=0.7, zorder=1)
    ax2.set_ylabel('dDeg/dDoD  [×10⁶ USD/MWh]', fontsize=9, alpha=0.5)
    ax2.tick_params(axis='y', labelsize=8, labelcolor='grey')

    ax.set_xlabel('Project year', fontsize=10)
    ax.set_ylabel('Value / year-1 value  (normalised)', fontsize=10)
    ax.set_title('Gradient Decomposition:  dDeg/dDoD = −e_cap × dot(subgrad, dual)\n'
                 'e_cap and |subgrad| are smooth/flat,  dot product drives all oscillation',
                 fontsize=10, pad=10)
    ax.set_xticks(d['year'])
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.42, 0.99))
    ax.grid(True, alpha=0.25, axis='y')

    _save(fig, 'fig2_component_decomposition.png')


# =============================================================================
# Figure 3,  Mean |subgradient| bars + n_nonzero overlay
# =============================================================================

def fig3_mean_abs_subgrad(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _style_ax(ax)

    ax.bar(d['year'], d['mean_abs_subgrad'], color='#8e44ad', alpha=0.85, width=0.7)
    ax.set_xlabel('Project year', fontsize=10)
    ax.set_ylabel('mean |subgrad_combined|  [USD/MWh]', fontsize=10)

    threshold_mas = d['mean_abs_subgrad'].mean() * 0.92
    low_mas_years = d.loc[d['mean_abs_subgrad'] < threshold_mas, 'year'].tolist()
    ax.set_title(f'Mean |Subgradient| per Year\n'
                 f'Anomalously low years: {low_mas_years},  LP basis switch signature',
                 fontsize=10, pad=10)
    ax.set_xticks(d['year'])

    if 'n_nonzero' in d.columns:
        axr = ax.twinx()
        axr.plot(d['year'], d['n_nonzero'], 'o--',
                 color='#e67e22', linewidth=1.4, markersize=5,
                 label='Active timesteps (non-zero subgrad)')
        axr.set_ylabel('Active timesteps [n]', color='#e67e22', fontsize=9)
        axr.tick_params(axis='y', colors='#e67e22', labelsize=8)
        axr.set_ylim(3600, 5000)
        axr.legend(fontsize=8, loc='lower right')

    _save(fig, 'fig3_mean_abs_subgrad.png')

# =============================================================================
# Figure 4,  Step 4a: alignment factor by year
# =============================================================================

def fig4_alignment_by_year(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _style_ax(ax)

    ax.bar(d['year'], d['alignment'], color=d['color'], alpha=0.85, width=0.7)
    ax.axhline(ALIGNMENT_THRESHOLD, linestyle='--', color='#2c3e50',
               alpha=0.55, linewidth=1.3, label=f'Regime boundary = {ALIGNMENT_THRESHOLD}')

    high_mean = d.loc[d['regime'] == 'High-alignment', 'alignment'].mean()
    low_mean  = d.loc[d['regime'] == 'Low-alignment',  'alignment'].mean()
    ax.axhline(high_mean, linestyle=':', color=C_HIGH, alpha=0.7, linewidth=1.2,
               label=f'High-regime mean = {high_mean:.0f}')
    ax.axhline(low_mean,  linestyle=':', color=C_LOW,  alpha=0.7, linewidth=1.2,
               label=f'Low-regime  mean = {low_mean:.0f}')

    ax.set_xlabel('Project year', fontsize=10)
    ax.set_ylabel('Alignment factor  =  dDeg/dDoD / (e_cap × |subgrad| × |dual|)', fontsize=9)
    ax.set_title('Step 4a,  Alignment Factor per Year\n'
                 'Two clean regimes; all years 7–12 permanently in high-alignment regime',
                 fontsize=10, pad=10)
    ax.set_xticks(d['year'])
    ax.legend(fontsize=8)
    _regime_legend(ax)

    _save(fig, 'fig4_step4a_alignment_by_year.png')


# =============================================================================
# Figure 5,  Step 4b: correlation bar chart
# =============================================================================

def fig5_correlation_bar(d: pd.DataFrame) -> None:
    components = [
        ('mean_dod',          'mean DoD'),
        ('e_cap_eff',         'e_cap_eff'),
        ('mean_abs_subgrad',  'mean |subgrad|'),
        ('mean_abs_dual',     'mean |dual|'),
        ('product_3',         'e_cap × |subgrad| × |dual|\n(scalar product)'),
        ('alignment',         'alignment factor\n(dot residual)'),
    ]

    labels, rvals, pvals, bar_cols = [], [], [], []
    for col, lbl in components:
        if col not in d.columns:
            continue
        r, p = stats.pearsonr(d[col], d['dDeg_dDoD'])
        labels.append(lbl)
        rvals.append(r)
        pvals.append(p)
        if col == 'alignment':
            bar_cols.append('#f39c12')
        elif abs(r) > 0.6:
            bar_cols.append('#27ae60')
        else:
            bar_cols.append(C_LOW)

    fig, ax = plt.subplots(figsize=(8, 5))
    _style_ax(ax)

    bars = ax.barh(labels, rvals, color=bar_cols, alpha=0.85)
    ax.axvline(0, color='#2c3e50', linewidth=0.9, alpha=0.5)
    ax.axvline( 0.5, color='#27ae60', linewidth=0.8, alpha=0.3, linestyle='--')
    ax.axvline(-0.5, color='#27ae60', linewidth=0.8, alpha=0.3, linestyle='--')

    for i, (r, p) in enumerate(zip(rvals, pvals)):
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        offset = 0.025 if r >= 0 else -0.025
        ha     = 'left'  if r >= 0 else 'right'
        ax.text(r + offset, i, f'{r:+.3f}  {stars}',
                va='center', ha=ha, fontsize=8.5)

    ax.set_xlim(-0.65, 1.25)
    ax.set_xlabel('Pearson r  with  dDeg/dDoD', fontsize=10)
    ax.set_title(f'Step 4b,  Correlation of Each Component with dDeg/dDoD\n'
                 f'(n = {len(d)} years, first battery lifetime)',
                 fontsize=10, pad=10)

    note = 'Gold bar = alignment factor\nBlue bars = individual scalar components'
    ax.text(0.97, 0.03, note, transform=ax.transAxes,
            fontsize=7.5, color='#7f6000', ha='right', va='bottom',
            bbox=dict(facecolor='#fefde8', edgecolor='#f39c12', alpha=0.90, pad=4))

    _save(fig, 'fig4_correlation_bar.png')

# =============================================================================
# Figure 6,  Step 4c: alignment vs actual gradient (the money plot)
# =============================================================================

def fig6_alignment_vs_gradient(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    _style_ax(ax)

    sc = ax.scatter(d['alignment'], d['dDeg_dDoD'] * 1e-6,
                    c=d['year'], cmap='plasma', s=85, zorder=3,
                    edgecolors='white', linewidths=0.5)

    for _, r in d.iterrows():
        ax.annotate(str(int(r.year)), (r.alignment, r.dDeg_dDoD * 1e-6),
                    fontsize=7.5, color='#2c3e50',
                    xytext=(4, 3), textcoords='offset points')

    rval, pval = stats.pearsonr(d['alignment'], d['dDeg_dDoD'])
    m, b = np.polyfit(d['alignment'], d['dDeg_dDoD'] * 1e-6, 1)
    xs = np.linspace(d['alignment'].min(), d['alignment'].max(), 60)
    ax.plot(xs, m * xs + b, '--', color='#e67e22', alpha=0.85, linewidth=1.5,
            label=f'Pearson r = {rval:+.3f}  (p = {pval:.4f})')

    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Project year', fontsize=8)

    ax.set_xlabel('Alignment factor', fontsize=10)
    ax.set_ylabel('dDeg/dDoD  [×10⁶ USD/MWh]', fontsize=10)
    ax.set_title(f'Alignment Factor vs Actual Gradient\n'
                 f'Near-perfect linear relationship (r = {rval:+.3f})',
                 fontsize=10, pad=10)
    ax.legend(fontsize=8)

    _save(fig, 'fig5_alignment_vs_gradient.png')

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("GRADIENT INVESTIGATION,  Steps 2–4 Analysis")
    print("=" * 70)

    print("\n[1/3] Loading CSVs...")
    mt, gt = load_data()

    print("\n[2/3] Preparing analysis dataframe...")
    d = prepare(mt, gt)
    print(f"  Years in analysis : {d['year'].tolist()}")
    print(f"  Replacement years : "
          f"{mt.loc[mt['replacement_this_year']==True, 'year'].tolist()}")

    print_correlations(d)

    print("[3/3] Generating figures...")
    fig1_gradient_per_year(d)
    fig2_component_decomposition(d)
    fig3_mean_abs_subgrad(d)
    fig5_correlation_bar(d)
    fig6_alignment_vs_gradient(d)

    print(f"\n✓ All figures saved to: {PLOTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()