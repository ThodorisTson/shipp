"""
multiyear_plots_midterm.py
==========================
Produces three standalone slide figures from the multiyear trajectory CSV:

  1. soh_trajectory_midterm.png     — SoH curve, two battery segments connected
  2. fd_pie_yr1_midterm.png         — pie chart for year 1 only
  3. fd_stacked_bar_midterm.png     — vertical stacked bar for all 20 years

Place this script in the same folder as the CSV.
Run:  python multiyear_plots_midterm.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
RESULTS_DIR = HERE.parent / 'Results'

# ── Toggle: which price dataset to plot ──────────────────────────────────────
PRICE_YEAR = '2022'   # <-- change to '2022' or '2019' to switch dataset
# ─────────────────────────────────────────────────────────────────────────────

_csv_candidates = sorted(
    RESULTS_DIR.glob(f'multiyear_trajectory_*dk{PRICE_YEAR}*.csv')
)
if not _csv_candidates:
    raise FileNotFoundError(
        f"No multiyear_trajectory_*dk{PRICE_YEAR}*.csv found in: {RESULTS_DIR}"
    )
CSV_PATH = _csv_candidates[-1]   # newest matching file
print(f"Reading CSV : {CSV_PATH.name}")

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = '#f7f9fc'
C_BAT1  = '#2166ac'    # battery 1 — dark blue
C_BAT2  = '#4a9edd'    # battery 2 — lighter blue
C_CONN  = '#aaaaaa'    # dotted connector between battery segments
C_80    = '#b5351b'
C_70    = '#E8873A'
C_REPL  = '#b5351b'
C_CYCLE = '#4a9edd'
C_CAL   = '#E8873A'
TXT     = '#0d1b2a'
GRID    = 'white'
EDGE    = '#cccccc'

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.facecolor':    BG,
    'figure.facecolor':  'white',
    'text.color':        TXT,
    'axes.edgecolor':    EDGE,
    'axes.labelcolor':   TXT,
    'xtick.color':       '#444444',
    'ytick.color':       '#444444',
    'grid.color':        GRID,
    'grid.linewidth':    0.8,
    'axes.grid':         True,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH).sort_values('year').reset_index(drop=True)

years    = df['year'].values
soh      = df['soh_pct'].values
fd_cycle = df['fd_cycle'].values
fd_cal   = df['fd_calendar'].values
repl_yrs = df.loc[df['replacement_this_year'] == True, 'year'].values

# Full SoH series: prepend year 0 = 100%
plot_years = np.concatenate([[0], years])
plot_soh   = np.concatenate([[100.0], soh])

# Segment break indices (SoH jumps up after replacement)
breaks     = [i for i in range(1, len(plot_soh)) if plot_soh[i] > plot_soh[i-1] + 5]
seg_edges  = [0] + breaks + [len(plot_soh)]
seg_colors = [C_BAT1, C_BAT2]   # extend for more replacements

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — SoH trajectory  (matches upper-left of degradation_plots_multiyear)
# ══════════════════════════════════════════════════════════════════════════════

# ── Segment split: same logic as _split_battery_segments() in original ────────
def _split_segments(years_full, soh_full, replacement_years):
    repl_set = set(replacement_years)
    segments, cur_x, cur_y = [], [], []
    for yr, soh in zip(years_full, soh_full):
        cur_x.append(yr); cur_y.append(soh)
        if yr in repl_set:
            segments.append((list(cur_x), list(cur_y)))
            cur_x, cur_y = [yr], [100.0]   # anchor fresh battery at same x
    if cur_x:
        segments.append((cur_x, cur_y))
    return segments

# ── Helper: linear crossing year ──────────────────────────────────────────────
def _crossing(soh_vals, yr_vals, thr):
    for i in range(len(soh_vals) - 1):
        if soh_vals[i] >= thr > soh_vals[i+1]:
            frac = (soh_vals[i] - thr) / (soh_vals[i] - soh_vals[i+1])
            return yr_vals[i] + frac * (yr_vals[i+1] - yr_vals[i])
    return None

C_LINE = '#4C72B0'   # single colour for all batteries
n_years = int(years.max())

fig1, ax = plt.subplots(figsize=(7.5, 4.8), facecolor='white')
fig1.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.13)

# Plot segments — all same colour
segments = _split_segments(list(plot_years), list(plot_soh), list(repl_yrs))
for sx, sy in segments:
    ax.plot(sx, sy, color=C_LINE, lw=2.0, marker='o', ms=3.5,
            zorder=3, label='_nolegend_')

# Solid connector between segment end and fresh-battery start (same colour)
for i in range(len(segments) - 1):
    x_conn = segments[i][0][-1]
    y_bot  = segments[i][1][-1]
    y_top  = segments[i+1][1][0]     # always 100.0
    ax.plot([x_conn, x_conn], [y_bot, y_top],
            color=C_LINE, lw=2.0, ls='-', alpha=1.0, zorder=4)

# Threshold lines + annotated crossing years
thr_cfg = {80.0: C_80, 70.0: C_70}
thr_labels = {80.0: '80%  [IEC/EV]', 70.0: '70%  [warranty]'}
crossing_handles = []
all_soh = list(plot_soh)
all_yr  = list(plot_years)
for thr, col in thr_cfg.items():
    ln = ax.axhline(thr, color=col, lw=1.2, ls='--', alpha=0.85)
    crossing_handles.append((ln, thr_labels[thr]))
    cross = _crossing(all_soh, all_yr, thr)
    if cross is not None and cross <= n_years:
        ax.axvline(cross, ls=':', lw=1.0, color=col, alpha=0.4, zorder=1)
        ha = 'right' if cross > n_years * 0.75 else 'left'
        offset = -1.0 if ha == 'right' else 1.0
        ax.annotate(
            f'{thr:.0f}%  yr {cross:.1f}',
            xy=(cross, thr), xytext=(cross + offset, thr + 2.0),
            fontsize=7.5, color=col, fontweight='bold', ha=ha,
            arrowprops=dict(arrowstyle='-', color=col, lw=0.6),
        )

# Replacement verticals — label at bottom
repl_handle = None
min_soh_val = float(np.min(plot_soh[1:]))   # exclude year-0 anchor
for ry in repl_yrs:
    repl_handle = ax.axvline(ry, color='red', lw=1.5, ls='-.', alpha=0.85, zorder=2)
    ax.text(ry - 0.3, min_soh_val - 3.5, f'Replace\nyr {ry}',
            fontsize=7, color='red', ha='right', va='top')

# Legend: single SoH entry + thresholds + replacement
leg_handles = [plt.Line2D([0], [0], color=C_LINE, lw=2.0, marker='o',
                           ms=4, label='Simulated SoH')]
for ln, lbl in crossing_handles:
    leg_handles.append(plt.Line2D([0], [0], ls='--', color=ln.get_color(),
                                   lw=1.2, label=lbl))
if repl_handle is not None:
    leg_handles.append(plt.Line2D([0], [0], ls='-.', color='red',
                                   lw=1.5, label='Battery replacement'))

ax.set_xlim(0, n_years)
ax.set_xticks(range(0, n_years + 1, 2))
ax.set_ylim(max(min_soh_val - 8, 50), 103)
ax.set_xlabel('Project year', fontsize=10)
ax.set_ylabel('State of Health  [%]', fontsize=10)
ax.set_title(
    f'State of Health Trajectory for a 20-year simulation  |  DK1 {PRICE_YEAR} prices',
    fontsize=11, fontweight='bold', pad=10, color=TXT)
ax.legend(handles=leg_handles, fontsize=8.5, framealpha=0.0,
          loc='lower left', handlelength=1.8, borderpad=0.3, labelspacing=0.35)
ax.grid(True, alpha=0.25)

out1 = HERE / f'soh_trajectory_midterm_{PRICE_YEAR}.png'
fig1.savefig(out1, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print(f"Saved → {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Pie: year 1 only
# ══════════════════════════════════════════════════════════════════════════════
yr1     = df[df['year'] == 1].iloc[0]
cy1     = yr1['fd_cycle']
cal1    = yr1['fd_calendar']
pct_cy  = cy1  / (cy1 + cal1) * 100
pct_cal = cal1 / (cy1 + cal1) * 100

fig2, ax2 = plt.subplots(figsize=(5, 4.5), facecolor='white')
fig2.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)

ax2.pie(
    [cy1, cal1],
    labels=[f'Cycle\n{pct_cy:.0f}%', f'Calendar\n{pct_cal:.0f}%'],
    colors=[C_CYCLE, C_CAL],
    startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2.5),
    textprops=dict(fontsize=13, color=TXT, fontweight='bold'),
    radius=0.88,
)
ax2.set_title(f'Year 1: fd decomposition\nCycle vs Calendar aging| DK1 {PRICE_YEAR} prices',
              fontsize=11, fontweight='bold', pad=14, color=TXT)

out2 = HERE / f'fd_pie_yr1_midterm_{PRICE_YEAR}.png'
fig2.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f"Saved → {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Stacked bar: all years
# ══════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(8, 4.8), facecolor='white')
fig3.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.13)

ax3.set_facecolor(BG)
ax3.grid(True, color=GRID, lw=0.8, axis='y', zorder=0)

# ── Bar percentage labels ─────────────────────────────────────────────────────
fd_total_arr = fd_cycle + fd_cal

bar_w = 0.65

ax3.bar(years, fd_cycle, width=bar_w, color=C_CYCLE, zorder=3,
        label='Cycle fd  (Shi Φ accumulation)')
ax3.bar(years, fd_cal, width=bar_w, bottom=fd_cycle, color=C_CAL, zorder=3,
        label='Calendar fd  (Xu ft_calendar)')
# Option A: label every bar — comment out if too busy
for i, yr in enumerate(years):
    pct_cyc = fd_cycle[i] / fd_total_arr[i] * 100
    pct_cal = fd_cal[i]   / fd_total_arr[i] * 100
    # Cycle label — midpoint of blue segment
    ax3.text(yr, fd_cycle[i] / 2,
             f'{pct_cyc:.0f}%', ha='center', va='center',
             fontsize=5.5, color='white', fontweight='bold', rotation=90)
    # Calendar label — midpoint of orange segment
    ax3.text(yr, fd_cycle[i] + fd_cal[i] / 2,
             f'{pct_cal:.0f}%', ha='center', va='center',
             fontsize=5.5, color='white', fontweight='bold', rotation=90)



# # Option C: mean percentages in legend — always safe, add to label strings above
# mean_cyc_pct = fd_cycle.mean() / fd_total_arr.mean() * 100
# mean_cal_pct = fd_cal.mean()   / fd_total_arr.mean() * 100

# ax3.bar(years, fd_cycle, width=bar_w, color=C_CYCLE, zorder=3,
#         label=f'Cycle fd  (Shi Φ accumulation)  —  {mean_cyc_pct:.0f}% avg')
# ax3.bar(years, fd_cal, width=bar_w, bottom=fd_cycle, color=C_CAL, zorder=3,
#         label=f'Calendar fd  (Xu ft_calendar)  —  {mean_cal_pct:.0f}% avg')

for ry in repl_yrs:
    ax3.axvline(ry, color=C_REPL, lw=1.4, ls='-.', alpha=0.80, zorder=4)

ax3.set_xlabel('Project year', fontsize=10)
ax3.set_ylabel('Annual  fd  contribution', fontsize=10)
ax3.set_title(rf'$f_d$ Decomposition: Cycle vs Calendar (20 yr) | DK1 {PRICE_YEAR} prices',
              fontsize=11, fontweight='bold', pad=10, color=TXT)
ax3.set_xlim(years.min() - 0.8, years.max() + 0.8)
ax3.set_xticks(years)
ax3.set_ylim(0, (fd_cycle + fd_cal).max() * 1.15)   # 15% headroom for legend
ax3.legend(fontsize=9, framealpha=0.0, loc='upper right',
           handlelength=1.6, borderpad=0.3, labelspacing=0.3)

out3 = HERE / f'fd_stacked_bar_midterm_{PRICE_YEAR}.png'
fig3.savefig(out3, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig3)
print(f"Saved → {out3}")