import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

# ── Constants ─────────────────────────────────────────────────────────────────
RHO, D = 1.225, 125.88009368
A_SWEPT = np.pi / 4 * D**2
RATED_POWER = 5_000_000.0
CUT_IN, CUT_OUT = 3.0, 25.0
N_TURBINES = 65
H_REF, H_HUB, ALPHA = 86.0, 90.0, 0.20

cp_ws  = np.array([3,4,5,6,7,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8,9,10,
                   10.1,10.2,10.3,10.4,10.5,10.6,10.7,10.8,10.9,11,11.1,
                   11.2,11.3,11.4,11.5,11.6,11.7,11.8,11.9,12,13,14,15,16,
                   17,18,19,20,21,22,23,24,25])
cp_val = np.array([0.208546508,0.385795061,0.449038264,0.474546985,0.480994449,
                   0.481172749,0.481235678,0.481305875,0.481238912,0.481167356,
                   0.481081935,0.481007003,0.480880409,0.480789285,0.480737341,
                   0.480111543,0.479218839,0.479120347,0.479022984,0.478834971,
                   0.478597234,0.478324162,0.477994289,0.477665338,0.477253698,
                   0.476819542,0.476368667,0.475896732,0.475404347,0.474814698,
                   0.469087611,0.456886723,0.445156758,0.433837552,0.422902868,
                   0.412332387,0.402110045,0.316270768,0.253224057,0.205881042,
                   0.169640239,0.141430529,0.119144335,0.101304591,0.086856409,
                   0.075029591,0.065256635,0.057109143,0.050263779,0.044470536])

def ws_to_farm_power_mw(ws_ref):
    ws_hub = ws_ref * (H_HUB / H_REF) ** ALPHA
    cp = np.interp(ws_hub, cp_ws, cp_val, left=0.0, right=cp_val[-1])
    p = cp * 0.5 * RHO * A_SWEPT * ws_hub**3
    p = np.minimum(p, RATED_POWER)
    p = np.where((ws_hub < CUT_IN) | (ws_hub > CUT_OUT), 0.0, p)
    return p * N_TURBINES / 1e6

# ── Load data ─────────────────────────────────────────────────────────────────
prices = pd.read_csv('/mnt/user-data/uploads/dk1_prices_2022.csv',
                     parse_dates=['timestamp']).set_index('timestamp').sort_index()
jan_p = prices['2022-01-01':'2022-01-31 23:00']

with open('/mnt/user-data/uploads/wind_resource_2022hourly_referenceHPP.yaml') as f:
    wr = yaml.safe_load(f)['wind_resource']
times  = pd.to_datetime(wr['time'], utc=True).tz_localize(None)
ws_arr = np.array(list(wr['wind_speed']))
wind_df = pd.DataFrame({'power_mw': ws_to_farm_power_mw(ws_arr)}, index=times)
jan_w = wind_df['2022-01-01':'2022-01-31 23:00']

t   = jan_p.index
prc = jan_p['price_eur_mwh'].values
pwr = jan_w['power_mw'].values
rated_farm = N_TURBINES * 5  # 325 MW

# ── Identify alignment/mismatch regions ───────────────────────────────────────
# Normalise both signals to [0,1] for comparison
prc_n = (prc - prc.min()) / (prc.max() - prc.min())
pwr_n = (pwr - pwr.min()) / (pwr.max() - pwr.min())
# "Moving together" = both above 0.5 or both below 0.5
aligned = ((prc_n > 0.5) & (pwr_n > 0.5)) | ((prc_n < 0.5) & (pwr_n < 0.5))

# ── Figure ────────────────────────────────────────────────────────────────────
PRICE_COL = '#c0392b'
WIND_COL  = '#2166ac'

fig, ax1 = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor('white')
ax1.set_facecolor('#f7f9fc')

# Shade brief alignment windows (Jan 14–15 region visible to committee)
# Use a subtle highlight so mismatch dominates visually
in_align = False
for i, aln in enumerate(aligned):
    if aln and not in_align:
        start_i = i
        in_align = True
    elif not aln and in_align:
        ax1.axvspan(t[start_i], t[i], color='#f4a261', alpha=0.18, zorder=0)
        in_align = False
if in_align:
    ax1.axvspan(t[start_i], t[-1], color='#f4a261', alpha=0.18, zorder=0)

ax1.grid(True, which='major', color='white', lw=1.0, zorder=0)
ax1.grid(True, which='minor', color='white', lw=0.4, zorder=0)

# ── Right y-axis: wind power ───────────────────────────────────────────────
ax2 = ax1.twinx()

# Wind fill + line
ax2.fill_between(t, pwr, alpha=0.18, color=WIND_COL, zorder=1)
ax2.plot(t, pwr, color=WIND_COL, lw=1.2, zorder=2, label='Wind power')
ax2.axhline(rated_farm, color=WIND_COL, lw=1.0, ls='--', alpha=0.5, zorder=2)
ax2.set_ylabel('Wind Farm Power (MW)', fontsize=12, color=WIND_COL)
ax2.tick_params(axis='y', labelcolor=WIND_COL, labelsize=10)
ax2.set_ylim(0, rated_farm * 1.12)
ax2.text(t[-1], rated_farm + 4, '325 MW', color=WIND_COL, fontsize=8,
         ha='right', va='bottom', alpha=0.7)

# ── Left y-axis: price ────────────────────────────────────────────────────
ax1.fill_between(t, prc, alpha=0.18, color=PRICE_COL, zorder=1)
ax1.plot(t, prc, color=PRICE_COL, lw=1.2, zorder=3, label='Electricity price')
ax1.set_ylabel('DK1 Day-Ahead Price (EUR MWh$^{-1}$)', fontsize=12, color=PRICE_COL)
ax1.tick_params(axis='y', labelcolor=PRICE_COL, labelsize=10)
ax1.set_ylim(0, prc.max() * 1.12)

# ── x-axis ────────────────────────────────────────────────────────────────
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax1.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax1.tick_params(axis='x', labelsize=10)
ax1.set_xlabel('January 2022', fontsize=11)

# ── Title ─────────────────────────────────────────────────────────────────
ax1.set_title('WP2 Site – Price and Wind Power, January 2022',
              fontsize=16, fontweight='bold', color='#0d1b2a', pad=10)

# ── Legend ────────────────────────────────────────────────────────────────
price_patch = mpatches.Patch(color=PRICE_COL, alpha=0.7, label='DK1 price (EUR MWh⁻¹)')
wind_patch  = mpatches.Patch(color=WIND_COL,  alpha=0.7, label='Wind farm power (MW)')
align_patch = mpatches.Patch(color='#f4a261', alpha=0.5, label='Partial co-movement')
ax1.legend(handles=[price_patch, wind_patch, align_patch],
           loc='upper left', fontsize=10, framealpha=0.9, edgecolor='#ccc')

plt.tight_layout()
fig.savefig('/mnt/user-data/outputs/jan2022_overlay.pdf', dpi=300,
            bbox_inches='tight', facecolor='white')
fig.savefig('/mnt/user-data/outputs/jan2022_overlay.png', dpi=200,
            bbox_inches='tight', facecolor='white')
import shutil; shutil.copy(__file__, '/mnt/user-data/outputs/plot_jan_overlay.py')
print("Done.")
