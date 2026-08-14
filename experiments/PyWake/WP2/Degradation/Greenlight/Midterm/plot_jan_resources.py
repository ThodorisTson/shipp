import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────
RHO          = 1.225      # kg/m³
D            = 125.88009368  # m  (from WP2_Wind_Farm.yaml)
A_SWEPT      = np.pi / 4 * D**2
RATED_POWER  = 5_000_000.0  # W
CUT_IN       = 3.0        # m/s
CUT_OUT      = 25.0       # m/s
N_TURBINES   = 65
H_REF        = 86.0       # ERA5 reference height [m]
H_HUB        = 90.0       # hub height [m]
ALPHA        = 0.20       # shear exponent

# ── Cp curve from WP2_Wind_Farm.yaml ─────────────────────────────────────────
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
    """ERA5 ref-height wind speed → total farm power [MW]."""
    # 1. Shear scaling to hub height
    ws_hub = ws_ref * (H_HUB / H_REF) ** ALPHA
    # 2. Cp interpolation (clamp outside table range to boundary values)
    cp = np.interp(ws_hub, cp_ws, cp_val, left=0.0, right=cp_val[-1])
    # 3. P = Cp * 0.5 * rho * A * v³  (single turbine, watts)
    p_single = cp * 0.5 * RHO * A_SWEPT * ws_hub**3
    # 4. Cap at rated, zero outside cut-in/cut-out
    p_single = np.minimum(p_single, RATED_POWER)
    p_single = np.where((ws_hub < CUT_IN) | (ws_hub > CUT_OUT), 0.0, p_single)
    # 5. Farm total in MW
    return p_single * N_TURBINES / 1e6

# ── Load price data ───────────────────────────────────────────────────────────
prices = pd.read_csv(Path(__file__).parent.parent / 'dk1_prices_2022.csv',parse_dates=['timestamp'])
prices = prices.set_index('timestamp').sort_index()
jan_prices = prices['2022-01-01':'2022-01-31 23:00']

# ── Load wind resource ────────────────────────────────────────────────────────
with open(Path(__file__).parent.parent / 'wind_resource_2022hourly_referenceHPP.yaml') as f:
    wr = yaml.safe_load(f)['wind_resource']

times  = pd.to_datetime(wr['time'], utc=True).tz_localize(None)
ws_arr = np.array(list(wr['wind_speed']))
wind_df = pd.DataFrame({'ws': ws_arr, 'power_mw': ws_to_farm_power_mw(ws_arr)},
                       index=times)
jan_wind = wind_df['2022-01-01':'2022-01-31 23:00']

print(f"Hours   – price: {len(jan_prices)}, wind: {len(jan_wind)}")
print(f"Price   : {jan_prices['price_eur_mwh'].min():.0f} – "
      f"{jan_prices['price_eur_mwh'].max():.0f} EUR/MWh")
print(f"Power   : {jan_wind['power_mw'].min():.0f} – "
      f"{jan_wind['power_mw'].max():.0f} MW  "
      f"(rated farm = {N_TURBINES*5} MW)")

# ── Figure ────────────────────────────────────────────────────────────────────
PRICE_COL = '#c0392b'
WIND_COL  = '#2166ac'
FILL_ALPHA = 0.15

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')

tick_days  = mdates.DayLocator(interval=7)
minor_days = mdates.DayLocator(interval=1)
fmt        = mdates.DateFormatter('%d %b')

for ax in axes:
    ax.set_facecolor('#f7f9fc')
    ax.grid(True, which='major', color='white', lw=1.0, zorder=0)
    ax.grid(True, which='minor', color='white', lw=0.4, zorder=0)
    ax.xaxis.set_major_locator(tick_days)
    ax.xaxis.set_minor_locator(minor_days)
    ax.xaxis.set_major_formatter(fmt)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlabel('January 2022', fontsize=11)

# ─── Left panel: price ───────────────────────────────────────────────────────
ax1 = axes[0]
t_p = jan_prices.index
p   = jan_prices['price_eur_mwh'].values
ax1.fill_between(t_p, p, alpha=FILL_ALPHA, color=PRICE_COL, zorder=1)
ax1.plot(t_p, p, color=PRICE_COL, lw=1.0, zorder=2)
ax1.set_title('DK1 Day-Ahead Price', fontsize=16, fontweight='bold',
              color='#0d1b2a', pad=8)
ax1.set_ylabel('Price (EUR MWh$^{-1}$)', fontsize=12)
ax1.set_ylim(bottom=0)

# ─── Right panel: wind power ─────────────────────────────────────────────────
ax2 = axes[1]
t_w = jan_wind.index
pw  = jan_wind['power_mw'].values
rated_farm = N_TURBINES * 5  # 325 MW

ax2.fill_between(t_w, pw, alpha=FILL_ALPHA, color=WIND_COL, zorder=1)
ax2.plot(t_w, pw, color=WIND_COL, lw=1.0, zorder=2)
ax2.axhline(rated_farm, color=WIND_COL, lw=1.0, ls='--', alpha=0.6,
            label=f'Rated capacity ({rated_farm} MW)', zorder=2)
ax2.set_title('Thesis Wind Farm Power Output', fontsize=16, fontweight='bold',
              color='#0d1b2a', pad=8)
ax2.set_ylabel('Power (MW)', fontsize=12)
ax2.set_ylim(bottom=0, top=rated_farm * 1.08)
ax2.legend(fontsize=9.5, loc='upper left', framealpha=0.85, edgecolor='#ccc')

plt.tight_layout(w_pad=3.0)

# Save script too
OUT = Path(__file__).parent
fig.savefig(OUT / 'jan2022_price_wind.pdf', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUT / 'jan2022_price_wind.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved.")
