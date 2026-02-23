"""
WP2 Comprehensive Wind Farm Analysis
=====================================

Complete analysis of WP2 offshore wind farm with visualizations:
- Wake modeling with NOJ and Bastankhah-Gaussian
- Power production time series
- Market price analysis
- Duration curves
- Power-price correlation analysis
- Wind resource characterization

Wind farm: WP2 Denmark (65 × 5MW, D=161.5m, hub=100m)
Location: Denmark North Sea (56.23°N, 8.59°E)

Author: Thodoris
Date: 2026-02-11
Based on: Fryslån analysis structure
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import time

# PyWake imports
from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake import NOJ
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.turbulence_models.crespo import CrespoHernandez
from py_wake.rotor_avg_models import RotorCenter, EqGridRotorAvg

# windIO for YAML loading
try:
    from windIO.utils.yml_utils import load_yaml
except ImportError:
    import yaml
    
    def _yaml_loader_with_include(base_path: Path):
        class Loader(yaml.SafeLoader):
            pass
        def include(loader, node):
            rel_path = loader.construct_scalar(node)
            inc_path = (base_path.parent / rel_path).resolve()
            try:
                with open(inc_path, "r", encoding="utf-8-sig") as f:
                    return yaml.load(f, Loader=_yaml_loader_with_include(inc_path))
            except UnicodeDecodeError:
                with open(inc_path, "r", encoding="cp1252") as f:
                    return yaml.load(f, Loader=_yaml_loader_with_include(inc_path))
        Loader.add_constructor("!include", include)
        return Loader
    
    def load_yaml(path):
        path = Path(path).resolve()
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                return yaml.load(f, Loader=_yaml_loader_with_include(path))
        except UnicodeDecodeError:
            with open(path, "r", encoding="cp1252") as f:
                return yaml.load(f, Loader=_yaml_loader_with_include(path))


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PLOTS_DIR = SCRIPT_DIR / 'wp2_analysis_plots'
PLOTS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("WP2 COMPREHENSIVE WIND FARM ANALYSIS")
print("=" * 80)
print(f"\nPlots will be saved to: {PLOTS_DIR.resolve()}\n")

# =============================================================================
# STEP 1: LOAD WP2 DATA
# =============================================================================

print("=" * 80)
print("STEP 1: Load WP2 Data")
print("=" * 80)

# Load site + wind resource
site_file = SCRIPT_DIR / 'wp2_site_weibull.yaml'
print(f"\nLoading site: {site_file}")
data = load_yaml(site_file)

site_dat = data['site']
wr = data['wind_resource']

# Extract Weibull parameters
sectors = wr['sectors']
wd_sectors = np.array([s['direction_deg'] for s in sectors])
A = np.array([s['weibull_a'] for s in sectors])
k_weibull = np.array([s['weibull_k'] for s in sectors])
freq = np.array([s['probability'] for s in sectors])
TI_const = np.array([s['turbulence_intensity'] for s in sectors])

ws = np.arange(4.5, 25.0, 1.0)
TI_data = np.full(len(ws), TI_const.mean())

print(f"  ✓ Site: {site_dat['name']}")
print(f"    Location: {site_dat['latitude']:.2f}°N, {site_dat['longitude']:.2f}°E")
print(f"    Mean Weibull A: {A.mean():.2f} m/s")
print(f"    Mean Weibull k: {k_weibull.mean():.2f}")
print(f"    Turbulence intensity: {TI_const.mean()*100:.1f}%")

# Load turbine
turbine_file = SCRIPT_DIR / 'wp2_turbine.yaml'
print(f"\nLoading turbine: {turbine_file}")
turbine_dat = load_yaml(turbine_file)

hh = float(turbine_dat['hub_height'])
rd = float(turbine_dat['rotor_diameter'])

perf = turbine_dat['performance']
p_ws = np.array(perf['power_curve']['power_wind_speeds'])
p = np.array(perf['power_curve']['power_values'])
ct_ws = np.array(perf['Ct_curve']['Ct_wind_speeds'])
ct = np.array(perf['Ct_curve']['Ct_values'])
rated_power = perf['rated_power']

cut_in = p_ws[np.where(p > 0)[0][0]]
cut_out = p_ws[-1]

print(f"  ✓ Turbine: {turbine_dat['name']}")
print(f"    Rotor diameter: {rd:.1f} m")
print(f"    Hub height: {hh} m")
print(f"    Rated power: {rated_power/1e6:.1f} MW")

# Load layout
layout_file = SCRIPT_DIR / 'wp2_turbine_coordinates.csv'
print(f"\nLoading layout: {layout_file}")
layout = pd.read_csv(layout_file)
x = layout['X_m'].values
y = layout['Y_m'].values

n_turbines = len(x)
total_capacity_mw = n_turbines * rated_power / 1e6

print(f"  ✓ Layout: {n_turbines} turbines")
print(f"    Total capacity: {total_capacity_mw:.1f} MW")
print(f"    Farm area: {(x.max()-x.min())/1000:.1f} × {(y.max()-y.min())/1000:.1f} km")

# =============================================================================
# STEP 2: CREATE PYWAKE OBJECTS
# =============================================================================

print("\n" + "=" * 80)
print("STEP 2: Create PyWake Objects")
print("=" * 80)

# Interpolate curves to 10,000 points (IEA 740 style)
int_speeds = np.linspace(min(p_ws.min(), ct_ws.min()), max(p_ws.max(), ct_ws.max()), 10000)
ps_int = np.interp(int_speeds, p_ws, p)
cts_int = np.interp(int_speeds, ct_ws, ct)

# Wind turbine
windTurbines = WindTurbine(
    name=turbine_dat['name'],
    diameter=rd,
    hub_height=hh,
    powerCtFunction=PowerCtTabular(int_speeds, ps_int, power_unit='W', ct=cts_int)
)

# Site (XRSite with Weibull) - make wind direction cyclic for 360° wraparound
# Add duplicate points at 0° (= 360°) and 360° to enable full circle interpolation
wd_cyclic = np.concatenate([wd_sectors - 360, wd_sectors, wd_sectors + 360])
freq_cyclic = np.concatenate([freq, freq, freq])
A_cyclic = np.concatenate([A, A, A])
k_cyclic = np.concatenate([k_weibull, k_weibull, k_weibull])
TI_const_cyclic = np.concatenate([TI_const, TI_const, TI_const])

site = XRSite(
    ds=xr.Dataset(
        data_vars={
            'Sector_frequency': ('wd', freq_cyclic),
            'Weibull_A': ('wd', A_cyclic),
            'Weibull_k': ('wd', k_cyclic),
            'TI': ('ws', TI_data),
        },
        coords={'wd': wd_cyclic, 'ws': ws}
    )
)
site.interp_method = 'linear'

print("\n  ✓ Created XRSite with Weibull parameters")
print("  ✓ Created WindTurbine with PowerCtTabular")

# Wind rose discretization
ws_sw = 1
wd_sw = 1
ws_py = np.arange(max(cut_in, ws.min()), min(cut_out, ws.max()) + ws_sw, ws_sw)
wd_py = np.arange(0, 360, wd_sw)
TI = np.interp(ws_py, ws, TI_data)

print(f"\n  Wind discretization:")
print(f"    Wind speeds: {len(ws_py)} bins ({ws_py.min():.0f}-{ws_py.max():.0f} m/s)")
print(f"    Wind directions: {len(wd_py)} bins (0-360°)")
print(f"    Total cases: {len(ws_py)*len(wd_py):,}")

# =============================================================================
# STEP 3: RUN WAKE MODELS
# =============================================================================

print("\n" + "=" * 80)
print("STEP 3: Run Wake Models")
print("=" * 80)

# NOJ baseline
print("\n  Running NOJ (Jensen) model...")
noj = NOJ(site, windTurbines, turbulenceModel=None, k=0.05, rotorAvgModel=RotorCenter())

t0 = time.time()
sim_noj = noj(x, y, time=False, ws=ws_py, wd=wd_py, TI=TI)
t_noj = time.time() - t0

aep_noj_gwh = float(xr.DataArray.to_numpy(sim_noj.aep(normalize_probabilities=False).sum()))
cf_noj = aep_noj_gwh / (n_turbines * rated_power * 8760 / 1e9)

print(f"    ✓ Completed in {t_noj:.2f}s")
print(f"    AEP: {aep_noj_gwh:.2f} GWh")
print(f"    Capacity factor: {cf_noj*100:.2f}%")

# Bastankhah-Gaussian
print("\n  Running Bastankhah-Gaussian model...")
bastankhah = Bastankhah_PorteAgel_2014(
    site, windTurbines, k=0.04,
    turbulenceModel=CrespoHernandez(),
    rotorAvgModel=EqGridRotorAvg(3)
)

t0 = time.time()
sim_bast = bastankhah(x, y, time=False, ws=ws_py, wd=wd_py, TI=TI)
t_bast = time.time() - t0

aep_bast_gwh = float(xr.DataArray.to_numpy(sim_bast.aep(normalize_probabilities=False).sum()))
cf_bast = aep_bast_gwh / (n_turbines * rated_power * 8760 / 1e9)

print(f"    ✓ Completed in {t_bast:.2f}s")
print(f"    AEP: {aep_bast_gwh:.2f} GWh")
print(f"    Capacity factor: {cf_bast*100:.2f}%")

diff_pct = (aep_bast_gwh / aep_noj_gwh - 1) * 100
print(f"\n  Difference: {diff_pct:+.2f}% (Bastankhah vs NOJ)")

# Use Bastankhah for power time series generation
print("\n  Using Bastankhah model for time series generation...")

# =============================================================================
# STEP 4: GENERATE POWER TIME SERIES FROM HOURLY WIND DATA
# =============================================================================

print("\n" + "=" * 80)
print("STEP 4: Generate Power Time Series")
print("=" * 80)

# Load hourly wind data
wind_resource_file = SCRIPT_DIR / 'wind_resource_2022hourly_referenceHPP.yaml'
print(f"\nLoading hourly wind data: {wind_resource_file}")

with open(wind_resource_file, 'r') as f:
    import yaml
    wind_hourly = yaml.safe_load(f)

# Extract wind data
time_stamps = pd.to_datetime(wind_hourly['wind_resource']['time'])
ws_hourly_86m = np.array(wind_hourly['wind_resource']['wind_speed'])

# Apply wind shear correction (86m → 100m)
alpha = 0.12  # Offshore shear exponent
ws_hourly_100m = ws_hourly_86m * (100.0 / 86.0) ** alpha

# Get wind direction if available
if 'wind_direction' in wind_hourly['wind_resource']:
    wd_hourly = np.array(wind_hourly['wind_resource']['wind_direction'])
else:
    # Use mean from sectors if not available
    wd_hourly = np.full(len(ws_hourly_100m), np.average(wd_sectors, weights=freq))

print(f"  ✓ Loaded {len(time_stamps)} hourly data points")
print(f"    Date range: {time_stamps[0]} to {time_stamps[-1]}")
print(f"    Mean wind speed: {ws_hourly_100m.mean():.2f} m/s at 100m")

# Calculate power for each hour using Bastankhah model
print("\n  Calculating hourly power production...")

# We'll use the farm-averaged power from the wake model
# For simplicity, we use the mean power curve with wake losses

# Get effective wind speeds accounting for wake losses
# Approximate: use mean Ct to estimate wake losses
mean_ct = np.interp(ws_hourly_100m, ct_ws, ct)

# Simple wake loss approximation (very rough!)
# For a more accurate approach, we'd run full simulations
# Here we'll use the capacity factor ratio as a scaling factor
wake_loss_factor = cf_bast / (cf_bast + 0.15)  # Assume ~15% wake losses

# Calculate power from turbine curve
power_per_turbine_w = np.interp(ws_hourly_100m, p_ws, p)
power_farm_mw = power_per_turbine_w * n_turbines * wake_loss_factor / 1e6

print(f"  ✓ Power time series generated")
print(f"    Mean power: {power_farm_mw.mean():.1f} MW")
print(f"    Max power: {power_farm_mw.max():.1f} MW")
print(f"    Capacity factor: {power_farm_mw.mean()/total_capacity_mw*100:.1f}%")

# Create DataFrame
power_df = pd.DataFrame({
    'time': time_stamps,
    'wind_speed_100m': ws_hourly_100m,
    'wind_direction': wd_hourly,
    'power_mw': power_farm_mw
})
power_df.set_index('time', inplace=True)

# =============================================================================
# STEP 5: LOAD PRICE DATA (IF AVAILABLE)
# =============================================================================

print("\n" + "=" * 80)
print("STEP 5: Load Electricity Price Data")
print("=" * 80)

# Try to load real ENTSO-E price data, otherwise create synthetic
price_file = SCRIPT_DIR / 'dk1_prices_2022.csv'

if price_file.exists():
    print(f"\n  Loading real DK1 day-ahead prices from {price_file.name}...")
    price_data = pd.read_csv(price_file, index_col=0, parse_dates=True)
    
    # Reindex to match wind data timestamps (handles missing hours)
    if len(price_data) != len(power_df):
        print(f"  ⚠️  Price data has {len(price_data)} hours, wind data has {len(power_df)} hours")
        print(f"     Reindexing and forward-filling {len(power_df) - len(price_data)} missing hours...")
        
        # Reindex to wind data timestamps
        price_data_reindexed = price_data.reindex(power_df.index, method='ffill')
        
        # Check how many were forward-filled
        filled_count = price_data_reindexed.isna().sum().iloc[0]
        if filled_count > 0:
            # If forward-fill didn't work (missing at start), backfill
            price_data_reindexed = price_data_reindexed.fillna(method='bfill')
        
        prices_eur_mwh = price_data_reindexed.iloc[:, 0].values
        missing_filled = len(power_df) - len(price_data)
        print(f"  ✓ Loaded real prices with {missing_filled} hours interpolated")
    else:
        prices_eur_mwh = price_data.iloc[:, 0].values
        print(f"  ✓ Loaded {len(prices_eur_mwh)} hourly prices from ENTSO-E (perfect match)")
        price_file = True  # Keep as loaded
else:
    price_file = None

if price_file is None:
    print("\n  Creating synthetic day-ahead prices (DK1 zone typical patterns)...")
    
    # Typical Danish price characteristics (based on historical data)
    np.random.seed(42)
    n_hours = len(power_df)
    
    # Base price with seasonal and daily patterns (convert Index to arrays!)
    hour_of_day = power_df.index.hour.values
    day_of_year = power_df.index.dayofyear.values
    
    # Seasonal component (€/MWh)
    seasonal = 20 * np.sin(2 * np.pi * day_of_year / 365)
    
    # Daily pattern (higher during day, lower at night)
    daily = 15 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    
    # Base price
    base_price = 50  # €/MWh
    
    # Add correlation with wind (negative: more wind → lower prices)
    wind_effect = -0.3 * (power_farm_mw / total_capacity_mw - 0.5) * 40
    
    # Random noise
    noise = np.random.normal(0, 8, n_hours)
    
    # Combine
    prices_eur_mwh = base_price + seasonal + daily + wind_effect + noise
    
    # Clip to reasonable range
    prices_eur_mwh = np.clip(prices_eur_mwh, 0, 200)

power_df['price_eur_mwh'] = prices_eur_mwh

print(f"  ✓ Price data created")
print(f"    Mean price: {prices_eur_mwh.mean():.1f} €/MWh")
print(f"    Price range: {prices_eur_mwh.min():.1f} - {prices_eur_mwh.max():.1f} €/MWh")
print(f"    Std dev: {prices_eur_mwh.std():.1f} €/MWh")

# =============================================================================
# STEP 6: CREATE VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 80)
print("STEP 6: Create Visualizations")
print("=" * 80)

stamp = f"{pd.Timestamp.now():%Y%m%d}"

# PLOT 1: Wind, Power, Prices (first month)
# -----------------------------------------------------------------------------
print("\n  Creating first-month time series plot...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

month_hours = min(30 * 24, len(power_df))
t_month = power_df.index[:month_hours]

# Wind speed
axes[0].plot(t_month, power_df['wind_speed_100m'].iloc[:month_hours], linewidth=1)
axes[0].set_ylabel("Wind speed [m/s]")
axes[0].set_title(f"WP2 Denmark Offshore: Wind, Power, and Day-Ahead Prices (First Month)")
axes[0].grid(True, alpha=0.3)

# Power
axes[1].plot(t_month, power_df['power_mw'].iloc[:month_hours], linewidth=1)
axes[1].axhline(total_capacity_mw, linestyle="--", linewidth=1, label="Capacity")
axes[1].set_ylabel("Power [MW]")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Price
axes[2].plot(t_month, power_df['price_eur_mwh'].iloc[:month_hours], linewidth=1)
axes[2].set_ylabel("Price [€/MWh]")
axes[2].set_xlabel("Time")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_file = PLOTS_DIR / f"wp2_wind_power_price_month_{stamp}.png"
plt.savefig(plot_file, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    ✓ Saved: {plot_file.name}")

# PLOT 2: Price duration curve
# -----------------------------------------------------------------------------
print("  Creating price duration curve...")

sorted_prices = np.sort(power_df['price_eur_mwh'].values)[::-1]
dur = np.arange(len(sorted_prices)) / len(sorted_prices) * 100

mean_price = power_df['price_eur_mwh'].mean()
p90 = np.percentile(power_df['price_eur_mwh'], 90)
p10 = np.percentile(power_df['price_eur_mwh'], 10)
spread = p90 - p10

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)

ax.plot(dur, sorted_prices, linewidth=2.5)
ax.fill_between(dur, 0, sorted_prices, alpha=0.15)

ax.axhline(mean_price, linestyle="--", linewidth=2, label=f"Mean: {mean_price:.1f} €/MWh")
ax.axvspan(0, 10, alpha=0.15, label="Top 10% prices")
ax.axvspan(90, 100, alpha=0.15, label="Bottom 10% prices")

ax.set_xlabel("Duration [%]")
ax.set_ylabel("Price [€/MWh]")
ax.set_title(f"DK Day-Ahead Price Duration Curve (8760 hours)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

info = (
    f"p90: {p90:.1f} €/MWh\n"
    f"p10: {p10:.1f} €/MWh\n"
    f"Spread: {spread:.1f} €/MWh"
)
ax.text(0.98, 0.02, info, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plot_file = PLOTS_DIR / f"wp2_price_duration_{stamp}.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"    ✓ Saved: {plot_file.name}")

# PLOT 3: Power duration curve
# -----------------------------------------------------------------------------
print("  Creating power duration curve...")

sorted_power = np.sort(power_df['power_mw'].values)[::-1]
duration = np.arange(len(sorted_power)) / len(sorted_power) * 100

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)

ax.plot(duration, sorted_power, linewidth=2.5, color="steelblue")
ax.axhline(power_df['power_mw'].mean(), linestyle="--", linewidth=2, color="red",
           label=f"Mean: {power_df['power_mw'].mean():.1f} MW")

ax.set_xlabel("Duration [%]")
ax.set_ylabel("Wind power [MW]")
ax.set_title(f"Wind Power Duration Curve – WP2 Denmark (8760 hours)")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plot_file = PLOTS_DIR / f"wp2_power_duration_{stamp}.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"    ✓ Saved: {plot_file.name}")

# PLOT 4: Wind power vs price (density)
# -----------------------------------------------------------------------------
print("  Creating power vs price density plot...")

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)

hb = ax.hexbin(power_df['power_mw'], power_df['price_eur_mwh'], 
               gridsize=60, mincnt=1, norm=colors.LogNorm())
fig.colorbar(hb, ax=ax, label="Count (log scale)")

ax.set_xlabel("Wind power [MW]")
ax.set_ylabel("Price [€/MWh]")
ax.set_title("Wind Power vs Day-Ahead Price (Density)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_file = PLOTS_DIR / f"wp2_power_vs_price_density_{stamp}.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"    ✓ Saved: {plot_file.name}")

# PLOT 5: Price vs power (binned median + IQR)
# -----------------------------------------------------------------------------
print("  Creating price vs power binned plot...")

bins = np.linspace(0, total_capacity_mw, 30)
bin_id = np.digitize(power_df['power_mw'].values, bins)

centers = 0.5 * (bins[:-1] + bins[1:])
med = np.full(len(centers), np.nan)
p25 = np.full(len(centers), np.nan)
p75 = np.full(len(centers), np.nan)

for k in range(1, len(bins)):
    vals = power_df['price_eur_mwh'].values[bin_id == k]
    if len(vals) > 20:
        med[k-1] = np.percentile(vals, 50)
        p25[k-1] = np.percentile(vals, 25)
        p75[k-1] = np.percentile(vals, 75)

mask = ~np.isnan(med)

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)

ax.plot(centers[mask], med[mask], linewidth=2, label="Median price")
ax.fill_between(centers[mask], p25[mask], p75[mask], alpha=0.2, label="IQR (25–75%)")

ax.set_xlabel("Wind power [MW]")
ax.set_ylabel("Price [€/MWh]")
ax.set_title("Price vs Wind Power (Binned Median and IQR)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
plot_file = PLOTS_DIR / f"wp2_price_vs_power_binned_{stamp}.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"    ✓ Saved: {plot_file.name}")

# PLOT 6: Wind rose
# -----------------------------------------------------------------------------
print("  Creating wind rose...")

try:
    from windrose import WindroseAxes
    
    fig = plt.figure(figsize=(10, 10))
    ax = WindroseAxes.from_ax(fig=fig)
    
    ax.bar(
        power_df['wind_direction'].values,
        power_df['wind_speed_100m'].values,
        normed=True,
        opening=0.8,
        edgecolor='white',
        cmap=plt.cm.viridis,
        bins=np.arange(0, 26, 2)
    )
    ax.set_legend(title="Wind speed [m/s]", loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_title("Wind Rose – WP2 Denmark (100m height)", pad=20)
    
    plt.tight_layout()
    plot_file = PLOTS_DIR / f"wp2_windrose_{stamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved: {plot_file.name}")
    
except ImportError:
    print("    ⚠️  windrose package not available, skipping wind rose plot")
    print("       Install with: pip install windrose")

# PLOT 7: Layout visualization
# -----------------------------------------------------------------------------
print("  Creating layout visualization...")

fig, ax = plt.subplots(figsize=(12, 10))

# Plot turbines
ax.scatter(x, y, s=200, c='blue', alpha=0.6, edgecolors='black', linewidth=1.5, 
           label='Turbines', zorder=3)

# Add rotor diameter circle to first turbine
circle = plt.Circle((x[0], y[0]), rd/2, fill=False, edgecolor='red', 
                     linestyle='--', linewidth=2, label='Rotor diameter', zorder=2)
ax.add_patch(circle)

# Mark first turbine
ax.plot(x[0], y[0], 'r*', markersize=20, label='Turbine 1', zorder=4)

ax.set_xlabel('X [m]', fontsize=12)
ax.set_ylabel('Y [m]', fontsize=12)
ax.set_title(f'WP2 Wind Farm Layout\n{n_turbines} × {rated_power/1e6:.1f} MW = {total_capacity_mw:.1f} MW', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.legend(fontsize=10)

# Add text box with farm metrics
textstr = f'Rotor diameter: {rd:.1f} m\n'
textstr += f'Hub height: {hh:.0f} m\n'
textstr += f'Specific power: {rated_power/(np.pi*(rd/2)**2):.0f} W/m²\n'
textstr += f'Farm area: {(x.max()-x.min())/1000:.2f} × {(y.max()-y.min())/1000:.2f} km'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plot_file = PLOTS_DIR / f"wp2_layout_{stamp}.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"    ✓ Saved: {plot_file.name}")

# PLOT 8: Wake model comparison
# -----------------------------------------------------------------------------
print("  Creating wake model comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

models = ['NOJ\n(Jensen)', 'Bastankhah-Gaussian\n+ Crespo-Hernandez']
aeps = [aep_noj_gwh, aep_bast_gwh]
cfs = [cf_noj * 100, cf_bast * 100]

x_pos = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x_pos - width/2, aeps, width, label='AEP [GWh]', alpha=0.8)
ax2 = ax.twinx()
bars2 = ax2.bar(x_pos + width/2, cfs, width, label='Capacity Factor [%]', 
                alpha=0.8, color='orange')

ax.set_ylabel('AEP [GWh/year]', fontsize=12)
ax2.set_ylabel('Capacity Factor [%]', fontsize=12)
ax.set_title('Wake Model Comparison – WP2 Denmark', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax.text(bar1.get_x() + bar1.get_width()/2., height1,
            f'{height1:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.text(bar2.get_x() + bar2.get_width()/2., height2,
             f'{height2:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add legends
ax.legend(loc='upper left', fontsize=10)
ax2.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plot_file = PLOTS_DIR / f"wp2_wake_model_comparison_{stamp}.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"    ✓ Saved: {plot_file.name}")

# =============================================================================
# STEP 7: SUMMARY REPORT
# =============================================================================

print("\n" + "=" * 80)
print("STEP 7: Summary Report")
print("=" * 80)

summary = f"""
WP2 DENMARK OFFSHORE WIND FARM ANALYSIS SUMMARY
{'=' * 80}

SITE INFORMATION:
  Location:          {site_dat['latitude']:.2f}°N, {site_dat['longitude']:.2f}°E
  Altitude:          {site_dat['altitude']} m
  Mean wind speed:   {ws_hourly_100m.mean():.2f} m/s at {hh}m
  Weibull A (avg):   {A.mean():.2f} m/s
  Weibull k (avg):   {k_weibull.mean():.2f}

WIND FARM SPECIFICATIONS:
  Turbines:          {n_turbines} × {rated_power/1e6:.1f} MW
  Total capacity:    {total_capacity_mw:.1f} MW
  Rotor diameter:    {rd:.1f} m
  Hub height:        {hh} m
  Specific power:    {rated_power/(np.pi*(rd/2)**2):.0f} W/m²
  Farm area:         {(x.max()-x.min())/1000:.1f} × {(y.max()-y.min())/1000:.1f} km

WAKE MODELING RESULTS:
  NOJ (Jensen):
    AEP:             {aep_noj_gwh:.2f} GWh/year
    Capacity factor: {cf_noj*100:.2f}%
    
  Bastankhah-Gaussian + Crespo:
    AEP:             {aep_bast_gwh:.2f} GWh/year
    Capacity factor: {cf_bast*100:.2f}%
    
  Difference:        {diff_pct:+.2f}% (Bastankhah vs NOJ)

POWER PRODUCTION:
  Mean power:        {power_df['power_mw'].mean():.1f} MW
  Max power:         {power_df['power_mw'].max():.1f} MW
  Annual energy:     {power_df['power_mw'].sum() / 1000:.1f} GWh

MARKET PRICES ({'real ENTSO-E DK1' if price_file else 'synthetic DK1'}):
  Mean price:        {mean_price:.1f} €/MWh
  Price range:       {prices_eur_mwh.min():.1f} - {prices_eur_mwh.max():.1f} €/MWh
  P90 - P10 spread:  {spread:.1f} €/MWh
  Negative prices:   {(prices_eur_mwh < 0).sum()} hours{' (min: ' + f'{prices_eur_mwh.min():.1f}' + ' €/MWh)' if (prices_eur_mwh < 0).sum() > 0 else ''}

PLOTS CREATED:
  All plots saved to: {PLOTS_DIR.resolve()}
  1. wp2_wind_power_price_month_{stamp}.png
  2. wp2_price_duration_{stamp}.png
  3. wp2_power_duration_{stamp}.png
  4. wp2_power_vs_price_density_{stamp}.png
  5. wp2_price_vs_power_binned_{stamp}.png
  6. wp2_windrose_{stamp}.png (if windrose available)
  7. wp2_layout_{stamp}.png
  8. wp2_wake_model_comparison_{stamp}.png

NEXT STEPS:
  • Validate against measured data if available
  • Use for SHIPP battery optimization
  • Compare with Fryslån results
  • Integrate with electricity market models

{'=' * 80}
"""

print(summary)

# Save summary to file
summary_file = PLOTS_DIR / f"wp2_analysis_summary_{stamp}.txt"
with open(summary_file, 'w') as f:
    f.write(summary)
print(f"\n✓ Summary saved to: {summary_file.name}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)