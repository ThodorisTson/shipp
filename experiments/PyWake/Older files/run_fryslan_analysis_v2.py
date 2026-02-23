"""
Windpark Fryslân - PyWake Analysis with ERA5 + ENTSO-E Day-Ahead Prices
========================================================================

Complete workflow matching SHIPP Example 2 structure:
1) Load processed ERA5 wind data
2) Create PyWake wind farm model  
3) Calculate power production with wake effects (FAST - vectorized!)
4) Fetch ENTSO-E day-ahead prices for NL bidding zone
5) Compute baseline revenue (same as Example 2)
6) Visualize results

Author: Thodoris
Date: January 2026
"""

from setup_paths import configure_paths
configure_paths()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
import time

START_TIME = time.perf_counter()

# PyWake imports
from pywake_windfarm import WindFarmModel, create_reference_offshore_windfarm, generate_synthetic_wind_data

# Choose horizon
RUN_FULL_YEAR = True  # set False to keep 30-day quick test

print("="*70)
print("WINDPARK FRYSLÂN - PYWAKE + ENTSO-E DAY-AHEAD PRICES (NL)")
print("="*70)

# =============================================================================
# CONFIGURATION
# =============================================================================

FRYSLAN_CONFIG = {
    'name': 'Windpark Fryslån',
    'latitude': 52.95,
    'longitude': 5.35,
    'n_turbines': 89,
    'turbine_rating_mw': 4.0,
    'hub_height': 108.5,
    'rotor_diameter': 132,
    'total_capacity_mw': 356,
}

# File paths (project directory)
PROJECT_DIR = Path(__file__).parent.absolute()
WIND_CSV = PROJECT_DIR / "wind_data_fryslan.csv"
PLOTS_DIR = PROJECT_DIR / "plots"  # Save plots in script directory

# Token is in shipp directory (two levels up from PyWake)
ENTSOE_TOKEN_FILE = PROJECT_DIR.parent.parent / "token_entsoe.txt"

# NL bidding zone EIC code
NL_BZ_EIC = "10YNL----------L"

# Baseload cap (like Example 2)
P_MAX_MW = FRYSLAN_CONFIG['total_capacity_mw']  # Full capacity

# PyWake performance tuning
CHUNK_SIZE = 1500  # Try 1000, 2000, or 4000 depending on RAM

# =============================================================================
# ENTSO-E API FUNCTIONS
# =============================================================================

def _fmt_entsoe_ts(ts_utc: pd.Timestamp) -> str:
    """Format UTC timestamp for ENTSO-E API: YYYYMMDDHHMM."""
    ts_utc = pd.Timestamp(ts_utc)
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    else:
        ts_utc = ts_utc.tz_convert("UTC")
    return ts_utc.strftime("%Y%m%d%H%M")


def fetch_entsoe_day_ahead_prices(
    security_token: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    bidding_zone_eic: str = "10YNL----------L",
    timeout_s: int = 60,
) -> pd.Series:
    """
    Fetch ENTSO-E day-ahead prices (EUR/MWh) for NL bidding zone.
    
    Uses Transparency Platform API:
      documentType=A44 (Price Document)
      in_Domain/out_Domain = bidding zone EIC
      periodStart/periodEnd in UTC
    
    Parameters
    ----------
    security_token : str
        Your ENTSO-E API token
    start_utc : pd.Timestamp
        Start time (UTC)
    end_utc : pd.Timestamp
        End time (UTC, exclusive)
    bidding_zone_eic : str
        EIC code for bidding zone (default: NL = 10YNL----------L)
    timeout_s : int
        Request timeout in seconds
        
    Returns
    -------
    pd.Series
        Hourly prices indexed by UTC timestamps
    """
    base_url = "https://web-api.tp.entsoe.eu/api"
    
    params = {
        "securityToken": security_token,
        "documentType": "A44",
        "in_Domain": bidding_zone_eic,
        "out_Domain": bidding_zone_eic,
        "periodStart": _fmt_entsoe_ts(start_utc),
        "periodEnd": _fmt_entsoe_ts(end_utc),
    }
    
    print(f"\n  Requesting ENTSO-E day-ahead prices...")
    print(f"    Zone: {bidding_zone_eic}")
    print(f"    Period: {start_utc} to {end_utc}")
    
    r = requests.get(base_url, params=params, timeout=timeout_s)
    
    if r.status_code != 200:
        raise RuntimeError(
            f"ENTSO-E request failed (HTTP {r.status_code}). "
            f"Response: {r.text[:500]}"
        )
    
    # Parse XML
    root = ET.fromstring(r.text)
    
    # Helper to find elements by tag suffix (namespace-agnostic)
    def findall_suffix(node, suffix):
        return [e for e in node.iter() if e.tag.endswith(suffix)]
    
    # Extract periods
    periods = findall_suffix(root, "Period")
    
    if not periods:
        # Check for error messages
        reasons = findall_suffix(root, "Reason")
        if reasons:
            txt = " | ".join(["".join(list(rsn.itertext())).strip() for rsn in reasons])
            raise RuntimeError(f"ENTSO-E returned no data. Reason: {txt}")
        raise RuntimeError("ENTSO-E returned no Period blocks")
    
    # Parse price points
    out = {}
    
    for period in periods:
        # Get period start time
        ti = None
        for x in period.iter():
            if x.tag.endswith("timeInterval"):
                ti = x
                break
        
        if ti is None:
            continue
        
        p_start = None
        for x in ti.iter():
            if x.tag.endswith("start"):
                # Parse timestamp - it might already have timezone info
                ts = pd.Timestamp(x.text)
                # Convert to UTC (handles both tz-naive and tz-aware)
                if ts.tzinfo is None:
                    p_start = ts.tz_localize("UTC")
                else:
                    p_start = ts.tz_convert("UTC")
                break
        
        if p_start is None:
            continue
        
        # Extract price points
        points = [e for e in period.iter() if e.tag.endswith("Point")]
        
        for pt in points:
            pos = None
            price = None
            
            for c in pt.iter():
                if c.tag.endswith("position"):
                    pos = int(c.text)
                elif c.tag.endswith("price.amount"):
                    price = float(c.text)
            
            if pos is None or price is None:
                continue
            
            # Calculate timestamp (position is 1-indexed)
            ts = p_start + pd.Timedelta(hours=pos - 1)
            out[ts] = price
    
    if not out:
        raise RuntimeError("Parsed ENTSO-E XML but found no price points")
    
    # Create series
    s = pd.Series(out).sort_index()
    s.name = "price_eur_mwh"
    
    print(f"    ✓ Retrieved {len(s)} hourly prices")
    
    return s


def align_prices_to_wind(price_series_utc: pd.Series, wind_index) -> np.ndarray:
    """
    Align ENTSO-E prices to wind data timestamps.
    
    Parameters
    ----------
    price_series_utc : pd.Series
        Hourly prices in UTC
    wind_index : pd.DatetimeIndex
        Wind data timestamps
        
    Returns
    -------
    np.ndarray
        Prices aligned to wind index
    """
    idx = pd.DatetimeIndex(wind_index)
    
    # Convert to UTC
    if idx.tz is None:
        idx_utc = idx.tz_localize("UTC")
    else:
        idx_utc = idx.tz_convert("UTC")
    
    # Reindex to wind timeline
    prices = price_series_utc.reindex(idx_utc)
    
    # Fill any gaps (forward fill then backward fill)
    prices = prices.ffill().bfill()
    
    return prices.to_numpy(dtype=float)

# =============================================================================
# STEP 1: LOAD WIND DATA
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Load Wind Data")
print("="*70)

if not WIND_CSV.exists():
    print(f"\n⚠️  Wind CSV not found: {WIND_CSV}")
    print("  Please run:")
    print("    1. python era5_download.py")
    print("    2. python era5_process.py")
    print("\nUsing synthetic wind (30 days) for demo...")
    
    wind_data = generate_synthetic_wind_data(hours=30*24, mean_ws=8.0)
    use_real_data = False
else:
    print(f"\n✓ Loading: {WIND_CSV}")
    wind_data = pd.read_csv(WIND_CSV, parse_dates=['timestamp'])
    wind_data = wind_data.set_index('timestamp').sort_index()
    use_real_data = True

print(f"  Records: {len(wind_data)} hours")
print(f"  Period: {wind_data.index[0]} to {wind_data.index[-1]}")
print(f"  Mean wind speed: {wind_data['wind_speed'].mean():.2f} m/s")

# =============================================================================
# STEP 2: CREATE WIND FARM MODEL
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Create PyWake Wind Farm Model")
print("="*70)

wf = WindFarmModel(
    n_turbines=FRYSLAN_CONFIG['n_turbines'],
    turbine_rating_mw=FRYSLAN_CONFIG['turbine_rating_mw'],
    rotor_diameter=FRYSLAN_CONFIG['rotor_diameter'],
    hub_height=FRYSLAN_CONFIG['hub_height'],
)

wf.create_layout(layout_type='grid', spacing=7.0)
wf.setup_wake_model(use_turbulence=True)

print(f"\n✓ Wind farm created")
print(f"  Turbines: {wf.n_turbines} × {wf.turbine_rating_mw}MW")
print(f"  Total capacity: {FRYSLAN_CONFIG['total_capacity_mw']} MW")
print(f"  Wake model: IEA37 Bastankhah-Gaussian (turbulence ON)")

# =============================================================================
# STEP 3: CALCULATE POWER PRODUCTION
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Calculate Power Production with Wake Effects")
print("="*70)

# Choose subset or full year
if RUN_FULL_YEAR:
    print(f"\n✓ Using full dataset: {len(wind_data)} hours")
    wind_data_subset = wind_data
else:
    print(f"\n⚠️  Using first 30 days for demonstration")
    wind_data_subset = wind_data.iloc[:30*24]

print(f"\nCalculating power for {len(wind_data_subset)} hours...")
pywake_start = time.perf_counter()

# FAST VECTORIZED VERSION with tunable chunk size
power_ts = wf.generate_timeseries(
    wind_data_subset['wind_speed'],
    wind_data_subset['wind_direction'],
    ti=0.06,
    chunk_size=CHUNK_SIZE,  # Tune for speed/memory
    verbose=True
)

pywake_time = time.perf_counter() - pywake_start

# Statistics
mean_power = float(power_ts.mean())
capacity_factor = mean_power / FRYSLAN_CONFIG['total_capacity_mw'] * 100
total_energy_gwh = float(power_ts.sum()) / 1000.0
n_days = len(wind_data_subset) / 24

print(f"\n✓ Power calculation complete")
print(f"  Mean power: {mean_power:.1f} MW")
print(f"  Capacity factor: {capacity_factor:.1f}%")
print(f"  Energy ({n_days:.0f} days): {total_energy_gwh:.1f} GWh")
print(f"  Calculation time: {pywake_time:.1f} seconds")

# =============================================================================
# STEP 4: FETCH ENTSO-E DAY-AHEAD PRICES
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Fetch ENTSO-E Day-Ahead Prices (NL)")
print("="*70)

if not ENTSOE_TOKEN_FILE.exists():
    print(f"\n✗ ENTSO-E token file not found: {ENTSOE_TOKEN_FILE}")
    print(f"  Expected location: C:\\Users\\User\\Desktop\\Thesis\\shipp\\token_entsoe.txt")
    print("\nTo get ENTSO-E API token:")
    print("  1. Register at: https://transparency.entsoe.eu/")
    print("  2. Get your API token")
    print(f"  3. Save to: {ENTSOE_TOKEN_FILE}")
    
    raise FileNotFoundError(
        f"ENTSO-E token required at: {ENTSOE_TOKEN_FILE}"
    )

print(f"\n✓ Loading ENTSO-E token from: {ENTSOE_TOKEN_FILE}")
token_entsoe = ENTSOE_TOKEN_FILE.read_text(encoding='utf-8').strip()

# Define time window from wind data
wind_idx = pd.DatetimeIndex(wind_data_subset.index)

if wind_idx.tz is None:
    start_utc = wind_idx[0].tz_localize('UTC')
    end_utc = (wind_idx[-1] + pd.Timedelta(hours=1)).tz_localize('UTC')
else:
    start_utc = wind_idx[0].tz_convert('UTC')
    end_utc = (wind_idx[-1] + pd.Timedelta(hours=1)).tz_convert('UTC')

# Fetch prices from ENTSO-E
price_series_utc = fetch_entsoe_day_ahead_prices(
    security_token=token_entsoe,
    start_utc=start_utc,
    end_utc=end_utc,
    bidding_zone_eic=NL_BZ_EIC,
)

# Align to wind data
data_price = align_prices_to_wind(price_series_utc, wind_data_subset.index)

print(f"\n✓ Prices loaded and aligned")
print(f"  Mean price: {float(np.mean(data_price)):.2f} €/MWh")
print(f"  Min: {float(np.min(data_price)):.2f} €/MWh")
print(f"  Max: {float(np.max(data_price)):.2f} €/MWh")

# =============================================================================
# STEP 5: BASELINE REVENUE (EXAMPLE 2 STYLE)
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Baseline Revenue (No Storage) - Example 2 Style")
print("="*70)

# Convert to arrays
dt = 1  # hours
power_mw = power_ts.to_numpy(dtype=float)
n = len(power_mw)

# Apply baseload cap (like Example 2)
power_capped = np.minimum(power_mw, float(P_MAX_MW))

# Calculate baseline revenue (same formula as Example 2)
baseline_revenue_eur = float(np.dot(data_price[:n], power_capped) * dt)
mean_price = float(np.mean(data_price[:n]))

# =============================================================================
# EXAMPLE 2 STYLE OUTPUT
# =============================================================================

print(f"\n{'='*70}")
print("BASELINE SCENARIO (NO STORAGE)")
print(f"{'='*70}")
print(f"Period: {wind_data_subset.index[0]} to {wind_data_subset.index[-1]}")
print(f"Timesteps: {n} hours (dt = {dt} h)")
print(f"{'-'*70}")

# Power statistics
print(f"\nPOWER PRODUCTION:")
print(f"  Wind farm capacity:     {FRYSLAN_CONFIG['total_capacity_mw']:.1f} MW")
print(f"  Mean power output:      {mean_power:.2f} MW")
print(f"  Capacity factor:        {capacity_factor:.1f}%")
print(f"  Total energy:           {total_energy_gwh:.2f} GWh")
print(f"  Wake losses:            {(1 - capacity_factor/100*1.78):.1f}%")  # Rough estimate

# Price statistics
print(f"\nELECTRICITY PRICES (NL):")
print(f"  Mean price:             {mean_price:.2f} €/MWh")
print(f"  Min price:              {float(np.min(data_price)):.2f} €/MWh")
print(f"  Max price:              {float(np.max(data_price)):.2f} €/MWh")
print(f"  Price volatility (std): {float(np.std(data_price)):.2f} €/MWh")

# Revenue
print(f"\nREVENUE (NO STORAGE):")
print(f"  Baseload cap (p_max):   {P_MAX_MW:.1f} MW")
print(f"  Period revenue:         {baseline_revenue_eur:,.0f} €")
print(f"  Per day:                {baseline_revenue_eur/n_days:,.0f} €/day")
print(f"  Per MWh:                {baseline_revenue_eur/total_energy_gwh/1000:.2f} €/MWh")

# Annualize (like Example 2)
days = 366 if n == 366 * 24 else 365
baseline_revenue_annual_eur = float((days * 24 / n) * baseline_revenue_eur)
print(f"\nANNUALIZED METRICS:")
print(f"  Annual energy:          {total_energy_gwh * 365 * 24 / n:.2f} GWh/year")
print(f"  Annual revenue:         {baseline_revenue_annual_eur/1e6:.2f} M€/year")
print(f"  Revenue per MW:         {baseline_revenue_annual_eur/FRYSLAN_CONFIG['total_capacity_mw']/1e3:.2f} k€/MW/year")

print(f"\n{'='*70}")

# =============================================================================
# STEP 6: VISUALIZATION
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Create Visualizations")
print("="*70)

# Create plots directory in script location
PLOTS_DIR.mkdir(exist_ok=True)
print(f"\nPlots will be saved to: {PLOTS_DIR.resolve()}")

stamp = f"{pd.Timestamp.now():%Y%m%d}"

# -----------------------------------------------------------------------------
# PLOT 1: Wind, Power, Prices (first month)
# -----------------------------------------------------------------------------
print("\nCreating first-month time series plot (wind, power, price)...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

month_hours = min(30 * 24, len(wind_data_subset))
t_month = wind_data_subset.index[:month_hours]

# Wind speed
axes[0].plot(
    t_month,
    wind_data_subset["wind_speed"].iloc[:month_hours].values,
    linewidth=1
)
axes[0].set_ylabel("Wind speed [m/s]")
axes[0].set_title("Windpark Fryslân: Wind, Power, and NL Day-Ahead Prices (First Month)")
axes[0].grid(True, alpha=0.3)

# Power
axes[1].plot(t_month, power_mw[:month_hours], linewidth=1)
axes[1].axhline(
    FRYSLAN_CONFIG["total_capacity_mw"],
    linestyle="--",
    linewidth=1,
    label="Capacity"
)
axes[1].set_ylabel("Power [MW]")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Price
axes[2].plot(t_month, data_price[:month_hours], linewidth=1)
axes[2].set_ylabel("Price [€/MWh]")
axes[2].set_xlabel("Time")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_file = PLOTS_DIR / f"fryslan_wind_power_price_month_{stamp}.png"
plt.savefig(plot_file, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✓ Saved: {plot_file.resolve()}")


# -----------------------------------------------------------------------------
# PLOT 2: Day-ahead price duration curve (market arbitrage signal)
# -----------------------------------------------------------------------------
print("\nCreating day-ahead price duration curve...")

sorted_prices = np.sort(data_price[:n])[::-1]
dur = np.arange(n) / n * 100

mean_price = float(np.mean(data_price[:n]))
p90 = float(np.percentile(data_price[:n], 90))
p10 = float(np.percentile(data_price[:n], 10))
spread = p90 - p10

high_hours = int(np.sum(data_price[:n] >= p90))
low_hours = int(np.sum(data_price[:n] <= p10))

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)

ax.plot(dur, sorted_prices, linewidth=2.5)
ax.fill_between(dur, 0, sorted_prices, alpha=0.15)

ax.axhline(mean_price, linestyle="--", linewidth=2, label=f"Mean price: {mean_price:.1f} €/MWh")
ax.axvspan(0, 10, alpha=0.15, label="Top 10% prices")
ax.axvspan(90, 100, alpha=0.15, label="Bottom 10% prices")

ax.set_xlabel("Duration [%]")
ax.set_ylabel("Price [€/MWh]")
ax.set_title(f"NL Day-Ahead Price Duration Curve ({n_days:.0f} days)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

info = (
    f"p90: {p90:.1f} €/MWh ({high_hours} h)\n"
    f"p10: {p10:.1f} €/MWh ({low_hours} h)\n"
    f"Spread (p90–p10): {spread:.1f} €/MWh"
)
ax.text(
    0.02, 0.02, info, transform=ax.transAxes,
    fontsize=9, family="monospace",
    bbox=dict(boxstyle="round", alpha=0.9),
    va="bottom", ha="left"
)

plt.tight_layout()
plot_file = PLOTS_DIR / f"fryslan_price_duration_{stamp}.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✓ Saved: {plot_file.resolve()}")

print("\nCreating power duration curve...")

# =============================================================================
# Plot 2.5 POWER DURATION CURVE (standalone)
# =============================================================================

sorted_power = np.sort(power_mw)[::-1]
duration = np.arange(len(sorted_power)) / len(sorted_power) * 100

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)

ax.plot(duration, sorted_power, linewidth=2.5, color="steelblue")
ax.axhline(
    mean_power,
    linestyle="--",
    linewidth=2,
    color="red",
    label=f"Mean power: {mean_power:.1f} MW"
)

ax.set_xlabel("Duration [%]")
ax.set_ylabel("Wind power [MW]")
ax.set_title(f"Wind Power Duration Curve – Windpark Fryslân ({n_days:.0f} days)")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plot_file = PLOTS_DIR / f"fryslan_power_duration_{pd.Timestamp.now():%Y%m%d}.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✓ Saved: {plot_file.resolve()}")

# -----------------------------------------------------------------------------
# PLOT 3: Wind power vs price (density)
# -----------------------------------------------------------------------------
print("\nCreating wind power vs price density plot...")

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)

import matplotlib.colors as colors

hb = ax.hexbin(power_mw, data_price[:n], gridsize=60, mincnt=1, norm=colors.LogNorm())
fig.colorbar(hb, ax=ax, label="Count (log scale)")

ax.set_xlabel("Wind power [MW]")
ax.set_ylabel("Price [€/MWh]")
ax.set_title("Wind Power vs NL Day-Ahead Price (Density)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_file = PLOTS_DIR / f"fryslan_power_vs_price_density_{stamp}.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✓ Saved: {plot_file.resolve()}")


# -----------------------------------------------------------------------------
# PLOT 4: Price vs power (binned median + IQR)
# -----------------------------------------------------------------------------
print("\nCreating price vs power (binned median + IQR) plot...")

bins = np.linspace(0, FRYSLAN_CONFIG["total_capacity_mw"], 30)
bin_id = np.digitize(power_mw, bins)

centers = 0.5 * (bins[:-1] + bins[1:])
med = np.full(len(centers), np.nan)
p25 = np.full(len(centers), np.nan)
p75 = np.full(len(centers), np.nan)

for k in range(1, len(bins)):
    vals = data_price[:n][bin_id == k]
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
plot_file = PLOTS_DIR / f"fryslan_price_vs_power_binned_{stamp}.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✓ Saved: {plot_file.resolve()}")


# -----------------------------------------------------------------------------
# WIND ROSE (Direction & Speed Distribution)
# -----------------------------------------------------------------------------
print("\nCreating wind rose...")

try:
    from windrose import WindroseAxes

    fig = plt.figure(figsize=(10, 10))
    ax = WindroseAxes.from_ax(fig=fig)

    ax.bar(
        wind_data_subset["wind_direction"],
        wind_data_subset["wind_speed"],
        normed=True,
        opening=0.8,
        edgecolor="white",
        bins=[0, 4, 8, 12, 16, 20, 100],
        cmap=plt.cm.viridis
    )

    ax.set_legend(title="Wind Speed [m/s]")
    ax.set_title(
        f"Wind Rose - Windpark Fryslân ({n_days:.0f} days)",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    plot_file = PLOTS_DIR / f"fryslan_windrose_{stamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {plot_file.resolve()}")

except ImportError:
    print("  Install windrose for better plots: pip install windrose")

# =============================================================================
# RUNTIME SUMMARY
# =============================================================================

total_time = time.perf_counter() - START_TIME

print("\n" + "="*70)
print("RUNTIME SUMMARY")
print("="*70)
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = total_time % 60
print(f"PyWake calculation: {pywake_time:.1f}s ({len(wind_data_subset)} hours)")
print(f"Total runtime: {hours:02d}:{minutes:02d}:{seconds:05.2f} (hh:mm:ss.ss)")
print(f"Speed: {len(wind_data_subset)/pywake_time:.0f} hours/second")
print("="*70)