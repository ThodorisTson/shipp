"""
WP2 Wind Resource Characterization
===================================

Visualize and analyze the wind resource at WP2 site including:
- Wind rose showing directional wind distribution
- Sector-wise Weibull statistics
- Wind speed frequency distribution

Author: Thodoris
Date: 2026-02-12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from wp2_common import quick_setup, load_wind_resource, load_yaml

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PLOTS_DIR = SCRIPT_DIR / "wp2_wind_resource_plots"
PLOTS_DIR.mkdir(exist_ok=True)

HPP_YAML = SCRIPT_DIR / "WP2_HPP.yaml"

# For synthetic hourly data generation
N_HOURS = 8760  # Full year
SEED = 42  # For reproducibility

print("=" * 80)
print("WP2 WIND RESOURCE CHARACTERIZATION")
print("=" * 80)

# =============================================================================
# LOAD SITE DATA
# =============================================================================

print("\nLoading site data...")
hpp = load_yaml(HPP_YAML)
site_dat = load_wind_resource(hpp, verbose=True)

print(f"✓ Site: {site_dat['name']}")
print(f"  Location: {site_dat['latitude']:.2f}°N, {site_dat['longitude']:.2f}°E")
print(f"  Altitude: {site_dat['altitude']} m")
print(f"  Mean Weibull A: {site_dat['weibull_A'].mean():.2f} m/s")
print(f"  Mean Weibull k: {site_dat['weibull_k'].mean():.2f}")
print(f"  Sectors: {len(site_dat['sector_centers'])}")

# =============================================================================
# GENERATE SYNTHETIC HOURLY WIND DATA FROM WEIBULL DISTRIBUTIONS
# =============================================================================

print("\nGenerating synthetic hourly wind data from Weibull distributions...")

np.random.seed(SEED)

# Generate wind direction samples based on sector frequencies
sector_centers = site_dat['sector_centers']
sector_freqs = site_dat['frequencies']
weibull_A = site_dat['weibull_A']
weibull_k = site_dat['weibull_k']

# Sample wind directions proportionally to sector frequencies
wind_directions = np.random.choice(
    sector_centers,
    size=N_HOURS,
    p=sector_freqs / sector_freqs.sum()
)

# For each direction, sample wind speed from corresponding Weibull distribution
wind_speeds = np.zeros(N_HOURS)

for i, wd in enumerate(wind_directions):
    # Find corresponding sector
    sector_idx = np.argmin(np.abs(sector_centers - wd))
    A = weibull_A[sector_idx]
    k = weibull_k[sector_idx]
    
    # Sample from Weibull distribution
    wind_speeds[i] = np.random.weibull(k) * A

# Add some random jitter to wind directions for smoother rose
wind_directions += np.random.normal(0, 5, N_HOURS)  # ±5° jitter
wind_directions = wind_directions % 360  # Keep in 0-360 range

print(f"✓ Generated {N_HOURS} hours of synthetic wind data")
print(f"  Mean wind speed: {wind_speeds.mean():.2f} m/s")
print(f"  95th percentile: {np.percentile(wind_speeds, 95):.2f} m/s")

# =============================================================================
# PLOT 1: WIND ROSE
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING WIND ROSE")
print("=" * 80)

try:
    from windrose import WindroseAxes
    
    fig = plt.figure(figsize=(10, 10))
    ax = WindroseAxes.from_ax(fig=fig)
    
    ax.bar(
        wind_directions,
        wind_speeds,
        normed=True,
        opening=0.8,
        edgecolor='white',
        cmap=plt.cm.viridis,
        bins=np.arange(0, 26, 2)
    )
    
    ax.set_legend(title="Wind speed [m/s]", loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_title(f"Wind Rose – WP2 Denmark (Hub Height)", pad=20, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plot_file = PLOTS_DIR / "wp2_windrose.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {plot_file.name}")
    
except ImportError:
    print("⚠️  windrose package not available!")
    print("   Install with: pip install windrose")
    print("   Skipping wind rose plot...")

# =============================================================================
# PLOT 2: SECTOR STATISTICS
# =============================================================================

print("\nGenerating sector statistics...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Frequency bar chart
ax1.bar(sector_centers, sector_freqs * 100, width=25, 
        color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
ax1.set_xlabel("Wind Direction [°]", fontsize=11)
ax1.set_ylabel("Frequency [%]", fontsize=11)
ax1.set_title("Wind Direction Frequency by Sector", fontsize=12, fontweight='bold')
ax1.set_xticks(np.arange(0, 360, 45))
ax1.grid(True, alpha=0.3, axis='y')

# Weibull parameters
x = np.arange(len(sector_centers))
width = 0.35

ax2_twin = ax2.twinx()
bars1 = ax2.bar(x - width/2, weibull_A, width, label='Weibull A', 
                color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
bars2 = ax2_twin.bar(x + width/2, weibull_k, width, label='Weibull k', 
                     color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)

ax2.set_xlabel("Sector Center [°]", fontsize=11)
ax2.set_ylabel("Weibull A [m/s]", fontsize=11, color='steelblue')
ax2_twin.set_ylabel("Weibull k [-]", fontsize=11, color='coral')
ax2.set_title("Weibull Parameters by Sector", fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f"{int(sc)}°" for sc in sector_centers], rotation=45)
ax2.tick_params(axis='y', labelcolor='steelblue')
ax2_twin.tick_params(axis='y', labelcolor='coral')
ax2.grid(True, alpha=0.3, axis='y')

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

plt.tight_layout()
plot_file = PLOTS_DIR / "wp2_sector_statistics.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✓ Saved: {plot_file.name}")

# =============================================================================
# PLOT 3: WIND SPEED DISTRIBUTION
# =============================================================================

print("\nGenerating wind speed distribution...")

fig, ax = plt.subplots(figsize=(10, 6))

# Histogram
n, bins, patches = ax.hist(wind_speeds, bins=50, density=True, 
                            alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)

# Overall Weibull fit (weighted average)
ws_range = np.linspace(0, 25, 200)
overall_A = np.average(weibull_A, weights=sector_freqs)
overall_k = np.average(weibull_k, weights=sector_freqs)

# Weibull PDF: f(x) = (k/A) * (x/A)^(k-1) * exp(-(x/A)^k)
weibull_pdf = (overall_k / overall_A) * (ws_range / overall_A)**(overall_k - 1) * \
              np.exp(-(ws_range / overall_A)**overall_k)

ax.plot(ws_range, weibull_pdf, 'r-', linewidth=2.5, 
        label=f'Weibull fit (A={overall_A:.2f}, k={overall_k:.2f})')

ax.axvline(wind_speeds.mean(), linestyle='--', linewidth=2, color='darkgreen',
           label=f'Mean: {wind_speeds.mean():.2f} m/s')

ax.set_xlabel("Wind Speed [m/s]", fontsize=11)
ax.set_ylabel("Probability Density", fontsize=11)
ax.set_title("Wind Speed Distribution", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Add statistics text box
stats_text = (
    f"Mean: {wind_speeds.mean():.2f} m/s\n"
    f"Median: {np.median(wind_speeds):.2f} m/s\n"
    f"Std dev: {wind_speeds.std():.2f} m/s\n"
    f"P95: {np.percentile(wind_speeds, 95):.2f} m/s\n"
    f"Max: {wind_speeds.max():.2f} m/s"
)
ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plot_file = PLOTS_DIR / "wp2_wind_speed_distribution.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✓ Saved: {plot_file.name}")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("WIND RESOURCE SUMMARY")
print("=" * 80)

print(f"\nSite: {site_dat['name']}")
print(f"Location: {site_dat['latitude']:.2f}°N, {site_dat['longitude']:.2f}°E")
print(f"Altitude: {site_dat['altitude']} m")

print("\nWeibull Parameters:")
print(f"  Overall A (weighted): {overall_A:.2f} m/s")
print(f"  Overall k (weighted): {overall_k:.2f}")

print("\nWind Statistics (synthetic 8760h sample):")
print(f"  Mean wind speed: {wind_speeds.mean():.2f} m/s")
print(f"  Median wind speed: {np.median(wind_speeds):.2f} m/s")
print(f"  95th percentile: {np.percentile(wind_speeds, 95):.2f} m/s")
print(f"  Max wind speed: {wind_speeds.max():.2f} m/s")

print("\nPredominant Wind Directions:")
# Find top 3 sectors by frequency
top_sectors = np.argsort(sector_freqs)[::-1][:3]
for i, idx in enumerate(top_sectors, 1):
    print(f"  {i}. {sector_centers[idx]:.0f}° ({sector_freqs[idx]*100:.1f}%)")

print("\n" + "=" * 80)
print("✓ WIND RESOURCE CHARACTERIZATION COMPLETE")
print("=" * 80)
print(f"\nPlots saved in: {PLOTS_DIR}/")
print("  - wp2_windrose.png (if windrose package available)")
print("  - wp2_sector_statistics.png")
print("  - wp2_wind_speed_distribution.png")
print("=" * 80)