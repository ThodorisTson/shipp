"""
Diagnostic: Find why sector probabilities sum to 0.913 instead of 1.0
=====================================================================

We're losing ~8.7% of wind data somewhere in sector assignment.
This script will identify exactly where.
"""

import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

# Load the hourly ERA5 data
HPP_YAML = Path(__file__).parent / "WP2_HPP.yaml"

def load_yaml(path):
    path = Path(path)
    def _include(loader, node):
        target = Path(loader.construct_scalar(node))
        if not target.is_absolute():
            target = path.parent / target
        with open(target, 'r') as fh:
            return yaml.safe_load(fh)
    yaml.add_constructor('!include', _include, Loader=yaml.SafeLoader)
    with open(path, 'r') as fh:
        return yaml.safe_load(fh)

hpp = load_yaml(HPP_YAML)
ts = hpp['site']['energy_resource']['time_series']['wind_resource']

ws = np.array(ts['wind_speed'], dtype=float)
wd = np.array(ts['wind_direction'], dtype=float)

print("=" * 72)
print("WIND DIRECTION DATA QUALITY CHECK")
print("=" * 72)

print(f"\nTotal data points: {len(wd):,}")
print(f"\nWind direction statistics:")
print(f"  Min:    {wd.min():.2f}°")
print(f"  Max:    {wd.max():.2f}°")
print(f"  Mean:   {wd.mean():.2f}°")
print(f"  Median: {np.median(wd):.2f}°")

# Check for NaN / inf
n_nan = np.isnan(wd).sum()
n_inf = np.isinf(wd).sum()
n_valid = len(wd) - n_nan - n_inf

print(f"\nData validity:")
print(f"  Valid:   {n_valid:,}  ({n_valid/len(wd)*100:.2f}%)")
print(f"  NaN:     {n_nan:,}")
print(f"  Inf:     {n_inf:,}")

# Check range
n_negative = (wd < 0).sum()
n_above360 = (wd >= 360).sum()
n_in_range = ((wd >= 0) & (wd < 360)).sum()

print(f"\nRange check:")
print(f"  [0, 360): {n_in_range:,}  ({n_in_range/len(wd)*100:.2f}%)")
print(f"  < 0:      {n_negative:,}")
print(f"  >= 360:   {n_above360:,}")

# Now test the sector assignment logic
print("\n" + "=" * 72)
print("SECTOR ASSIGNMENT TEST")
print("=" * 72)

N_SECTORS = 12
SECTOR_WIDTH = 360.0 / N_SECTORS
centers = np.arange(SECTOR_WIDTH/2, 360, SECTOR_WIDTH)

def sector_mask_current(wd_deg, center, width):
    """Current implementation from wp2_common.py"""
    lo = (center - width / 2) % 360
    hi = (center + width / 2) % 360
    if lo < hi:
        return (wd_deg >= lo) & (wd_deg < hi)
    else:
        return (wd_deg >= lo) | (wd_deg < hi)

def sector_mask_fixed(wd_deg, center, width):
    """Fixed version — handle wraparound properly"""
    lo = center - width / 2
    hi = center + width / 2
    
    if lo >= 0 and hi <= 360:
        # Normal case: sector doesn't wrap
        return (wd_deg >= lo) & (wd_deg < hi)
    elif lo < 0:
        # Wraps at lower boundary: [lo+360, 360) OR [0, hi)
        return (wd_deg >= lo + 360) | (wd_deg < hi)
    else:  # hi > 360
        # Wraps at upper boundary: [lo, 360) OR [0, hi-360)
        return (wd_deg >= lo) | (wd_deg < hi - 360)

print("\nTesting CURRENT sector assignment:")
print(f"{'Sector':>6} {'Center':>8} {'Bounds':>16} {'Count':>10} {'Freq %':>8}")
print("-" * 60)

assigned_current = np.zeros(len(wd), dtype=bool)
freq_current = []

for i, center in enumerate(centers):
    mask = sector_mask_current(wd, center, SECTOR_WIDTH)
    count = mask.sum()
    freq = count / len(wd)
    freq_current.append(freq)
    assigned_current |= mask
    
    lo = (center - SECTOR_WIDTH/2) % 360
    hi = (center + SECTOR_WIDTH/2) % 360
    
    print(f"{i+1:6d} {center:8.0f}° [{lo:5.0f}, {hi:5.0f}) {count:10,} {freq*100:7.2f}%")

unassigned_current = (~assigned_current).sum()
sum_current = sum(freq_current)

print(f"\n{'TOTAL':>6} {'':>8} {'':>16} {assigned_current.sum():10,} {sum_current*100:7.2f}%")
print(f"{'UNASSIGNED':>6} {'':>8} {'':>16} {unassigned_current:10,} {(unassigned_current/len(wd))*100:7.2f}%")

print("\n" + "-" * 72)
print("\nTesting FIXED sector assignment:")
print(f"{'Sector':>6} {'Center':>8} {'Bounds':>16} {'Count':>10} {'Freq %':>8}")
print("-" * 60)

assigned_fixed = np.zeros(len(wd), dtype=bool)
freq_fixed = []

for i, center in enumerate(centers):
    mask = sector_mask_fixed(wd, center, SECTOR_WIDTH)
    count = mask.sum()
    freq = count / len(wd)
    freq_fixed.append(freq)
    assigned_fixed |= mask
    
    lo = center - SECTOR_WIDTH/2
    hi = center + SECTOR_WIDTH/2
    
    if lo < 0:
        bounds_str = f"[{lo+360:.0f}, 360) OR [0, {hi:.0f})"
    elif hi > 360:
        bounds_str = f"[{lo:.0f}, 360) OR [0, {hi-360:.0f})"
    else:
        bounds_str = f"[{lo:.0f}, {hi:.0f})"
    
    print(f"{i+1:6d} {center:8.0f}° {bounds_str:>16} {count:10,} {freq*100:7.2f}%")

unassigned_fixed = (~assigned_fixed).sum()
sum_fixed = sum(freq_fixed)

print(f"\n{'TOTAL':>6} {'':>8} {'':>16} {assigned_fixed.sum():10,} {sum_fixed*100:7.2f}%")
print(f"{'UNASSIGNED':>6} {'':>8} {'':>16} {unassigned_fixed:10,} {(unassigned_fixed/len(wd))*100:7.2f}%")

# Find which data points are unassigned in current method
if unassigned_current > 0:
    print("\n" + "=" * 72)
    print(f"ANALYZING {unassigned_current:,} UNASSIGNED DATA POINTS (CURRENT METHOD)")
    print("=" * 72)
    
    unassigned_wd = wd[~assigned_current]
    
    print(f"\nWind directions of unassigned points:")
    print(f"  Min:    {unassigned_wd.min():.2f}°")
    print(f"  Max:    {unassigned_wd.max():.2f}°")
    print(f"  Mean:   {unassigned_wd.mean():.2f}°")
    print(f"  Median: {np.median(unassigned_wd):.2f}°")
    
    # Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(unassigned_wd, bins=36, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Wind Direction [°]')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {unassigned_current:,} Unassigned Data Points\n(Current Sector Assignment Method)')
    ax.grid(True, alpha=0.3)
    
    # Mark sector boundaries
    for center in centers:
        ax.axvline(center, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'unassigned_wind_directions.png', dpi=200)
    print(f"\n  → Saved histogram: unassigned_wind_directions.png")

print("\n" + "=" * 72)
print("DIAGNOSIS COMPLETE")
print("=" * 72)

# Also check: what do OUR fitted frequencies sum to?
from wp2_common import load_yaml as load_yaml_common, load_wind_resource

print("\n" + "=" * 72)
print("CHECKING OUR WEIBULL SECTOR FREQUENCIES")
print("=" * 72)

hpp_check = load_yaml_common(HPP_YAML)
wind_res = load_wind_resource(hpp_check, verbose=False)

our_freq_sum = wind_res['frequencies'].sum()
print(f"\nOur fitted Weibull sector frequencies:")
print(f"  Sum: {our_freq_sum:.10f}")

if abs(our_freq_sum - 1.0) < 1e-10:
    print(f"  ✅ Perfectly normalized!")
else:
    print(f"  ⚠️  Not exactly 1.0 (off by {abs(our_freq_sum - 1.0):.2e})")

print("\n" + "=" * 72)

if abs(sum_current - 1.0) > 0.01:
    print(f"\n⚠️  CURRENT method loses {(1-sum_current)*100:.1f}% of data!")
    print(f"   Frequency sum: {sum_current:.6f} (should be 1.000000)")

if abs(sum_fixed - 1.0) < 0.001:
    print(f"\n✅ FIXED method accounts for all data!")
    print(f"   Frequency sum: {sum_fixed:.6f}")
else:
    print(f"\n❌ FIXED method still has issues!")
    print(f"   Frequency sum: {sum_fixed:.6f}")
