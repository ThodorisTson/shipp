"""
Download DK1 Day-Ahead Electricity Prices from ENTSO-E
=======================================================

This script downloads real electricity prices for Denmark DK1 zone 
to use with WP2 analysis instead of synthetic prices.

Requirements:
    pip install entsoe-py pandas

Setup:
    1. Get free API token from: https://transparency.entsoe.eu/
    2. Register and go to "Account Settings" → "Web API Security Token"
    3. Save token in environment variable or in this script

Author: Thodoris
Date: 2026-02-11
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Your ENTSO-E API token (get from https://transparency.entsoe.eu/)
# Option 1: File named 'token_entsoe' in script directory
# Option 2: Set environment variable: export ENTSOE_API_TOKEN=your-token-here
# Option 3: Uncomment and paste your token below:
# API_TOKEN = "your-token-here"

import os

# Try to read token from file first - check multiple locations and file names
TOKEN_FILE = None
API_TOKEN = None

# File name variations to search for
token_filenames = ['token_entsoe.txt', 'token_entsoe']

# Search locations in order of priority:
# The script is in: shipp/experiments/PyWake/WP2/
# We want to find: shipp/token_entsoe.txt (3 levels up)
search_dirs = [
    Path(__file__).parent,                          # Same directory as script (WP2/)
    Path(__file__).parent.parent,                   # 1 level up (PyWake/)
    Path(__file__).parent.parent.parent,            # 2 levels up (experiments/)
    Path(__file__).parent.parent.parent.parent,     # 3 levels up (shipp/) ← YOUR TOKEN IS HERE
]

print("Searching for token file...")
for search_dir in search_dirs:
    for filename in token_filenames:
        token_path = search_dir / filename
        if token_path.exists():
            TOKEN_FILE = token_path
            print(f"✓ Found token file: {TOKEN_FILE}")
            with open(TOKEN_FILE, 'r') as f:
                API_TOKEN = f.read().strip()
            print(f"✓ Loaded API token from file")
            break
    if API_TOKEN:
        break

if API_TOKEN is None or API_TOKEN == '':
    # Fall back to environment variable
    API_TOKEN = os.getenv('ENTSOE_API_TOKEN', None)
    if API_TOKEN:
        print("✓ Loaded API token from environment variable")

if API_TOKEN is None or API_TOKEN == '':
    print("=" * 80)
    print("ERROR: No ENTSO-E API token found!")
    print("=" * 80)
    print("\nSearched for token files in:")
    for search_dir in search_dirs:
        print(f"  - {search_dir / 'token_entsoe.txt'}")
        print(f"  - {search_dir / 'token_entsoe'}")
    print("\nPlease set your API token in one of these ways:")
    print("\n1. Create a file named 'token_entsoe.txt' or 'token_entsoe'")
    print(f"   in: {Path(__file__).parent.parent.parent.parent}")
    print("   (the shipp/ directory where your .venv is)")
    print("\n2. Environment variable:")
    print("   $env:ENTSOE_API_TOKEN='your-token-here'  (PowerShell)")
    print("   export ENTSOE_API_TOKEN=your-token-here   (Linux/Mac)")
    print("\n3. Or edit this script and set API_TOKEN directly")
    print("\nGet a free token from: https://transparency.entsoe.eu/")
    print("  → Register → Account Settings → Web API Security Token")
    print("=" * 80)
    exit(1)

# # Date range (2022 for WP2 analysis)
# START_DATE = pd.Timestamp('2022-01-01', tz='UTC')
# END_DATE = pd.Timestamp('2022-12-31 23:00:00', tz='UTC')

# # Output file
# OUTPUT_DIR = Path(__file__).parent
# OUTPUT_FILE = OUTPUT_DIR / 'dk1_prices_2022.csv'

# Date range (2019 for WP2 analysis)
START_DATE = pd.Timestamp('2019-01-01', tz='UTC')
END_DATE = pd.Timestamp('2019-12-31 23:00:00', tz='UTC')

# Output file
OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / 'dk1_prices_2019.csv'

# Denmark DK1 zone code
COUNTRY_CODE = 'DK_1'  # DK1 = Western Denmark (where WP2 is located)

print("=" * 80)
print("DOWNLOADING DK1 DAY-AHEAD ELECTRICITY PRICES")
print("=" * 80)
print(f"\nZone: {COUNTRY_CODE}")
print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
print(f"\nOutput location:")
print(f"  Directory: {OUTPUT_DIR}")
print(f"  File: {OUTPUT_FILE.name}")
print(f"  Full path: {OUTPUT_FILE}")
print("=" * 80)

# =============================================================================
# DOWNLOAD DATA
# =============================================================================

try:
    from entsoe import EntsoePandasClient
except ImportError:
    print("\n" + "=" * 80)
    print("ERROR: entsoe-py package not installed!")
    print("=" * 80)
    print("\nInstall with:")
    print("  pip install entsoe-py")
    print("\nor:")
    print("  pip install entsoe-py pandas")
    print("=" * 80)
    exit(1)

print("\n✓ Initializing ENTSO-E client...")
client = EntsoePandasClient(api_key=API_TOKEN)

print("✓ Downloading day-ahead prices...")
print("  (This may take 10-30 seconds...)")

try:
    # Download day-ahead prices (Document type: A44)
    prices = client.query_day_ahead_prices(COUNTRY_CODE, start=START_DATE, end=END_DATE)
    
    print(f"\n✓ Downloaded {len(prices)} hourly price points")
    print(f"  Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"  Mean price: {prices.mean():.2f} €/MWh")
    print(f"  Min price: {prices.min():.2f} €/MWh")
    print(f"  Max price: {prices.max():.2f} €/MWh")
    
except Exception as e:
    print("\n" + "=" * 80)
    print("ERROR downloading data!")
    print("=" * 80)
    print(f"\n{e}")
    print("\nPossible issues:")
    print("  - Invalid API token")
    print("  - Network connection problem")
    print("  - ENTSO-E API is down")
    print("  - Too many requests (rate limit)")
    print("=" * 80)
    exit(1)

# =============================================================================
# SAVE TO CSV
# =============================================================================

print(f"\n✓ Saving to: {OUTPUT_FILE}")

# Convert to DataFrame with proper column name
df = pd.DataFrame({'price_eur_mwh': prices})
df.to_csv(OUTPUT_FILE)

print("✓ Saved successfully!")

# =============================================================================
# VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

# Reload to verify
df_check = pd.read_csv(OUTPUT_FILE, index_col=0, parse_dates=True)

print(f"\n✓ File contains {len(df_check)} rows")
print(f"✓ Columns: {list(df_check.columns)}")
print(f"✓ Date range: {df_check.index[0]} to {df_check.index[-1]}")

# Check for missing data
missing = df_check['price_eur_mwh'].isna().sum()
if missing > 0:
    print(f"\n⚠️  Warning: {missing} missing values found")
    print("   Consider interpolating or handling these gaps")
else:
    print(f"\n✓ No missing values")

# Summary statistics
print("\nPrice statistics:")
print(f"  Mean: {df_check['price_eur_mwh'].mean():.2f} €/MWh")
print(f"  Median: {df_check['price_eur_mwh'].median():.2f} €/MWh")
print(f"  Std dev: {df_check['price_eur_mwh'].std():.2f} €/MWh")
print(f"  Min: {df_check['price_eur_mwh'].min():.2f} €/MWh")
print(f"  Max: {df_check['price_eur_mwh'].max():.2f} €/MWh")

# Price distribution
print("\nPrice percentiles:")
for p in [10, 25, 50, 75, 90]:
    val = df_check['price_eur_mwh'].quantile(p/100)
    print(f"  p{p}: {val:.2f} €/MWh")

# Arbitrage potential
p90 = df_check['price_eur_mwh'].quantile(0.9)
p10 = df_check['price_eur_mwh'].quantile(0.1)
spread = p90 - p10

print(f"\nBattery arbitrage potential:")
print(f"  Spread (p90-p10): {spread:.2f} €/MWh")
print(f"  High-price hours (>p90): {(df_check['price_eur_mwh'] > p90).sum()} hours")
print(f"  Low-price hours (<p10): {(df_check['price_eur_mwh'] < p10).sum()} hours")

print("\n" + "=" * 80)
print("SUCCESS!")
print("=" * 80)
print(f"\n✓ DK1 prices saved to: {OUTPUT_FILE.name}")
print(f"✓ Ready to use with run_wp2_analysis_comprehensive.py")
print("\nThe analysis script will automatically detect and use this file.")
print("=" * 80)