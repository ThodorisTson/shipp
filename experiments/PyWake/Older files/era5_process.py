"""
ERA5 Data Processing for PyWake
================================

Processes downloaded ERA5 NetCDF files for use with PyWake:
- Extracts wind components
- Averages over spatial area
- Extrapolates to hub height
- Saves as CSV for PyWake

Author: Thodoris
Date: January 2026
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

FRYSLAN_CONFIG = {
    'name': 'Windpark Fryslân',
    'latitude': 52.95,
    'longitude': 5.35,
    'hub_height': 108.5,
    'total_capacity_mw': 356,
}

# =============================================================================
# PROCESSING
# =============================================================================

def process_era5_netcdf(netcdf_file, output_csv=None):
    """
    Process ERA5 NetCDF file for PyWake.
    
    Parameters
    ----------
    netcdf_file : str
        Path to ERA5 NetCDF file
    output_csv : str, optional
        Output CSV filename. If None, uses 'wind_data_fryslan.csv' in project directory.
        
    Returns
    -------
    df : pandas.DataFrame
        Processed wind data with columns: timestamp, wind_speed, wind_direction
    """
    # Default output path
    if output_csv is None:
        project_dir = Path(__file__).parent.absolute()
        output_csv = project_dir / 'wind_data_fryslan.csv'
    
    print("\n" + "="*70)
    print("PROCESSING ERA5 DATA")
    print("="*70)
    print(f"\nInput: {netcdf_file}")
    
    # Load NetCDF
    ds = xr.open_dataset(netcdf_file)
    
    print(f"  Dimensions: {dict(ds.sizes)}")
    
    # Average over spatial dimensions (latitude, longitude)
    # Keep time dimension intact!
    ds_point = ds[["u100", "v100"]].mean(dim=["latitude", "longitude"])
    
    print(f"  After spatial average: {dict(ds_point.sizes)}")
    
    # Extract arrays
    u100 = ds_point["u100"].to_numpy()
    v100 = ds_point["v100"].to_numpy()
    
    # ERA5 uses 'valid_time' as dimension name
    time = pd.to_datetime(ds_point["valid_time"].to_numpy())
    
    print(f"\n✓ Loaded {len(time)} hourly records")
    print(f"  Period: {time[0]} to {time[-1]}")
    
    # Calculate wind speed and direction at 100m
    ws_100m = np.sqrt(u100**2 + v100**2)
    
    # Wind direction (meteorological: direction FROM which wind blows)
    wd_100m = (180 + np.degrees(np.arctan2(u100, v100))) % 360
    
    # Extrapolate to hub height (108.5m)
    # IJsselmeer is nearshore: use alpha = 0.12
    hub_height = FRYSLAN_CONFIG['hub_height']
    alpha = 0.12
    
    ws_hub = ws_100m * (hub_height / 100.0)**alpha
    
    print(f"\n✓ Extrapolated to {hub_height}m (alpha={alpha})")
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': time,
        'wind_speed': ws_hub,
        'wind_direction': wd_100m
    })
    
    # Remove NaN
    df_clean = df.dropna()
    
    if len(df_clean) < len(df):
        print(f"  ⚠️  Removed {len(df) - len(df_clean)} NaN records")
    
    # Statistics
    print(f"\n✓ Wind statistics at {hub_height}m:")
    print(f"  Mean: {df_clean['wind_speed'].mean():.2f} m/s")
    print(f"  Median: {df_clean['wind_speed'].median():.2f} m/s")
    print(f"  Std: {df_clean['wind_speed'].std():.2f} m/s")
    print(f"  Max: {df_clean['wind_speed'].max():.2f} m/s")
    print(f"  Min: {df_clean['wind_speed'].min():.2f} m/s")
    
    # Distribution
    print(f"\n  Wind speed distribution:")
    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 100)]
    for low, high in bins:
        count = ((df_clean['wind_speed'] >= low) & (df_clean['wind_speed'] < high)).sum()
        pct = count / len(df_clean) * 100
        print(f"    {low:2d}-{high:2d} m/s: {pct:5.1f}%")
    
    # Save
    df_clean.to_csv(output_csv, index=False)
    print(f"\n✓ Saved: {output_csv}")
    
    ds.close()
    
    return df_clean


def create_validation_plots(df, output_file=None):
    """
    Create validation plots for processed wind data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Wind data with timestamp, wind_speed, wind_direction
    output_file : str, optional
        Output plot filename. If None, uses 'plots/era5_validation.png' in project directory.
    """
    # Default output path
    if output_file is None:
        project_dir = Path(__file__).parent.absolute()
        output_file = project_dir / 'plots' / 'era5_validation.png'
    
    print("\n" + "="*70)
    print("CREATING VALIDATION PLOTS")
    print("="*70)
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Set timestamp as index if not already
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Time series (first 30 days)
    sample = df.iloc[:min(30*24, len(df))]
    axes[0, 0].plot(sample.index, sample['wind_speed'], linewidth=0.8, color='steelblue')
    axes[0, 0].set_ylabel('Wind Speed [m/s]')
    axes[0, 0].set_title('Wind Speed Time Series (First 30 Days)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Wind speed histogram
    axes[0, 1].hist(df['wind_speed'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 1].axvline(df['wind_speed'].mean(), color='red', linestyle='--',
                       linewidth=2, label=f"Mean: {df['wind_speed'].mean():.2f} m/s")
    axes[0, 1].set_xlabel('Wind Speed [m/s]')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Wind Speed Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Wind direction distribution
    axes[1, 0].hist(df['wind_direction'], bins=36, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Wind Direction [°]')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Wind Direction Distribution')
    axes[1, 0].set_xticks([0, 90, 180, 270, 360])
    axes[1, 0].set_xticklabels(['N', 'E', 'S', 'W', 'N'])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Monthly statistics (if full year)
    if len(df) > 31*24:
        df_monthly = df.groupby(df.index.month)['wind_speed'].agg(['mean', 'std'])
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        month_labels = [months[i-1] for i in df_monthly.index]
        
        axes[1, 1].bar(range(len(df_monthly)), df_monthly['mean'],
                       yerr=df_monthly['std'], alpha=0.7, capsize=5,
                       edgecolor='black', color='green')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Wind Speed [m/s]')
        axes[1, 1].set_title('Monthly Mean Wind Speed')
        axes[1, 1].set_xticks(range(len(df_monthly)))
        axes[1, 1].set_xticklabels(month_labels, rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    else:
        # Daily average for short periods
        df_daily = df.resample('D')['wind_speed'].mean()
        axes[1, 1].plot(df_daily.index, df_daily.values, marker='o', markersize=3)
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Wind Speed [m/s]')
        axes[1, 1].set_title('Daily Mean Wind Speed')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

# =============================================================================
# MAIN INTERFACE
# =============================================================================

def process_fryslan_data(netcdf_file=None, create_plots=True):
    """
    Main function to process Windpark Fryslån ERA5 data.
    
    Parameters
    ----------
    netcdf_file : str, optional
        Path to NetCDF file. If None, looks for latest in era5_data/ directory
    create_plots : bool
        Whether to create validation plots (default: True)
        
    Returns
    -------
    df : pandas.DataFrame
        Processed wind data
    """
    # Find NetCDF file if not specified
    if netcdf_file is None:
        project_dir = Path(__file__).parent.absolute()
        data_dir = project_dir / 'era5_data'
        
        if not data_dir.exists():
            print(f"✗ Data directory not found: {data_dir}")
            print("  Run: python era5_download.py")
            return None
        
        # Look for combined file first, then any .nc file
        combined = data_dir / 'era5_fryslan_combined.nc'
        
        if combined.exists():
            netcdf_file = str(combined)
        else:
            nc_files = list(data_dir.glob('era5_fryslan_*.nc'))
            if len(nc_files) == 0:
                print(f"✗ No ERA5 files found in: {data_dir}")
                print("  Run: python era5_download.py")
                return None
            
            # Use most recent file
            netcdf_file = str(sorted(nc_files)[-1])
    
    # Process
    df = process_era5_netcdf(netcdf_file)
    
    # Create plots
    if create_plots and df is not None:
        create_validation_plots(df)
    
    return df

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process ERA5 data for PyWake'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input NetCDF file (default: auto-detect latest)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file (default: wind_data_fryslan.csv in project directory)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Do not create validation plots'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*70)
    print("ERA5 DATA PROCESSING - WINDPARK FRYSLÂN")
    print("="*70)
    print(f"\nWind Farm: {FRYSLAN_CONFIG['name']}")
    print(f"Hub Height: {FRYSLAN_CONFIG['hub_height']} m")
    
    # Process
    df = process_fryslan_data(
        netcdf_file=args.input,
        create_plots=not args.no_plots
    )
    
    # Summary
    if df is not None:
        print("\n" + "="*70)
        print("✓ PROCESSING SUCCESS")
        print("="*70)
        print(f"\nOutput: {args.output}")
        print(f"Records: {len(df)} hours")
        print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Mean wind speed: {df['wind_speed'].mean():.2f} m/s")
        
        # Check if realistic for IJsselmeer
        mean_ws = df['wind_speed'].mean()
        if 7 <= mean_ws <= 10:
            print(f"  ✓ Realistic for IJsselmeer")
        else:
            print(f"  ⚠️  Unusual for IJsselmeer (expected 7-10 m/s)")
        
        print("\n📝 Next steps:")
        print("  python run_fryslan_analysis.py")
        print("="*70)
    else:
        print("\n✗ Processing failed")
