"""
ERA5 Wind Data Downloader for Windpark Fryslân
===============================================

Simplified, automated ERA5 downloader.
- Auto-configures API credentials
- Downloads wind data for Windpark Fryslån
- Saves raw NetCDF files
- No processing (handled separately)

Author: Thodoris
Date: January 2026
"""

import cdsapi
import xarray as xr
from pathlib import Path
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

FRYSLAN_CONFIG = {
    'name': 'Windpark Fryslân',
    'latitude': 52.95,
    'longitude': 5.35,
    'n_turbines': 89,
    'turbine_rating_mw': 4.0,
    'hub_height': 108.5,
    'rotor_diameter': 132,
    'total_capacity_mw': 356,
    'location': 'IJsselmeer, Netherlands'
}

# =============================================================================
# API CONFIGURATION
# =============================================================================

def setup_cdsapi_credentials(token=None):
    """
    Setup CDS API credentials in project directory.
    
    Stores .cdsapirc in the PyWake experiments folder, not home directory.
    This keeps credentials with the project.
    
    If token is provided, creates .cdsapirc file.
    If token is None, checks if .cdsapirc exists.
    
    Parameters
    ----------
    token : str, optional
        Your CDS API personal access token
        
    Returns
    -------
    bool
        True if credentials are configured, False otherwise
    """
    # Use project directory instead of home directory
    project_dir = Path(__file__).parent.absolute()
    cdsapirc_path = project_dir / ".cdsapirc"
    
    if token is not None:
        # Create new credentials file
        content = f"""url: https://cds.climate.copernicus.eu/api
key: {token}
"""
        cdsapirc_path.write_text(content, encoding="utf-8")
        print(f"✓ Credentials saved to: {cdsapirc_path}")
        
        # Set environment variable so cdsapi finds it
        import os
        os.environ['CDSAPI_RC'] = str(cdsapirc_path)
        
        return True
    
    elif cdsapirc_path.exists():
        # Credentials already exist in project
        print(f"✓ Using project credentials: {cdsapirc_path}")
        
        # Set environment variable so cdsapi finds it
        import os
        os.environ['CDSAPI_RC'] = str(cdsapirc_path)
        
        return True
    
    else:
        # Check home directory as fallback
        home_cdsapirc = Path.home() / ".cdsapirc"
        if home_cdsapirc.exists():
            print(f"✓ Found credentials in home directory: {home_cdsapirc}")
            print(f"  (Consider copying to project: {cdsapirc_path})")
            return True
        
        # No credentials found
        print(f"✗ No credentials found")
        print(f"  Checked: {cdsapirc_path}")
        print(f"  Checked: {home_cdsapirc}")
        return False


def get_credentials_interactive():
    """
    Prompt user for credentials if not configured.
    
    Returns
    -------
    bool
        True if configured successfully
    """
    print("\n" + "="*70)
    print("CDS API CREDENTIALS REQUIRED")
    print("="*70)
    print("\nYour CDS API token is needed to download ERA5 data.")
    print("\nTo get your token:")
    print("  1. Go to: https://cds.climate.copernicus.eu/how-to-api")
    print("  2. Login/register")
    print("  3. Copy your 'Personal Access Token'")
    print("\n⚠️  SECURITY: Never share your token publicly!")
    print("="*70)
    
    token = input("\nEnter your CDS API token (or 'skip' to use existing): ").strip()
    
    if token.lower() == 'skip':
        return setup_cdsapi_credentials()
    else:
        return setup_cdsapi_credentials(token)

# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_era5_month(year, month, output_dir=None):
    """
    Download ERA5 data for a single month.
    
    Parameters
    ----------
    year : int
        Year (e.g., 2024)
    month : int
        Month (1-12)
    output_dir : str, optional
        Output directory. If None, uses 'era5_data' in project directory.
        
    Returns
    -------
    str or None
        Path to downloaded file, or None if failed
    """
    # Default to project directory
    if output_dir is None:
        project_dir = Path(__file__).parent.absolute()
        output_dir = project_dir / 'era5_data'
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Output filename
    output_file = f"{output_dir}/era5_fryslan_{year}_{month:02d}.nc"
    
    # Check if already exists
    if Path(output_file).exists():
        file_size = Path(output_file).stat().st_size / 1e6  # MB
        print(f"  ✓ Already downloaded: {output_file} ({file_size:.1f} MB)")
        return output_file
    
    # Configuration
    lat = FRYSLAN_CONFIG['latitude']
    lon = FRYSLAN_CONFIG['longitude']
    
    # Small area around Fryslân (±0.25°)
    area = [
        lat + 0.25,  # North
        lon - 0.25,  # West
        lat - 0.25,  # South
        lon + 0.25,  # East
    ]
    
    # Days in month
    import calendar
    _, last_day = calendar.monthrange(year, month)
    days = [f"{d:02d}" for d in range(1, last_day + 1)]
    
    print(f"  Downloading {year}-{month:02d}...")
    print(f"    Area: {area[0]:.2f}°N, {area[1]:.2f}°E to {area[2]:.2f}°N, {area[3]:.2f}°E")
    print(f"    Days: {len(days)}, Hours: {len(days) * 24}")
    
    try:
        c = cdsapi.Client()
        
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "100m_u_component_of_wind",
                    "100m_v_component_of_wind"
                ],
                "year": str(year),
                "month": f"{month:02d}",
                "day": days,
                "time": [
                    "00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
                    "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
                    "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
                    "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"
                ],
                "area": area,
                "format": "netcdf",
            },
            output_file
        )
        
        file_size = Path(output_file).stat().st_size / 1e6  # MB
        print(f"    ✓ Downloaded: {output_file} ({file_size:.1f} MB)")
        return output_file
        
    except Exception as e:
        print(f"    ✗ Download failed: {e}")
        return None


def download_era5_year(year, output_dir=None):
    """
    Download ERA5 data for entire year (month by month).
    
    Parameters
    ----------
    year : int
        Year to download (e.g., 2024)
    output_dir : str, optional
        Output directory. If None, uses 'era5_data' in project directory.
        
    Returns
    -------
    list
        List of downloaded files
    """
    if output_dir is None:
        project_dir = Path(__file__).parent.absolute()
        output_dir = project_dir / 'era5_data'
    print("\n" + "="*70)
    print(f"DOWNLOADING ERA5 DATA: {year}")
    print("="*70)
    print(f"Location: {FRYSLAN_CONFIG['name']}")
    print(f"Coordinates: {FRYSLAN_CONFIG['latitude']}°N, {FRYSLAN_CONFIG['longitude']}°E")
    print(f"Variables: 100m U/V wind components")
    print(f"Months: January - December")
    print("="*70)
    
    downloaded_files = []
    
    for month in range(1, 13):
        file_path = download_era5_month(year, month, output_dir)
        if file_path:
            downloaded_files.append(file_path)
        print()  # Blank line between months
    
    print("="*70)
    print(f"DOWNLOAD COMPLETE: {len(downloaded_files)}/12 months")
    print("="*70)
    
    if len(downloaded_files) < 12:
        print(f"⚠️  Warning: Only {len(downloaded_files)} months downloaded successfully")
    
    return downloaded_files


def download_era5_months(year, months, output_dir=None):
    """
    Download ERA5 data for specific months.
    
    Parameters
    ----------
    year : int
        Year (e.g., 2024)
    months : list of int
        Months to download (e.g., [1, 2, 3])
    output_dir : str, optional
        Output directory. If None, uses 'era5_data' in project directory.
        
    Returns
    -------
    list
        List of downloaded files
    """
    if output_dir is None:
        project_dir = Path(__file__).parent.absolute()
        output_dir = project_dir / 'era5_data'
    print("\n" + "="*70)
    print(f"DOWNLOADING ERA5 DATA: {year}")
    print("="*70)
    print(f"Location: {FRYSLAN_CONFIG['name']}")
    print(f"Months: {months}")
    print("="*70)
    
    downloaded_files = []
    
    for month in months:
        file_path = download_era5_month(year, month, output_dir)
        if file_path:
            downloaded_files.append(file_path)
        print()
    
    print("="*70)
    print(f"DOWNLOAD COMPLETE: {len(downloaded_files)}/{len(months)} months")
    print("="*70)
    
    return downloaded_files


def combine_monthly_files(monthly_files, output_file=None):
    """
    Combine monthly NetCDF files into single file.
    
    Parameters
    ----------
    monthly_files : list
        List of monthly NetCDF files
    output_file : str, optional
        Combined output file. If None, uses 'era5_data/era5_fryslan_combined.nc'
        
    Returns
    -------
    str
        Path to combined file
    """
    if len(monthly_files) == 0:
        print("✗ No files to combine")
        return None
    
    if len(monthly_files) == 1:
        print(f"✓ Single file, no combining needed: {monthly_files[0]}")
        return monthly_files[0]
    
    # Default output path
    if output_file is None:
        first_file = Path(monthly_files[0])
        output_file = first_file.parent / 'era5_fryslan_combined.nc'
    
    print(f"\nCombining {len(monthly_files)} monthly files...")
    
    # Load and concatenate
    datasets = [xr.open_dataset(f) for f in monthly_files]
    combined = xr.concat(datasets, dim='valid_time')
    
    # Save
    combined.to_netcdf(output_file)
    
    # Close datasets
    for ds in datasets:
        ds.close()
    
    file_size = Path(output_file).stat().st_size / 1e6  # MB
    print(f"✓ Combined file: {output_file} ({file_size:.1f} MB)")
    print(f"  Total hours: {len(combined.valid_time)}")
    
    return output_file

# =============================================================================
# MAIN INTERFACE
# =============================================================================

def download_fryslan_data(year=2024, months=None, combine=True):
    """
    Main function to download Windpark Fryslân wind data.
    
    This is the simplified interface you should use.
    
    Parameters
    ----------
    year : int
        Year to download (default: 2024)
    months : list of int, optional
        Specific months to download (default: all 12 months)
        Example: [1, 2, 3] for Jan-Mar
    combine : bool
        Whether to combine monthly files (default: True)
        
    Returns
    -------
    str
        Path to final NetCDF file (combined or single month)
        
    Examples
    --------
    >>> # Download full year
    >>> download_fryslan_data(2024)
    
    >>> # Download specific months
    >>> download_fryslan_data(2024, months=[1, 6, 12])
    
    >>> # Download single month
    >>> download_fryslan_data(2024, months=[1])
    """
    # Step 1: Check/setup credentials
    if not setup_cdsapi_credentials():
        if not get_credentials_interactive():
            print("\n✗ Cannot proceed without credentials")
            return None
    
    # Step 2: Download data
    if months is None:
        # Full year
        files = download_era5_year(year)
    else:
        # Specific months
        files = download_era5_months(year, months)
    
    if len(files) == 0:
        print("\n✗ No files downloaded")
        return None
    
    # Step 3: Combine if needed
    if combine and len(files) > 1:
        combined_file = combine_monthly_files(files)
        return combined_file
    else:
        return files[0] if len(files) == 1 else files

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download ERA5 wind data for Windpark Fryslân'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        default=2024,
        help='Year to download (default: 2024)'
    )
    
    parser.add_argument(
        '--months',
        type=int,
        nargs='+',
        help='Specific months to download (1-12). If not specified, downloads all.'
    )
    
    parser.add_argument(
        '--token',
        type=str,
        help='CDS API token (if not already configured)'
    )
    
    parser.add_argument(
        '--no-combine',
        action='store_true',
        help='Do not combine monthly files'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: download only January'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*70)
    print("ERA5 DATA DOWNLOADER - WINDPARK FRYSLÂN")
    print("="*70)
    print(f"\nWind Farm: {FRYSLAN_CONFIG['name']}")
    print(f"Location: {FRYSLAN_CONFIG['location']}")
    print(f"Capacity: {FRYSLAN_CONFIG['total_capacity_mw']} MW")
    print(f"Turbines: {FRYSLAN_CONFIG['n_turbines']} × {FRYSLAN_CONFIG['turbine_rating_mw']}MW")
    
    # Setup credentials if provided
    if args.token:
        setup_cdsapi_credentials(args.token)
    
    # Test mode
    if args.test:
        print("\n🧪 TEST MODE: Downloading January only")
        args.months = [1]
    
    # Download
    result = download_fryslan_data(
        year=args.year,
        months=args.months,
        combine=not args.no_combine
    )
    
    # Summary
    if result:
        print("\n" + "="*70)
        print("✓ DOWNLOAD SUCCESS")
        print("="*70)
        if isinstance(result, list):
            print(f"\nDownloaded files:")
            for f in result:
                print(f"  - {f}")
        else:
            print(f"\nFinal file: {result}")
        
        print("\n📝 Next steps:")
        print("  1. Process data: python process_era5.py")
        print("  2. Run PyWake: python run_fryslan_analysis.py")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ DOWNLOAD FAILED")
        print("="*70)
        print("\nTroubleshooting:")
        print("  1. Check API token is correct")
        print("  2. Verify internet connection")
        print("  3. Check CDS service status: https://cds.climate.copernicus.eu")
        print("  4. Accept dataset terms: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels")
        sys.exit(1)
