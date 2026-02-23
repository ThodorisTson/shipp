"""
WP2 Common Functions - Shared Setup for PyWake Analysis
========================================================

This module centralizes all YAML loading, interpolation, and PyWake object
construction to eliminate code duplication across scripts.

Author: Thodoris
Date: 2026-02-11
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
import xarray as xr

# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================

DEFAULT_CONFIG = {
    'interp_n': 2000,           # Curve interpolation points (10000 for final validation)
    'wd_step': 5,               # Wind direction step (degrees) - use 1 for final
    'ws_step': 1,               # Wind speed step (m/s)
    'ws_min': 4,                # Minimum wind speed (m/s)
    'ws_max': 24,               # Maximum wind speed (m/s)
    'hub_height': 120,          # Hub height (m) - parameterized!
    'shear_alpha': 0.2,        # Wind shear exponent (ocean: 0.14, land: 0.2)
}

# =============================================================================
# YAML LOADING WITH INCLUDE SUPPORT
# =============================================================================

def load_yaml_with_includes(yaml_path):
    """
    Load YAML file with !include directive support.
    
    Parameters
    ----------
    yaml_path : Path or str
        Path to YAML file
        
    Returns
    -------
    dict
        Parsed YAML content
    """
    yaml_path = Path(yaml_path)
    
    # Custom constructor for !include
    def yaml_include(loader, node):
        filepath = Path(loader.construct_scalar(node))
        if not filepath.is_absolute():
            filepath = yaml_path.parent / filepath
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    
    yaml.add_constructor('!include', yaml_include, Loader=yaml.SafeLoader)
    
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

# =============================================================================
# SITE LOADING
# =============================================================================

def load_wp2_site(site_yaml_path):
    """
    Load WP2 site + Weibull data from YAML file
    Structure expected:

    site:
    wind_resource:
        sectors:
    """
    data = load_yaml_with_includes(site_yaml_path)

    site_info = data['site']
    wr = data['wind_resource']
    sectors = wr['sectors']

    wd = np.array([s['direction_deg'] for s in sectors])
    A = np.array([s['weibull_a'] for s in sectors])
    k = np.array([s['weibull_k'] for s in sectors])
    freq = np.array([s['probability'] for s in sectors])

    freq = freq / freq.sum()

    return {
        'name': site_info.get('name', 'WP2 site'),
        'latitude': site_info.get('latitude'),
        'longitude': site_info.get('longitude'),
        'altitude': site_info.get('altitude', 0),
        'weibull_A': A,
        'weibull_k': k,
        'sector_centers': wd,
        'frequencies': freq,
        'ti': np.mean([s.get('turbulence_intensity', 0.1) for s in sectors]),
    }

# =============================================================================
# TURBINE LOADING
# =============================================================================

def load_wp2_turbine(turbine_yaml_path, interp_n=None):
    """
    Load WP2 turbine data from YAML file with curve interpolation.
    
    Parameters
    ----------
    turbine_yaml_path : Path or str
        Path to turbine YAML file
    interp_n : int, optional
        Number of interpolation points (default: from config)
        
    Returns
    -------
    dict
        Turbine data with keys: name, diameter, hub_height, rated_power,
        ws_interp, power_interp, ct_interp
    """
    if interp_n is None:
        interp_n = DEFAULT_CONFIG['interp_n']
    
    turbine_dat = load_yaml_with_includes(turbine_yaml_path)
    
    # Extract power curve
    pc = turbine_dat['performance']['power_curve']
    ws_pc = np.array(pc['power_wind_speeds'])
    power_pc = np.array(pc['power_values']) #already in W
    
    # Extract Ct curve
    ct_curve = turbine_dat['performance']['Ct_curve']
    ws_ct = np.array(ct_curve['Ct_wind_speeds'])
    ct_values = np.array(ct_curve['Ct_values'])
    
    # Interpolate to dense grid (IEA 740 style)
    ws_min_interp = min(ws_pc.min(), ws_ct.min())
    ws_max_interp = max(ws_pc.max(), ws_ct.max())
    ws_interp = np.linspace(ws_min_interp, ws_max_interp, interp_n)
    
    power_interp = np.interp(ws_interp, ws_pc, power_pc)
    ct_interp = np.interp(ws_interp, ws_ct, ct_values)
    
    return {
        'name': turbine_dat['name'],
        'diameter': turbine_dat['rotor_diameter'],
        'hub_height': turbine_dat['hub_height'],
        'rated_power': turbine_dat['performance']['rated_power'], 
        'ws_interp': ws_interp,
        'power_interp': power_interp,
        'ct_interp': ct_interp,
    }

# =============================================================================
# LAYOUT LOADING
# =============================================================================

def load_wp2_layout(layout_csv_path):
    """
    Load WP2 turbine layout from CSV file.
    
    Parameters
    ----------
    layout_csv_path : Path or str
        Path to layout CSV file
        
    Returns
    -------
    tuple
        (x, y, n_turbines) where x and y are arrays in meters
    """
    layout_df = pd.read_csv(layout_csv_path)

    # Allow both naming conventions:
    if 'x' in layout_df.columns and 'y' in layout_df.columns:
        x = layout_df['x'].values
        y = layout_df['y'].values

    elif 'X_m' in layout_df.columns and 'Y_m' in layout_df.columns:
        x = layout_df['X_m'].values
        y = layout_df['Y_m'].values

    else:
        raise KeyError(
            "Layout CSV must contain either columns "
            "('x','y') or ('X_m','Y_m'). "
            f"Found columns: {list(layout_df.columns)}"
        )

    n_turbines = len(x)

    return x, y, n_turbines


# =============================================================================
# PYWAKE OBJECT CONSTRUCTION
# =============================================================================

def build_pywake_objects(site_dat, turbine_dat, config=None):
    """
    Build PyWake XRSite and WindTurbine objects from loaded data.

    Parameters
    ----------
    site_dat : dict
        Site data from load_wp2_site()
    turbine_dat : dict
        Turbine data from load_wp2_turbine()
    config : dict, optional
        Configuration overrides (default: DEFAULT_CONFIG)

    Returns
    -------
    tuple
        (site, windturbine, ws_bins, wd_bins)
    """

    # -----------------------------
    # Merge configuration
    # -----------------------------
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(config)
        config = cfg

    # -----------------------------
    # Extract Weibull sector data
    # -----------------------------
    wd = site_dat['sector_centers']
    freq = site_dat['frequencies']
    A = site_dat['weibull_A']
    k = site_dat['weibull_k']

    # Triplicate for cyclic interpolation
    wd_cyclic = np.concatenate([wd - 360, wd, wd + 360])
    freq_cyclic = np.concatenate([freq, freq, freq])
    A_cyclic = np.concatenate([A, A, A])
    k_cyclic = np.concatenate([k, k, k])

    # Wind speed grid
    ws = np.arange(0, 30, 0.5)

    # -----------------------------
    # Proper XRSite construction
    # -----------------------------
    ds = xr.Dataset(
        data_vars={
            'Sector_frequency': ('wd', freq_cyclic),
            'Weibull_A': ('wd', A_cyclic),
            'Weibull_k': ('wd', k_cyclic),
            'TI': ('ws', np.full(len(ws), site_dat['ti'])),
        },
        coords={
            'wd': wd_cyclic,
            'ws': ws
        }
    )

    site = XRSite(
        ds=ds,
        initial_position=np.array([[0, 0]]),
        interp_method='linear',
        shear=None,
    )

    # -----------------------------
    # Turbine construction
    # -----------------------------
    windturbine = WindTurbine(
        name=turbine_dat['name'],
        diameter=turbine_dat['diameter'],
        hub_height=turbine_dat['hub_height'],
        powerCtFunction=PowerCtTabular(
            ws=turbine_dat['ws_interp'],
            power=turbine_dat['power_interp'],
            power_unit='W',
            ct=turbine_dat['ct_interp'],
        )
    )

    # -----------------------------
    # Simulation bins
    # -----------------------------
    ws_bins = np.arange(
        config['ws_min'],
        config['ws_max'] + config['ws_step'],
        config['ws_step']
    )

    wd_bins = np.arange(0, 360, config['wd_step'])

    return site, windturbine, ws_bins, wd_bins

# =============================================================================
# WAKE MODEL FACTORY
# =============================================================================

def get_wake_model(model_name, site, windturbine, **kwargs):
    """
    Factory function to create wake models by name.
    
    Parameters
    ----------
    model_name : str
        Wake model name: 'NOJ', 'Bastankhah', 'TurboNOJ', 'TurboPark', etc.
    site : XRSite
        PyWake site object
    windturbine : WindTurbine
        PyWake wind turbine object
    **kwargs
        Additional model-specific parameters
        
    Returns
    -------
    WakeModel
        Configured PyWake wake model
    """
    from py_wake.deficit_models.noj import NOJ
    from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
    #from py_wake.deficit_models.gaussian import BastankhahGaussian
    from py_wake.turbulence_models.crespo import CrespoHernandez
    from py_wake.superposition_models import LinearSum
    from py_wake.rotor_avg_models import EqGridRotorAvg

    
    if model_name.upper() == 'NOJ':
        return NOJ(site, windturbine, **kwargs)
    
    elif model_name.upper() == 'BASTANKHAH':
        # Bastankhah-Gaussian with Crespo turbulence
        return Bastankhah_PorteAgel_2014(
            site, 
            windturbine,
            k=0.032,
            turbulenceModel=CrespoHernandez(),
            superpositionModel=LinearSum(),
            rotorAvgModel=EqGridRotorAvg(3),   # 3×3 equal-spaced grid
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown wake model: {model_name}")

# =============================================================================
# WIND SHEAR CORRECTION
# =============================================================================

def apply_wind_shear(ws_ref, z_ref, z_target, alpha=None):
    """
    Apply power-law wind shear correction.
    
    Parameters
    ----------
    ws_ref : array_like
        Wind speed at reference height
    z_ref : float
        Reference height (m)
    z_target : float
        Target height (m)
    alpha : float, optional
        Shear exponent (default: from config)
        
    Returns
    -------
    array_like
        Wind speed at target height
    """
    if alpha is None:
        alpha = DEFAULT_CONFIG['shear_alpha']
    
    return ws_ref * (z_target / z_ref) ** alpha

# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================

def print_site_summary(site_dat):
    """Print formatted summary of site data."""
    print(f"\n✓ Site: {site_dat['name']}")
    print(f"  Location: {site_dat['latitude']:.2f}°N, {site_dat['longitude']:.2f}°E")
    print(f"  Altitude: {site_dat['altitude']} m")
    print(f"  Mean Weibull A: {site_dat['weibull_A'].mean():.2f} m/s")
    print(f"  Mean Weibull k: {site_dat['weibull_k'].mean():.2f}")
    print(f"  Turbulence intensity: {site_dat['ti']*100:.1f}%")
    print(f"  Sectors: {len(site_dat['sector_centers'])}")

def print_turbine_summary(turbine_dat):
    """Print formatted summary of turbine data."""
    print(f"\n✓ Turbine: {turbine_dat['name']}")
    print(f"  Rotor diameter: {turbine_dat['diameter']:.1f} m")
    print(f"  Hub height: {turbine_dat['hub_height']:.1f} m")
    print(f"  Rated power: {turbine_dat['rated_power']/1e6:.1f} MW")
    area = np.pi * (turbine_dat['diameter']/2)**2
    print(f"  Specific power: {turbine_dat['rated_power']/area:.0f} W/m²")

def print_layout_summary(x, y, rated_power):
    """Print formatted summary of layout."""
    n_turbines = len(x)
    total_capacity = n_turbines * rated_power / 1e6
    print(f"\n✓ Layout: {n_turbines} turbines")
    print(f"  Total capacity: {total_capacity:.1f} MW")
    print(f"  Farm area: {(x.max()-x.min())/1000:.1f} × {(y.max()-y.min())/1000:.1f} km")

def print_config_summary(config):
    """Print formatted summary of configuration."""
    print(f"\n✓ Configuration:")
    print(f"  Interpolation points: {config.get('interp_n', DEFAULT_CONFIG['interp_n'])}")
    print(f"  Wind direction step: {config.get('wd_step', DEFAULT_CONFIG['wd_step'])}°")
    print(f"  Wind speed step: {config.get('ws_step', DEFAULT_CONFIG['ws_step'])} m/s")
    print(f"  Hub height: {config.get('hub_height', DEFAULT_CONFIG['hub_height'])} m")

# =============================================================================
# QUICK SETUP FUNCTION (ALL-IN-ONE)
# =============================================================================

def quick_setup(site_yaml, turbine_yaml, layout_csv, config=None, verbose=True):
    """
    One-function setup: load everything and return PyWake objects.
    
    Parameters
    ----------
    site_yaml : Path or str
        Path to site YAML
    turbine_yaml : Path or str
        Path to turbine YAML
    layout_csv : Path or str
        Path to layout CSV
    config : dict, optional
        Configuration overrides
    verbose : bool, optional
        Print summaries (default: True)
        
    Returns
    -------
    dict
        Dictionary with keys: site, windturbine, x, y, n_turbines,
        site_dat, turbine_dat, ws_bins, wd_bins, config
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    if verbose:
        print("=" * 80)
        print("LOADING WP2 CONFIGURATION")
        print("=" * 80)
    
    # Load data
    site_dat = load_wp2_site(site_yaml)
    turbine_dat = load_wp2_turbine(turbine_yaml, config.get('interp_n'))
    x, y, n_turbines = load_wp2_layout(layout_csv)
    
    if verbose:
        print_site_summary(site_dat)
        print_turbine_summary(turbine_dat)
        print_layout_summary(x, y, turbine_dat['rated_power'])
        print_config_summary(config)
    
    # Build PyWake objects
    site, windturbine, ws_bins, wd_bins = build_pywake_objects(site_dat, turbine_dat, config)
    
    return {
        'site': site,
        'windturbine': windturbine,
        'x': x,
        'y': y,
        'n_turbines': n_turbines,
        'site_dat': site_dat,
        'turbine_dat': turbine_dat,
        'ws_bins': ws_bins,
        'wd_bins': wd_bins,
        'config': config,
    }
