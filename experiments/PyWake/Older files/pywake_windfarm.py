"""
PyWake Wind Farm Model for SHIPP Integration
=============================================

This module creates a state-of-the-art wind farm model using PyWake to replace
the basic renewable.ninja API approach used in SHIPP example 2.

Key improvements over renewable.ninja:
- Wake effects modeling
- Turbulence modeling
- More realistic power production estimates
- Flexible wind farm layout configuration

Author: Thodoris
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# PyWake imports
from py_wake.wind_turbines import WindTurbine
from py_wake.site import UniformSite
from py_wake.site.xrsite import XRSite
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.turbulence_models import CrespoHernandez


class WindFarmModel:
    """
    Wind farm model using PyWake for realistic power production estimates.
    
    This class handles:
    - Wind turbine specification
    - Wind farm layout
    - Wake modeling
    - Power production calculation
    - Time series generation
    """
    
    def __init__(self, 
                 n_turbines=10,
                 turbine_rating_mw=5.0,
                 rotor_diameter=136,
                 hub_height=90,
                 ct_curve=None,
                 power_curve=None):
        """
        Initialize wind farm model.
        
        Parameters
        ----------
        n_turbines : int
            Number of turbines in the farm
        turbine_rating_mw : float
            Rated power of each turbine in MW
        rotor_diameter : float
            Rotor diameter in meters (default: DTU 10MW reference turbine)
        hub_height : float
            Hub height in meters
        ct_curve : tuple (ws, ct), optional
            Wind speed and thrust coefficient curve
        power_curve : tuple (ws, power), optional
            Wind speed and power curve in MW
        """
        self.n_turbines = n_turbines
        self.turbine_rating_mw = turbine_rating_mw
        self.rotor_diameter = rotor_diameter
        self.hub_height = hub_height
        
        # Create wind turbine with power curve
        self.wind_turbine = self._create_turbine(ct_curve, power_curve)
        
        # Wind farm layout (will be set later)
        self.x = None
        self.y = None
        
        # Wake model configuration (set by setup_wake_model)
        self.wake_model_class = None
        self.turbulence_model = None
        self.wake_model_configured = False
        
    def _create_turbine(self, ct_curve=None, power_curve=None):
        """
        Create wind turbine object with power and thrust curves.
        
        Uses simplified curves if not provided.
        """
        if power_curve is None:
            # Generic offshore wind turbine curve
            ws = np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25])
            # Simplified power curve (fraction of rated power)
            power_frac = np.array([0, 0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.78, 0.88, 0.95, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0])
            power_mw = power_frac * self.turbine_rating_mw
        else:
            ws, power_mw = power_curve
            
        if ct_curve is None:
            # Generic thrust coefficient curve
            ct = np.array([0, 0, 0.8, 0.82, 0.83, 0.82, 0.80, 0.77, 0.73, 0.68, 0.62, 0.55, 0.48, 0.42, 0.30, 0.25])
        else:
            _, ct = ct_curve
        
        # Convert power from MW to kW for PyWake
        power_kw = power_mw * 1000
        
        # Create power-ct table (PyWake 2.6+ format)
        # Format: ws, power (kW), ct
        from py_wake.wind_turbines import WindTurbine
        from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
        
        power_ct_func = PowerCtTabular(
            ws=ws,
            power=power_kw,
            power_unit='kW',
            ct=ct,
            ws_cutin=ws[np.where(power_kw > 0)[0][0]] if np.any(power_kw > 0) else ws[0],
            ws_cutout=ws[-1]
        )
        
        # Create custom wind turbine
        wt = WindTurbine(
            name=f'Generic_{self.turbine_rating_mw}MW',
            diameter=self.rotor_diameter,
            hub_height=self.hub_height,
            powerCtFunction=power_ct_func
        )
        return wt
        
    def create_layout(self, layout_type='grid', spacing=5.0):
        """
        Create wind farm layout.
        
        Parameters
        ----------
        layout_type : str
            'grid', 'row', or 'custom'
        spacing : float
            Spacing in rotor diameters
        
        Returns
        -------
        x, y : array
            Turbine coordinates in meters
        """
        D = self.rotor_diameter
        
        if layout_type == 'grid':
            # Grid layout - common for offshore
            n_rows = int(np.ceil(np.sqrt(self.n_turbines)))
            n_cols = int(np.ceil(self.n_turbines / n_rows))
            
            x_coords = []
            y_coords = []
            for i in range(n_rows):
                for j in range(n_cols):
                    if len(x_coords) < self.n_turbines:
                        x_coords.append(j * spacing * D)
                        y_coords.append(i * spacing * D)
            
            self.x = np.array(x_coords)
            self.y = np.array(y_coords)
            
        elif layout_type == 'row':
            # Single row layout
            self.x = np.arange(self.n_turbines) * spacing * D
            self.y = np.zeros(self.n_turbines)
            
        else:
            raise ValueError(f"Unknown layout type: {layout_type}")
        
        return self.x, self.y
    
    def setup_wake_model(self, use_turbulence=True):
        """
        Setup wake model with optional turbulence.
        
        Parameters
        ----------
        use_turbulence : bool
            Whether to include turbulence modeling
            
        Note
        ----
        In PyWake 2.6+, wake models are initialized during simulation,
        so we just store the model class and parameters here.
        """
        # Store wake model class and parameters for later use
        self.use_turbulence = use_turbulence
        self.wake_model_class = IEA37SimpleBastankhahGaussian
        
        # Store turbulence model if needed
        if use_turbulence:
            self.turbulence_model = CrespoHernandez()
        else:
            self.turbulence_model = None
        
        # Set a flag that wake model is configured
        self.wake_model_configured = True
    
    def calculate_aep(self, site, wd=None, ws=None):
        """
        Calculate Annual Energy Production.
        
        Parameters
        ----------
        site : UniformSite or XRSite
            Site object with wind conditions
        wd : array, optional
            Wind directions to evaluate (degrees)
        ws : array, optional
            Wind speeds to evaluate (m/s)
            
        Returns
        -------
        aep_gwh : float
            Annual energy production in GWh
        """
        if not self.wake_model_configured:
            self.setup_wake_model()
        
        # Initialize wake model with site and wind turbines (PyWake 2.6+ API)
        if self.turbulence_model is not None:
            wake_model = self.wake_model_class(
                site, self.wind_turbine,
                turbulenceModel=self.turbulence_model
            )
        else:
            wake_model = self.wake_model_class(site, self.wind_turbine)
        
        # Run simulation
        sim_res = wake_model(self.x, self.y, wd=wd, ws=ws)
        
        # Calculate AEP
        aep_gwh = sim_res.aep().sum() / 1e9  # Convert from Wh to GWh
        
        return aep_gwh
    
    def generate_timeseries(
        self,
        wind_speed_ts,
        wind_direction_ts,
        ti=0.06,
        shear=0.14,
        chunk_size=2000,
        verbose=True,
    ):
        """
        Generate power production time series from wind data (VECTORIZED).
        
        This version calls PyWake once per chunk, not once per hour!
        Speed: ~60-100x faster than per-hour loops.
        
        Parameters
        ----------
        wind_speed_ts : pandas.Series or array-like
            Wind speed time series at hub height (m/s)
        wind_direction_ts : pandas.Series or array-like
            Wind direction time series (degrees, meteorological "from")
        ti : float
            Turbulence intensity (default: 0.06)
        shear : float
            Wind shear exponent (default: 0.14 for offshore)
        chunk_size : int
            Number of hours evaluated per PyWake call (memory/speed tradeoff)
            Recommended: 1000-4000 depending on RAM
            Default: 2000 (safe for most systems)
        verbose : bool
            Print progress information
            
        Returns
        -------
        power_ts : pandas.Series
            Total wind farm power production in MW
        """
        # Convert inputs to numpy arrays once
        if isinstance(wind_speed_ts, pd.Series):
            ws_array = wind_speed_ts.to_numpy(dtype=float)
            index = wind_speed_ts.index
        else:
            ws_array = np.asarray(wind_speed_ts, dtype=float)
            index = None
        
        if isinstance(wind_direction_ts, pd.Series):
            wd_array = wind_direction_ts.to_numpy(dtype=float)
        else:
            wd_array = np.asarray(wind_direction_ts, dtype=float)
        
        if len(ws_array) != len(wd_array):
            raise ValueError(
                f"Wind speed and direction length mismatch: {len(ws_array)} vs {len(wd_array)}"
            )
        
        n_steps = len(ws_array)
        power_array = np.zeros(n_steps, dtype=float)
        
        if not self.wake_model_configured:
            self.setup_wake_model()
        
        # Create site (time series provided via ws/wd arrays)
        from py_wake.site.shear import PowerShear
        site = UniformSite(
            p_wd=[1],
            ti=ti,
            shear=PowerShear(h_ref=self.hub_height, alpha=shear),
        )
        
        # Initialize wake model
        if self.turbulence_model is not None:
            wake_model = self.wake_model_class(
                site, self.wind_turbine,
                turbulenceModel=self.turbulence_model
            )
        else:
            wake_model = self.wake_model_class(site, self.wind_turbine)
        
        if verbose:
            print(f"Calculating power for {n_steps} timesteps (VECTORIZED)...")
            print(f"  Chunk size: {chunk_size} hours per PyWake call")
        
        # Helper: sum over turbine dimension robustly
        def _sum_over_turbines(power_da):
            """Sum over turbine dimension (handles different PyWake versions)."""
            # Common turbine dimension names
            for dim in ("wt", "turbine", "iwt"):
                if dim in power_da.dims:
                    return power_da.sum(dim=dim)
            # Fallback: sum over all dims except time-like
            non_time_dims = [d for d in power_da.dims if d not in ("time",)]
            if non_time_dims:
                return power_da.sum(dim=non_time_dims)
            return power_da
        
        # CHUNKED VECTORIZED SIMULATION
        # Key optimization: ONE PyWake call per chunk instead of per hour!
        for i in range(0, n_steps, chunk_size):
            end_idx = min(i + chunk_size, n_steps)
            batch_ws = ws_array[i:end_idx]
            batch_wd = wd_array[i:end_idx]
            
            # VECTORIZED CALL: One simulation for entire chunk!
            # CRITICAL: time=True makes wd/ws be treated as PAIRED time series
            # Without this, PyWake creates Cartesian product (grid) → wrong shape!
            sim_res = wake_model(self.x, self.y, wd=batch_wd, ws=batch_ws, time=True)
            
            # Power is typically in W, convert to MW
            farm_power_w = _sum_over_turbines(sim_res.Power).values
            power_array[i:end_idx] = farm_power_w / 1e6
            
            # Progress reporting (every 5th chunk or at start/end)
            if verbose and (i == 0 or end_idx == n_steps or (i // chunk_size) % 5 == 0):
                print(f"  Progress: {end_idx}/{n_steps} timesteps ({end_idx/n_steps*100:.1f}%)")
        
        if verbose:
            print(f"  ✓ Complete! Mean power: {power_array.mean():.2f} MW")
        
        # Return as pandas Series
        if index is not None:
            return pd.Series(power_array, index=index, name="wind_power_mw")
        return pd.Series(power_array, name="wind_power_mw")
    
    def get_power_timeseries_for_shipp(self, wind_speed, wind_direction, 
                                       dt=1, ti=0.06):
        """
        Generate power time series in format compatible with SHIPP Example 2.
        
        This is a drop-in replacement for the renewable.ninja API call.
        
        Parameters
        ----------
        wind_speed : array-like
            Wind speed at hub height [m/s], length n
        wind_direction : array-like
            Wind direction [degrees], length n
        dt : float
            Timestep in hours (default: 1 for hourly)
        ti : float
            Turbulence intensity
            
        Returns
        -------
        power_mw : np.ndarray
            Wind farm power in MW, length n
            
        Example
        -------
        >>> # Replace renewable.ninja call in Example 2:
        >>> # OLD: data_power = get_renewables_ninja(...)
        >>> # NEW:
        >>> wind_data = load_wind_data('wind.csv')
        >>> wf = create_reference_offshore_windfarm(n_turbines=10, turbine_mw=5.0)
        >>> data_power = wf.get_power_timeseries_for_shipp(
        ...     wind_data['wind_speed'].values,
        ...     wind_data['wind_direction'].values
        ... )
        """
        # Use the vectorized generate_timeseries
        power_ts = self.generate_timeseries(
            wind_speed,
            wind_direction,
            ti=ti
        )
        
        # Return as numpy array (what Example 2 expects)
        return power_ts.values
    
    def plot_layout(self):
        """Plot wind farm layout."""
        if self.x is None or self.y is None:
            raise ValueError("Layout not created yet. Call create_layout() first.")
        
        plt.figure(figsize=(10, 8))
        plt.scatter(self.x, self.y, s=200, c='blue', marker='o', alpha=0.6)
        
        # Add turbine labels
        for i, (x, y) in enumerate(zip(self.x, self.y)):
            plt.text(x, y, str(i+1), ha='center', va='center', color='white', fontsize=8, fontweight='bold')
        
        # Add rotor diameter circles
        from matplotlib.patches import Circle
        for x, y in zip(self.x, self.y):
            circle = Circle((x, y), self.rotor_diameter/2, fill=False, 
                          edgecolor='gray', linestyle='--', alpha=0.3)
            plt.gca().add_patch(circle)
        
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title(f'Wind Farm Layout - {self.n_turbines} × {self.turbine_rating_mw}MW Turbines')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        return plt.gcf()

def create_reference_offshore_windfarm(n_turbines=10, turbine_mw=5.0):
    """
    Create a reference offshore wind farm for SHIPP optimization.
    
    This is a convenience function that sets up a typical offshore wind farm
    configuration suitable for energy storage optimization studies.
    
    Parameters
    ----------
    n_turbines : int
        Number of turbines
    turbine_mw : float
        Turbine rating in MW
        
    Returns
    -------
    wf_model : WindFarmModel
        Configured wind farm model
    """
    # Create model with DTU 10MW reference turbine parameters (scaled)
    wf_model = WindFarmModel(
        n_turbines=n_turbines,
        turbine_rating_mw=turbine_mw,
        rotor_diameter=136,  # Scaled for 5MW
        hub_height=90
    )
    
    # Create grid layout with 7D spacing (typical for offshore)
    wf_model.create_layout(layout_type='grid', spacing=7.0)
    
    # Setup wake model with turbulence
    wf_model.setup_wake_model(use_turbulence=True)
    
    return wf_model

def generate_synthetic_wind_data(hours=8760, mean_ws=9.0, ti=0.06):
    """
    Generate synthetic wind speed and direction time series.
    
    This is a placeholder for real wind data (ERA5, mesoscale models, etc.)
    
    Parameters
    ----------
    hours : int
        Number of hours to generate
    mean_ws : float
        Mean wind speed (m/s)
    ti : float
        Turbulence intensity
        
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with 'wind_speed' and 'wind_direction' columns
    """
    # Create hourly timestamp
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
    
    # Generate wind speed using Weibull-like distribution
    # Add temporal correlation using AR(1) process
    rng = np.random.RandomState(42)
    
    # Shape and scale for Weibull (offshore conditions)
    k_weibull = 2.0  # Shape parameter
    scale_weibull = mean_ws / 0.886  # Scale to get desired mean
    
    # Generate base Weibull samples
    ws_base = rng.weibull(k_weibull, hours) * scale_weibull
    
    # Add temporal correlation (AR1 with correlation coefficient)
    rho = 0.95  # High correlation for realistic wind
    ws_array = np.zeros(hours)
    ws_array[0] = ws_base[0]
    
    for i in range(1, hours):
        ws_array[i] = rho * ws_array[i-1] + (1-rho) * ws_base[i]
    
    # Add turbulence
    ws_turb = ws_array + rng.normal(0, ti * ws_array)
    ws_turb = np.maximum(ws_turb, 0)  # No negative wind speeds
    
    # Generate wind direction (with some prevailing direction)
    # Offshore typically has prevailing westerlies
    wd_base = rng.vonmises(np.radians(270), 2, hours)  # Concentrated around 270°
    wd_array = np.degrees(wd_base) % 360
    
    # Add temporal correlation to wind direction
    wd_smooth = np.zeros(hours)
    wd_smooth[0] = wd_array[0]
    rho_wd = 0.98
    
    for i in range(1, hours):
        wd_smooth[i] = rho_wd * wd_smooth[i-1] + (1-rho_wd) * wd_array[i]
        wd_smooth[i] = wd_smooth[i] % 360
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'wind_speed': ws_turb,
        'wind_direction': wd_smooth
    })
    df.set_index('timestamp', inplace=True)
    
    return df

if __name__ == '__main__':
    """
    Example usage and testing of the wind farm model.
    """
    print("=" * 70)
    print("PyWake Wind Farm Model - Testing")
    print("=" * 70)
    
    # Create reference wind farm
    print("\n1. Creating reference offshore wind farm...")
    wf = create_reference_offshore_windfarm(n_turbines=10, turbine_mw=5.0)
    print(f"   Created: {wf.n_turbines} × {wf.turbine_rating_mw}MW wind farm")
    print(f"   Total capacity: {wf.n_turbines * wf.turbine_rating_mw:.0f} MW")
    
    # Plot layout
    print("\n2. Plotting wind farm layout...")
    fig = wf.plot_layout()
    plt.show(block=False)
    
    # Generate synthetic wind data
    print("\n3. Generating synthetic wind data (1 year, hourly)...")
    wind_data = generate_synthetic_wind_data(hours=8760, mean_ws=9.0)
    print(f"   Mean wind speed: {wind_data['wind_speed'].mean():.2f} m/s")
    print(f"   Max wind speed: {wind_data['wind_speed'].max():.2f} m/s")
    
    # Calculate power time series (using smaller sample for speed)
    print("\n4. Calculating power production time series...")
    print("   Note: Using first 168 hours (1 week) for demonstration")
    sample_data = wind_data.iloc[:168]  # First week
    
    power_ts = wf.generate_timeseries(
        sample_data['wind_speed'],
        sample_data['wind_direction'],
        ti=0.06
    )
    
    print(f"   Mean power: {power_ts.mean():.2f} MW")
    print(f"   Capacity factor: {power_ts.mean() / (wf.n_turbines * wf.turbine_rating_mw) * 100:.1f}%")
    
    # Plot power time series
    print("\n5. Plotting power production...")
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(power_ts.index, power_ts.values, linewidth=1)
    plt.ylabel('Power [MW]')
    plt.title('Wind Farm Power Production (1 week)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.hist(power_ts.values, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Power [MW]')
    plt.ylabel('Frequency')
    plt.title('Power Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 70)
    print("Testing complete!")
    print("=" * 70)