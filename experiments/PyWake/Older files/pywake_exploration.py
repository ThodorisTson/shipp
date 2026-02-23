"""
PyWake Model Deep Dive - Complete Guide
========================================

This guide explains the PyWake wind farm model in detail and how to use it
effectively for your SHIPP optimization thesis work.

Author: Thodoris
Date: January 2026
"""

# =============================================================================
# PART 1: UNDERSTANDING THE MODEL STRUCTURE
# =============================================================================

"""
The WindFarmModel class has 4 main components:

1. TURBINE SPECIFICATION
   - Power curve: How much power at each wind speed
   - Thrust coefficient (Ct): How much wind the turbine "blocks"
   - Physical dimensions: rotor diameter, hub height

2. WIND FARM LAYOUT
   - Turbine positions (x, y coordinates)
   - Spacing between turbines (affects wake interactions)

3. WAKE MODEL
   - IEA37SimpleBastankhahGaussian: Industry-standard Gaussian wake model
   - Models how turbines affect downstream turbines
   - Optional turbulence modeling

4. POWER CALCULATION
   - Single timestep: calculate_aep()
   - Time series: generate_timeseries()
"""

# =============================================================================
# PART 2: QUICK START EXAMPLES
# =============================================================================

from setup_paths import configure_paths
configure_paths()

from pywake_windfarm import (
    WindFarmModel,
    create_reference_offshore_windfarm,
    generate_synthetic_wind_data
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Example 1: Create a simple wind farm
# -----------------------------------------------------------------------------

def example_1_basic_windfarm():
    """Create and visualize a basic wind farm."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Wind Farm Creation")
    print("="*70)
    
    # Create a 10-turbine, 5MW wind farm
    wf = create_reference_offshore_windfarm(n_turbines=10, turbine_mw=5.0)
    
    print(f"Wind farm created:")
    print(f"  - Number of turbines: {wf.n_turbines}")
    print(f"  - Turbine rating: {wf.turbine_rating_mw} MW")
    print(f"  - Total capacity: {wf.n_turbines * wf.turbine_rating_mw} MW")
    print(f"  - Rotor diameter: {wf.rotor_diameter} m")
    print(f"  - Hub height: {wf.hub_height} m")
    
    # Plot layout
    wf.plot_layout()
    plt.savefig('windfarm_layout.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Layout saved to: windfarm_layout.png")
    
    return wf


# -----------------------------------------------------------------------------
# Example 2: Understanding the power curve
# -----------------------------------------------------------------------------

def example_2_power_curve():
    """Explore the turbine power curve."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Turbine Power Curve")
    print("="*70)
    
    wf = create_reference_offshore_windfarm(n_turbines=1, turbine_mw=5.0)
    
    # Test at different wind speeds
    wind_speeds = np.linspace(0, 25, 100)
    
    # Get power and Ct for each wind speed
    # PyWake 2.6+ returns power in W and ct as scalars/arrays
    power_w, ct = wf.wind_turbine.powerCtFunction(wind_speeds)
    
    # Convert to MW for plotting
    powers = power_w / 1e6
    cts = ct
    
    # Plot power curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(wind_speeds, powers, linewidth=2, color='steelblue')
    ax1.axhline(wf.turbine_rating_mw, color='red', linestyle='--', 
                label=f'Rated power: {wf.turbine_rating_mw} MW')
    ax1.set_xlabel('Wind Speed [m/s]')
    ax1.set_ylabel('Power [MW]')
    ax1.set_title('Turbine Power Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot thrust coefficient
    ax2.plot(wind_speeds, cts, linewidth=2, color='orangered')
    ax2.set_xlabel('Wind Speed [m/s]')
    ax2.set_ylabel('Thrust Coefficient Ct [-]')
    ax2.set_title('Thrust Coefficient Curve')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('turbine_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Curves saved to: turbine_curves.png")
    
    # Key points
    print(f"\nKey characteristics:")
    print(f"  - Cut-in wind speed: ~3 m/s (when power starts)")
    print(f"  - Rated wind speed: ~12 m/s (when rated power reached)")
    print(f"  - Cut-out wind speed: >25 m/s (when turbine stops)")
    print(f"  - Peak Ct: {max(cts):.3f} (maximum wind blocking)")
    
    return wf


# -----------------------------------------------------------------------------
# Example 3: Understanding wake effects
# -----------------------------------------------------------------------------

def example_3_wake_effects():
    """Compare power with and without wake effects."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Wake Effects Analysis")
    print("="*70)
    
    # Create two wind farms: one with wakes, one without
    n_turb = 5
    
    # Wind farm 1: With wake effects
    wf_with_wake = WindFarmModel(
        n_turbines=n_turb,
        turbine_rating_mw=5.0,
        rotor_diameter=136,
        hub_height=90
    )
    wf_with_wake.create_layout(layout_type='row', spacing=5.0)  # 5D spacing
    wf_with_wake.setup_wake_model(use_turbulence=True)
    
    # Simulate constant wind conditions
    from py_wake.site import UniformSite
    from py_wake.site.shear import PowerShear
    site = UniformSite(p_wd=[1], ti=0.06, shear=PowerShear(h_ref=wf_with_wake.hub_height, alpha=0.14))
    
    # Calculate power for single turbine (no wake)
    ws_test = 10.0  # m/s
    wd_test = 270.0  # degrees (wind from west, hitting turbines in row)
    
    # Single turbine power (no wake losses)
    power_single, _ = wf_with_wake.wind_turbine.powerCtFunction(ws_test)
    power_single_mw = power_single / 1e6
    
    # Initialize wake model
    from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
    wake_model = IEA37SimpleBastankhahGaussian(site, wf_with_wake.wind_turbine)
    
    # Run simulation with wake
    sim_res = wake_model(wf_with_wake.x, wf_with_wake.y, wd=[wd_test], ws=[ws_test])
    
    # Extract individual turbine powers
    turbine_powers = sim_res.Power.values.flatten() / 1e6  # Convert to MW
    
    print(f"\nWind conditions:")
    print(f"  - Wind speed: {ws_test} m/s")
    print(f"  - Wind direction: {wd_test}° (wind from west)")
    print(f"  - Layout: {n_turb} turbines in a row (5D spacing)")
    
    print(f"\nPower output per turbine:")
    for i, power in enumerate(turbine_powers):
        wake_loss_pct = (1 - power/power_single_mw) * 100
        print(f"  Turbine {i+1}: {power:.2f} MW ({wake_loss_pct:.1f}% wake loss)")
    
    total_power_with_wake = turbine_powers.sum()
    total_power_no_wake = power_single_mw * n_turb
    total_wake_loss_pct = (1 - total_power_with_wake/total_power_no_wake) * 100
    
    print(f"\nTotal wind farm:")
    print(f"  - Power without wake: {total_power_no_wake:.2f} MW")
    print(f"  - Power with wake: {total_power_with_wake:.2f} MW")
    print(f"  - Total wake loss: {total_wake_loss_pct:.1f}%")
    
    # Visualize wake deficit
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(1, n_turb + 1)
    width = 0.35
    
    ax.bar(x - width/2, [power_single_mw]*n_turb, width, 
           label='No wake', alpha=0.7, color='green')
    ax.bar(x + width/2, turbine_powers, width, 
           label='With wake', alpha=0.7, color='red')
    
    ax.set_xlabel('Turbine Number')
    ax.set_ylabel('Power [MW]')
    ax.set_title('Wake Effect on Individual Turbines (5D Spacing, Wind from West)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('wake_effects.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Wake analysis saved to: wake_effects.png")
    
    return wf_with_wake, total_wake_loss_pct


# -----------------------------------------------------------------------------
# Example 4: Impact of turbine spacing
# -----------------------------------------------------------------------------

def example_4_spacing_study():
    """Study how spacing affects wake losses."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Turbine Spacing Study")
    print("="*70)
    
    spacings = [3, 5, 7, 9, 12]  # In rotor diameters (D)
    wake_losses = []
    
    from py_wake.site import UniformSite
    from py_wake.site.shear import PowerShear
    from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
    
    hub_height = 90
    site = UniformSite(p_wd=[1], ti=0.06, shear=PowerShear(h_ref=hub_height, alpha=0.14))
    ws_test = 10.0
    wd_test = 270.0
    
    for spacing in spacings:
        wf = WindFarmModel(n_turbines=5, turbine_rating_mw=5.0)
        wf.create_layout(layout_type='row', spacing=spacing)
        
        # Single turbine power
        power_single, _ = wf.wind_turbine.powerCtFunction(ws_test)
        power_single_mw = power_single / 1e6
        
        # With wake
        wake_model = IEA37SimpleBastankhahGaussian(site, wf.wind_turbine)
        sim_res = wake_model(wf.x, wf.y, wd=[wd_test], ws=[ws_test])
        total_power_with_wake = sim_res.Power.sum() / 1e6
        
        # Calculate loss
        total_power_no_wake = power_single_mw * 5
        wake_loss_pct = (1 - total_power_with_wake/total_power_no_wake) * 100
        wake_losses.append(wake_loss_pct)
        
        print(f"  Spacing {spacing}D: {wake_loss_pct:.1f}% wake loss")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(spacings, wake_losses, 'o-', linewidth=2, markersize=8, color='steelblue')
    plt.xlabel('Turbine Spacing [D]')
    plt.ylabel('Wake Loss [%]')
    plt.title('Wake Losses vs. Turbine Spacing (5 turbines in row, wind from west)')
    plt.grid(True, alpha=0.3)
    plt.axhline(10, color='red', linestyle='--', alpha=0.5, label='10% loss threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('spacing_study.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Spacing study saved to: spacing_study.png")
    
    print(f"\n Key insight:")
    print(f"  - 3D spacing: {wake_losses[0]:.1f}% loss (too close)")
    print(f"  - 7D spacing: {wake_losses[2]:.1f}% loss (typical offshore)")
    print(f"  - 12D spacing: {wake_losses[4]:.1f}% loss (widely spaced)")


# -----------------------------------------------------------------------------
# Example 5: Time series generation
# -----------------------------------------------------------------------------

def example_5_timeseries():
    """Generate realistic power time series."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Power Time Series Generation")
    print("="*70)
    
    # Create wind farm
    wf = create_reference_offshore_windfarm(n_turbines=10, turbine_mw=5.0)
    
    # Generate wind data (1 week for demonstration)
    wind_data = generate_synthetic_wind_data(hours=168, mean_ws=9.0)
    
    print(f"Wind data generated:")
    print(f"  - Duration: {len(wind_data)} hours")
    print(f"  - Mean wind speed: {wind_data['wind_speed'].mean():.2f} m/s")
    print(f"  - Max wind speed: {wind_data['wind_speed'].max():.2f} m/s")
    print(f"  - Mean direction: {wind_data['wind_direction'].mean():.1f}°")
    
    print(f"\nCalculating power with wake effects...")
    print(f"  (This takes ~1 min for 168 hours)")
    
    # Calculate power
    power_ts = wf.generate_timeseries(
        wind_data['wind_speed'],
        wind_data['wind_direction'],
        ti=0.06
    )
    
    # Statistics
    capacity_factor = (power_ts.mean() / (wf.n_turbines * wf.turbine_rating_mw)) * 100
    
    print(f"\nPower production statistics:")
    print(f"  - Mean power: {power_ts.mean():.2f} MW")
    print(f"  - Max power: {power_ts.max():.2f} MW")
    print(f"  - Capacity factor: {capacity_factor:.1f}%")
    print(f"  - Total energy (1 week): {power_ts.sum():.0f} MWh")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Wind speed
    axes[0].plot(wind_data.index, wind_data['wind_speed'], linewidth=1, color='blue')
    axes[0].set_ylabel('Wind Speed [m/s]')
    axes[0].set_title('Wind Conditions and Power Production (1 Week)')
    axes[0].grid(True, alpha=0.3)
    
    # Power output
    axes[1].plot(power_ts.index, power_ts.values, linewidth=1, color='green')
    axes[1].axhline(wf.n_turbines * wf.turbine_rating_mw, 
                    color='red', linestyle='--', label='Installed capacity')
    axes[1].set_ylabel('Power [MW]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Capacity factor over time
    cf_rolling = (power_ts / (wf.n_turbines * wf.turbine_rating_mw) * 100).rolling(24).mean()
    axes[2].plot(cf_rolling.index, cf_rolling.values, linewidth=2, color='orange')
    axes[2].axhline(capacity_factor, color='black', linestyle='--', 
                   label=f'Mean CF: {capacity_factor:.1f}%')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Capacity Factor [%]')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('timeseries_example.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Time series saved to: timeseries_example.png")
    
    return wf, power_ts


# =============================================================================
# PART 3: CUSTOMIZATION EXAMPLES
# =============================================================================

def example_6_custom_turbine():
    """Create wind farm with custom turbine specifications."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Custom Turbine Specification")
    print("="*70)
    
    # Define custom 8MW turbine (similar to Vestas V164)
    ws = np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25])
    
    # Power curve (MW)
    power_mw = np.array([0, 0, 0.5, 1.2, 2.2, 3.5, 5.0, 6.3, 7.3, 7.8, 7.95, 8.0, 8.0, 8.0, 8.0, 8.0])
    
    # Thrust coefficient
    ct = np.array([0, 0, 0.82, 0.83, 0.84, 0.83, 0.81, 0.78, 0.74, 0.69, 0.63, 0.56, 0.49, 0.43, 0.30, 0.25])
    
    # Create wind farm with custom turbine
    wf = WindFarmModel(
        n_turbines=15,
        turbine_rating_mw=8.0,
        rotor_diameter=164,  # Vestas V164
        hub_height=105,
        power_curve=(ws, power_mw),
        ct_curve=(ws, ct)
    )
    
    wf.create_layout(layout_type='grid', spacing=7.0)
    
    print(f"Custom wind farm created:")
    print(f"  - Turbine: 8MW, 164m rotor")
    print(f"  - Total capacity: {wf.n_turbines * wf.turbine_rating_mw} MW")
    print(f"  - Layout: {wf.n_turbines} turbines in grid (7D spacing)")
    
    return wf


def example_7_layout_comparison():
    """Compare different wind farm layouts."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Layout Comparison")
    print("="*70)
    
    from py_wake.site import UniformSite
    from py_wake.site.shear import PowerShear
    from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
    
    layouts = {
        'Row (5D)': ('row', 5.0),
        'Row (7D)': ('row', 7.0),
        'Grid (7D)': ('grid', 7.0)
    }
    
    results = {}
    hub_height = 90
    site = UniformSite(p_wd=[1], ti=0.06, shear=PowerShear(h_ref=hub_height, alpha=0.14))
    
    for name, (layout_type, spacing) in layouts.items():
        wf = WindFarmModel(n_turbines=10, turbine_rating_mw=5.0)
        wf.create_layout(layout_type=layout_type, spacing=spacing)
        
        # Calculate AEP with realistic wind rose
        # (Simplified: using uniform wind from west)
        wake_model = IEA37SimpleBastankhahGaussian(site, wf.wind_turbine)
        
        # Test multiple wind directions
        wind_dirs = np.arange(0, 360, 30)
        wind_speeds = [8, 10, 12]  # m/s
        
        total_power = 0
        n_cases = 0
        
        for wd in wind_dirs:
            for ws in wind_speeds:
                sim_res = wake_model(wf.x, wf.y, wd=[wd], ws=[ws])
                total_power += sim_res.Power.sum() / 1e6
                n_cases += 1
        
        avg_power = total_power / n_cases
        results[name] = avg_power
        
        print(f"  {name}: {avg_power:.2f} MW average")
    
    print(f"\n Best layout: {max(results, key=results.get)}")
    
    return results


# =============================================================================
# RUN ALL EXAMPLES
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("PyWake Model Exploration - Running All Examples")
    print("="*70)
    
    # Run examples
    wf1 = example_1_basic_windfarm()
    wf2 = example_2_power_curve()
    wf3, wake_loss = example_3_wake_effects()
    example_4_spacing_study()
    wf5, power_ts = example_5_timeseries()
    wf6 = example_6_custom_turbine()
    results = example_7_layout_comparison()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nGenerated files:")
    print("  - windfarm_layout.png")
    print("  - turbine_curves.png")
    print("  - wake_effects.png")
    print("  - spacing_study.png")
    print("  - timeseries_example.png")
    print("\nNext steps:")
    print("  1. Review the generated plots")
    print("  2. Modify parameters to explore sensitivity")
    print("  3. Integrate with SHIPP optimization")
    print("="*70)