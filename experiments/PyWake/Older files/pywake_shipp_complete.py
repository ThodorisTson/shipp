"""
PyWake + SHIPP Integration Guide
=================================

Practical examples of using PyWake wind farm model with SHIPP optimization.
This replaces the renewable.ninja approach from Example 2.

Author: Thodoris
Date: January 2026
"""

from setup_paths import configure_paths
configure_paths()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shipp.components import Storage, OpSchedule
from pywake_windfarm import create_reference_offshore_windfarm, generate_synthetic_wind_data
from pywake_shipp_integration import prepare_wind_data_for_shipp

# =============================================================================
# EXAMPLE 1: Basic Integration - PyWake → SHIPP
# =============================================================================

def example_1_basic_integration():
    """
    Simplest possible integration: Create wind farm, generate power, 
    prepare for SHIPP.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic PyWake → SHIPP Integration")
    print("="*70)
    
    # Step 1: Create wind farm
    print("\n[1] Creating wind farm...")
    wf = create_reference_offshore_windfarm(n_turbines=10, turbine_mw=5.0)
    print(f"    ✓ {wf.n_turbines} × {wf.turbine_rating_mw}MW = {wf.n_turbines * wf.turbine_rating_mw}MW capacity")
    
    # Step 2: Generate wind data (use small sample for speed)
    print("\n[2] Generating wind data (24 hours)...")
    wind_data = generate_synthetic_wind_data(hours=24, mean_ws=9.0)
    print(f"    ✓ Mean wind speed: {wind_data['wind_speed'].mean():.2f} m/s")
    
    # Step 3: Calculate power with wake effects
    print("\n[3] Calculating power production...")
    power_ts = wf.generate_timeseries(
        wind_data['wind_speed'],
        wind_data['wind_direction'],
        ti=0.06
    )
    cf = (power_ts.mean() / (wf.n_turbines * wf.turbine_rating_mw)) * 100
    print(f"    ✓ Mean power: {power_ts.mean():.2f} MW")
    print(f"    ✓ Capacity factor: {cf:.1f}%")
    
    # Step 4: Prepare for SHIPP
    print("\n[4] Preparing data for SHIPP...")
    shipp_input = prepare_wind_data_for_shipp(power_ts)
    print(f"    ✓ Power data: {len(shipp_input['power_mw'])} timesteps")
    print(f"    ✓ Price data: {len(shipp_input['price_eur_mwh'])} timesteps")
    
    # Step 5: Create SHIPP storage
    print("\n[5] Creating battery storage...")
    storage = Storage(
        name='battery',
        E_capacity=100,  # MWh - roughly 2 hours at full power
        P_capacity=25,   # MW - half of wind farm capacity
        eff_in=0.95,
        eff_out=0.95
    )
    print(f"    ✓ Storage: {storage.E_capacity}MWh / {storage.P_capacity}MW")
    print(f"    ✓ Round-trip efficiency: {storage.eff_in * storage.eff_out * 100:.1f}%")
    
    print("\n[6] Data ready for SHIPP optimization!")
    print("    Next: Use OpSchedule to optimize battery dispatch")
    
    return wf, power_ts, shipp_input, storage


# =============================================================================
# EXAMPLE 2: Compare with/without Wake Effects
# =============================================================================

def example_2_wake_impact_on_revenue():
    """
    Show how wake effects impact both power production and revenue.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Impact of Wake Effects on Revenue")
    print("="*70)
    
    # Create wind farm
    wf = create_reference_offshore_windfarm(n_turbines=10, turbine_mw=5.0)
    wind_data = generate_synthetic_wind_data(hours=168, mean_ws=9.0)  # 1 week
    
    # Calculate power WITH wake effects
    print("\n[1] Calculating power WITH wake effects...")
    power_with_wake = wf.generate_timeseries(
        wind_data['wind_speed'],
        wind_data['wind_direction'],
        ti=0.06
    )
    
    # Calculate power WITHOUT wake effects (theoretical)
    # Just multiply single turbine power by number of turbines
    print("\n[2] Calculating power WITHOUT wake effects...")
    power_no_wake = []
    for ws in wind_data['wind_speed']:
        p_single, _ = wf.wind_turbine.powerCtFunction(ws)
        power_no_wake.append((p_single / 1e6) * wf.n_turbines)
    power_no_wake = pd.Series(power_no_wake, index=power_with_wake.index)
    
    # Calculate wake losses
    wake_loss_pct = ((power_no_wake.mean() - power_with_wake.mean()) / 
                      power_no_wake.mean() * 100)
    
    print(f"\n[3] Power production comparison:")
    print(f"    Without wake: {power_no_wake.mean():.2f} MW")
    print(f"    With wake:    {power_with_wake.mean():.2f} MW")
    print(f"    Wake loss:    {wake_loss_pct:.1f}%")
    
    # Prepare for SHIPP and calculate revenue
    shipp_with_wake = prepare_wind_data_for_shipp(power_with_wake)
    shipp_no_wake = prepare_wind_data_for_shipp(power_no_wake)
    
    # Calculate revenue (no storage, just selling at spot price)
    revenue_with_wake = (power_with_wake.values * 
                         shipp_with_wake['price_eur_mwh']).sum()
    revenue_no_wake = (power_no_wake.values * 
                       shipp_no_wake['price_eur_mwh']).sum()
    
    revenue_loss_pct = ((revenue_no_wake - revenue_with_wake) / 
                        revenue_no_wake * 100)
    
    print(f"\n[4] Revenue impact (1 week, no storage):")
    print(f"    Without wake: {revenue_no_wake:,.0f} €")
    print(f"    With wake:    {revenue_with_wake:,.0f} €")
    print(f"    Revenue loss: {revenue_loss_pct:.1f}%")
    
    print(f"\n[5] Key insight:")
    print(f"    Wake effects reduce BOTH power ({wake_loss_pct:.1f}%) and revenue ({revenue_loss_pct:.1f}%)")
    print(f"    This is why using PyWake is important for realistic optimization!")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Power comparison
    axes[0].plot(power_no_wake.index, power_no_wake.values, 
                 label='No wake', alpha=0.7, linewidth=1.5)
    axes[0].plot(power_with_wake.index, power_with_wake.values, 
                 label='With wake', alpha=0.7, linewidth=1.5)
    axes[0].set_ylabel('Power [MW]')
    axes[0].set_title('Power Production: With vs. Without Wake Effects')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Difference
    diff = power_no_wake - power_with_wake
    axes[1].fill_between(diff.index, 0, diff.values, 
                         alpha=0.5, color='red', label='Wake losses')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Power Loss [MW]')
    axes[1].set_title('Wake-Induced Power Losses')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wake_impact_revenue.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: wake_impact_revenue.png")
    
    return power_with_wake, power_no_wake, shipp_with_wake


# =============================================================================
# EXAMPLE 3: Sensitivity to Wind Farm Size
# =============================================================================

def example_3_wind_farm_sizing():
    """
    Explore how wind farm size affects storage requirements.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Wind Farm Sizing Study")
    print("="*70)
    
    # Test different wind farm sizes
    farm_sizes = [5, 10, 15, 20]  # Number of turbines
    turbine_mw = 5.0
    
    results = {}
    
    wind_data = generate_synthetic_wind_data(hours=168, mean_ws=9.0)
    
    for n_turb in farm_sizes:
        print(f"\n[{farm_sizes.index(n_turb)+1}] Testing {n_turb} turbines ({n_turb * turbine_mw}MW)...")
        
        # Create wind farm
        wf = create_reference_offshore_windfarm(
            n_turbines=n_turb,
            turbine_mw=turbine_mw
        )
        
        # Calculate power
        power_ts = wf.generate_timeseries(
            wind_data['wind_speed'],
            wind_data['wind_direction'],
            ti=0.06
        )
        
        # Statistics
        mean_power = power_ts.mean()
        std_power = power_ts.std()
        max_power = power_ts.max()
        
        # Suggested storage size (rule of thumb: 2-4 hours of mean power)
        suggested_e_cap = mean_power * 3  # 3 hours
        suggested_p_cap = max_power * 0.5  # 50% of max power
        
        results[n_turb] = {
            'capacity': n_turb * turbine_mw,
            'mean_power': mean_power,
            'std_power': std_power,
            'max_power': max_power,
            'suggested_e_cap': suggested_e_cap,
            'suggested_p_cap': suggested_p_cap
        }
        
        print(f"    Mean power: {mean_power:.1f} MW")
        print(f"    Std power:  {std_power:.1f} MW")
        print(f"    Suggested storage: {suggested_e_cap:.0f}MWh / {suggested_p_cap:.0f}MW")
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Suggested Storage Sizing")
    print("="*70)
    print(f"{'Turbines':<10} {'Capacity':<12} {'Mean Power':<12} {'Storage E':<12} {'Storage P':<12}")
    print("-" * 70)
    for n_turb, data in results.items():
        print(f"{n_turb:<10} {data['capacity']:.0f}MW{'':<7} {data['mean_power']:.1f}MW{'':<6} "
              f"{data['suggested_e_cap']:.0f}MWh{'':<6} {data['suggested_p_cap']:.0f}MW")
    
    return results


# =============================================================================
# EXAMPLE 4: Complete SHIPP Optimization Setup
# =============================================================================

def example_4_complete_shipp_setup():
    """
    Complete example showing full PyWake → SHIPP workflow.
    This is what you'll actually use in your thesis.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Complete SHIPP Optimization Setup")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # STEP 1: Define wind farm parameters
    # -------------------------------------------------------------------------
    print("\n[STEP 1: Wind Farm Configuration]")
    
    n_turbines = 10
    turbine_mw = 5.0
    total_capacity_mw = n_turbines * turbine_mw
    
    print(f"  Wind farm size: {n_turbines} × {turbine_mw}MW = {total_capacity_mw}MW")
    
    # -------------------------------------------------------------------------
    # STEP 2: Create wind farm model
    # -------------------------------------------------------------------------
    print("\n[STEP 2: Creating Wind Farm Model]")
    
    wf = create_reference_offshore_windfarm(
        n_turbines=n_turbines,
        turbine_mw=turbine_mw
    )
    wf.plot_layout()
    plt.savefig('final_windfarm_layout.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Wind farm created")
    print(f"  ✓ Layout saved: final_windfarm_layout.png")
    
    # -------------------------------------------------------------------------
    # STEP 3: Generate/load wind data
    # -------------------------------------------------------------------------
    print("\n[STEP 3: Wind Data]")
    
    # For thesis: Replace this with real ERA5 or mesoscale data
    wind_data = generate_synthetic_wind_data(hours=168, mean_ws=9.0)
    
    print(f"  ✓ Wind data: {len(wind_data)} hours")
    print(f"  ✓ Mean wind speed: {wind_data['wind_speed'].mean():.2f} m/s")
    
    # -------------------------------------------------------------------------
    # STEP 4: Calculate power production
    # -------------------------------------------------------------------------
    print("\n[STEP 4: Power Production Calculation]")
    print("  (This will take ~1 minute for 168 hours)")
    
    power_ts = wf.generate_timeseries(
        wind_data['wind_speed'],
        wind_data['wind_direction'],
        ti=0.06
    )
    
    cf = (power_ts.mean() / total_capacity_mw) * 100
    print(f"  ✓ Power calculated")
    print(f"  ✓ Mean: {power_ts.mean():.2f} MW")
    print(f"  ✓ Capacity factor: {cf:.1f}%")
    
    # -------------------------------------------------------------------------
    # STEP 5: Prepare SHIPP input
    # -------------------------------------------------------------------------
    print("\n[STEP 5: Preparing SHIPP Input]")
    
    shipp_input = prepare_wind_data_for_shipp(power_ts)
    
    print(f"  ✓ SHIPP input prepared")
    print(f"  ✓ Mean price: {shipp_input['price_eur_mwh'].mean():.2f} €/MWh")
    
    # -------------------------------------------------------------------------
    # STEP 6: Define storage parameters
    # -------------------------------------------------------------------------
    print("\n[STEP 6: Battery Storage Configuration]")
    
    # Rule of thumb: E_capacity = 2-4 hours of mean power
    #                P_capacity = 40-60% of max power
    e_capacity = power_ts.mean() * 3  # 3 hours
    p_capacity = power_ts.max() * 0.5  # 50% of max
    
    storage = Storage(
        name='battery',
        E_capacity=e_capacity,
        P_capacity=p_capacity,
        eff_in=0.95,
        eff_out=0.95
    )
    
    print(f"  ✓ Storage configured")
    print(f"  ✓ E_capacity: {storage.E_capacity:.0f} MWh")
    print(f"  ✓ P_capacity: {storage.P_capacity:.0f} MW")
    print(f"  ✓ E/P ratio: {storage.E_capacity/storage.P_capacity:.1f} hours")
    
    # -------------------------------------------------------------------------
    # STEP 7: Create OpSchedule for optimization
    # -------------------------------------------------------------------------
    print("\n[STEP 7: SHIPP Optimization Setup]")
    
    # Create OpSchedule
    op = OpSchedule(
        fixed=Storage(name='wind', P_capacity=total_capacity_mw),
        storage=storage,
        name='wind_battery_system'
    )
    
    print(f"  ✓ OpSchedule created")
    print(f"  ✓ Wind: {op.fixed.P_capacity} MW (fixed)")
    print(f"  ✓ Storage: {op.storage.E_capacity}MWh / {op.storage.P_capacity}MW")
    
    # -------------------------------------------------------------------------
    # STEP 8: Ready for optimization
    # -------------------------------------------------------------------------
    print("\n[STEP 8: Ready for Optimization]")
    print("  ")
    print("  Next steps to run optimization:")
    print("  ")
    print("  ```python")
    print("  # Set wind power as input")
    print("  op.set_power_profile(")
    print("      power_in=shipp_input['power_mw'],")
    print("      P_R=None  # No baseload requirement yet")
    print("  )")
    print("  ")
    print("  # Set prices")
    print("  op.set_market_prices(shipp_input['price_eur_mwh'])")
    print("  ")
    print("  # Run optimization")
    print("  from shipp.tools import solve_lp")
    print("  results = solve_lp(")
    print("      op,")
    print("      solver='glpk',  # or 'gurobi', 'cplex'")
    print("      verbose=True")
    print("  )")
    print("  ")
    print("  # Analyze results")
    print("  print(f'Total revenue: {results[\"objective\"]:.0f} €')")
    print("  print(f'Storage utilization: {results[\"storage_cycles\"]:.1f} cycles')")
    print("  ```")
    
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    
    return wf, power_ts, shipp_input, storage, op


# =============================================================================
# RUN EXAMPLES
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("PyWake + SHIPP Integration Examples")
    print("="*70)
    
    # Run examples
    print("\nRunning examples...")
    print("Note: Example 2 and 4 take longer due to time series calculations")
    
    # Example 1: Basic integration
    wf1, power1, shipp1, storage1 = example_1_basic_integration()
    
    # Example 2: Wake impact
    power_wake, power_no_wake, shipp_wake = example_2_wake_impact_on_revenue()
    
    # Example 3: Sizing study
    sizing_results = example_3_wind_farm_sizing()
    
    # Example 4: Complete setup
    wf4, power4, shipp4, storage4, op4 = example_4_complete_shipp_setup()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  - wake_impact_revenue.png")
    print("  - final_windfarm_layout.png")
    print("\nYou now have:")
    print("  1. Understanding of PyWake wake effects")
    print("  2. Complete integration workflow")
    print("  3. Ready-to-use SHIPP setup")
    print("\nNext: Run actual SHIPP optimization and analyze results!")
    print("="*70)
