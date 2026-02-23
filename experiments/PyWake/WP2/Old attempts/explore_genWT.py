"""
Explore HyDesign's Generic Wind Turbine Lookup Table (genWT_v3.nc)

This version automatically looks for genWT_v3.nc in the current directory.

Usage:
    python explore_genWT_simple.py

Requirements:
    - xarray
    - h5netcdf
    - matplotlib
    - numpy
    - pyyaml
"""

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import xarray as xr
except ImportError:
    print("ERROR: xarray not installed")
    print("Install with: pip install xarray h5netcdf matplotlib pyyaml numpy")
    exit(1)

def explore_genWT(filename):
    """
    Explore the genWT lookup table structure and contents
    """
    print("="*70)
    print("HyDesign Generic Wind Turbine Lookup Table Analysis")
    print("="*70)
    print()
    
    # Open the dataset
    genWT = xr.open_dataset(filename, engine='h5netcdf')
    
    print("1. DATASET STRUCTURE")
    print("-" * 70)
    print(genWT)
    print()
    
    # Extract key information
    sp_values = genWT.sp.values
    ws_values = genWT.ws.values
    
    print("2. AVAILABLE SPECIFIC POWER VALUES")
    print("-" * 70)
    print(f"Specific Power (sp) range: {sp_values.min():.0f} - {sp_values.max():.0f} W/m²")
    print(f"Available sp values: {sp_values}")
    print()
    
    print("3. WIND SPEED RANGE")
    print("-" * 70)
    print(f"Min: {ws_values.min():.1f} m/s")
    print(f"Max: {ws_values.max():.1f} m/s")
    print(f"Number of points: {len(ws_values)}")
    print()
    
    print("4. DATA VARIABLES")
    print("-" * 70)
    for var in genWT.data_vars:
        print(f"{var}:")
        print(f"  Shape: {genWT[var].shape}")
        print(f"  Dimensions: {genWT[var].dims}")
        print(f"  Description: {get_var_description(var)}")
    print()
    
    print("5. WP2 APPLICATION")
    print("-" * 70)
    sp_wp2 = 244  # W/m²
    p_rated_wp2 = 5  # MW
    
    print(f"WP2 Specifications:")
    print(f"  Specific Power: {sp_wp2} W/m²")
    print(f"  Rated Power: {p_rated_wp2} MW")
    print()
    
    # Calculate rotor parameters
    A = (p_rated_wp2 * 1e6) / sp_wp2
    d = np.sqrt(4 * A / np.pi)
    print(f"Calculated from sp = P_rated / A:")
    print(f"  Rotor Area: {A:.1f} m²")
    print(f"  Rotor Diameter: {d:.1f} m")
    print()
    
    # Check if sp=244 needs interpolation
    if sp_wp2 in sp_values:
        print(f"✓ Exact match: sp={sp_wp2} W/m² exists in lookup table")
        idx = np.where(sp_values == sp_wp2)[0][0]
        pc_wp2 = genWT.pc.isel(sp=idx).values
        ct_wp2 = genWT.ct.isel(sp=idx).values
        ws_wp2 = ws_values
    else:
        print(f"⚠ Interpolation needed: sp={sp_wp2} W/m² not in table")
        print(f"  Will interpolate between available sp values")
        print(f"  Closest values: {find_bracket(sp_values, sp_wp2)}")
        
        # Interpolate
        genWT_interp = genWT.interp(sp=sp_wp2)
        pc_wp2 = genWT_interp.pc.values
        ct_wp2 = genWT_interp.ct.values
        ws_wp2 = genWT_interp.ws.values
    
    print()
    print("6. EXTRACTED WP2 CURVES")
    print("-" * 70)
    print(f"Power Curve (pc):")
    print(f"  Rated Power: {pc_wp2.max():.3f} (normalized)")
    print(f"  Cut-in wind speed: {ws_values[pc_wp2 > 0.001][0]:.1f} m/s")
    print(f"  Rated wind speed: {ws_values[pc_wp2 >= pc_wp2.max() * 0.99][0]:.1f} m/s")
    print()
    print(f"Thrust Coefficient (ct):")
    print(f"  Maximum Ct: {ct_wp2.max():.3f}")
    print(f"  Ct at rated: {ct_wp2[pc_wp2 >= pc_wp2.max() * 0.99][0]:.3f}")
    print()
    
    genWT.close()
    
    return ws_wp2, pc_wp2, ct_wp2, sp_wp2, p_rated_wp2, d

def get_var_description(var_name):
    """Get description for each variable"""
    descriptions = {
        'ws': 'Wind speed array for power/Ct curves',
        'pc': 'Power curve (normalized to 1.0 at rated)',
        'ct': 'Thrust coefficient curve'
    }
    return descriptions.get(var_name, 'Unknown variable')

def find_bracket(arr, value):
    """Find the two values that bracket the target value"""
    arr_sorted = np.sort(arr)
    idx = np.searchsorted(arr_sorted, value)
    if idx == 0:
        return f"{arr_sorted[0]:.0f} W/m² (extrapolation)"
    elif idx == len(arr_sorted):
        return f"{arr_sorted[-1]:.0f} W/m² (extrapolation)"
    else:
        return f"{arr_sorted[idx-1]:.0f} - {arr_sorted[idx]:.0f} W/m²"

def plot_wp2_curves(ws, pc, ct, sp, p_rated, d, save_fig=True):
    """
    Create visualization of WP2 turbine curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot power curve
    ax1.plot(ws, pc * p_rated, 'b-', linewidth=2, label='Power Curve')
    ax1.axhline(p_rated, color='r', linestyle='--', alpha=0.5, label=f'Rated: {p_rated} MW')
    ax1.set_xlabel('Wind Speed [m/s]', fontsize=12)
    ax1.set_ylabel('Power [MW]', fontsize=12)
    ax1.set_title(f'WP2 Turbine Power Curve\nsp={sp} W/m², D={d:.1f}m, P={p_rated} MW', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 25)
    ax1.set_ylim(0, p_rated * 1.1)
    
    # Plot thrust coefficient
    ax2.plot(ws, ct, 'g-', linewidth=2, label='Ct Curve')
    ax2.set_xlabel('Wind Speed [m/s]', fontsize=12)
    ax2.set_ylabel('Thrust Coefficient [-]', fontsize=12)
    ax2.set_title(f'WP2 Turbine Thrust Coefficient\nsp={sp} W/m²', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 25)
    ax2.set_ylim(0, max(ct) * 1.1)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('wp2_turbine_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Saved plot: wp2_turbine_curves.png")
    
    plt.show()
    return fig

def compare_specific_powers(filename):
    """
    Compare curves for different specific power values
    """
    genWT = xr.open_dataset(filename, engine='h5netcdf')
    
    sp_values = genWT.sp.values
    ws = genWT.ws.values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Select a few sp values to compare
    sp_compare = [150, 210, 244, 270, 330, 360]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sp_compare)))
    
    for i, sp in enumerate(sp_compare):
        if sp in sp_values:
            idx = np.where(sp_values == sp)[0][0]
            pc = genWT.pc.isel(sp=idx).values
            ct = genWT.ct.isel(sp=idx).values
        else:
            genWT_interp = genWT.interp(sp=sp)
            pc = genWT_interp.pc.values
            ct = genWT_interp.ct.values
        
        label = f'sp={sp} W/m²' + (' (WP2)' if sp == 244 else '')
        linewidth = 3 if sp == 244 else 1.5
        linestyle = '-' if sp == 244 else '--'
        
        ax1.plot(ws, pc, color=colors[i], linewidth=linewidth, 
                linestyle=linestyle, label=label)
        ax2.plot(ws, ct, color=colors[i], linewidth=linewidth,
                linestyle=linestyle, label=label)
    
    ax1.set_xlabel('Wind Speed [m/s]', fontsize=12)
    ax1.set_ylabel('Power (normalized)', fontsize=12)
    ax1.set_title('Power Curves - Different Specific Powers', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 25)
    
    ax2.set_xlabel('Wind Speed [m/s]', fontsize=12)
    ax2.set_ylabel('Thrust Coefficient [-]', fontsize=12)
    ax2.set_title('Ct Curves - Different Specific Powers', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 25)
    
    plt.tight_layout()
    plt.savefig('genWT_comparison_sp.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot: genWT_comparison_sp.png")
    
    plt.show()
    genWT.close()
    return fig

def export_wp2_curves_to_yaml(ws, pc, ct, sp, p_rated, d, hub_height=115):
    """
    Export WP2 curves in YAML format compatible with PyWake/IEA 740
    """
    import yaml
    
    # Scale power curve to actual MW values
    pc_mw = (pc * p_rated * 1e6).tolist()  # Convert to Watts
    
    turbine_data = {
        'name': 'WP2_5MW_GenericTurbine',
        'description': f'Generic 5 MW turbine with sp={sp} W/m² from HyDesign lookup table',
        'manufacturer': 'Generic (HyDesign)',
        'hub_height': hub_height,
        'rotor_diameter': float(d),
        'tilt': 5.0,
        'performance': {
            'rated_power': int(p_rated * 1e6),
            'cutin_wind_speed': 3.0,
            'cutout_wind_speed': 25.0,
            'rated_wind_speed': 12.0,
            'power_curve': {
                'power_wind_speeds': ws.tolist(),
                'power_values': pc_mw,
                'notes': f'Generated from HyDesign genWT_v3.nc with sp={sp} W/m²'
            },
            'Ct_curve': {
                'Ct_wind_speeds': ws.tolist(),
                'Ct_values': ct.tolist(),
                'notes': 'Generated from HyDesign genWT_v3.nc'
            }
        },
        'specifications': {
            'blade_length_m': float(d/2),
            'swept_area_m2': float(np.pi * (d/2)**2),
            'specific_power_Wm2': sp,
            'max_cp': 0.45,
            'generator_type': 'Generic'
        }
    }
    
    filename = 'wp2_turbine_from_hydesign.yaml'
    with open(filename, 'w') as f:
        yaml.dump(turbine_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Exported turbine data to: {filename}")
    return filename

def main():
    # Look for genWT_v3.nc in current directory
    filename = 'genWT_v3.nc'
    
    if not os.path.exists(filename):
        print("ERROR: genWT_v3.nc not found in current directory!")
        print(f"Current directory: {os.getcwd()}")
        print("\nPlease make sure genWT_v3.nc is in the same folder as this script.")
        print("Or run with: python explore_genWT.py <path_to_genWT_v3.nc>")
        exit(1)
    
    print(f"Found: {filename}")
    print(f"Location: {os.path.abspath(filename)}\n")
    
    print("\n" + "="*70)
    print("STEP 1: Exploring Lookup Table")
    print("="*70 + "\n")
    
    # Explore the file
    ws, pc, ct, sp, p_rated, d = explore_genWT(filename)
    
    print("\n" + "="*70)
    print("STEP 2: Visualizing WP2 Curves")
    print("="*70 + "\n")
    
    # Plot WP2 specific curves
    plot_wp2_curves(ws, pc, ct, sp, p_rated, d)
    
    print("\n" + "="*70)
    print("STEP 3: Comparing Different Specific Powers")
    print("="*70 + "\n")
    
    # Compare with other sp values
    compare_specific_powers(filename)
    
    print("\n" + "="*70)
    print("STEP 4: Exporting to YAML Format")
    print("="*70 + "\n")
    
    # Export to YAML
    export_wp2_curves_to_yaml(ws, pc, ct, sp, p_rated, d)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. wp2_turbine_curves.png - Power and Ct curves for WP2")
    print("  2. genWT_comparison_sp.png - Comparison of different sp values")
    print("  3. wp2_turbine_from_hydesign.yaml - Turbine spec in YAML format")
    print("\nNext steps:")
    print("  - Review the plots to verify curves look reasonable")
    print("  - Use the YAML file with PyWake for wake modeling")
    print("  - Integrate with your WP2 IEA 740 system file")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()