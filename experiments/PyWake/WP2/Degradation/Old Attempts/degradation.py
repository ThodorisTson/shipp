"""
WP2 Battery Degradation Analysis
=================================

Step 1: Analyze battery degradation from SHIPP optimization results

This module takes the battery dispatch from SHIPP (storage_p, storage_e)
and calculates degradation metrics:
  • Cycle counting (equivalent full cycles)
  • Depth of discharge distribution
  • Capacity fade estimation
  • State of health (SoH) evolution

Usage:
  from degradation import analyze_degradation
  
  # After running SHIPP optimization
  degradation = analyze_degradation(
      storage_p=os_fixed.storage_p[0].data,
      storage_e=os_fixed.storage_e[0].data,
      e_cap_nominal=300.0,  # MWh
      battery_params=setup['battery']
  )
  
  print(f"Total cycles: {degradation['total_cycles']:.1f}")
  print(f"Capacity fade: {degradation['capacity_fade_percent']:.2f}%")
"""

import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import rainflow


# =============================================================================
# CYCLE COUNTING
# =============================================================================

def count_equivalent_full_cycles(storage_p: List[float], 
                                 storage_e: List[float],
                                 e_cap: float) -> float:
    """
    Count equivalent full cycles (simple method).
    
    One full cycle = charging e_cap MWh then discharging e_cap MWh.
    
    Parameters
        ----------
    storage_p : list
        Battery power [MW] at each timestep (+ = discharging, - = charging)
        SHIPP convention: negative during charge, positive during discharge.
        See: OpSchedule.get_power_partition() which uses np.maximum(p,0)
        for discharge and np.minimum(p,0) for charge.
    storage_e : list
        Battery energy [MWh] at each timestep (state of charge)
    e_cap : float
        Nominal energy capacity [MWh]
    
    Returns
    -------
    float
        Equivalent full cycles
    """
    # Total energy discharged (p>0 catches discharging)
    total_discharge_MWh = sum([p for p in storage_p if p > 0])    
    # One full cycle discharges e_cap
    cycles = total_discharge_MWh / e_cap
    
    return cycles


def calculate_dod_distribution(storage_e: List[float], 
                               e_cap: float,
                               n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate depth of discharge (DoD) distribution.
    
    DoD = (e_cap - e_current) / e_cap
    
    Returns histogram of DoD values.
    """
    soc = np.array(storage_e)
    dod = (e_cap - soc) / e_cap
    
    # Histogram
    counts, bin_edges = np.histogram(dod, bins=n_bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return bin_centers, counts


# =============================================================================
# CAPACITY FADE MODELS
# =============================================================================

def simple_linear_fade(cycles: float, 
                       nominal_cycles: int = 4000,
                       eol_capacity: float = 0.80) -> float:
    """
    Simple linear capacity fade model.
    
    Capacity decreases linearly from 100% at 0 cycles to eol_capacity at nominal_cycles.
    
    Parameters
    ----------
    cycles : float
        Number of equivalent full cycles
    nominal_cycles : int
        Manufacturer's rated cycle life (e.g., 4000 for LFP)
    eol_capacity : float
        End-of-life capacity retention (e.g., 0.80 = 80%)
    
    Returns
    -------
    float
        Current capacity retention [0-1]
    """
    if cycles >= nominal_cycles:
        return eol_capacity
    
    # Linear fade: 100% → eol_capacity over nominal_cycles
    fade_rate = (1.0 - eol_capacity) / nominal_cycles
    capacity = 1.0 - fade_rate * cycles
    
    return max(capacity, eol_capacity)


def rainflow_cycle_counting(storage_e):
    """ASTM E1049 rainflow counting via rainflow library."""
    cycles = []
    for rng, mean, count, i_start, i_end in rainflow.extract_cycles(storage_e):
        cycles.append({
            'depth_MWh': rng,           # cycle range (depth)
            'mean_MWh': mean,           # mean SoC during cycle
            'count': count,             # 0.5 or 1.0 (half vs full cycle)
            'dod': rng / max(storage_e) # normalized DoD
        })
    return cycles


# =============================================================================
# MAIN DEGRADATION ANALYSIS
# =============================================================================

def analyze_degradation(storage_p: List[float],
                       storage_e: List[float],
                       e_cap_nominal: float,
                       battery_params: Dict) -> Dict:
    """
    Complete degradation analysis from SHIPP optimization results.
    
    Parameters
    ----------
    storage_p : list
        Battery power dispatch [MW]
    storage_e : list
        Battery state of charge [MWh]
    e_cap_nominal : float
        Nominal energy capacity [MWh]
    battery_params : dict
        Battery parameters from wp2_common.load_battery()
        
    Returns
    -------
    dict
        Degradation metrics:
        - total_cycles: equivalent full cycles
        - capacity_retention: current capacity [0-1]
        - capacity_fade_percent: capacity lost [%]
        - soh: state of health [%]
        - dod_distribution: (bin_centers, counts)
        - cycle_depth_distribution: list of (depth, count)
    """
    # Count cycles
    total_cycles = count_equivalent_full_cycles(storage_p, storage_e, e_cap_nominal)
    
    # Calculate capacity fade
    nominal_cycles = battery_params.get('n_full_load_cycles', 4000)
    eol_capacity = battery_params.get('eol_capacity_fraction', 0.80)
    
    capacity_retention = simple_linear_fade(total_cycles, nominal_cycles, eol_capacity)
    capacity_fade_percent = (1.0 - capacity_retention) * 100
    soh = capacity_retention * 100  # State of health [%]
    
    # DoD distribution
    dod_bins, dod_counts = calculate_dod_distribution(storage_e, e_cap_nominal)
    
    # Rainflow cycles (simplified)
    cycle_depths = rainflow_cycle_counting(storage_e)
    
    return {
        'total_cycles': total_cycles,
        'capacity_retention': capacity_retention,
        'capacity_fade_percent': capacity_fade_percent,
        'soh': soh,
        'dod_distribution': (dod_bins, dod_counts),
        'cycle_depth_distribution': cycle_depths,
        'e_cap_degraded': e_cap_nominal * capacity_retention,
        'p_cap_degraded': battery_params['power_capacity_W'] / 1e6 * capacity_retention,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_degradation_analysis(degradation: Dict, 
                              storage_e: List[float],
                              time_vec: np.ndarray,
                              save_path: str = None):
    """
    Create comprehensive degradation analysis plots.
    
    4 subplots:
    1. SoC time series with degraded capacity limit
    2. DoD distribution histogram
    3. Capacity fade over cycles
    4. Cycle depth distribution
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: SoC time series
    ax = axes[0, 0]
    ax.plot(time_vec, storage_e, linewidth=1)
    ax.axhline(degradation['e_cap_degraded'], color='red', linestyle='--', 
               label=f"Degraded capacity ({degradation['soh']:.1f}% SoH)")
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('State of Charge [MWh]')
    ax.set_title('Battery State of Charge')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: DoD distribution
    ax = axes[0, 1]
    dod_bins, dod_counts = degradation['dod_distribution']
    ax.bar(dod_bins, dod_counts, width=0.08, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Depth of Discharge')
    ax.set_ylabel('Frequency')
    ax.set_title('DoD Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Capacity fade curve
    ax = axes[1, 0]
    cycles_range = np.linspace(0, degradation['total_cycles'] * 1.5, 100)
    capacity_curve = [simple_linear_fade(c) * 100 for c in cycles_range]
    ax.plot(cycles_range, capacity_curve, linewidth=2)
    ax.scatter([degradation['total_cycles']], [degradation['soh']], 
               color='red', s=100, zorder=5, label='Current')
    ax.axhline(80, color='red', linestyle='--', alpha=0.5, label='EOL (80%)')
    ax.set_xlabel('Equivalent Full Cycles')
    ax.set_ylabel('Capacity Retention [%]')
    ax.set_title('Capacity Fade Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cycle depth distribution
    ax = axes[1, 1]
    cycle_depths = degradation['cycle_depth_distribution']
    if cycle_depths:
        depths = [c['depth_MWh'] for c in cycle_depths]
        counts = [c['count'] for c in cycle_depths]
        ax.hist(depths, bins=20, weights=counts, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Cycle Depth [MWh]')
    ax.set_ylabel('Number of Cycles')
    ax.set_title('Cycle Depth Distribution (Rainflow)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"✓ Saved degradation analysis: {save_path}")
    
    return fig


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_degradation_report(degradation: Dict, period_days: float):
    """
    Print a formatted degradation analysis report.
    """
    print("\n" + "=" * 72)
    print("BATTERY DEGRADATION ANALYSIS")
    print("=" * 72)
    
    print(f"\nSimulation period: {period_days:.1f} days")
    print(f"\nCycle analysis:")
    print(f"  Equivalent full cycles: {degradation['total_cycles']:.2f}")
    print(f"  Cycles per day: {degradation['total_cycles'] / period_days:.3f}")
    print(f"  Cycles per year: {degradation['total_cycles'] / period_days * 365:.1f}")
    
    print(f"\nCapacity degradation:")
    print(f"  State of Health (SoH): {degradation['soh']:.2f}%")
    print(f"  Capacity retention: {degradation['capacity_retention']:.4f}")
    print(f"  Capacity fade: {degradation['capacity_fade_percent']:.2f}%")
    print(f"  Degraded capacity: {degradation['e_cap_degraded']:.2f} MWh")
    print(f"  Degraded power: {degradation['p_cap_degraded']:.2f} MW")
    
    print(f"\nDepth of discharge:")
    dod_bins, dod_counts = degradation['dod_distribution']
    mean_dod = np.average(dod_bins, weights=dod_counts)
    print(f"  Mean DoD: {mean_dod:.2f}")
    print(f"  Max DoD: {max(dod_bins):.2f}")
    
    print("\n" + "=" * 72)
