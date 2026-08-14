"""
WP2 PyWake Analysis - Wake Model Comparison
============================================

Compare NOJ and Bastankhah & Porté-Agel (2014) wake models by reading 
their summary files. This is 85× faster and follows clean architecture.

Usage:
    1. First run: python run_wp2_noj.py
    2. Then run: python run_wp2_bastankhah.py
    3. Finally run: python run_wp2_comparison.py

Author: Thodoris
Date: 2026-02-12
"""

import re
from pathlib import Path
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PLOTS_DIR = SCRIPT_DIR / "wp2_comparison_plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Summary file paths
NOJ_SUMMARY = SCRIPT_DIR / "wp2_single_model_plots" / "wp2_noj_summary.txt"
BAST_SUMMARY = SCRIPT_DIR / "wp2_single_model_plots" / "wp2_bastankhah2014_summary.txt"

# Revenue assumption (parameterized)
PRICE_EUR_MWH = 50.0  # €/MWh - update with real prices if available

# =============================================================================
# PARSE SUMMARY FILES
# =============================================================================

def parse_summary_file(filepath):
    """
    Parse a structured summary TXT file and extract key metrics.
    
    Returns
    -------
    dict
        Dictionary with keys: aep_gwh, cf_pct, wake_loss_pct, runtime_s,
        n_turbines, rated_power_mw, total_capacity_mw, rotor_diameter_m,
        hub_height_m, wd_step, ws_step, interp_n, total_sims
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Summary file not found: {filepath}\n"
            f"Please run the corresponding model script first!"
        )
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract key metrics using regex
    data = {}
    
    # Results section
    data['aep_gwh'] = float(re.search(r'AEP:\s+([\d.]+)\s+GWh/year', content).group(1))
    data['cf_pct'] = float(re.search(r'Capacity Factor:\s+([\d.]+)%', content).group(1))
    data['wake_loss_pct'] = float(re.search(r'Wake Losses:\s+([\d.]+)%', content).group(1))
    data['runtime_s'] = float(re.search(r'Runtime:\s+([\d.]+)\s+s', content).group(1))
    
    # Farm configuration
    data['n_turbines'] = int(re.search(r'Turbines:\s+(\d+)', content).group(1))
    data['rated_power_mw'] = float(re.search(r'Rated power:\s+([\d.]+)\s+MW', content).group(1))
    data['total_capacity_mw'] = float(re.search(r'Total capacity:\s+([\d.]+)\s+MW', content).group(1))
    data['rotor_diameter_m'] = float(re.search(r'Rotor diameter:\s+([\d.]+)\s+m', content).group(1))
    data['hub_height_m'] = float(re.search(r'Hub height:\s+([\d.]+)\s+m', content).group(1))
    
    # Simulation settings
    data['wd_step'] = float(re.search(r'Wind direction step:\s+([\d.]+)°', content).group(1))
    data['ws_step'] = float(re.search(r'Wind speed step:\s+([\d.]+)\s+m/s', content).group(1))
    data['interp_n'] = int(re.search(r'Interpolation points:\s+([\d,]+)', content).group(1).replace(',', ''))
    data['total_sims'] = int(re.search(r'Total simulations:\s+([\d,]+)', content).group(1).replace(',', ''))
    
    return data

print("=" * 80)
print("WP2 PYWAKE ANALYSIS - WAKE MODEL COMPARISON")
print("=" * 80)

print("\nReading summary files...")

try:
    noj_data = parse_summary_file(NOJ_SUMMARY)
    print(f"✓ Loaded NOJ results from {NOJ_SUMMARY.name}")
except FileNotFoundError as e:
    print(f"\n❌ Error: {e}")
    print("   Please run: python run_wp2_noj.py")
    exit(1)

try:
    bast_data = parse_summary_file(BAST_SUMMARY)
    print(f"✓ Loaded Bastankhah results from {BAST_SUMMARY.name}")
except FileNotFoundError as e:
    print(f"\n❌ Error: {e}")
    print("   Please run: python run_wp2_bastankhah.py")
    exit(1)

# =============================================================================
# COMPARISON SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

print(f"\nWind Farm: {noj_data['n_turbines']} × {noj_data['rated_power_mw']:.1f} MW = {noj_data['total_capacity_mw']:.1f} MW")
print(f"Rotor diameter: {noj_data['rotor_diameter_m']:.1f} m")
print(f"Hub height: {noj_data['hub_height_m']:.1f} m\n")

# Table header
print(f"{'Model':<35} {'AEP (GWh/yr)':>15} {'CF (%)':>10} {'Wake Loss (%)':>15} {'Runtime (s)':>12}")
print("-" * 90)

# NOJ results
print(f"{'NOJ (Jensen)':<35} {noj_data['aep_gwh']:>15.2f} {noj_data['cf_pct']:>10.2f} {noj_data['wake_loss_pct']:>15.2f} {noj_data['runtime_s']:>12.2f}")

# Bastankhah results
print(f"{'Bastankhah & Porté-Agel (2014)':<35} {bast_data['aep_gwh']:>15.2f} {bast_data['cf_pct']:>10.2f} {bast_data['wake_loss_pct']:>15.2f} {bast_data['runtime_s']:>12.2f}")

# Differences
diff_aep = bast_data['aep_gwh'] - noj_data['aep_gwh']
diff_aep_pct = (bast_data['aep_gwh'] / noj_data['aep_gwh'] - 1) * 100
diff_cf = bast_data['cf_pct'] - noj_data['cf_pct']
diff_wake_loss = bast_data['wake_loss_pct'] - noj_data['wake_loss_pct']
diff_runtime = bast_data['runtime_s'] - noj_data['runtime_s']

print("-" * 90)
print(f"{'Difference (Bast - NOJ)':<35} {diff_aep:>+15.2f} {diff_cf:>+10.2f} {diff_wake_loss:>+15.2f} {diff_runtime:>+12.2f}")
print(f"{'Relative difference':<35} {diff_aep_pct:>+14.2f}%")

# =============================================================================
# REVENUE ANALYSIS
# =============================================================================

print("\n" + "=" * 80)

revenue_noj = noj_data['aep_gwh'] * 1000 * PRICE_EUR_MWH / 1e6
revenue_bast = bast_data['aep_gwh'] * 1000 * PRICE_EUR_MWH / 1e6
revenue_diff = revenue_bast - revenue_noj

print(f"\nRevenue Estimate (at {PRICE_EUR_MWH:.1f} €/MWh):")
print(f"  NOJ:                    {revenue_noj:>8.2f} M€/year")
print(f"  Bastankhah (2014):      {revenue_bast:>8.2f} M€/year")
print(f"  Difference:             {revenue_diff:>+8.2f} M€/year ({diff_aep_pct:+.2f}%)")

# =============================================================================
# COMPARISON PLOTS
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING COMPARISON PLOTS")
print("=" * 80)

models = ['NOJ', 'Bastankhah\n(2014)']

# AEP Comparison
fig, ax = plt.subplots(figsize=(7, 5))
aeps = [noj_data['aep_gwh'], bast_data['aep_gwh']]
bars = ax.bar(models, aeps, color=['#1f77b4', '#ff7f0e'])
ax.set_ylabel("AEP [GWh/year]", fontsize=11)
ax.set_title("Wake Model AEP Comparison", fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, aep in zip(bars, aeps):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{aep:.1f}',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "wp2_aep_comparison.png", dpi=200)
plt.close(fig)
print("✓ Saved: wp2_aep_comparison.png")

# Capacity Factor Comparison
fig, ax = plt.subplots(figsize=(7, 5))
cfs = [noj_data['cf_pct'], bast_data['cf_pct']]
bars = ax.bar(models, cfs, color=['#1f77b4', '#ff7f0e'])
ax.set_ylabel("Capacity Factor [%]", fontsize=11)
ax.set_title("Wake Model Capacity Factor Comparison", fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, cf in zip(bars, cfs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{cf:.2f}%',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "wp2_cf_comparison.png", dpi=200)
plt.close(fig)
print("✓ Saved: wp2_cf_comparison.png")

# Wake Loss Comparison
fig, ax = plt.subplots(figsize=(7, 5))
wake_losses = [noj_data['wake_loss_pct'], bast_data['wake_loss_pct']]
bars = ax.bar(models, wake_losses, color=['#1f77b4', '#ff7f0e'])
ax.set_ylabel("Wake Losses [%]", fontsize=11)
ax.set_title("Wake Model Wake Loss Comparison", fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, wl in zip(bars, wake_losses):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{wl:.2f}%',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "wp2_wake_loss_comparison.png", dpi=200)
plt.close(fig)
print("✓ Saved: wp2_wake_loss_comparison.png")

# Runtime Comparison
fig, ax = plt.subplots(figsize=(7, 5))
runtimes = [noj_data['runtime_s'], bast_data['runtime_s']]
bars = ax.bar(models, runtimes, color=['#1f77b4', '#ff7f0e'])
ax.set_ylabel("Runtime [s]", fontsize=11)
ax.set_title("Wake Model Runtime Comparison", fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, rt in zip(bars, runtimes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{rt:.2f}s',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "wp2_runtime_comparison.png", dpi=200)
plt.close(fig)
print("✓ Saved: wp2_runtime_comparison.png")

# =============================================================================
# SAVE COMPARISON SUMMARY
# =============================================================================

comparison_file = PLOTS_DIR / "wp2_comparison_summary.txt"

with open(comparison_file, "w") as f:
    f.write("WP2 Wake Model Comparison Summary\n")
    f.write("="*60 + "\n\n")
    
    f.write("MODELS COMPARED\n")
    f.write("-"*30 + "\n")
    f.write("Model 1: NOJ (Jensen, 1983)\n")
    f.write("Model 2: Bastankhah & Porté-Agel (2014)\n\n")
    
    f.write("FARM CONFIGURATION\n")
    f.write("-"*30 + "\n")
    f.write(f"Turbines: {noj_data['n_turbines']}\n")
    f.write(f"Rated power: {noj_data['rated_power_mw']:.1f} MW\n")
    f.write(f"Total capacity: {noj_data['total_capacity_mw']:.1f} MW\n")
    f.write(f"Rotor diameter: {noj_data['rotor_diameter_m']:.1f} m\n")
    f.write(f"Hub height: {noj_data['hub_height_m']:.1f} m\n\n")
    
    f.write("RESULTS COMPARISON\n")
    f.write("-"*30 + "\n")
    f.write(f"{'Metric':<25} {'NOJ':>12} {'Bastankhah':>12} {'Difference':>12}\n")
    f.write("-"*64 + "\n")
    f.write(f"{'AEP [GWh/year]':<25} {noj_data['aep_gwh']:>12.2f} {bast_data['aep_gwh']:>12.2f} {diff_aep:>+12.2f}\n")
    f.write(f"{'Capacity Factor [%]':<25} {noj_data['cf_pct']:>12.2f} {bast_data['cf_pct']:>12.2f} {diff_cf:>+12.2f}\n")
    f.write(f"{'Wake Losses [%]':<25} {noj_data['wake_loss_pct']:>12.2f} {bast_data['wake_loss_pct']:>12.2f} {diff_wake_loss:>+12.2f}\n")
    f.write(f"{'Runtime [s]':<25} {noj_data['runtime_s']:>12.2f} {bast_data['runtime_s']:>12.2f} {diff_runtime:>+12.2f}\n\n")
    
    f.write("RELATIVE DIFFERENCE\n")
    f.write("-"*30 + "\n")
    f.write(f"AEP: {diff_aep_pct:+.2f}% ({diff_aep:+.2f} GWh/year)\n")
    f.write(f"This represents {revenue_diff:+.2f} M€/year at {PRICE_EUR_MWH:.1f} €/MWh\n\n")
    
    f.write("SIMULATION SETTINGS\n")
    f.write("-"*30 + "\n")
    f.write(f"Wind direction step: {noj_data['wd_step']:.0f}°\n")
    f.write(f"Wind speed step: {noj_data['ws_step']:.0f} m/s\n")
    f.write(f"Interpolation points: {noj_data['interp_n']}\n")
    f.write(f"Total simulations: {noj_data['total_sims']}\n")

print(f"\n✓ Comparison summary saved to: {comparison_file.name}")

print("\n" + "=" * 80)
print("✓ COMPARISON COMPLETE")
print("=" * 80)
print(f"\nPlots saved in: {PLOTS_DIR}/")
print("  - wp2_aep_comparison.png")
print("  - wp2_cf_comparison.png")
print("  - wp2_wake_loss_comparison.png")
print("  - wp2_runtime_comparison.png")
print(f"  - {comparison_file.name}")
print("=" * 80)