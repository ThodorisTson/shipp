"""
WP2 PyWake Analysis - NOJ (Jensen) Wake Model
============================================

Baseline AEP calculation using NOJ wake model with:
- Clean refactored setup via wp2_common.py
- Key diagnostics (AEP, CF, wake loss, runtime)
- Standard visual outputs (layout, wake map, per-turbine power)
- Concise farm/config summary for reproducibility

Author: Thodoris
Date: 2026-02-11
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from py_wake.flow_map import XYGrid


from wp2_common import quick_setup, get_wake_model

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PLOTS_DIR = SCRIPT_DIR / "wp2_single_model_plots"
PLOTS_DIR.mkdir(exist_ok=True)

SITE_YAML = SCRIPT_DIR / "wp2_site_weibull.yaml"
TURBINE_YAML = SCRIPT_DIR / "wp2_turbine_nrel5mw.yaml"
LAYOUT_CSV = SCRIPT_DIR / "wp2_turbine_coordinates.csv"

# Coarser for iteration, refine for final validation
CONFIG = {
    "interp_n": 2000,  # 2000 for dev, 10000 for final
    "wd_step": 5,      # 5° for dev, 1° for final
    "ws_step": 1,      # keep 1 m/s
}

MODEL_NAME = "NOJ"

# Representative case for plots
WD_PLOT = 270
WS_PLOT = 10

# =============================================================================
# SETUP
# =============================================================================

print("=" * 80)
print("WP2 PYWAKE ANALYSIS - NOJ (JENSEN) WAKE MODEL")
print("=" * 80)

setup = quick_setup(SITE_YAML, TURBINE_YAML, LAYOUT_CSV, config=CONFIG)

site = setup["site"]
windturbine = setup["windturbine"]
x, y = setup["x"], setup["y"]
ws_bins, wd_bins = setup["ws_bins"], setup["wd_bins"]

total_capacity_mw = setup["n_turbines"] * setup["turbine_dat"]["rated_power"] / 1e6

# =============================================================================
# RUN WAKE MODEL
# =============================================================================

print("\n" + "=" * 80)
print("RUNNING NOJ WAKE MODEL")
print("=" * 80)

start_time = time.time()

wf_model = get_wake_model(MODEL_NAME, site, windturbine)
sim_res = wf_model(x, y, ws=ws_bins, wd=wd_bins)

elapsed = time.time() - start_time

# =============================================================================
# RESULTS
# =============================================================================

# Quick sanity checks (optional, keep for now)
ws_min = float(sim_res.localWind.ws.min())
ws_max = float(sim_res.localWind.ws.max())
print("Wind speed min/max:", ws_min, ws_max)

print("Power at 15 m/s:", windturbine.power(15))
print("Power at 10 m/s:", windturbine.power(10))
print("Power at 3 m/s:", windturbine.power(3))

# AEP from PyWake
# In your PyWake version, aep() is already in GWh (your raw value is ~1403.96)
aep_gwh = float(sim_res.aep().sum())
print("Raw AEP value:", aep_gwh)

capacity_factor = aep_gwh * 1e3 / (total_capacity_mw * 8760)

# Ideal (no-wake) AEP: compute for 1 turbine and scale by N
# IMPORTANT: Do NOT divide by 1e9 here, because aep() is already in GWh
single_turbine_res = wf_model([x[0]], [y[0]], ws=ws_bins, wd=wd_bins)
ideal_single_aep_gwh = float(single_turbine_res.aep().sum())

ideal_aep_gwh = ideal_single_aep_gwh * setup["n_turbines"]
wake_loss_pct = (1 - aep_gwh / ideal_aep_gwh) * 100

print(f"\n✓ Simulation complete in {elapsed:.2f}s")
print(f"  AEP: {aep_gwh:.2f} GWh/year")
print(f"  Capacity factor: {capacity_factor*100:.2f}%")
print(f"  Wake losses: {wake_loss_pct:.2f}%")

# Concise reproducibility summary
print("\n--- FARM SUMMARY ---")
print(f"Site: {setup['site_dat']['name']}")
print(f"Location: {setup['site_dat']['latitude']:.2f}°N, {setup['site_dat']['longitude']:.2f}°E")
print(f"Turbines: {setup['n_turbines']} × {setup['turbine_dat']['rated_power']/1e6:.1f} MW")
print(f"Total capacity: {total_capacity_mw:.1f} MW")
print(f"Rotor diameter: {setup['turbine_dat']['diameter']:.1f} m")
print(f"Hub height: {setup['turbine_dat']['hub_height']:.1f} m")
print(f"Wind directions: {len(wd_bins)} (step {CONFIG['wd_step']}°)")
print(f"Wind speeds: {len(ws_bins)} (step {CONFIG['ws_step']} m/s)")
print(f"Interpolation points: {CONFIG['interp_n']}")
print(f"Total simulations: {len(wd_bins) * len(ws_bins):,}")

# Optional diagnostics (safe to keep)
print("Sum of sector frequencies:", float(setup["site_dat"]["frequencies"].sum()))
print("Probability sum check:", float(sim_res.localWind.P.sum()))

# =============================================================================
# PLOTS
# =============================================================================

# Layout plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x, y, s=20)
ax.set_title("WP2 Turbine Layout")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "wp2_layout.png", dpi=200)
plt.close(fig)

# Wake map plot (PyWake version requires Grid object)
sim_case = wf_model(x, y, ws=WS_PLOT, wd=WD_PLOT)

x_grid = np.linspace(min(x) - 1000, max(x) + 1000, 200)
y_grid = np.linspace(min(y) - 1000, max(y) + 1000, 200)
grid = XYGrid(x=x_grid, y=y_grid)

flow_map = sim_case.flow_map(grid)

fig, ax = plt.subplots(figsize=(8, 6))
flow_map.plot_wake_map(ax=ax)
ax.set_title(f"NOJ Wake Map – WS={WS_PLOT} m/s, WD={WD_PLOT}°")

# Add summary statistics (thesis-grade, clean white background)
textstr = (
    f"AEP: {aep_gwh:.1f} GWh/yr\n"
    f"CF: {capacity_factor*100:.1f}%\n"
    f"Wake loss: {wake_loss_pct:.1f}%"
)
ax.text(
    0.02, 0.98,
    textstr,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "wp2_noj_wake_map.png", dpi=200)
plt.close(fig)

# Per-turbine power distribution at representative case
power_per_turbine = sim_res.Power.sel(ws=WS_PLOT, wd=WD_PLOT).values.flatten()

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(np.arange(len(power_per_turbine)), power_per_turbine / 1e6)
ax.set_ylabel("Power [MW]")
ax.set_title(f"Per-Turbine Power – NOJ (WS={WS_PLOT} m/s, WD={WD_PLOT}°)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "wp2_noj_power_distribution.png", dpi=200)
plt.close(fig)

# Power Duration Curve
print("\nGenerating power duration curve...")

# Calculate farm power for all (wd, ws) bins weighted by probability
# Sum power across all turbines, then weight by wind probability
farm_power = sim_res.Power.sum(dim='wt') / 1e6  # Total farm power in MW
probabilities = sim_res.localWind.P  # Probability for each (wd, ws) bin

# Flatten and create weighted samples for duration curve
# Each (wd, ws) combination is repeated proportionally to its probability
farm_power_flat = farm_power.values.flatten()
prob_flat = probabilities.values.flatten()

# Create duration curve by sorting power weighted by probability
# We'll sample based on probabilities to create realistic duration curve
n_samples = 8760  # Hourly resolution for full year
np.random.seed(42)  # For reproducibility
sample_indices = np.random.choice(
    len(farm_power_flat), 
    size=n_samples, 
    p=prob_flat/prob_flat.sum()
)
power_samples = farm_power_flat[sample_indices]

# Sort in descending order for duration curve
sorted_power = np.sort(power_samples)[::-1]
duration_pct = np.arange(len(sorted_power)) / len(sorted_power) * 100

# Calculate statistics
mean_power = power_samples.mean()
p50_power = np.percentile(power_samples, 50)
capacity_pct = (mean_power / total_capacity_mw) * 100

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(duration_pct, sorted_power, linewidth=2.5, color="steelblue")
ax.fill_between(duration_pct, 0, sorted_power, alpha=0.15, color="steelblue")

ax.axhline(mean_power, linestyle="--", linewidth=2, color="red",
           label=f"Mean: {mean_power:.1f} MW ({capacity_pct:.1f}% CF)")
ax.axhline(total_capacity_mw, linestyle=":", linewidth=1.5, color="gray",
           label=f"Capacity: {total_capacity_mw:.1f} MW")

ax.set_xlabel("Duration [%]", fontsize=11)
ax.set_ylabel("Farm Power [MW]", fontsize=11)
ax.set_title("NOJ Power Duration Curve", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Add statistics text box
stats_text = (
    f"Hours > 50% capacity: {(sorted_power > total_capacity_mw*0.5).sum()/len(sorted_power)*100:.1f}%\n"
    f"Hours > 75% capacity: {(sorted_power > total_capacity_mw*0.75).sum()/len(sorted_power)*100:.1f}%\n"
    f"Hours > 90% capacity: {(sorted_power > total_capacity_mw*0.9).sum()/len(sorted_power)*100:.1f}%"
)
ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(PLOTS_DIR / "wp2_noj_power_duration.png", dpi=200)
plt.close(fig)
print("✓ Saved: wp2_noj_power_duration.png")

# =============================================================================
# SAVE STRUCTURED SUMMARY
# =============================================================================

summary_file = PLOTS_DIR / f"wp2_{MODEL_NAME.lower()}_summary.txt"

with open(summary_file, "w") as f:
    f.write(f"WP2 {MODEL_NAME} Wake Model Analysis\n")
    f.write("="*60 + "\n\n")

    f.write("RESULTS\n")
    f.write("-"*30 + "\n")
    f.write(f"AEP: {aep_gwh:.2f} GWh/year\n")
    f.write(f"Capacity Factor: {capacity_factor*100:.2f}%\n")
    f.write(f"Wake Losses: {wake_loss_pct:.2f}%\n")
    f.write(f"Runtime: {elapsed:.2f} s\n\n")

    f.write("FARM CONFIGURATION\n")
    f.write("-"*30 + "\n")
    f.write(f"Turbines: {setup['n_turbines']}\n")
    f.write(f"Rated power: {setup['turbine_dat']['rated_power']/1e6:.1f} MW\n")
    f.write(f"Total capacity: {total_capacity_mw:.1f} MW\n")
    f.write(f"Rotor diameter: {setup['turbine_dat']['diameter']:.1f} m\n")
    f.write(f"Hub height: {setup['turbine_dat']['hub_height']:.1f} m\n\n")

    f.write("SIMULATION SETTINGS\n")
    f.write("-"*30 + "\n")
    f.write(f"Wind direction step: {CONFIG['wd_step']}°\n")
    f.write(f"Wind speed step: {CONFIG['ws_step']} m/s\n")
    f.write(f"Interpolation points: {CONFIG['interp_n']}\n")
    f.write(f"Total simulations: {len(wd_bins)*len(ws_bins)}\n")

print(f"\n✓ Summary saved to {summary_file.name}")

print("\n✓ NOJ analysis complete.")
print("=" * 80)