"""
plot_wp2_turbine_curves.py
==========================
Plots the NREL 5 MW power curve and thrust coefficient curve
directly from WP2_Wind_Farm.yaml.

Power curve is derived as:
    P(v) = Cp(v) * 0.5 * rho * A * v^3   [capped at rated_power]

Run:  python plot_wp2_turbine_curves.py
Output: wp2_turbine_curves.png  (same folder as this script)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# ── Load YAML ─────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
with open(HERE / 'WP2_Wind_Farm.yaml') as f:
    wf = yaml.safe_load(f)

turb = wf['turbines']
perf = turb['performance']

# ── Parameters ────────────────────────────────────────────────────────────────
RHO         = 1.225                       # kg/m³  (standard air density)
D           = turb['rotor_diameter']      # 125.88 m
A           = np.pi / 4 * D**2           # swept area [m²]
RATED_W     = perf['rated_power']         # 5 000 000 W
RATED_MW    = RATED_W / 1e6

cp_ws  = np.array(perf['Cp_curve']['Cp_wind_speeds'])
cp_val = np.array(perf['Cp_curve']['Cp_values'])
ct_ws  = np.array(perf['Ct_curve']['Ct_wind_speeds'])
ct_val = np.array(perf['Ct_curve']['Ct_values'])

# ── Derived power curve ───────────────────────────────────────────────────────
ws_plot = np.linspace(0, 25, 500)

cp_interp = np.interp(ws_plot, cp_ws, cp_val, left=0.0, right=0.0)
ct_interp = np.interp(ws_plot, ct_ws, ct_val, left=0.0, right=0.0)

power_w = cp_interp * 0.5 * RHO * A * ws_plot**3
power_w = np.minimum(power_w, RATED_W)          # cap at rated
power_w = np.where(ws_plot < 3.0, 0.0, power_w) # below cut-in → 0
power_w = np.where(ws_plot > 25.0, 0.0, power_w)# above cut-out → 0
power_mw = power_w / 1e6

ct_interp = np.where(ws_plot < 3.0,  0.0, ct_interp)
ct_interp = np.where(ws_plot > 25.0, 0.0, ct_interp)

# ── Style ─────────────────────────────────────────────────────────────────────
BG     = '#f7f9fc'
C_POW  = '#2166ac'
C_CT   = '#b5351b'
TXT    = '#0d1b2a'
EDGE   = '#cccccc'

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.facecolor':    BG,
    'figure.facecolor':  'white',
    'text.color':        TXT,
    'axes.edgecolor':    EDGE,
    'axes.labelcolor':   TXT,
    'xtick.color':       '#444444',
    'ytick.color':       '#444444',
    'grid.color':        'white',
    'grid.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor='white')
fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.15, wspace=0.32)

# ── Left: power curve ─────────────────────────────────────────────────────────
ax1.fill_between(ws_plot, power_mw, color=C_POW, alpha=0.10)
ax1.plot(ws_plot, power_mw, color=C_POW, lw=2.4)
ax1.axhline(RATED_MW, color=C_POW, lw=1.0, ls='--', alpha=0.55,
            label=f'Rated  {RATED_MW:.0f} MW')
ax1.set_xlabel('Wind speed  (m s⁻¹)', fontsize=10)
ax1.set_ylabel('Power  (MW)', fontsize=10)
ax1.set_xlim(0, 25)
ax1.set_ylim(bottom=0)
ax1.grid(True)
ax1.legend(fontsize=9, framealpha=0.0)
ax1.set_title('Power Curve', fontsize=11, fontweight='bold', color=C_POW, pad=8)

# ── Right: Ct curve ───────────────────────────────────────────────────────────
ax2.fill_between(ws_plot, ct_interp, color=C_CT, alpha=0.10)
ax2.plot(ws_plot, ct_interp, color=C_CT, lw=2.4)
ax2.set_xlabel('Wind speed  (m s⁻¹)', fontsize=10)
ax2.set_ylabel('Thrust coefficient  Cₜ  (–)', fontsize=10)
ax2.set_xlim(0, 25)
ax2.set_ylim(bottom=0)
ax2.grid(True)
ax2.set_title('Thrust Coefficient Curve', fontsize=11, fontweight='bold', color=C_CT, pad=8)

# ── Figure title ──────────────────────────────────────────────────────────────
fig.suptitle(
    f'NREL 5 MW Reference Turbine  —  D = {D:.1f} m  |  A = {A/1e4:.2f} ×10⁴ m²  '
    f'|  Hub height = {turb["hub_height"]:.0f} m',
    fontsize=11, fontweight='bold', color=TXT, y=0.97
)

# ── Save ──────────────────────────────────────────────────────────────────────
out = HERE / 'wp2_turbine_curves.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved → {out}")
