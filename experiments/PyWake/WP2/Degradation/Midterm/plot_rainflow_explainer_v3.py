import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── ASTM E1049 rainflow ───────────────────────────────────────────────────────
def _extract_peaks(x):
    pts = [x[0]]
    for i in range(1, len(x) - 1):
        if (x[i] > x[i-1] and x[i] >= x[i+1]) or \
           (x[i] < x[i-1] and x[i] <= x[i+1]):
            pts.append(x[i])
    pts.append(x[-1])
    return np.array(pts)

def rainflow_count(signal):
    pts = _extract_peaks(signal)
    stack, cycles = [], []
    for pt in pts:
        stack.append(pt)
        while len(stack) >= 4:
            X = abs(stack[-1] - stack[-2])
            Y = abs(stack[-2] - stack[-3])
            if X >= Y:
                cycles.append((Y / 2.0, (stack[-2] + stack[-3]) / 2.0, 1.0))
                stack.pop(-2); stack.pop(-2)
            else:
                break
    for i in range(len(stack) - 1):
        cycles.append((abs(stack[i+1] - stack[i]) / 2.0,
                       (stack[i] + stack[i+1]) / 2.0, 0.5))
    return cycles

# ── Synthetic signals ─────────────────────────────────────────────────────────
t = np.linspace(0, 1, 800)

# Wing spar: 80 piecewise nodes alternating hi/lo → ~41 cycles
rng = np.random.default_rng(13)
n   = 80
st  = np.linspace(0, 1, n)
hi  = rng.uniform(0.45, 1.00, n // 2)
lo  = rng.uniform(-1.00, -0.35, n // 2)
sv  = np.empty(n); sv[0::2] = hi; sv[1::2] = lo
sv[0] = 0.1; sv[-1] = 0.05
stress = np.interp(t, st, sv)

# Battery SoC: 54 piecewise nodes → ~29 cycles
rng2   = np.random.default_rng(7)
soc_t  = np.linspace(0, 1, 54)
highs  = rng2.uniform(0.62, 0.88, 27)
lows   = rng2.uniform(0.15, 0.42, 27)
vals   = np.empty(54); vals[0::2] = highs; vals[1::2] = lows
soc    = np.interp(t, soc_t, vals)

cycles_stress = rainflow_count(stress)
cycles_soc    = rainflow_count(soc)
print(f"Stress: {len(cycles_stress)} cycles  |  SoC: {len(cycles_soc)} cycles")

# ── Style ─────────────────────────────────────────────────────────────────────
STRESS_COL = '#b5351b'
SOC_COL    = '#2166ac'
BG         = '#f7f9fc'

fig = plt.figure(figsize=(13, 7.2), facecolor='white')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.38,
                         top=0.85, bottom=0.14, left=0.07, right=0.97)

ax_s  = fig.add_subplot(gs[0, 0])
ax_b  = fig.add_subplot(gs[0, 1])
ax_sc = fig.add_subplot(gs[1, 0])
ax_bc = fig.add_subplot(gs[1, 1])

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.grid(True, color='white', lw=0.8, zorder=0)
    ax.tick_params(labelsize=9)

# ─── Stress signal ────────────────────────────────────────────────────────────
style_ax(ax_s)
ax_s.plot(t, stress, color=STRESS_COL, lw=1.1, zorder=2)
ax_s.axhline(0, color='#888', lw=0.6, ls='--', zorder=1)
ax_s.set_xlim(0, 1); ax_s.set_ylim(-1.35, 1.35)
ax_s.set_ylabel('Normalised stress  (–)', fontsize=10)
ax_s.set_xlabel('Time  (–)', fontsize=10)
ax_s.set_title('Wing Spar Load Spectrum', fontsize=13, fontweight='bold',
               color='#0d1b2a', pad=6)

# ─── SoC signal ───────────────────────────────────────────────────────────────
style_ax(ax_b)
ax_b.fill_between(t, soc, 0, alpha=0.12, color=SOC_COL, zorder=1)
ax_b.plot(t, soc, color=SOC_COL, lw=1.3, zorder=2)
ax_b.axhline(0.9, color=SOC_COL, lw=0.7, ls='--', alpha=0.5, zorder=1)
ax_b.axhline(0.1, color=SOC_COL, lw=0.7, ls='--', alpha=0.5, zorder=1)
ax_b.text(1.01, 0.9, 'SoC$_{max}$', va='center', fontsize=7.5,
          color=SOC_COL, alpha=0.75, transform=ax_b.get_yaxis_transform())
ax_b.text(1.01, 0.1, 'SoC$_{min}$', va='center', fontsize=7.5,
          color=SOC_COL, alpha=0.75, transform=ax_b.get_yaxis_transform())
ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1.08)
ax_b.set_ylabel('State of Charge  (–)', fontsize=10)
ax_b.set_xlabel('Time  (–)', fontsize=10)
ax_b.set_title('Battery Dispatch  (SoC)', fontsize=13, fontweight='bold',
               color='#0d1b2a', pad=6)

# ─── Stress scatter ───────────────────────────────────────────────────────────
style_ax(ax_sc)
sc1 = ax_sc.scatter([c[1] for c in cycles_stress],
                    [c[0] for c in cycles_stress],
                    c=[c[2] for c in cycles_stress],
                    cmap='Reds', vmin=0.4, vmax=1.0,
                    s=55, edgecolors='#6b1a0a', linewidths=0.4, zorder=3, alpha=0.85)
ax_sc.set_xlabel('Cycle mean  (–)', fontsize=10)
ax_sc.set_ylabel('Cycle amplitude  (–)', fontsize=10)
ax_sc.set_title(f'Extracted Cycles  (n = {len(cycles_stress)})',
                fontsize=11, color='#0d1b2a', pad=5)
cb1 = fig.colorbar(sc1, ax=ax_sc, pad=0.02)
cb1.set_label('Count', fontsize=8); cb1.ax.tick_params(labelsize=8)

# ─── SoC scatter ─────────────────────────────────────────────────────────────
style_ax(ax_bc)
sc2 = ax_bc.scatter([c[1] for c in cycles_soc],
                    [c[0] for c in cycles_soc],
                    c=[c[2] for c in cycles_soc],
                    cmap='Blues', vmin=0.4, vmax=1.0,
                    s=55, edgecolors='#08306b', linewidths=0.4, zorder=3, alpha=0.85)
ax_bc.set_xlabel('Cycle mean SoC  (–)', fontsize=10)
ax_bc.set_ylabel('Cycle amplitude  ½ΔSoC  (–)', fontsize=10)
ax_bc.set_title(f'Extracted Cycles  (n = {len(cycles_soc)})',
                fontsize=11, color='#0d1b2a', pad=5)
cb2 = fig.colorbar(sc2, ax=ax_bc, pad=0.02)
cb2.set_label('Count', fontsize=8); cb2.ax.tick_params(labelsize=8)

# ── Column headers ────────────────────────────────────────────────────────────
fig.text(0.27, 0.935, 'STRUCTURAL FATIGUE', ha='center', fontsize=11,
         fontweight='bold', color=STRESS_COL, alpha=0.85)
fig.text(0.73, 0.935, 'BATTERY DEGRADATION', ha='center', fontsize=11,
         fontweight='bold', color=SOC_COL, alpha=0.85)
fig.text(0.50, 0.910, '(artificially constructed signals)', ha='center',
         fontsize=9.5, color='#555', style='italic')

# # ── Bottom sentence ───────────────────────────────────────────────────────────
# fig.text(
#     0.5, 0.037,
#     'Rainflow counting extracts equivalent cycles from irregular signals'
#     ' — the method is identical in both domains.',
#     ha='center', va='center', fontsize=11.5, color='#1a1a2e', style='italic',
#     bbox=dict(boxstyle='round,pad=0.45', fc='#eef2f7', ec='#b0bec5',
#               alpha=0.95, lw=0.8)
# )

# fig.savefig('/mnt/user-data/outputs/rainflow_explainer_v3.pdf', dpi=300,
#             bbox_inches='tight', facecolor='white')
# fig.savefig('/mnt/user-data/outputs/rainflow_explainer_v3.png', dpi=200,
#             bbox_inches='tight', facecolor='white')
# import shutil; shutil.copy(__file__, '/mnt/user-data/outputs/plot_rainflow_explainer_v3.py')
# print("Saved.")

from pathlib import Path
OUT = Path(__file__).parent
fig.savefig(OUT / 'rainflow_explainer_v3.pdf', dpi=300,
            bbox_inches='tight', facecolor='white')
fig.savefig(OUT / 'rainflow_explainer_v3.png', dpi=200,
            bbox_inches='tight', facecolor='white')
print(f"Saved to {OUT}")