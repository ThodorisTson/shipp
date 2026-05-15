import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── palette (matches sigma_effect_real) ──────────────────────────────────────
BG            = '#f7f9fc'
CONVEX_CURVE  = '#6baed6'   # light blue    — curve line
CONVEX_ARROW  = '#08306b'   # dark navy     — arrow & point
NONCONV_CURVE = '#fc8d59'   # light orange-red — curve line
NONCONV_ARROW = '#7f0000'   # dark crimson  — arrow & point
CONVEX        = CONVEX_ARROW   # titles / labels
NONCONV       = NONCONV_ARROW
GRID    = 'white'
EDGE    = '#cccccc'
TXT     = '#0d1b2a'
ANNOT   = '#555555'

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
    'grid.color':        GRID,
    'grid.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), facecolor='white')
fig.subplots_adjust(left=0.06, right=0.96, top=0.76, bottom=0.20, wspace=0.10)

x = np.linspace(0, 1, 400)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT  — convex bowl
# ══════════════════════════════════════════════════════════════════════════════
ax1 = axes[0]
ax1.set_facecolor(BG)
ax1.grid(True, color=GRID, lw=0.8, zorder=0)

y_cvx = 1.8 * (x - 0.5) ** 2 + 0.06

# tangent at x0 = 0.22
x0    = 0.22
y0    = 1.8 * (x0 - 0.5) ** 2 + 0.06
slope = 2 * 1.8 * (x0 - 0.5)
xt    = np.linspace(0.04, 0.52, 200)
yt    = slope * (xt - x0) + y0

# shaded fill under bowl
ax1.fill_between(x, y_cvx.min(), y_cvx, color=CONVEX_CURVE, alpha=0.12, zorder=1)
ax1.plot(x, y_cvx, color=CONVEX_CURVE, lw=2.4, zorder=3, solid_capstyle='round')

# tangent line
ax1.plot(xt, yt, color=CONVEX_ARROW, lw=1.3, ls='--', alpha=0.55, zorder=2)

# point of tangency
ax1.plot(x0, y0, 'o', color='white', ms=6.5, zorder=5,
         markeredgecolor=CONVEX_ARROW, markeredgewidth=1.8)

# gradient arrow (tangent direction)
ax1.annotate('', xy=(x0 + 0.20, y0 + slope * 0.20),
             xytext=(x0, y0),
             arrowprops=dict(arrowstyle='->', color=CONVEX_ARROW, lw=2.2,
                             mutation_scale=13))

# global minimum marker
xmin = 0.50
ymin = 1.8 * (xmin - 0.5) ** 2 + 0.06
ax1.plot(xmin, ymin, '*', color=CONVEX, ms=10, zorder=6)
ax1.text(xmin, ymin - 0.055, 'global\nmin', ha='center', va='top',
         fontsize=7.5, color=CONVEX)

ax1.set_xlim(0, 1);  ax1.set_ylim(-0.05, 0.58)
ax1.axis('off')

# thin bounding box
for sp in ['bottom', 'left']:
    ax1.spines[sp].set_visible(True)
    ax1.spines[sp].set_color(EDGE)
    ax1.spines[sp].set_linewidth(0.6)

ax1.set_title('convex\ngradient reliable', fontsize=12,
              fontweight='bold', color=CONVEX, pad=8, linespacing=1.5)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — non-convex wavy
# ══════════════════════════════════════════════════════════════════════════════
ax2 = axes[1]
ax2.set_facecolor(BG)
ax2.grid(True, color=GRID, lw=0.8, zorder=0)

y_ncvx = 0.20 * np.sin(4.8 * np.pi * x) + 0.30 * (x - 0.5) ** 2 + 0.22

# Point at x=0.38: descending slope leading into the first local minimum (~0.52)
# Arrow will point down-right toward local min; global min (star) is in second valley (~0.73)
x0b     = 0.38
y0b     = 0.20 * np.sin(4.8 * np.pi * x0b) + 0.30 * (x0b - 0.5) ** 2 + 0.22
slope_b = (0.20 * 4.8 * np.pi * np.cos(4.8 * np.pi * x0b)
           + 2 * 0.30 * (x0b - 0.5))
dx_arr  = 0.14                          # arrow horizontal reach
xt2     = np.linspace(0.20, 0.58, 200)
yt2     = slope_b * (xt2 - x0b) + y0b

# shaded fill (light red band)
ax2.fill_between(x, y_ncvx.min(), y_ncvx, color=NONCONV_CURVE, alpha=0.12, zorder=1)
ax2.plot(x, y_ncvx, color=NONCONV_CURVE, lw=2.4, zorder=3, solid_capstyle='round')

# tangent line
ax2.plot(xt2, yt2, color=NONCONV_ARROW, lw=1.3, ls='--', alpha=0.55, zorder=2)

# point of tangency
ax2.plot(x0b, y0b, 'o', color='white', ms=6.5, zorder=5,
         markeredgecolor=NONCONV_ARROW, markeredgewidth=1.8)

# gradient arrow — points down-right into LOCAL minimum (misleading)
ax2.annotate('', xy=(x0b + dx_arr, y0b + slope_b * dx_arr),
             xytext=(x0b, y0b),
             arrowprops=dict(arrowstyle='->', color=NONCONV_ARROW, lw=2.4,
                             mutation_scale=14))

# local minimum marker (where arrow leads — wrong target)
idx_local = np.argmin(y_ncvx[int(0.45*len(x)):int(0.62*len(x))]) + int(0.45*len(x))
xl, yl = x[idx_local], y_ncvx[idx_local]
ax2.text(xl, yl - 0.055, 'local\nmin', ha='center', va='top',
         fontsize=7.5, color=NONCONV_ARROW, alpha=0.7)

# true global minimum (different valley — star in gold)
idx_true = np.argmin(y_ncvx[int(0.65*len(x)):]) + int(0.65*len(x))
xtm, ytm = x[idx_true], y_ncvx[idx_true]
ax2.plot(xtm, ytm, '*', color='#e6a817', ms=10, zorder=6,
         markeredgecolor='#c47d00', markeredgewidth=0.6)
ax2.text(xtm, ytm - 0.055, 'global\nmin', ha='center', va='top',
         fontsize=7.5, color='#c47d00')

ax2.set_xlim(0, 1);  ax2.set_ylim(-0.05, 0.62)
ax2.axis('off')

for sp in ['bottom', 'left']:
    ax2.spines[sp].set_visible(True)
    ax2.spines[sp].set_color(EDGE)
    ax2.spines[sp].set_linewidth(0.6)

ax2.set_title('non-convex\ngradient misleads', fontsize=12,
              fontweight='bold', color=NONCONV, pad=8, linespacing=1.5)

# ── figure-level annotation ───────────────────────────────────────────────────
fig.text(0.515, 0.965,
         'Why Convexity Matters for Gradient-Based Optimisation',
         ha='center', fontsize=12, fontweight='bold', color=TXT)
fig.text(0.515, 0.920,
         'Arrow = gradient direction from marked point   ★ = true global minimum',
         ha='center', fontsize=9, color=ANNOT, style='italic')

plt.savefig('/mnt/user-data/outputs/convexity_sketch.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print("Saved.")
