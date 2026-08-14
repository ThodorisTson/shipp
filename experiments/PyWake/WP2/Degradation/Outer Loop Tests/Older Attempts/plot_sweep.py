from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from datetime import datetime

# ── Windows-Safe Path Setup ──────────────────────────────────────────────────
# __file__ gets the absolute path of this script on your Windows machine
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The CSVs are located in the "Plan B Results" subfolder
RESULTS_DIR = os.path.join(SCRIPT_DIR, "Plan B Results")

# Priority order: list COARSE first, ZOOM last. 
CSV_PATHS = [
    os.path.join(RESULTS_DIR, "planB_lifetime_sweep_8760h_20yr_20260603_181702.csv"),   # coarse, 88-pt
    os.path.join(RESULTS_DIR, "planB_lifetime_sweep_8760h_20yr_20260603_221835.csv"),   # zoom, 24-pt
]

# Save the generated plots directly into the Results folder as well
OUTPUT_DIR = RESULTS_DIR

# ── Configuration ────────────────────────────────────────────────────────────
# Headline NPV columns (these track HEADLINE_BASIS; here = marginal).
SCENARIOS = {
    "no_deg": ("npv_no_deg",   "No degradation"),
    "xu":     ("npv_with_xu",  "Xu degradation"),
    "shi":    ("npv_with_shi", "Shi degradation"),
}
EUR_TO_MEUR = 1e-6

# Line-subplot slices: P-slice at the degraded-optimum E=550 (zoom-only value, so
# P spans 150-225); E-slice at P=175 (dense across both grids, E 300-1100).
FIXED_E_FOR_P_SLICE = 550
FIXED_P_FOR_E_SLICE = 175

# Zoom-local box for the quadratic fit (must match the zoom run extent).
QUAD_FIT_E_RANGE = (450, 700)
QUAD_FIT_P_RANGE = (150, 225)

SLIDE_SCENARIO = "xu"          # which model the single slide heatmap shows
SHOW_SAMPLE_POINTS = True
CMAP = "RdYlGn"

C = {"no_deg": "#1f77b4", "xu": "#d62728", "shi": "#2ca02c"}
MARK = {"no_deg": "*", "xu": "*", "shi": "*"}


# ── Data layer ───────────────────────────────────────────────────────────────
def load_and_pool(paths):
    """Read sweep CSVs, tag source, concat, dedupe on (e_cap,p_cap) keeping the
    finer (later-listed) run. NPV columns converted to MEUR. Returns a DataFrame."""
    frames = []
    for prio, p in enumerate(paths):
        # Double check file existence to output clear Windows path errors if missing
        if not os.path.exists(p):
            raise FileNotFoundError(f"Could not find CSV file at specified Windows path: {p}")
            
        df = pd.read_csv(p)
        df["__prio"] = prio                      # higher = finer (wins dupes)
        df["__source"] = os.path.basename(p)
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values("__prio")
    pooled = pooled.drop_duplicates(subset=["e_cap", "p_cap"], keep="last")
    for key, (col, _) in SCENARIOS.items():
        if col in pooled.columns:
            pooled[col] = pooled[col] * EUR_TO_MEUR
    return pooled.reset_index(drop=True)


def grid_optimum(df, col):
    """(E, P, NPV) at the max of `col` over the pooled, non-NaN data."""
    valid = df.dropna(subset=[col])
    row = valid.loc[valid[col].idxmax()]
    return float(row["e_cap"]), float(row["p_cap"]), float(row[col])


def fit_quadratic(df, col, e_range, p_range):
    """Fit z = b0 + b1 E + b2 P + b3 E^2 + b4 EP + b5 P^2 on the LOCAL box,
    return (E*, P*, z*, is_max, R2). Falls back to grid optimum if the
    stationary point is not a maximum (Hessian not negative-definite)."""
    m = df.dropna(subset=[col])
    m = m[(m.e_cap.between(*e_range)) & (m.p_cap.between(*p_range))]
    E, P, z = m.e_cap.values.astype(float), m.p_cap.values.astype(float), m[col].values
    A = np.column_stack([np.ones_like(E), E, P, E**2, E*P, P**2])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    b0, b1, b2, b3, b4, b5 = coef
    # R^2
    z_hat = A @ coef
    ss_res = float(np.sum((z - z_hat) ** 2))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # Stationary point: [[2b3, b4],[b4, 2b5]] [E,P] = [-b1,-b2]
    H = np.array([[2 * b3, b4], [b4, 2 * b5]])
    is_max = (b3 < 0) and (np.linalg.det(H) > 0)   # neg-definite Hessian
    try:
        star = np.linalg.solve(H, np.array([-b1, -b2]))
        Es, Ps = float(star[0]), float(star[1])
        # clip into the fit box
        Es = min(max(Es, e_range[0]), e_range[1])
        Ps = min(max(Ps, p_range[0]), p_range[1])
        zs = float(np.array([1, Es, Ps, Es**2, Es*Ps, Ps**2]) @ coef)
    except np.linalg.LinAlgError:
        Es = Ps = zs = float("nan")
        is_max = False
    return Es, Ps, zs, is_max, r2


def _tri(df, col):
    """Triangulation + values over the non-NaN samples for `col` (x=P, y=E)."""
    m = df.dropna(subset=[col])
    x, y, z = m.p_cap.values.astype(float), m.e_cap.values.astype(float), m[col].values
    return mtri.Triangulation(x, y), z, (x, y)


# ── Figure 1: full multi-panel (thesis style) ────────────────────────────────
def plot_multipanel(df, grid_opt, quad, outpath):
    fig = plt.figure(figsize=(16, 11.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.30)

    # Top row: three contour panels
    for j, key in enumerate(["no_deg", "xu", "shi"]):
        col, label = SCENARIOS[key]
        ax = fig.add_subplot(gs[0, j])
        tri, z, (xs, ys) = _tri(df, col)
        levels = 18
        cf = ax.tricontourf(tri, z, levels=levels, cmap=CMAP)
        ax.tricontour(tri, z, levels=levels, colors="k", linewidths=0.25, alpha=0.4)
        fig.colorbar(cf, ax=ax, label="MEUR", shrink=0.9)
        if SHOW_SAMPLE_POINTS:
            ax.scatter(xs, ys, s=6, c="k", alpha=0.35, linewidths=0, zorder=3)
        Eo, Po, No = grid_opt[key]
        ax.scatter([Po], [Eo], marker="*", s=320, c=C[key],
                   edgecolors="k", linewidths=1.2, zorder=5)
        # quadratic optimum (degraded models only; no-deg opt lies outside fit box)
        if key in quad and quad[key][3]:           # is_max
            Eq, Pq = quad[key][0], quad[key][1]
            ax.scatter([Pq], [Eq], marker="o", s=120, facecolors="none",
                       edgecolors=C[key], linewidths=2.0, zorder=6)
        ax.set_title(f"NPV, {label} [MEUR]", fontweight="bold")
        ax.set_xlabel("Power capacity P [MW]")
        ax.set_ylabel("Energy capacity E [MWh]")

    # Bottom-left: NPV vs P at fixed E
    axp = fig.add_subplot(gs[1, 0:1])
    sl = df[df.e_cap == FIXED_E_FOR_P_SLICE].dropna(subset=["npv_no_deg"]).sort_values("p_cap")
    for key in ["no_deg", "xu", "shi"]:
        col, label = SCENARIOS[key]
        axp.plot(sl.p_cap, sl[col], marker="o", color=C[key], label=label)
    axp.set_title(f"NPV vs P,  fixed E = {FIXED_E_FOR_P_SLICE} MWh")
    axp.set_xlabel("Power capacity P [MW]"); axp.set_ylabel("Lifetime NPV [MEUR]")
    axp.legend(fontsize=8); axp.grid(alpha=0.3)

    # Bottom-middle+right: NPV vs E at fixed P, with Xu degradation-gap shading
    axe = fig.add_subplot(gs[1, 1:3])
    sl = df[df.p_cap == FIXED_P_FOR_E_SLICE].dropna(subset=["npv_no_deg"]).sort_values("e_cap")
    for key in ["no_deg", "xu", "shi"]:
        col, label = SCENARIOS[key]
        axe.plot(sl.e_cap, sl[col], marker="o", color=C[key], label=label)
    axe.fill_between(sl.e_cap, sl["npv_with_xu"], sl["npv_no_deg"],
                     color=C["xu"], alpha=0.10, label="Xu degradation cost")
    axe.set_title(f"NPV vs E,  fixed P = {FIXED_P_FOR_E_SLICE} MW")
    axe.set_xlabel("Energy capacity E [MWh]"); axe.set_ylabel("Lifetime NPV [MEUR]")
    axe.legend(fontsize=8); axe.grid(alpha=0.3)

    # ── Shared optima legend (values per model + quadratic circles) ──────────
    from matplotlib.lines import Line2D
    handles = []
    for key in ["no_deg", "xu", "shi"]:
        Eo, Po, No = grid_opt[key]
        handles.append(Line2D([0], [0], marker="*", color="w",
                              markerfacecolor=C[key], markeredgecolor="k",
                              markersize=16,
                              label=f"{SCENARIOS[key][1]} optimum  ({Eo:.0f} MWh / {Po:.0f} MW, {No:.1f} MEUR)"))
    for key in ["xu", "shi"]:
        if quad[key][3]:                                   # is_max
            Eq, Pq, Nq, _, r2 = quad[key]
            handles.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor="none", markeredgecolor=C[key],
                                  markeredgewidth=2.0, markersize=12,
                                  label=f"{SCENARIOS[key][1]} quadratic opt  ({Eq:.0f} MWh / {Pq:.0f} MW, {Nq:.1f} MEUR, R²={r2:.3f})"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=2, fontsize=10, framealpha=0.95)

    # fig.suptitle("Lifetime NPV design space (pooled coarse + zoom runs, marginal basis)\n"
    #              "Stars = grid optimum.  Circles = quadratic-fit optimum (degraded models).  "
    #              "Dots = solved samples (density varies by run).",
    #              fontweight="bold", fontsize=13)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Figure 2: single slide heatmap ───────────────────────────────────────────
def plot_slide_heatmap(df, scenario, grid_opt, quad_opt, outpath):
    col, label = SCENARIOS[scenario]
    fig, ax = plt.subplots(figsize=(9, 7))
    tri, z, (xs, ys) = _tri(df, col)
    cf = ax.tricontourf(tri, z, levels=20, cmap=CMAP)
    ax.tricontour(tri, z, levels=20, colors="k", linewidths=0.25, alpha=0.35)
    fig.colorbar(cf, ax=ax, label="Lifetime NPV [MEUR]")

    if SHOW_SAMPLE_POINTS:
        ax.scatter(xs, ys, s=8, c="k", alpha=0.30, linewidths=0, zorder=3,
                   label="solved samples")

    # degraded grid optimum (this scenario)
    Eo, Po, No = grid_opt[scenario]
    ax.scatter([Po], [Eo], marker="*", s=420, c=C[scenario], edgecolors="k",
               linewidths=1.3, zorder=6,
               label=f"{label} optimum  ({Eo:.0f}/{Po:.0f} MW, {No:.1f} MEUR)")
    # continuous (quadratic) optimum — offset marker so it doesn't sit under the star
    Eq, Pq, Nq, is_max, r2 = quad_opt
    if is_max:
        ax.scatter([Pq], [Eq], marker="o", s=90, facecolors="none",
                   edgecolors="k", linewidths=1.6, zorder=7,
                   label=f"quad fit  ({Eq:.0f}/{Pq:.0f} MW, R²={r2:.3f})")
    # no-degradation optimum for contrast (from the coarse run, carried in pool)
    En, Pn, Nn = grid_opt["no_deg"]
    ax.scatter([Pn], [En], marker="D", s=130, c=C["no_deg"], edgecolors="k",
               linewidths=1.2, zorder=6,
               label=f"no-deg optimum  ({En:.0f}/{Pn:.0f} MW, {Nn:.1f} MEUR)")

    ax.set_xlabel("Power capacity P [MW]")
    ax.set_ylabel("Energy capacity E [MWh]")
    # ax.set_title(f"Degradation pulls the optimum inward — {label}\n"
    #              f"(pooled sweep, marginal NPV; optimum is a broad plateau, not a point)",
    #              fontweight="bold", fontsize=12)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Verify the output folder exists before processing
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = load_and_pool(CSV_PATHS)
    n_valid = df["npv_no_deg"].notna().sum()
    print(f"Pooled {len(df)} unique (E,P) cells from {len(CSV_PATHS)} runs "
          f"({n_valid} solved, {len(df) - n_valid} pruned/NaN).")

    # Generate your custom timestamp tag
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    grid_opt = {k: grid_optimum(df, col) for k, (col, _) in SCENARIOS.items()}
    print("\nGrid optima (max over pooled):")
    for k, (E, P, N) in grid_opt.items():
        print(f"  {SCENARIOS[k][1]:<16}: E={E:.0f} MWh  P={P:.0f} MW  NPV={N:.2f} MEUR")

    quad = {k: fit_quadratic(df, SCENARIOS[k][0], QUAD_FIT_E_RANGE, QUAD_FIT_P_RANGE)
            for k in ("xu", "shi")}
    print("\nContinuous (quadratic, zoom-local) degraded optima:")
    for k in ("xu", "shi"):
        Eq, Pq, Nq, is_max, r2 = quad[k]
        tag = "max" if is_max else "NOT a max (fell back conceptually to grid)"
        print(f"  {SCENARIOS[k][1]:<16}: E*={Eq:.0f}  P*={Pq:.0f}  NPV*={Nq:.2f}  "
              f"R²={r2:.3f}  [{tag}]")

    # Injected run_ts into the output filenames
    mp = os.path.join(OUTPUT_DIR, f"sweep_multipanel_pooled_{run_ts}.png")
    sh = os.path.join(OUTPUT_DIR, f"sweep_slide_heatmap_{SLIDE_SCENARIO}_{run_ts}.png")
    
    plot_multipanel(df, grid_opt, quad, mp)
    plot_slide_heatmap(df, SLIDE_SCENARIO, grid_opt, quad[SLIDE_SCENARIO], sh)
    print(f"\nWrote:\n  {mp}\n  {sh}")


if __name__ == "__main__":
    main()