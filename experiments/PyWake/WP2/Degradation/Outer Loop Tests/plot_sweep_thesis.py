r"""
plot_sweep_thesis.py
====================
Individual thesis-ready figures from the Plan B 2D parameter sweep.

Location: Outer Loop Tests/   (thesis_style.py is one folder up in Degradation/)

Auto-discovers the two CSV files:
  COARSE  — planB_lifetime_sweep_*<tag>*.csv  (full domain, wide E/P grid)
  ZOOM    — planB_lifetime_sweep_*<tag>*.csv  (refined region, dense E/P grid)

Strategy per memory note: two SEPARATE figures, never pooled.
  1. Full-domain heatmap with dashed box marking the refined region  → Fig sweep-A
  2. Zoomed heatmap showing the design result                        → Fig sweep-B
  3. NPV vs E at fixed P=175 MW (all three degradation scenarios)    → Fig sweep-C
  4. NPV vs P at fixed E=550 MWh (Xu grid optimum)                  → Fig sweep-D

CAPTION VALUES printed at runtime:
  Grid optima (E*, P*, NPV*) for each scenario from each run
  Quadratic-fit optimum with R² (zoom only)

CONFIGURATION (edit at top of this file):
  COARSE_TAG   — substring matching the coarse-run CSV filename
  ZOOM_TAG     — substring matching the zoom-run CSV filename
  SCENARIO     — "xu" or "shi" (which degraded model is the PRIMARY heatmap)
  FIXED_P_MW   — P slice for NPV-vs-E plot
  FIXED_E_MWH  — E slice for NPV-vs-P plot  (set to the Xu grid optimum)
"""
from __future__ import annotations
import sys
from pathlib import Path

# ── Path guard: finds thesis_style.py in any parent folder ───────────────────
for _d in Path(__file__).resolve().parents:
    if (_d / "thesis_style.py").exists():
        sys.path.insert(0, str(_d))
        break
else:
    raise FileNotFoundError("thesis_style.py not found in any parent folder")

import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')   # uncomment if running headless
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from thesis_style import (apply_thesis_style, figsize, TUDELFT,
                          FS_BASE, FS_LABEL, FS_ANNOT)

P = apply_thesis_style(palette="brand", usetex=False)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
COARSE_TAG   = "221934"    # substring uniquely identifying the coarse CSV
ZOOM_TAG     = "014612"    # substring uniquely identifying the zoom CSV
SCENARIO     = "xu"        # primary degraded model for heatmaps ("xu" or "shi")
FIXED_P_MW   = 175         # fixed P for the NPV-vs-E line plot
FIXED_E_MWH  = 550         # fixed E for the NPV-vs-P line plot (Xu grid opt from zoom)
EUR_TO_MEUR  = 1e-6

# Zoom region boundaries — drawn as dashed box on the full heatmap
ZOOM_E_MIN, ZOOM_E_MAX = 450, 750    # MWh
ZOOM_P_MIN, ZOOM_P_MAX = 125, 250    # MW

# White-space padding on the refined heatmaps. The no-deg diamond sits on the
# grid corner closest to the axes; without padding it is clipped by the frame.
ZOOM_PAD_P = 8       # MW  of margin on each side of the P axis
ZOOM_PAD_E = 14      # MWh of margin on each side of the E axis

# Quadratic-fit domain (zoom only)
QUAD_E_RANGE = (450, 750)
QUAD_P_RANGE = (125, 250)

# ── Colour assignments ────────────────────────────────────────────────────────
C_NODEG = TUDELFT["navy"]      # no-degradation scenario
C_XU    = TUDELFT["darkred"]   # Xu degradation scenario
C_SHI   = TUDELFT["blue"]      # Shi degradation scenario
C_SHADE = {                    # fill_between alpha for degradation gap
    "xu":  (C_XU,  0.12),
    "shi": (C_SHI, 0.10),
}
C_SCENARIO = {"no_deg": C_NODEG, "xu": C_XU, "shi": C_SHI}

CMAP_NPV = "RdYlGn"           # NPV landscape: red=low, green=high

SCENARIO_LABEL = {
    "no_deg": "No degradation",
    "xu":     "Xu et al. (2016)",
    "shi":    "Shi et al. (2018)",
}
NPV_COL = {
    "no_deg": "npv_no_deg",
    "xu":     "npv_with_xu",
    "shi":    "npv_with_shi",
}

# ═══════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _load_csv(results_dir: Path, tag: str) -> tuple[pd.DataFrame, str]:
    """Load CSV whose filename contains `tag`, pick most-recent by mtime."""
    candidates = sorted(results_dir.glob(f"planB_lifetime_sweep*{tag}*.csv"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No sweep CSV with tag '{tag}' in {results_dir}")
    chosen = candidates[-1]
    df = pd.read_csv(chosen)
    for col in NPV_COL.values():
        if col in df.columns:
            df[col] = df[col] * EUR_TO_MEUR
    print(f"  [{tag}] loaded: {chosen.name}  ({len(df)} rows)")
    return df, chosen.stem


def grid_optimum(df: pd.DataFrame, scenario: str) -> tuple[float, float, float]:
    """(E, P, NPV_MEUR) at the grid optimum for `scenario`."""
    col = NPV_COL[scenario]
    v = df.dropna(subset=[col])
    row = v.loc[v[col].idxmax()]
    return float(row["e_cap"]), float(row["p_cap"]), float(row[col])


def fit_quadratic(df: pd.DataFrame, scenario: str,
                  e_range: tuple, p_range: tuple
                  ) -> tuple[float, float, float, bool, float]:
    """Fit z = b0+b1E+b2P+b3E²+b4EP+b5P² on the local box.
    Returns (E*, P*, NPV*, is_max, R²)."""
    col = NPV_COL[scenario]
    m = df.dropna(subset=[col])
    m = m[m.e_cap.between(*e_range) & m.p_cap.between(*p_range)]
    E, P_, z = m.e_cap.values.astype(float), m.p_cap.values.astype(float), m[col].values
    A = np.column_stack([np.ones_like(E), E, P_, E**2, E*P_, P_**2])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    b0, b1, b2, b3, b4, b5 = coef
    H = np.array([[2*b3, b4], [b4, 2*b5]])
    is_max = bool((b3 < 0) and (np.linalg.det(H) > 0))
    try:
        star = np.linalg.solve(H, [-b1, -b2])
        Es, Ps = float(np.clip(star[0], *e_range)), float(np.clip(star[1], *p_range))
        zs = float(np.array([1, Es, Ps, Es**2, Es*Ps, Ps**2]) @ coef)
    except np.linalg.LinAlgError:
        Es = Ps = zs = float("nan")
        is_max = False
    z_hat = A @ coef
    ss_res = float(np.sum((z - z_hat)**2))
    ss_tot = float(np.sum((z - z.mean())**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return Es, Ps, zs, is_max, r2


def _save(fig, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  ✓ {stem}.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIG A — FULL-DOMAIN HEATMAP (coarse) with refinement box
# ═══════════════════════════════════════════════════════════════════════════

def plot_full_heatmap(df: pd.DataFrame, scenario: str,
                      grid_opt_all: dict, out_dir: Path) -> None:
    """
    Full-domain NPV landscape (coarse sweep) with a dashed box marking the
    refined-zoom region.  Shows grid optima for both the focal scenario and
    the no-deg baseline for contrast.

    CAPTION: "NPV landscape over the full design space (E: 150–1500 MWh,
    P: 75–325 MW) for the {scenario} degradation model. Stars mark the grid
    optima; the dashed rectangle bounds the refined sweep region."
    """
    col = NPV_COL[scenario]
    m   = df.dropna(subset=[col])
    tri = mtri.Triangulation(m.p_cap.values, m.e_cap.values)

    fig, ax = plt.subplots(figsize=figsize(0.72, aspect=0.95))

    cf = ax.tricontourf(tri, m[col].values, levels=18, cmap=CMAP_NPV)
    ax.tricontour(tri, m[col].values, levels=18,
                  colors="k", linewidths=0.20, alpha=0.35)
    cb = fig.colorbar(cf, ax=ax, shrink=0.88)
    cb.set_label("Lifetime NPV  (MEUR)")

    # Sample dots
    ax.scatter(m.p_cap, m.e_cap, s=5, c="k", alpha=0.30,
               linewidths=0, zorder=3)

    # Focal scenario optimum (star). Omitted for no-deg: it coincides with the
    # no-deg diamond drawn below, which is the marker reused across the other
    # panels for the no-deg optimum position.
    if scenario != "no_deg":
        Eo, Po, No = grid_opt_all[scenario]
        ax.scatter([Po], [Eo], marker="*", s=340, c=C_SCENARIO[scenario],
                   edgecolors="k", linewidths=1.0, zorder=6,
                   label=f"{SCENARIO_LABEL[scenario]}  E={Eo:.0f}/P={Po:.0f}  {No:.1f} MEUR")

    # No-deg optimum for contrast (diamond)
    En, Pn, Nn = grid_opt_all["no_deg"]
    ax.scatter([Pn], [En], marker="D", s=90, c=C_NODEG,
               edgecolors="k", linewidths=0.9, zorder=6,
               label=f"No degradation   E={En:.0f}/P={Pn:.0f}  {Nn:.1f} MEUR")

    # Refinement box
    rect = Rectangle((ZOOM_P_MIN, ZOOM_E_MIN),
                      ZOOM_P_MAX - ZOOM_P_MIN, ZOOM_E_MAX - ZOOM_E_MIN,
                      linewidth=1.4, edgecolor=TUDELFT["navy"],
                      facecolor="none", linestyle="--", zorder=8)
    ax.add_patch(rect)

    ax.set_xlabel("Power capacity  P  (MW)")
    ax.set_ylabel("Energy capacity  E  (MWh)")
    # Marker legend intentionally omitted; star/diamond/box explained in caption.

    _save(fig, f"fig_sweep_full_{scenario}", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# FIG B — ZOOM HEATMAP (refined sweep)
# ═══════════════════════════════════════════════════════════════════════════

def plot_zoom_heatmap(df: pd.DataFrame, scenario: str,
                      grid_opt_all: dict, quad: tuple,
                      out_dir: Path, levels=None) -> None:
    """
    Refined-region NPV landscape (zoom sweep). Shows grid optimum (star),
    quadratic-fit optimum (circle), and no-deg optimum (diamond).

    CAPTION: "Refined NPV landscape (E: 300–600 MWh, P: 75–200 MW) for the
    {scenario} model. Star = grid optimum; circle = quadratic-fit optimum
    (E*≈{Es:.0f} MWh, P*≈{Ps:.0f} MW, R²={r2:.3f})."
    """
    col = NPV_COL[scenario]
    m   = df.dropna(subset=[col])
    tri = mtri.Triangulation(m.p_cap.values, m.e_cap.values)

    fig, ax = plt.subplots(figsize=figsize(0.68, aspect=0.88))

    lv = 16 if levels is None else levels
    cf = ax.tricontourf(tri, m[col].values, levels=lv, cmap=CMAP_NPV)
    ax.tricontour(tri, m[col].values, levels=lv,
                  colors="k", linewidths=0.20, alpha=0.35)
    cb = fig.colorbar(cf, ax=ax, shrink=0.88)
    cb.set_label("Lifetime NPV  (MEUR)")

    ax.scatter(m.p_cap, m.e_cap, s=7, c="k", alpha=0.30,
               linewidths=0, zorder=3)

    # Grid optimum
    Eo, Po, No = grid_opt_all[scenario]
    ax.scatter([Po], [Eo], marker="*", s=340, c=C_SCENARIO[scenario],
               edgecolors="k", linewidths=1.0, zorder=6,
               label=f"Grid opt  E={Eo:.0f}/P={Po:.0f}  {No:.1f} MEUR")

    # Quadratic optimum
    Es, Ps, Ns, is_max, r2 = quad
    if is_max:
        ax.scatter([Ps], [Es], marker="o", s=110, facecolors="none",
                   edgecolors=C_SCENARIO[scenario], linewidths=1.8, zorder=7,
                   label=f"Quad opt  E={Es:.0f}/P={Ps:.0f}  R²={r2:.3f}")
        print(f"  [Fig zoom {scenario}] quad opt: E*={Es:.0f} MWh, P*={Ps:.0f} MW, "
              f"NPV*={Ns:.2f} MEUR, R²={r2:.4f}  → put in caption")

    # No-deg optimum for contrast
    En, Pn, Nn = grid_opt_all["no_deg"]
    ax.scatter([Pn], [En], marker="D", s=90, c=C_NODEG,
               edgecolors="k", linewidths=0.9, zorder=6,
               label=f"No-deg opt  E={En:.0f}/P={Pn:.0f}  {Nn:.1f} MEUR")

    ax.set_xlabel("Power capacity  P  (MW)")
    ax.set_ylabel("Energy capacity  E  (MWh)")
    # White margin around the filled region so the corner no-deg diamond is not
    # clipped by the axes frame.
    ax.set_xlim(ZOOM_P_MIN - ZOOM_PAD_P, ZOOM_P_MAX + ZOOM_PAD_P)
    ax.set_ylim(ZOOM_E_MIN - ZOOM_PAD_E, ZOOM_E_MAX + ZOOM_PAD_E)
    # Marker legend intentionally omitted; star/circle/diamond explained in caption.

    _save(fig, f"fig_sweep_zoom_{scenario}", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# FIG C — NPV vs E at fixed P (using zoom for density, coarse for range)
# ═══════════════════════════════════════════════════════════════════════════

def plot_npv_vs_E(df_zoom: pd.DataFrame, df_coarse: pd.DataFrame,
                  fixed_P: int, grid_opt_zoom: dict,
                  out_dir: Path) -> None:
    """
    NPV vs energy capacity E at fixed power P.
    Zoom provides dense sampling in the optimum region; coarse extends range.

    CAPTION: "Lifetime NPV as a function of energy capacity at P = {fixed_P} MW.
    Degradation shifts and compresses the optimum toward smaller batteries."
    """
    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.50))

    for df, lw, ls, alpha_fill in [
        (df_coarse, 1.0, (0, (4, 3)), 0.0),   # coarse: dashed background
        (df_zoom,   1.5, "solid",     0.10),   # zoom:   solid foreground
    ]:
        sl = df[df.p_cap == fixed_P].sort_values("e_cap")
        sl_nd = sl.dropna(subset=[NPV_COL["no_deg"]])
        sl_xu = sl.dropna(subset=[NPV_COL["xu"]])
        sl_sh = sl.dropna(subset=[NPV_COL["shi"]])

        if len(sl_nd):
            ax.plot(sl_nd.e_cap, sl_nd[NPV_COL["no_deg"]],
                    color=C_NODEG, lw=lw, ls=ls, marker="o", markersize=3.5)
        if len(sl_xu):
            ax.plot(sl_xu.e_cap, sl_xu[NPV_COL["xu"]],
                    color=C_XU, lw=lw, ls=ls, marker="o", markersize=3.5)
        if len(sl_sh):
            ax.plot(sl_sh.e_cap, sl_sh[NPV_COL["shi"]],
                    color=C_SHI, lw=lw, ls=ls, marker="o", markersize=3.5)

        # Degradation-gap shading (zoom only, between no-deg and Xu)
        if alpha_fill > 0 and len(sl_nd) and len(sl_xu):
            common_E = sorted(set(sl_nd.e_cap) & set(sl_xu.e_cap))
            nd_v = sl_nd.set_index("e_cap").loc[common_E, NPV_COL["no_deg"]].values
            xu_v = sl_xu.set_index("e_cap").loc[common_E, NPV_COL["xu"]].values
            ax.fill_between(common_E, xu_v, nd_v,
                            color=C_XU, alpha=alpha_fill,
                            label="Degradation cost (Xu)")

    # Vertical lines at Xu and Shi grid optima
    for sc, col_r in [("xu", C_XU), ("shi", C_SHI)]:
        Eo, Po, No = grid_opt_zoom[sc]
        if Po == fixed_P:
            ax.axvline(Eo, color=col_r, lw=0.9, ls=":", alpha=0.7)

    # Legend
    handles = [
        Line2D([0], [0], color=C_NODEG, lw=1.5, marker="o", markersize=4,
               label=f"No degradation"),
        Line2D([0], [0], color=C_XU,    lw=1.5, marker="o", markersize=4,
               label=f"Xu et al. (2016)"),
        Line2D([0], [0], color=C_SHI,   lw=1.5, marker="o", markersize=4,
               label=f"Shi et al. (2018)"),
        Patch(facecolor=C_XU, alpha=0.18, label="Degradation cost  (Xu − no-deg)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=FS_ANNOT)

    ax.set_xlabel("Energy capacity  E  (MWh)")
    ax.set_ylabel("Lifetime NPV  (MEUR)")
    ax.axhline(0, color=P["neutral"], lw=0.7, ls="--", alpha=0.5)

    _save(fig, f"fig_sweep_npv_vs_E_p{fixed_P}", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# FIG D — NPV vs P at fixed E (zoom only)
# ═══════════════════════════════════════════════════════════════════════════

def plot_npv_vs_P(df_zoom: pd.DataFrame, fixed_E: int,
                  grid_opt_zoom: dict, out_dir: Path) -> None:
    """
    NPV vs power capacity P at fixed energy E (Xu grid optimum from zoom).

    CAPTION: "Lifetime NPV as a function of power capacity at E = {fixed_E} MWh.
    Increasing P beyond the optimum reduces NPV due to rising capex."
    """
    sl = df_zoom[df_zoom.e_cap == fixed_E].sort_values("p_cap")

    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.50))

    for sc, col_r in [("no_deg", C_NODEG), ("xu", C_XU), ("shi", C_SHI)]:
        col = NPV_COL[sc]
        s = sl.dropna(subset=[col])
        if len(s):
            ax.plot(s.p_cap, s[col], color=col_r, lw=1.5, marker="o", markersize=4,
                    label=SCENARIO_LABEL[sc])

    # Mark Xu grid optimum
    Eo, Po, No = grid_opt_zoom["xu"]
    if Eo == fixed_E:
        ax.axvline(Po, color=C_XU, lw=0.9, ls=":", alpha=0.7)
        ax.text(Po + 1, ax.get_ylim()[0] if ax.get_ylim()[0] != 0.0 else -5,
                f"P*={Po:.0f} MW", fontsize=FS_ANNOT, color=C_XU)

    ax.axhline(0, color=P["neutral"], lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Power capacity  P  (MW)")
    ax.set_ylabel("Lifetime NPV  (MEUR)")
    ax.legend(frameon=False, fontsize=FS_ANNOT)

    _save(fig, f"fig_sweep_npv_vs_P_e{fixed_E}", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    HERE        = Path(__file__).parent
    RESULTS_DIR = HERE / "Plan B Results"
    OUT_DIR     = HERE / "Thesis Figures"

    print("\nLoading sweep data...")
    df_coarse, _ = _load_csv(RESULTS_DIR, COARSE_TAG)
    df_zoom,   _ = _load_csv(RESULTS_DIR, ZOOM_TAG)

    print("\nGrid optima:")
    grid_coarse = {}
    grid_zoom   = {}
    for sc in ("no_deg", "xu", "shi"):
        grid_coarse[sc] = grid_optimum(df_coarse, sc)
        grid_zoom[sc]   = grid_optimum(df_zoom,   sc)
        Ec, Pc, Nc = grid_coarse[sc]
        Ez, Pz, Nz = grid_zoom[sc]
        print(f"  {SCENARIO_LABEL[sc]:<22}  "
              f"coarse: E={Ec:.0f}/P={Pc:.0f} {Nc:.2f} MEUR  |  "
              f"zoom: E={Ez:.0f}/P={Pz:.0f} {Nz:.2f} MEUR")

    print("\nQuadratic fit (zoom, full domain within zoom box):")
    quad = {}
    for sc in ("xu", "shi"):
        Es, Ps, Ns, is_max, r2 = fit_quadratic(
            df_zoom, sc, QUAD_E_RANGE, QUAD_P_RANGE)
        quad[sc] = (Es, Ps, Ns, is_max, r2)
        tag = "maximum ✓" if is_max else "NOT a max"
        print(f"  {SCENARIO_LABEL[sc]:<22}  E*={Es:.0f}  P*={Ps:.0f}  "
              f"NPV*={Ns:.2f} MEUR  R²={r2:.4f}  [{tag}]  → put in caption")

    print(f"\nProducing figures in: {OUT_DIR}")
    print("─" * 60)

    # Fig A — Full-domain heatmaps (coarse)
    plot_full_heatmap(df_coarse, SCENARIO,    grid_coarse, OUT_DIR)
    # Also produce the complementary degraded model and no-deg baseline
    other_sc = "shi" if SCENARIO == "xu" else "xu"
    plot_full_heatmap(df_coarse, other_sc,    grid_coarse, OUT_DIR)
    plot_full_heatmap(df_coarse, "no_deg",    grid_coarse, OUT_DIR)

    # Fig B — Zoom heatmaps (refined). Shared color levels across Xu and Shi so
    # the two panels are directly comparable (same colorbar).
    zoom_vals = np.concatenate([
        df_zoom.dropna(subset=[NPV_COL["xu"]])[NPV_COL["xu"]].values,
        df_zoom.dropna(subset=[NPV_COL["shi"]])[NPV_COL["shi"]].values,
    ])
    lo, hi = float(zoom_vals.min()), float(zoom_vals.max())
    pad = 0.02 * (hi - lo)
    zoom_levels = np.linspace(lo - pad, hi + pad, 17)
    plot_zoom_heatmap(df_zoom, SCENARIO, grid_zoom, quad[SCENARIO], OUT_DIR,
                      levels=zoom_levels)
    plot_zoom_heatmap(df_zoom, other_sc, grid_zoom, quad[other_sc], OUT_DIR,
                      levels=zoom_levels)

    # Fig C — NPV vs E (zoom + coarse for range)
    plot_npv_vs_E(df_zoom, df_coarse, FIXED_P_MW, grid_zoom, OUT_DIR)

    # Fig D — NPV vs P (zoom only)
    plot_npv_vs_P(df_zoom, FIXED_E_MWH, grid_zoom, OUT_DIR)

    print("─" * 60)
    print("Done.  Caption values printed above — copy into .tex.")


if __name__ == "__main__":
    main()