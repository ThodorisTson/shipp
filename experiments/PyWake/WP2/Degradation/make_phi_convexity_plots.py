"""
make_phi_convexity_plots.py
===========================
Generates three thesis-quality figures for the weekly report on
dual-Phi convexity validation.

    Figure 1 — phi_extrapolation.png
        Two-panel: Phi_shi vs Xu S_delta across full DoD range (top),
        second derivatives showing convexity sign (bottom).
        Replicates the HTML widget as a proper matplotlib figure.

    Figure 2 — option_b_sigma.png
        Two-panel: S_sigma distribution across dispatch cycles (left),
        per-cycle gradient error vs cycle mean SoC (right).
        Documents the Option B zero-cost formal claim.

    Figure 3 — option_c_convexity.png
        Three-panel: effective gradient scatter (left), S_sigma vs DoD
        correlation (middle), binned monotonicity check (right).
        Pure-Phi bars overlaid to show violations are S_sigma scatter.

Run from the project root (same directory as degradation_xu.py):
    python make_phi_convexity_plots.py

Outputs written to ./Figures_PhiConvexity/ (created if absent).
Requires: numpy, matplotlib, scipy.ndimage
Optional: rainflow (for real cycles from storage_e_fixed.npy)
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
DARK = False   # set True for dark-background slides

BG   = "#1a1a1a" if DARK else "white"
FG   = "#e0e0e0" if DARK else "#2c2c2c"
GRID = "#333333" if DARK else "#e8e8e8"

C_SHI  = "#2878BD"   # blue  — Phi_shi gradient function
C_XU   = "#C94C2A"   # red   — Xu S_delta reporting function
C_AMB  = "#BA7517"   # amber — non-convex region
C_BLU  = "#4A90D9"   # light blue — extrapolated convex region
C_GRN  = "#3B6D11"   # green — fitted range
C_VIOL = "#C94C2A"   # violation bars
C_OK   = "#3B8B3B"   # convex bars

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": FG, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.grid": True,
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "normal",
    "savefig.dpi": 180, "savefig.bbox": "tight",
    "savefig.facecolor": BG,
})

OUT = Path(__file__).parent / "Figures_PhiConvexity"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Xu / Shi model parameters
# ---------------------------------------------------------------------------
K1, K2_EXP, K3C = 1.40e5, -0.501, -1.23e5
K_SIGMA, SIGMA_REF = 1.04, 0.50
K3_SHI, K4_SHI = 3.2418e-5, 1.1785
BOUNDARY, FIT_FLOOR = 0.1437, 0.15


def s_dod(d):
    d = np.clip(np.asarray(d, float), 1e-6, 1.0)
    D = K1 * d**K2_EXP + K3C
    return 1.0 / D

def phi_shi(d):
    d = np.clip(np.asarray(d, float), 1e-9, 1.0)
    return K3_SHI * d**K4_SHI

def phi_shi_prime(d):
    d = np.clip(np.asarray(d, float), 1e-9, 1.0)
    return K3_SHI * K4_SHI * d**(K4_SHI - 1)

def phi_shi_d2(d):
    d = np.clip(np.asarray(d, float), 1e-9, 1.0)
    return K3_SHI * K4_SHI * (K4_SHI - 1) * d**(K4_SHI - 2)

def s_dod_d2(d):
    d = np.clip(np.asarray(d, float), 1e-6, 1.0)
    D   = K1 * d**K2_EXP + K3C
    Dp  = K1 * K2_EXP * d**(K2_EXP - 1)
    Dpp = K1 * K2_EXP * (K2_EXP - 1) * d**(K2_EXP - 2)
    return 2 * Dp**2 / D**3 - Dpp / D**2

def s_soc(sigma):
    return np.exp(K_SIGMA * (np.asarray(sigma, float) - SIGMA_REF))

def phi_shi_prime_with_stress(d, sigma):
    return phi_shi_prime(d) * s_soc(sigma)


# ---------------------------------------------------------------------------
# Cycle loading helpers (mirrors check_phi_convexity.py priority logic)
# ---------------------------------------------------------------------------

def _rainflow_minimal(storage_e: np.ndarray, e_cap: float) -> List[Dict]:
    """4-point ASTM E1049 rainflow counter — no external package needed."""
    soc = np.asarray(storage_e, float) / e_cap
    n   = len(soc)
    d   = np.diff(soc)
    tp  = [0]
    for i in range(1, n - 1):
        if d[i-1] * d[i] <= 0 and abs(d[i-1]) + abs(d[i]) > 1e-9:
            tp.append(i)
    tp.append(n - 1)
    tp = np.array(tp, int)
    tv = soc[tp]

    cycles: List[Dict] = []
    sv, si = [], []

    def emit(v1, v2, i1, i2, count):
        dod  = abs(v2 - v1)
        mean = 0.5 * (v1 + v2)
        cycles.append({"dod": float(np.clip(dod, 1e-6, 1.0)),
                        "soc_mean": float(np.clip(mean, 0, 1)),
                        "count": float(count),
                        "i_start": int(tp[i1]), "i_end": int(tp[i2])})

    for k, (v, orig) in enumerate(zip(tv, range(len(tv)))):
        sv.append(v); si.append(orig)
        while len(sv) >= 4:
            s0,s1,s2,s3 = sv[-4],sv[-3],sv[-2],sv[-1]
            i0,i1,i2,i3 = si[-4],si[-3],si[-2],si[-1]
            if abs(s2-s1) <= abs(s3-s0):
                emit(s1,s2,i1,i2,1.0); del sv[-3:-1]; del si[-3:-1]
            else:
                break
    for j in range(len(sv)-1):
        emit(sv[j], sv[j+1], si[j], si[j+1], 0.5)

    return [c for c in cycles if c["dod"] >= 0.005]


def load_cycles() -> tuple[np.ndarray, float, List[Dict], str]:
    """Return (storage_e, e_cap, cycles, source_label)."""
    npy = Path(__file__).parent / "Results" / "storage_e_fixed.npy"
    cap = Path(__file__).parent / "Results" / "e_cap_fixed.npy"
    if npy.exists() and cap.exists():
        storage_e = np.load(npy)
        e_cap     = float(np.load(cap).flat[0])
        source    = "SHIPP LP 2022"
    else:
        print("  [WARN] Results/storage_e_fixed.npy not found — generating synthetic cycles.")
        rng = np.random.default_rng(42)
        e_cap = 300.0
        dods   = np.clip(np.exp(rng.normal(-1.4, 0.85, 500)), 0.02, 0.80)
        sigmas = np.array([rng.uniform(0.10 + d/2, 0.90 - d/2) for d in dods])
        cycles = [{"dod": float(d), "soc_mean": float(s), "count": 0.5,
                   "i_start": 0, "i_end": 1}
                  for d, s in zip(dods, sigmas)]
        return np.zeros(100), e_cap, cycles, "synthetic"

    try:
        import rainflow as _rf
        cycles = []
        for rng_, mean, count, i0, i1 in _rf.extract_cycles(storage_e):
            cycles.append({"dod": float(rng_)/e_cap, "soc_mean": float(mean)/e_cap,
                            "count": float(count), "i_start": int(i0), "i_end": int(i1)})
        source += " (rainflow lib)"
    except ImportError:
        cycles = _rainflow_minimal(storage_e, e_cap)
        source += " (minimal rainflow)"

    return storage_e, e_cap, cycles, source


# ===========================================================================
# Figure 1 — Phi_shi extrapolation diagnostic
# ===========================================================================

def fig1_extrapolation():
    d_all  = np.linspace(0.02, 0.80, 500)
    d_fit  = d_all[d_all >= FIT_FLOOR]
    d_xtra = d_all[d_all < FIT_FLOOR]      # below fit floor
    d_nc   = d_all[d_all < BOUNDARY]       # Xu non-convex region
    d_cnv  = d_all[(d_all >= BOUNDARY) & (d_all < FIT_FLOOR)]  # convex extrap

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6),
                                    gridspec_kw={"height_ratios": [1.1, 1]})
    fig.suptitle("Φ_shi extrapolation below fit floor  δ = 0.15", fontsize=12)

    # ── Region shading ──────────────────────────────────────────────────────
    for ax in (ax1, ax2):
        ax.axvspan(0.02,  BOUNDARY,  alpha=0.10, color=C_AMB, label=None)
        ax.axvspan(BOUNDARY, FIT_FLOOR, alpha=0.07, color=C_BLU, label=None)
        ax.axvspan(FIT_FLOOR, 0.80,     alpha=0.06, color=C_GRN, label=None)
        ax.axvline(BOUNDARY,  color=C_AMB, lw=1.2, ls="--", alpha=0.8)
        ax.axvline(FIT_FLOOR, color=FG,    lw=1.0, ls="--", alpha=0.5)

    # ── Panel 1: function values ─────────────────────────────────────────────
    ax1.plot(d_all, s_dod(d_all)*1e5,    color=C_XU,  lw=2.0, label=r"Xu $S_\delta(\delta)$ — reporting")
    ax1.plot(d_all, phi_shi(d_all)*1e5,  color=C_SHI, lw=2.0, ls="--", label=r"$\Phi_{shi}(\delta)=k_3\cdot\delta^{k_4}$ — gradient")
    ax1.set_ylabel(r"$\Phi(\delta) \times 10^5$")
    ax1.set_xlim(0.02, 0.80)
    ax1.legend(fontsize=9, framealpha=0.3)

    # Region labels
    ax1.text(0.07, ax1.get_ylim()[1]*0.88 if ax1.get_ylim()[1]>0 else 2.8,
             "Xu\nnon-convex", ha="center", va="top", fontsize=8,
             color=C_AMB, style="italic")
    ax1.text(0.143, ax1.get_ylim()[1]*0.55 if ax1.get_ylim()[1]>0 else 1.8,
             "extrap.", ha="center", va="top", fontsize=8,
             color=C_BLU, style="italic")
    ax1.text(0.47, 0.3, "fitted [0.15, 0.80]  R²=0.963",
             ha="center", va="bottom", fontsize=8, color=C_GRN)

    # ── Panel 2: second derivatives ──────────────────────────────────────────
    xu_d2  = np.clip(s_dod_d2(d_all),   -3e-5, 1e-4)
    shi_d2 = np.clip(phi_shi_d2(d_all), -3e-5, 1e-4)

    ax2.plot(d_all, xu_d2*1e5,  color=C_XU,  lw=2.0, label=r"$S_\delta''(\delta)$")
    ax2.plot(d_all, shi_d2*1e5, color=C_SHI, lw=2.0, ls="--",
             label=r"$\Phi_{shi}''(\delta)$")
    ax2.axhline(0, color=FG, lw=1.4, alpha=0.7, zorder=3)
    ax2.set_ylabel(r"$\Phi''(\delta) \times 10^5$")
    ax2.set_xlabel(r"Cycle depth of discharge  $\delta$")
    ax2.set_xlim(0.02, 0.80)
    ax2.set_ylim(-2.5, 12)
    ax2.legend(fontsize=9, framealpha=0.3)

    # Annotations on panel 2
    ax2.text(0.07,  -1.8,  r"$S_\delta'' < 0$",  ha="center", color=C_XU,  fontsize=9)
    ax2.text(0.47, 10.5, r"$\Phi_{shi}'' > 0$ everywhere  ✓",
             ha="center", color=C_SHI, fontsize=9)
    ax2.annotate("sign change\nδ ≈ 0.144",
                 xy=(BOUNDARY, 0), xytext=(0.22, 5),
                 arrowprops=dict(arrowstyle="->", color=C_AMB, lw=1.2),
                 fontsize=8, color=C_AMB, ha="center")

    fig.tight_layout()
    path = OUT / "phi_extrapolation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ===========================================================================
# Figure 2 — Option B: S_sigma analysis
# ===========================================================================

def fig2_option_b(cycles: List[Dict], source: str):
    dods   = np.array([c["dod"]      for c in cycles])
    sigmas = np.array([c["soc_mean"] for c in cycles])
    counts = np.array([c["count"]    for c in cycles])
    s_vals = s_soc(sigmas)

    sigma_mid = 0.50  # window midpoint = sigma_ref → S_sigma = 1.0
    phi_live  = phi_shi_prime(dods) * s_vals
    phi_fixed = phi_shi_prime(dods)  # k3_eff = k3 since S_sigma(sigma_ref)=1
    rel_err   = (phi_fixed - phi_live) / np.maximum(phi_live, 1e-30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(f"Option B — fold $S_\\sigma$ into $k_{{3,eff}}$ at $\\bar{{\\sigma}}$  ({source})",
                 fontsize=11)

    # ── Left: S_sigma distribution ────────────────────────────────────────────
    ax1.hist(s_vals, bins=30, color=C_SHI, alpha=0.75, edgecolor="none",
             weights=counts, density=True)
    ax1.axvline(1.0, color=C_XU, lw=1.8, ls="--",
                label=r"$S_\sigma(\bar{\sigma}) = 1.0$  (σ̄ = σ_ref)")
    sigma_wt = float(np.average(sigmas, weights=dods * counts))
    ax1.axvline(s_soc(sigma_wt), color=C_AMB, lw=1.5, ls=":",
                label=f"$S_\\sigma(\\bar{{\\sigma}}_{{wt}})$ = {s_soc(sigma_wt):.3f}  "
                      f"(σ̄_wt = {sigma_wt:.3f})")
    ax1.set_xlabel(r"$S_\sigma(\sigma_{cycle})$")
    ax1.set_ylabel("Density (cycle-count weighted)")
    ax1.set_title(r"$S_\sigma$ distribution across dispatch cycles")
    ax1.legend(fontsize=8, framealpha=0.3)

    # Span annotation
    ax1.annotate("", xy=(s_vals.max(), 0.3), xytext=(s_vals.min(), 0.3),
                 arrowprops=dict(arrowstyle="<->", color=FG, lw=1.2))
    ax1.text(s_vals.mean(), 0.35,
             f"span = {s_vals.max()-s_vals.min():.2f}",
             ha="center", fontsize=8, color=FG)

    # ── Right: per-cycle gradient error ───────────────────────────────────────
    sc = ax2.scatter(sigmas, rel_err * 100, c=dods, cmap="viridis",
                     s=8, alpha=0.55, linewidths=0)
    ax2.axhline(0, color=FG, lw=1.4, ls="-", alpha=0.6, zorder=3,
                label="zero error")
    ax2.axhline(np.mean(rel_err)*100, color=C_XU, lw=1.4, ls="--",
                label=f"mean = {np.mean(rel_err)*100:+.1f}%")
    ax2.axvline(SIGMA_REF, color=C_SHI, lw=1.2, ls=":",
                label=r"σ_ref = 0.50")
    cb = fig.colorbar(sc, ax=ax2, pad=0.02)
    cb.set_label("Cycle DoD δ", fontsize=9)
    ax2.set_xlabel(r"Cycle mean SoC  $\sigma_{cycle}$")
    ax2.set_ylabel("Gradient error  (Φ′_fixed − Φ′_live) / Φ′_live  [%]")
    ax2.set_title("Per-cycle error when fixing σ̄ = σ_ref")
    ax2.legend(fontsize=8, framealpha=0.3, loc="upper left")

    # Text box
    textstr = (f"Max |err| = {np.max(np.abs(rel_err))*100:.0f}%\n"
               f"Mean err  = {np.mean(rel_err)*100:+.1f}%\n"
               f"k3_eff    = k3  (unchanged)")
    ax2.text(0.97, 0.03, textstr, transform=ax2.transAxes,
             fontsize=8, va="bottom", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=BG,
                       edgecolor=GRID, alpha=0.8))

    fig.tight_layout()
    path = OUT / "option_b_sigma.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ===========================================================================
# Figure 3 — Option C: empirical convexity check
# ===========================================================================

def fig3_option_c(cycles: List[Dict], source: str):
    dods   = np.array([c["dod"]      for c in cycles])
    sigmas = np.array([c["soc_mean"] for c in cycles])
    counts = np.array([c["count"]    for c in cycles])

    sort_idx = np.argsort(dods)
    dods_s   = dods[sort_idx]
    sigmas_s = sigmas[sort_idx]
    counts_s = counts[sort_idx]
    s_vals_s = s_soc(sigmas_s)

    phi_eff  = phi_shi_prime(dods_s) * s_vals_s   # composite with S_sigma
    phi_pure = phi_shi_prime(dods_s)               # Phi_shi only

    corr = float(np.corrcoef(dods_s, s_vals_s)[0, 1])

    # Binned monotonicity
    n_bins = min(15, len(cycles) // 5)
    edges  = np.linspace(dods_s.min(), dods_s.max(), n_bins + 1)
    mids, eff_means, pure_means = [], [], []
    for j in range(n_bins):
        m = (dods_s >= edges[j]) & (dods_s < edges[j+1])
        if m.sum() < 2: continue
        mids.append(0.5 * (edges[j] + edges[j+1]))
        eff_means.append(np.average(phi_eff[m],  weights=counts_s[m]))
        pure_means.append(np.average(phi_pure[m], weights=counts_s[m]))

    mids       = np.array(mids)
    eff_means  = np.array(eff_means)
    pure_means = np.array(pure_means)

    d_eff  = np.gradient(eff_means,  mids)
    d_pure = np.gradient(pure_means, mids)
    n_viol      = int(np.sum(d_eff  < 0))
    n_viol_pure = int(np.sum(d_pure < 0))

    # Theoretical envelope
    d_dense   = np.linspace(dods_s.min(), dods_s.max(), 400)
    phi_ref   = phi_shi_prime(d_dense)
    phi_hi    = phi_ref * float(s_soc(0.90))
    phi_lo    = phi_ref * float(s_soc(0.10))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(f"Option C — empirical convexity verification  ({source},  {len(cycles)} cycles)",
                 fontsize=11)

    # ── Panel 1: effective gradient scatter ──────────────────────────────────
    sc = ax1.scatter(dods_s, phi_eff*1e5, c=sigmas_s, cmap="RdYlBu",
                     s=8 + 25*(counts_s/counts_s.max()),
                     alpha=0.55, linewidths=0, vmin=0.10, vmax=0.90)
    ax1.plot(d_dense, phi_ref*1e5, "k-", lw=1.8,
             label=r"$\Phi'_{shi}$ at $\sigma_{ref}$")
    ax1.fill_between(d_dense, phi_lo*1e5, phi_hi*1e5,
                     alpha=0.12, color=C_SHI,
                     label=r"$S_\sigma$ envelope [10%, 90%]")
    cb = fig.colorbar(sc, ax=ax1, pad=0.02)
    cb.set_label("σ (mean SoC)", fontsize=8)
    ax1.set_xlabel(r"Cycle DoD $\delta$")
    ax1.set_ylabel(r"$\Phi'_{shi}(\delta)\cdot S_\sigma \times 10^5$")
    ax1.set_title("Effective gradient per cycle\n(colour = σ, size ∝ count)")
    ax1.legend(fontsize=8, framealpha=0.3)

    # ── Panel 2: S_sigma vs DoD ───────────────────────────────────────────────
    ax2.scatter(dods_s, s_vals_s, s=7, alpha=0.45, color=C_SHI,
                linewidths=0)
    ax2.axhline(1.0, color=C_XU, lw=1.4, ls="--",
                label=r"$S_\sigma = 1$  at σ_ref = 0.50")
    if len(dods_s) > 2:
        m, b = np.polyfit(dods_s, s_vals_s, 1)
        ax2.plot(d_dense, m*d_dense + b, color=C_XU, lw=1.4, alpha=0.7,
                 label=f"linear fit  r = {corr:+.3f}")
    ax2.set_xlabel(r"Cycle DoD $\delta$")
    ax2.set_ylabel(r"$S_\sigma(\sigma_{cycle})$")
    ax2.set_title(r"SoC stress vs cycle depth")
    ax2.legend(fontsize=8, framealpha=0.3)
    ax2.annotate(f"r = {corr:+.3f}\n{'weak — no systematic bias' if abs(corr)<0.3 else 'moderate correlation'}",
                 xy=(0.05, 0.90), xycoords="axes fraction", fontsize=8,
                 color=C_XU if abs(corr)>0.3 else C_GRN)

    # ── Panel 3: dPhi'/ddelta sign check, composite + pure side by side ──────
    bar_w = (mids[1] - mids[0]) * 0.38 if len(mids) > 1 else 0.03
    colors_eff = [C_VIOL if v < 0 else C_OK for v in d_eff]
    ax3.bar(mids - bar_w*0.55, d_eff*1e5,  width=bar_w, color=colors_eff,
            alpha=0.85, label=r"$\Delta\Phi'_{eff}/\Delta\delta$  (with $S_\sigma$)")
    ax3.bar(mids + bar_w*0.55, d_pure*1e5, width=bar_w, color=C_SHI,
            alpha=0.50, label=r"$\Delta\Phi'_{shi}/\Delta\delta$  (pure, $S_\sigma$=1)")
    ax3.axhline(0, color=FG, lw=1.4, alpha=0.7)
    ax3.set_xlabel(r"Cycle DoD $\delta$ (bin centre)")
    ax3.set_ylabel(r"$\Delta\Phi' / \Delta\delta \times 10^5$ (binned)")

    verdict = (f"Φ_shi: 0 violations  ✓\n"
               f"Composite: {n_viol}/{len(mids)} bins ← S_σ scatter")
    title_c = C_GRN if n_viol_pure == 0 else C_VIOL
    ax3.set_title(f"Monotonicity — {n_viol}/{len(mids)} apparent violations\n"
                  f"(pure Φ_shi: {n_viol_pure} violations)", color=title_c)
    ax3.legend(fontsize=8, framealpha=0.3)

    # Annotation box
    ax3.text(0.97, 0.97, verdict, transform=ax3.transAxes,
             fontsize=8, va="top", ha="right", color=FG,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=BG,
                       edgecolor=GRID, alpha=0.85))

    fig.tight_layout()
    path = OUT / "option_c_convexity.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("make_phi_convexity_plots.py")
    print("=" * 60)
    print(f"  Output dir : {OUT.resolve()}")
    print(f"  Dark mode  : {DARK}")

    print("\n[1/3] Phi_shi extrapolation diagnostic...")
    fig1_extrapolation()

    print("\n[2/3] Loading cycles for Option B / C...")
    storage_e, e_cap, cycles, source = load_cycles()
    print(f"      {len(cycles)} cycles  ({source})")

    print("\n[3/3] Option B — S_sigma analysis...")
    fig2_option_b(cycles, source)

    print("\n[4/4] Option C — empirical convexity check...")
    fig3_option_c(cycles, source)

    print(f"\nDone. Three figures in {OUT}/")
    print("  phi_extrapolation.png  — drop into 'Phi architecture' slide")
    print("  option_b_sigma.png     — drop into 'Option B' slide")
    print("  option_c_convexity.png — drop into 'Option C' slide")
