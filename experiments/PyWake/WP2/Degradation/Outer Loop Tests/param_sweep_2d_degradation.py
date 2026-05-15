"""
param_sweep_2d_degradation.py
==============================
2D parameter sweep over nominal battery energy capacity (e_cap_nom) and power
capacity (p_cap_nom). For each (e_cap_nom, p_cap_nom) design point, a 20-year
multi-year simulation loop is run:

  Year k:
    1. Effective capacity E_k = e_cap_nom * SoH_{k-1}   (p_cap fixed, from thesis)
    2. Inner LP (SHIPP, fixed_cap=True) -> dispatch schedule -> annual revenue
    3. Xu degradation model (post-processing) -> fd_k (cycle + calendar)
    4. Cumulative fd updated, SoH updated via SEI fade curve
    5. If SoH < 0.70 -> replacement triggered, fd_cum reset, cost recorded

IMPORTANT — what this sweep IS and IS NOT:
  - IS:  degradation-informed design selection. Capital + replacement costs enter NPV.
         Dispatch adapts to degraded capacity each year.
  - IS NOT: degradation-aware dispatch. The inner LP maximises revenue only.
            No degradation penalty in the objective. Degradation is post-processing.

This is a deliberate Plan B deliverable. It gives a 2D landscape of lifetime NPV
vs (E, P) design before Path 3 NLP (degradation-aware dispatch) is complete.

Imports required from your existing codebase:
  - shipp: solve_lp_pyomo, Storage, Production, TimeSeries
  - degradation_xu: compute_fd_annual_xu  (see wrapper below if name differs)

Usage:
  python param_sweep_2d_degradation.py --year 2022 --grid 10
  python param_sweep_2d_degradation.py --year 2019 --grid 12 --e_range 150 450 --p_range 75 225

Author: Thodoris Tsonopoulos  |  TU Delft  |  2025
"""

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy_financial as npf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Path setup ────────────────────────────────────────────────────────────────
# Adjust this to point to your SHIPP src/ and your degradation scripts
SCRIPT_DIR   = Path(__file__).parent
SHIPP_SRC    = SCRIPT_DIR.parent / "src"          # adjust if shipp is installed
DEG_SRC      = SCRIPT_DIR.parent                  # folder containing degradation_xu.py
DATA_DIR     = SCRIPT_DIR.parent / "Results"
OUTPUT_DIR   = SCRIPT_DIR / "Results_Sweep2D"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for p in [str(SHIPP_SRC), str(DEG_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from shipp.kernel_pyomo import solve_lp_pyomo
from shipp.classes import Storage, Production, TimeSeries

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — XU DEGRADATION (direct calls to validated degradation_xu.py API)
# ═══════════════════════════════════════════════════════════════════════════════
#
# The previous version tried to import a non-existent function 'compute_fd_annual'
# and fell back to a self-contained rainflow implementation using scipy.signal.
# Both were wrong.  The correct approach calls three functions that exist in the
# validated degradation_xu.py:
#
#   rainflow_cycle_counting(storage_e, e_cap)         → List[Dict]
#   compute_fd(cycles, sigma_mean, t_total_seconds)   → (fd, fd_cycle, fd_cal)
#   sei_capacity_loss(fd_cum)                          → L (capacity LOSS fraction)
#
# No fallback is provided.  The rainflow library is already a project dependency
# (imported at the top of degradation_xu.py as `import rainflow`).
# ═══════════════════════════════════════════════════════════════════════════════

from degradation_xu import (
    rainflow_cycle_counting,   # ASTM E1049 via the rainflow library, returns List[Dict]
    compute_fd,                # accumulates fd from cycle list + calendar term
    sei_capacity_loss,         # SEI two-exponential fade curve → capacity LOSS L
)


def xu_soh_from_fd_cum(fd_cum: float) -> float:
    """
    Capacity retention (SoH) from cumulative fd.
    SoH = 1 - L, where L = sei_capacity_loss(fd_cum).
    Replacement triggers when SoH < repl_threshold (e.g. 0.70).
    """
    return 1.0 - sei_capacity_loss(fd_cum)


def compute_fd_annual(
    e_t:      np.ndarray,
    e_cap:    float,
    dt_hours: float = 1.0,
    T_C:      float = 25.0,
) -> tuple[float, float]:
    """
    Compute (fd_cycle, fd_calendar) for one year from a SoC energy trajectory.

    Parameters
    ----------
    e_t      : Full SoC energy time series [MWh], length n+1 (initial state + n
               timestep-end states, as returned by os.storage_e[0].data).
               Do NOT truncate to length n — the terminal state closes the final
               cycle pair.  Matches the pattern in _run_multiyear v4 line 570.
    e_cap    : Effective energy capacity this year [MWh].
    dt_hours : Timestep [h], default 1.0.
    T_C      : Cell temperature [°C], default 25 (S_T = 1, drops out).

    Returns
    -------
    fd_cycle    : Cycle aging contribution (Xu rainflow model, Eq. 2.9).
    fd_calendar : Calendar aging contribution (Xu model, Eq. 2.15).

    Notes
    -----
    - rainflow_cycle_counting() operates on the raw MWh array and normalises
      to DoD/SoC internally.  Do not pre-normalise.
    - compute_fd() returns (fd_total, fd_cycle, fd_calendar).
    - Temperature is constant at 25°C throughout (S_T = 1 everywhere).
    - t_total_seconds = len(e_t) * dt * 3600 matches the convention in
      analyze_degradation() (len of the full n+1 array, consistent with v4).
    """
    e_t = np.asarray(e_t, dtype=float)

    # Cycle counting — library operates on MWh directly, normalises internally
    cycles = rainflow_cycle_counting(storage_e=e_t.tolist(), e_cap=e_cap)

    # Calendar aging inputs — match analyze_degradation() convention exactly
    sigma_mean      = float(np.mean(e_t)) / max(e_cap, 1e-9)
    t_total_seconds = float(len(e_t)) * dt_hours * 3600.0

    # compute_fd returns (fd_total, fd_cycle, fd_calendar)
    _, fd_cycle, fd_cal = compute_fd(
        cycles          = cycles,
        sigma_mean      = sigma_mean,
        t_total_seconds = t_total_seconds,
        T_C             = T_C,
    )

    return float(fd_cycle), float(fd_cal)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SINGLE DESIGN POINT: 20-YEAR MULTI-YEAR LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_20year_loop(
    e_cap_nom:     float,
    p_cap_nom:     float,
    price_data:    np.ndarray,
    wind_data:     np.ndarray,
    n_years:       int   = 20,
    discount_rate: float = 0.05,
    repl_threshold:float = 0.70,
    lambda_E:      float = 150_000.0,  # EUR/MWh  battery energy capex
    lambda_P:      float = 100_000.0,  # EUR/MW   battery power capex
    p_min:         float = 0.0,
    p_max:         float = 200.0,
    dt:            float = 1.0,        # hours
    soc_min_frac:  float = 0.10,       # lower SoC window (10%)
    soc_max_frac:  float = 0.90,       # upper SoC window (90%)
    eta_in:        float = 0.95,
    eta_out:       float = 0.95,
    pyo_solver:    str   = "gurobi",
) -> dict:
    """
    Run the 20-year degradation-informed simulation for one (e_cap_nom, p_cap_nom).

    The inner LP is a pure revenue-maximising dispatch (no degradation penalty).
    Degradation is computed post-hoc using the Xu model.  Battery is replaced
    when SoH < repl_threshold.

    Returns
    -------
    dict with keys:
        npv_lifetime      : lifetime NPV [EUR] = PV(revenues) - capex - PV(replacements)
        n_replacements    : number of battery replacements over 20 years
        soh_trajectory    : list of SoH at end of each year (len = n_years)
        fd_annual_list    : list of total fd per year
        revenue_annual    : list of annual (undiscounted) revenue [EUR]
        first_eol_year    : year of first replacement (None if none)
    """
    n         = len(price_data)            # 8760 for full year
    price_ts  = TimeSeries(price_data, dt)
    wind_ts   = TimeSeries(wind_data, dt)
    prod_null = Production(TimeSeries(np.zeros(n), dt), p_cost=0.0)

    # Initial capex at t=0 (no discounting — paid at start of project)
    initial_capex = lambda_E * e_cap_nom + lambda_P * p_cap_nom

    # State variables
    SoH         = 1.0
    fd_cum      = 0.0
    total_npv   = -initial_capex
    n_repl      = 0
    first_eol   = None

    soh_traj      = []
    fd_annual_all = []
    rev_annual    = []

    for k in range(1, n_years + 1):

        # ── Effective capacity this year ──────────────────────────────────────
        # Energy capacity scales with SoH; power capacity is held fixed (thesis §2.8).
        e_cap_eff = e_cap_nom * SoH
        e_min_eff = e_cap_eff * soc_min_frac   # absolute SoC lower bound

        # ── Set up Storage object for this year ───────────────────────────────
        # e_cost=0, p_cost=0: capital costs handled manually outside the LP.
        stor_batt = Storage(
            p_cap    = p_cap_nom,
            e_cap    = e_cap_eff,
            p_cost   = 0.0,
            e_cost   = 0.0,
            eff_in   = eta_in,
            eff_out  = eta_out,
        )
        stor_null = Storage(
            p_cap    = 0.0,
            e_cap    = 0.0,
            p_cost   = 0.0,
            e_cost   = 0.0,
            eff_in   = 1.0,
            eff_out  = 1.0,
        )
        prod_wind = Production(wind_ts, p_cost=0.0)

        # ── Solve inner LP (pure revenue maximisation, fixed capacity) ────────
        # n_year=1 here: annuity factor doesn't change optimal dispatch for
        # fixed_cap=True.  We use os.revenue (raw annual EUR) for NPV manually.
        try:
            os = solve_lp_pyomo(
                price_ts, prod_wind, prod_null, stor_batt, stor_null,
                discount_rate=discount_rate,
                n_year=1,
                p_min=p_min,
                p_max=p_max,
                n=n,
                name_solver=pyo_solver,
                fixed_cap=True,
            )
        except RuntimeError:
            # LP infeasible for this capacity — mark and break
            warnings.warn(
                f"LP infeasible at year {k}, e_cap_eff={e_cap_eff:.1f} MWh. "
                "Stopping loop for this design point."
            )
            break

        # ── Annual revenue (undiscounted) ─────────────────────────────────────
        annual_rev = os.revenue   # EUR, raw annual from SHIPP
        discount_f = (1.0 + discount_rate) ** (-k)
        total_npv += annual_rev * discount_f
        rev_annual.append(annual_rev)

        # ── Xu degradation (post-processing) ──────────────────────────────────
        e_t       = np.array(os.storage_e[0].data)   # SoC energy [MWh], length n+1
        # Full n+1 array — terminal state needed to close final cycle pair

        fd_cycle, fd_cal = compute_fd_annual(e_t, e_cap_eff, dt)
        fd_k     = fd_cycle + fd_cal
        fd_cum  += fd_k
        fd_annual_all.append(fd_k)

        # ── SoH update ────────────────────────────────────────────────────────
        SoH = xu_soh_from_fd_cum(fd_cum)
        soh_traj.append(SoH)

        # ── Replacement check ─────────────────────────────────────────────────
        if SoH < repl_threshold:
            if first_eol is None:
                first_eol = k
            # Replace with fresh battery at nominal capacity.
            # Cost discounted at the replacement year.
            repl_cost  = lambda_E * e_cap_nom   # energy capex only (no power upgrade)
            total_npv -= repl_cost * discount_f
            n_repl    += 1
            # Reset degradation state
            fd_cum = 0.0
            SoH    = 1.0

    return {
        "npv_lifetime":   total_npv,
        "n_replacements": n_repl,
        "soh_trajectory": soh_traj,
        "fd_annual_list": fd_annual_all,
        "revenue_annual": rev_annual,
        "first_eol_year": first_eol,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — 2D SWEEP DRIVER
# ═══════════════════════════════════════════════════════════════════════════════

def run_2d_sweep(
    price_data:     np.ndarray,
    wind_data:      np.ndarray,
    e_cap_range:    np.ndarray,
    p_cap_range:    np.ndarray,
    sweep_kwargs:   dict,
    verbose:        bool = True,
) -> dict:
    """
    Run the 20-year loop for all (e_cap, p_cap) combinations.

    Returns a dict of 2D result arrays, shape (n_e, n_p).
    Rows = e_cap_range, columns = p_cap_range.
    """
    n_e = len(e_cap_range)
    n_p = len(p_cap_range)
    total_pts = n_e * n_p

    # Pre-allocate result grids
    grid_npv       = np.full((n_e, n_p), np.nan)
    grid_nrepl     = np.full((n_e, n_p), np.nan)
    grid_fd_mean   = np.full((n_e, n_p), np.nan)
    grid_eol       = np.full((n_e, n_p), np.nan)
    grid_rev_mean  = np.full((n_e, n_p), np.nan)

    t_start = time.time()
    done    = 0

    for i, e_cap in enumerate(e_cap_range):
        for j, p_cap in enumerate(p_cap_range):
            done += 1
            if verbose:
                elapsed = time.time() - t_start
                eta     = (elapsed / done) * (total_pts - done) if done > 1 else 0.0
                print(
                    f"  [{done:3d}/{total_pts}]  e_cap={e_cap:6.0f} MWh  "
                    f"p_cap={p_cap:5.0f} MW   elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                    end="\r",
                )

            result = run_20year_loop(
                e_cap_nom  = e_cap,
                p_cap_nom  = p_cap,
                price_data = price_data,
                wind_data  = wind_data,
                **sweep_kwargs,
            )

            grid_npv[i, j]      = result["npv_lifetime"]
            grid_nrepl[i, j]    = result["n_replacements"]
            grid_fd_mean[i, j]  = np.mean(result["fd_annual_list"]) \
                                   if result["fd_annual_list"] else np.nan
            grid_eol[i, j]      = result["first_eol_year"] \
                                   if result["first_eol_year"] else np.nan
            grid_rev_mean[i, j] = np.mean(result["revenue_annual"]) \
                                   if result["revenue_annual"] else np.nan

    if verbose:
        print(f"\n  Sweep complete in {time.time()-t_start:.1f}s")

    return {
        "npv":       grid_npv,
        "n_repl":    grid_nrepl,
        "fd_mean":   grid_fd_mean,
        "eol_year":  grid_eol,
        "rev_mean":  grid_rev_mean,
        "e_cap":     e_cap_range,
        "p_cap":     p_cap_range,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PLOTTING  (thesis visual style from memory)
# ═══════════════════════════════════════════════════════════════════════════════

BG_COLOR = "#f7f9fc"
C_BLUE   = "#2166ac"
C_RED    = "#b5351b"

def plot_2d_sweep(grids: dict, timestamp: str, e_ref: float = 300.0, p_ref: float = 150.0):
    """
    Four-panel heatmap: NPV, number of replacements, mean annual fd, first EoL year.
    The reference design point (300 MWh / 150 MW) is marked on each panel.
    """
    E = grids["e_cap"]
    P = grids["p_cap"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        "2D Parameter Sweep: 20-Year Lifetime NPV\n"
        r"Energy Capacity $\bar{E}$ vs Power Capacity $\bar{P}$",
        fontsize=14, fontweight="bold", y=1.01,
    )

    panels = [
        (axes[0, 0], grids["npv"] * 1e-6,  "Lifetime NPV [M EUR]",          "RdYlGn"),
        (axes[0, 1], grids["n_repl"],        "Number of Replacements [–]",    "YlOrRd_r"),
        (axes[1, 0], grids["fd_mean"] * 1e3, "Mean Annual $f_d$ [×10⁻³]",    "YlOrRd"),
        (axes[1, 1], grids["eol_year"],       "First Battery EoL [year]",      "RdYlGn"),
    ]

    for ax, data, title, cmap in panels:
        ax.set_facecolor(BG_COLOR)
        im = ax.contourf(P, E, data, levels=20, cmap=cmap)
        cs = ax.contour(P, E, data, levels=10, colors="black", linewidths=0.4, alpha=0.4)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

        # Mark the reference design point
        ax.scatter(
            [p_ref], [e_ref], marker="*", s=200, color="white",
            edgecolors="black", linewidths=0.8, zorder=5,
            label=f"Reference ({e_ref:.0f} MWh / {p_ref:.0f} MW)",
        )
        ax.legend(fontsize=7, loc="upper left", framealpha=0.85)

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(title, fontsize=9)

        ax.set_xlabel(r"Power Capacity $\bar{P}$ [MW]", fontsize=10)
        ax.set_ylabel(r"Energy Capacity $\bar{E}$ [MWh]", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"sweep2d_heatmaps_{timestamp}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"  Saved heatmap → {out_path}")
    plt.close(fig)

    # ── Additional plot: E/P ratio ridge ──────────────────────────────────────
    _plot_ep_ratio(grids, timestamp, e_ref, p_ref)


def _plot_ep_ratio(grids: dict, timestamp: str, e_ref: float, p_ref: float):
    """
    NPV along lines of constant E/P ratio, and the NPV ridge (argmax_P for each E).
    Useful for communicating what the gradient is trying to find.
    """
    E   = grids["e_cap"]
    P   = grids["p_cap"]
    NPV = grids["npv"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    # Panel A: NPV slices along constant E/P ratio diagonals
    ax = axes[0]
    ax.set_facecolor(BG_COLOR)
    ratios = [1.5, 2.0, 2.5, 3.0]   # E/P in MWh/MW (hours of storage)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(ratios)))
    for ratio, col in zip(ratios, colors):
        # For each e_cap value, find the p_cap closest to e_cap/ratio
        npv_slice = []
        e_vals    = []
        for i, e in enumerate(E):
            p_target = e / ratio
            if p_target < P[0] or p_target > P[-1]:
                continue
            j = np.argmin(np.abs(P - p_target))
            npv_slice.append(NPV[i, j] * 1e-6)
            e_vals.append(e)
        ax.plot(e_vals, npv_slice, color=col, linewidth=1.8,
                label=f"E/P = {ratio:.1f} h")
    ax.axvline(e_ref, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel(r"Energy Capacity $\bar{E}$ [MWh]", fontsize=11)
    ax.set_ylabel("Lifetime NPV [M EUR]", fontsize=11)
    ax.set_title("NPV along constant E/P ratio lines", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    # Panel B: Optimal P for each E (ridge line) — shows sensitivity split
    ax2 = axes[1]
    ax2.set_facecolor(BG_COLOR)
    opt_p = P[np.nanargmax(NPV, axis=1)]   # p_cap that maximises NPV at each e_cap
    ax2.plot(opt_p, E, color=C_BLUE, linewidth=2.0, label="NPV-optimal P")
    ax2.axvline(p_ref, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.axhline(e_ref, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.scatter([p_ref], [e_ref], s=120, color=C_RED, zorder=5,
                label="Reference design")
    ax2.set_xlabel(r"NPV-Optimal $\bar{P}$ [MW]", fontsize=11)
    ax2.set_ylabel(r"Energy Capacity $\bar{E}$ [MWh]", fontsize=11)
    ax2.set_title("Optimal power capacity per energy level", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.tick_params(labelsize=9)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"sweep2d_ridge_{timestamp}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"  Saved ridge plot → {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SAVE / LOAD RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(grids: dict, timestamp: str):
    """Save all 2D arrays to a single .npz file and a human-readable CSV summary."""
    npz_path = OUTPUT_DIR / f"sweep2d_results_{timestamp}.npz"
    np.savez(
        npz_path,
        e_cap   = grids["e_cap"],
        p_cap   = grids["p_cap"],
        npv     = grids["npv"],
        n_repl  = grids["n_repl"],
        fd_mean = grids["fd_mean"],
        eol_year= grids["eol_year"],
        rev_mean= grids["rev_mean"],
    )
    print(f"  Saved arrays → {npz_path}")

    # Summary CSV: one row per (e_cap, p_cap) pair
    rows = []
    for i, e in enumerate(grids["e_cap"]):
        for j, p in enumerate(grids["p_cap"]):
            rows.append((
                e, p,
                grids["npv"][i, j],
                grids["n_repl"][i, j],
                grids["fd_mean"][i, j],
                grids["eol_year"][i, j],
                grids["rev_mean"][i, j],
            ))
    header = "e_cap_MWh,p_cap_MW,npv_EUR,n_replacements,fd_mean_annual,first_eol_year,rev_mean_annual_EUR"
    csv_path = OUTPUT_DIR / f"sweep2d_results_{timestamp}.csv"
    np.savetxt(
        csv_path,
        np.array(rows, dtype=float),
        delimiter=",",
        header=header,
        comments="",
        fmt=["%.1f", "%.1f", "%.2f", "%.0f", "%.6f", "%.1f", "%.2f"],
    )
    print(f"  Saved CSV   → {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def load_wp2_data(year: int, n: int = 8760) -> tuple[np.ndarray, np.ndarray]:
    """
    Load WP2 price and wind data. Adjust paths to match your data files.
    Falls back to dummy data with a warning if files are not found.
    """
    # ── Prices ─────────────────────────────────────────────────────────────
    price_candidates = [
        DATA_DIR / f"price_DK1_{year}_hourly.npy",
        DATA_DIR / f"DK1_price_{year}.npy",
        DATA_DIR.parent / f"data/price_DK1_{year}.npy",
    ]
    price_data = None
    for p in price_candidates:
        if p.exists():
            price_data = np.load(p)[:n]
            print(f"  Loaded prices from {p}")
            break

    # ── Wind power ──────────────────────────────────────────────────────────
    wind_candidates = [
        DATA_DIR / f"wind_wp2_{year}_hourly.npy",
        DATA_DIR / "wp2_wind_power.npy",
        DATA_DIR.parent / "data/wind_wp2.npy",
    ]
    wind_data = None
    for p in wind_candidates:
        if p.exists():
            wind_data = np.load(p)[:n]
            print(f"  Loaded wind   from {p}")
            break

    if price_data is None or wind_data is None:
        warnings.warn(
            "Could not find price or wind data files. "
            "Using dummy data (50 EUR/MWh flat price, 80 MW constant wind). "
            "Update load_wp2_data() with your actual file paths."
        )
        price_data = 50.0 * np.ones(n)
        wind_data  = 80.0 * np.ones(n)

    return price_data[:n], wind_data[:n]


def main():
    parser = argparse.ArgumentParser(
        description="2D parameter sweep: e_cap × p_cap with 20-year degradation loop"
    )
    parser.add_argument("--year",    type=int,   default=2022,
                        help="Price/wind year (2019 or 2022)")
    parser.add_argument("--grid",    type=int,   default=10,
                        help="Number of grid points per axis (default 10 → 100 LP solves × 20 years)")
    parser.add_argument("--n_years", type=int,   default=20)
    parser.add_argument("--e_range", type=float, nargs=2, default=[150.0, 450.0],
                        metavar=("E_MIN", "E_MAX"),
                        help="Energy capacity sweep range [MWh], default 150–450")
    parser.add_argument("--p_range", type=float, nargs=2, default=[75.0, 225.0],
                        metavar=("P_MIN", "P_MAX"),
                        help="Power capacity sweep range [MW], default 75–225")
    parser.add_argument("--solver",  type=str,   default="gurobi",
                        help="Pyomo solver name (gurobi, cplex, glpk)")
    parser.add_argument("--lambda_E", type=float, default=150_000.0,
                        help="Battery energy capex [EUR/MWh]")
    parser.add_argument("--lambda_P", type=float, default=100_000.0,
                        help="Battery power capex [EUR/MW]")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}")
    print(f"  2D Parameter Sweep  |  year={args.year}  |  grid={args.grid}×{args.grid}")
    print(f"  E range: {args.e_range[0]:.0f}–{args.e_range[1]:.0f} MWh")
    print(f"  P range: {args.p_range[0]:.0f}–{args.p_range[1]:.0f} MW")
    print(f"  n_years={args.n_years}  |  solver={args.solver}")
    total_lps = args.grid ** 2 * args.n_years
    print(f"  Total LP solves: {total_lps:,}  (grid² × n_years)")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    price_data, wind_data = load_wp2_data(args.year)

    # ── Define sweep grid ─────────────────────────────────────────────────────
    e_cap_range = np.linspace(args.e_range[0], args.e_range[1], args.grid)
    p_cap_range = np.linspace(args.p_range[0], args.p_range[1], args.grid)

    # ── Shared kwargs for every grid point ────────────────────────────────────
    sweep_kwargs = dict(
        n_years        = args.n_years,
        discount_rate  = 0.05,
        repl_threshold = 0.70,
        lambda_E       = args.lambda_E,
        lambda_P       = args.lambda_P,
        p_min          = 0.0,
        p_max          = 200.0,
        dt             = 1.0,
        soc_min_frac   = 0.10,
        soc_max_frac   = 0.90,
        eta_in         = 0.95,
        eta_out        = 0.95,
        pyo_solver     = args.solver,
    )

    # ── Run sweep ─────────────────────────────────────────────────────────────
    print("Running sweep...")
    grids = run_2d_sweep(
        price_data   = price_data,
        wind_data    = wind_data,
        e_cap_range  = e_cap_range,
        p_cap_range  = p_cap_range,
        sweep_kwargs = sweep_kwargs,
        verbose      = True,
    )

    # ── Print summary at reference point ──────────────────────────────────────
    e_ref, p_ref = 300.0, 150.0
    i_ref = np.argmin(np.abs(e_cap_range - e_ref))
    j_ref = np.argmin(np.abs(p_cap_range - p_ref))
    print(f"\n  Reference point ({e_ref:.0f} MWh / {p_ref:.0f} MW):")
    print(f"    Lifetime NPV     = {grids['npv'][i_ref, j_ref]*1e-6:.2f} M EUR")
    print(f"    N replacements   = {grids['n_repl'][i_ref, j_ref]:.0f}")
    print(f"    Mean annual fd   = {grids['fd_mean'][i_ref, j_ref]:.4f}")
    print(f"    First EoL year   = {grids['eol_year'][i_ref, j_ref]}")

    best_flat = np.nanargmax(grids["npv"])
    i_best, j_best = np.unravel_index(best_flat, grids["npv"].shape)
    print(f"\n  Best NPV point:")
    print(f"    e_cap = {e_cap_range[i_best]:.0f} MWh,  p_cap = {p_cap_range[j_best]:.0f} MW")
    print(f"    NPV   = {grids['npv'][i_best, j_best]*1e-6:.2f} M EUR")

    # ── Save and plot ──────────────────────────────────────────────────────────
    save_results(grids, timestamp)
    plot_2d_sweep(grids, timestamp, e_ref=e_ref, p_ref=p_ref)

    print(f"\n  All outputs written to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
