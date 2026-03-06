"""
WP2 Battery Degradation — Plots, Reports, and DST Validation
=============================================================

All visualisation and reporting for the Xu et al. (2016) degradation model.
Split from degradation_xu.py to keep the physical model file concise.

Imports everything it needs from degradation_xu — nothing here does any
degradation computation, it only presents results.

Public API
----------
    print_degradation_report(degradation, period_days, enabled) -> None
    plot_degradation_analysis(degradation, storage_e, time_vec,
                              save_path, show, verbose,
                              lifetime_years, eol_thresholds, p) -> List[Figure]
    validate_xu_dst(save_path, show, T_cell_C) -> Figure
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from degradation_xu import (
    XuModelParams,
    XU_LMO,
    s_temp,
    fc_cycle,
    xu_capacity_curve,
)


def print_degradation_report(degradation: Dict,
                              period_days: float,
                              enabled: bool = True) -> None:
    """Print a formatted degradation report based on Xu model results.

    Args:
        degradation: Output of analyze_degradation().
        period_days: Simulation duration [days].
        enabled:     If False, no output is printed.
    """
    if not enabled:
        return

    print("\n" + "=" * 72)
    print("BATTERY DEGRADATION — Xu et al. (2016) Semi-Empirical Model")
    print("=" * 72)

    meta   = degradation.get("meta", {})
    stats  = degradation.get("xu_cycle_stats", {})
    cycles = float(degradation["total_cycles"])

    print(f"\nSimulation period      : {period_days:.1f} days")
    print(f"Temperature assumption : {meta.get('T_cell_C', 25):.0f}°C  "
          f"(S_T = {s_temp(meta.get('T_cell_C', 25.0)):.4f})")
    print(f"Mean SoC               : {meta.get('sigma_mean', 0)*100:.1f}%")

    print("\nCycle metrics:")
    print(f"  Equivalent full cycles (EFC) : {cycles:.2f}")
    print(f"  EFC per day                  : {cycles/max(period_days, 1):.3f}")
    print(f"  EFC per year (extrapolated)  : {cycles/max(period_days, 1)*365:.1f}")
    if stats:
        print(f"  Rainflow cycles (weighted)   : {stats.get('n_rainflow_cycles', 0):.1f}")
        print(f"  Mean rainflow DoD            : {stats.get('mean_dod', 0)*100:.1f}%")
        print(f"  Mean rainflow SoC            : {stats.get('mean_soc', 0)*100:.1f}%")

    print("\nXu degradation (fd decomposition):")
    fd = float(degradation["fd"])
    print(f"  fd_total    : {fd:.6f}")
    print(f"  fd_cycle    : {degradation['fd_cycle']:.6f}  "
          f"({100*degradation['fd_cycle']/max(fd, 1e-30):.1f}%)")
    print(f"  fd_calendar : {degradation['fd_calendar']:.6f}  "
          f"({100*degradation['fd_calendar']/max(fd, 1e-30):.1f}%)")

    print("\nCapacity state:")
    print(f"  State of Health (SoH)   : {degradation['soh']:.3f}%")
    print(f"  Capacity retention      : {degradation['capacity_retention']:.6f}")
    print(f"  Capacity fade           : {degradation['capacity_fade_percent']:.4f}%")
    print(f"  Degraded energy cap     : {degradation['e_cap_degraded']:.3f} MWh")
    print(f"  Degraded power cap      : {degradation['p_cap_degraded']:.3f} MW")

    eol_years = degradation.get("eol_years", {})
    if eol_years:
        print("\nProjected End-of-Life (same cycling pattern extrapolated):")
        labels = {0.80: "IEC/EV convention",
                  0.70: "warranty typical",
                  0.60: "grid-storage operational"}
        for thr in sorted(eol_years.keys(), reverse=True):
            yr = eol_years[thr]
            note = labels.get(thr, "")
            yr_str = f"{yr:.1f} yr" if yr is not None else "> 200 yr"
            print(f"  SoH \u2265 {thr*100:.0f}%  \u2192  EoL at {yr_str}"
                  + (f"  [{note}]" if note else ""))

    print("\n" + "=" * 72)


# =============================================================================
# Capacity fade curve helper
# =============================================================================

def plot_degradation_analysis(
    degradation: Dict,
    storage_e: List[float],
    time_vec: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
    verbose: bool = False,
    lifetime_years: float = 15.0,
    eol_thresholds: List[float] = None,
    p: XuModelParams = XU_LMO,
) -> List["plt.Figure"]:
    """Save each degradation panel as a separate, clearly labelled figure.

    Six figures are produced and, if save_path is given, saved next to it:

        <stem>_1_soc.<ext>              Battery State of Charge
        <stem>_2_discharge_level.<ext>  Hourly Discharge Level Distribution
        <stem>_3_sei_fade.<ext>         Xu SEI Capacity Fade curve
        <stem>_4_soh_projection.<ext>   SoH projection over design lifetime
        <stem>_5_rainflow_dod.<ext>     Rainflow Cycle DoD histogram
        <stem>_6_rainflow_scatter.<ext> Rainflow DoD vs Mean SoC scatter

    Args:
        degradation:    Output dict from analyze_degradation().
        storage_e:      SOC time-series [MWh].
        time_vec:       Time axis (days).
        save_path:      Base path for output images.  Suffix and number are
                        inserted automatically before the extension.
                        If None, figures are not saved.
        show:           If True, calls plt.show() on each figure.
        verbose:        If True, prints each saved path.
        lifetime_years: Design lifetime for the SoH projection panel.
        p:              Xu model parameters.

    Returns:
        List of 6 matplotlib Figure objects.
    """
    from pathlib import Path as _Path

    if eol_thresholds is None:
        eol_thresholds = list(degradation.get("meta", {}).get(
            "eol_thresholds", [0.80, 0.60]))

    # Colour palette for EoL threshold lines (one colour per threshold)
    _thr_colours = {0.80: "#e74c3c", 0.70: "#e67e22", 0.60: "#8e44ad"}
    def _thr_colour(thr: float) -> str:
        return _thr_colours.get(thr, "#555555")

    meta       = degradation.get("meta", {})
    fd_now     = float(degradation["fd"])
    soh_now    = float(degradation["soh"])
    T_C        = float(meta.get("T_cell_C", 25.0))
    t_sim_h    = float(meta.get("t_total_hours", 8760.0))
    sigma_mean = float(meta.get("sigma_mean", 0.5))
    pct_cyc    = 100 * degradation["fd_cycle"]    / max(fd_now, 1e-30)
    pct_cal    = 100 * degradation["fd_calendar"] / max(fd_now, 1e-30)
    cycles     = degradation.get("cycle_depth_distribution", [])

    # Shared suptitle suffix so every figure is identifiable on its own
    sup = (f"Xu (2016) Degradation  |  T = {T_C:.0f}°C  |  "
           f"fd = {fd_now:.4f}  |  SoH = {soh_now:.2f}%  |  "
           f"Cycle {pct_cyc:.0f}% / Calendar {pct_cal:.0f}%")

    def _save(fig: "plt.Figure", suffix: str) -> None:
        if save_path is not None:
            sp = _Path(save_path)
            out = sp.with_name(sp.stem + suffix + sp.suffix)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            if verbose:
                print(f"  \u2713 Saved: {out.name}")
        if show:
            plt.show()

    figs: List["plt.Figure"] = []

    # ------------------------------------------------------------------
    # Figure 1 — Battery State of Charge
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_vec, storage_e, linewidth=0.6, color="steelblue", label="SoC")
    ax.axhline(
        degradation["e_cap_degraded"],
        linestyle="--", color="red", linewidth=1.5,
        label=f"Degraded capacity after {t_sim_h/8760:.1f} yr  (SoH = {soh_now:.2f}%)"
    )
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("State of Charge [MWh]")
    ax.set_title("Battery State of Charge")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.suptitle(sup, fontsize=9, y=1.01)
    plt.tight_layout()
    _save(fig, "_1_soc")
    figs.append(fig)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2 — Hourly Discharge Level Distribution
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    dod_bins, dod_counts = degradation["dod_distribution"]
    width = float(0.8 * (dod_bins[1] - dod_bins[0])) if len(dod_bins) > 1 else 0.08
    ax.bar(dod_bins * 100, dod_counts, width=width * 100,
           alpha=0.75, edgecolor="black", color="steelblue")
    ax.set_xlabel("Instantaneous discharge level [% of capacity]")
    ax.set_ylabel("Number of hours at this level")
    ax.set_title(
        "Hourly Discharge Level Distribution\n"
        "How many hours was the battery sitting at each discharge level?\n"
        "(hourly snapshot — not the same as cycle depth)"
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle(sup, fontsize=9, y=1.01)
    plt.tight_layout()
    _save(fig, "_2_discharge_level")
    figs.append(fig)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3 — Xu SEI Capacity Fade Curve
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    fd_hi     = max(fd_now * 3.0, 0.28)
    fd_range  = np.linspace(0.0, fd_hi, 400)
    cap_curve = xu_capacity_curve(fd_range, p)

    ax.plot(fd_range, cap_curve, linewidth=2, color="steelblue",
            label="Xu (2016) SEI model — LMO parameters (Table I)")
    for thr in sorted(eol_thresholds, reverse=True):
        ax.axhline(thr * 100, linestyle="--", color=_thr_colour(thr), alpha=0.8,
                   label=f"EoL threshold {thr*100:.0f}%  "
                         f"({'IEC/EV' if thr == 0.80 else 'grid-storage' if thr == 0.60 else f'{thr*100:.0f}%'})")
    ax.scatter([fd_now], [soh_now], s=150, zorder=5, color="red",
               label=f"Current state  (fd = {fd_now:.4f},  SoH = {soh_now:.2f}%)")
    ax.annotate(
        f"fd breakdown:\n"
        f"  Cycle aging    {degradation['fd_cycle']:.5f}  ({pct_cyc:.0f}%)\n"
        f"  Calendar aging {degradation['fd_calendar']:.5f}  ({pct_cal:.0f}%)\n"
        f"  Total fd       {fd_now:.5f}",
        xy=(0.36, 0.64), xycoords="axes fraction", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9)
    )
    ax.set_xlabel("Degradation index fd  (0 = new,  higher = more degraded)")
    ax.set_ylabel("Capacity retention [%]")
    ax.set_title(
        "Xu (2016) SEI Capacity Fade — LMO\n"
        "fd accumulates from cycle and calendar aging; "
        "the nonlinear curve reflects faster early degradation (SEI film formation)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.suptitle(sup, fontsize=9, y=1.01)
    plt.tight_layout()
    _save(fig, "_3_sei_fade")
    figs.append(fig)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4 — SoH Projection over Design Lifetime
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    fd_per_yr = fd_now / max(t_sim_h / 8760.0, 1e-6)
    years     = np.linspace(0, lifetime_years, 400)
    soh_proj  = xu_capacity_curve(years * fd_per_yr, p)

    ax.plot(years, soh_proj, linewidth=2.5, color="steelblue",
            label=f"Projected SoH  (fd/yr = {fd_per_yr:.5f})")

    # Draw a horizontal + vertical line for each EoL threshold
    eol_years_dict = degradation.get("eol_years", {})
    for thr in sorted(eol_thresholds, reverse=True):
        col = _thr_colour(thr)
        note = ('IEC/EV' if thr == 0.80 else
                'grid-storage' if thr == 0.60 else f'{thr*100:.0f}%')
        ax.axhline(thr * 100, linestyle="--", color=col, alpha=0.7,
                   label=f"EoL {thr*100:.0f}% [{note}]")
        # EoL crossing vertical line
        yr_eol = eol_years_dict.get(thr)
        if yr_eol is not None and yr_eol <= lifetime_years:
            ax.axvline(yr_eol, linestyle=":", color=col, alpha=0.6)
            ha = "left" if yr_eol < lifetime_years * 0.75 else "right"
            ax.annotate(
                f"{thr*100:.0f}% EoL\n\u2248 {yr_eol:.1f} yr",
                xy=(yr_eol + (0.2 if ha == "left" else -0.2), thr * 100 + 0.8),
                fontsize=8.5, color=col, ha=ha, fontweight="bold"
            )

    sim_yr = t_sim_h / 8760.0
    ax.scatter([sim_yr], [soh_now], s=150, zorder=5, color="red",
               label=f"End of simulation  ({sim_yr:.1f} yr,  SoH = {soh_now:.2f}%)")

    ax.set_xlim(0, lifetime_years)
    min_thr = min(eol_thresholds) * 100 if eol_thresholds else 60.0
    ax.set_ylim(max(float(soh_proj.min()) - 3, min_thr - 8), 102)
    ax.set_xlabel("Battery age [years]")
    ax.set_ylabel("State of Health [%]  (100% = new)")
    ax.set_title(
        f"SoH Projection over {lifetime_years:.0f}-Year Design Lifetime\n"
        "Assumes same operating pattern repeats each year"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.suptitle(sup, fontsize=9, y=1.01)
    plt.tight_layout()
    _save(fig, "_4_soh_projection")
    figs.append(fig)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 5 — Rainflow Cycle DoD Histogram
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    if cycles:
        dods = np.array([c["dod"]   for c in cycles]) * 100
        cnts = np.array([c["count"] for c in cycles])
        ax.hist(dods, bins=20, weights=cnts, color="steelblue",
                alpha=0.75, edgecolor="black")
        mean_dod_cyc = float(np.average(dods, weights=cnts))
        ax.axvline(mean_dod_cyc, linestyle="--", color="red",
                   label=f"Weighted mean DoD = {mean_dod_cyc:.1f}%")
        ax.legend(fontsize=9)
    ax.set_xlabel("Cycle depth of discharge [% of nominal capacity]")
    ax.set_ylabel("Weighted cycle count")
    ax.set_title(
        "Rainflow Cycle DoD Distribution\n"
        "Depth of each identified charge/discharge cycle "
        "(feeds into Xu S_\u03b4 stress factor \u2014 deeper = more damage per cycle)"
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle(sup, fontsize=9, y=1.01)
    plt.tight_layout()
    _save(fig, "_5_rainflow_dod")
    figs.append(fig)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 6 — Rainflow DoD vs Mean SoC Scatter
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    if cycles:
        dods_s = np.array([c["dod"]      for c in cycles]) * 100
        socs_s = np.array([c["soc_mean"] for c in cycles]) * 100
        cnts_s = np.array([c["count"]    for c in cycles])
        sc = ax.scatter(socs_s, dods_s, c=cnts_s, cmap="viridis",
                        s=25, alpha=0.7, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Cycle count (0.5 = half-cycle)")
    ax.set_xlabel("Mean SoC during the cycle [%]")
    ax.set_ylabel("Cycle DoD [%]")
    ax.set_title(
        "Rainflow Cycles \u2014 Cycle DoD vs Mean SoC\n"
        "Each dot = one identified cycle.  "
        "Upper-right = deepest cycles at highest SoC = most damaging in Xu model"
    )
    ax.grid(True, alpha=0.3)
    plt.suptitle(sup, fontsize=9, y=1.01)
    plt.tight_layout()
    _save(fig, "_6_rainflow_scatter")
    figs.append(fig)
    plt.close(fig)

    return figs



# =============================================================================
# Validation against Xu et al. (2016) Fig. 5
# =============================================================================

def validate_xu_dst(
    save_path: Optional[str] = "xu_validation_dst.png",
    show: bool = True,
    T_cell_C: float = 21.0,
) -> "plt.Figure":
    """Reproduce Fig. 5 of Xu et al. (2016): model vs. experimental DST data.

    Produces a two-panel figure mirroring the paper's layout:
      Left  — digitised experimental data from Fig. 5a.
      Right — this implementation's model reproduction (Fig. 5b equivalent).

    Model assumptions
    -----------------
    Each DST test cycle is modelled as a SINGLE full rainflow cycle with:
      DoD  = soc_start − soc_stop  (the full swing from start to stop)
      σ    = (soc_start + soc_stop) / 2  (mean SoC)

    This is the correct minimal approximation: the discharge half and the
    1C recharge half form one complete rainflow cycle.  The DST profile
    does create small internal micro-cycles (visible in Fig. 4c of the paper),
    but without the proprietary DST signal those cannot be reproduced exactly.
    Omitting them is consistent with the paper's own note that their 14%
    error comes from "the absence of the original SoC profile".

    Temperature is fixed at 21°C as stated in Section V.B of the paper
    ("simulated average cell temperature was 21°C").

    Digitised reference data
    ------------------------
    ~10 points per curve manually extracted from Fig. 5a.  Used as scatter
    markers on the left panel so the model curves on the right can be
    compared directly against experiment.

    Args:
        save_path:  Path to save the output figure.  None = don't save.
        show:       If True, calls plt.show().
        T_cell_C:   Cell temperature [°C].  Paper uses 21°C.

    Returns:
        Matplotlib Figure object.
    """
    p = XU_LMO

    # ------------------------------------------------------------------
    # 7 DST test conditions from Fig. 5 legend: (label, soc_start, soc_stop)
    # ------------------------------------------------------------------
    dst_cases = [
        ("100-25@20C", 1.00, 0.25),   # DoD=75%, mean SoC=62.5%  — most degradation
        ("100-40@20C", 1.00, 0.40),   # DoD=60%, mean SoC=70%
        ("85-25@20C",  0.85, 0.25),   # DoD=60%, mean SoC=55%
        ("100-50@20C", 1.00, 0.50),   # DoD=50%, mean SoC=75%
        ("75-25@20C",  0.75, 0.25),   # DoD=50%, mean SoC=50%
        ("75-45@20C",  0.75, 0.45),   # DoD=30%, mean SoC=60%
        ("75-65@20C",  0.75, 0.65),   # DoD=10%, mean SoC=70%  — least degradation
    ]

    colours = ["black", "red", "green", "navy", "teal", "magenta", "goldenrod"]

    # ------------------------------------------------------------------
    # Digitised reference data — Fig. 5a, Xu et al. (2016)
    # Manually extracted; ~10 points per curve.
    # ------------------------------------------------------------------
    fig5a_data = {
        "100-25@20C": [
            (0, 100.0), (300, 97.0), (700, 94.0), (1200, 91.5),
            (2000, 89.0), (3000, 86.5), (4500, 84.0), (6000, 82.0),
            (7500, 80.0), (9000, 78.5),
        ],
        "100-40@20C": [
            (0, 100.0), (400, 97.5), (900, 95.5), (1800, 93.0),
            (3000, 90.5), (4500, 88.0), (6000, 86.0), (7500, 84.0),
            (9000, 82.0),
        ],
        "85-25@20C": [
            (0, 100.0), (500, 98.0), (1200, 96.0), (2200, 94.0),
            (3500, 91.5), (5000, 89.5), (6500, 87.5), (8000, 86.0),
            (9000, 85.0),
        ],
        "100-50@20C": [
            (0, 100.0), (600, 98.0), (1500, 96.5), (2800, 94.5),
            (4500, 92.5), (6000, 91.0), (7500, 89.5), (9000, 88.0),
        ],
        "75-25@20C": [
            (0, 100.0), (700, 98.5), (1800, 97.0), (3000, 95.5),
            (4500, 94.0), (6000, 92.5), (7500, 91.0), (9000, 90.0),
        ],
        "75-45@20C": [
            (0, 100.0), (1000, 99.0), (2500, 98.0), (4000, 97.0),
            (5500, 96.0), (7000, 95.5), (9000, 94.5),
        ],
        "75-65@20C": [
            (0, 100.0), (1500, 99.5), (3000, 99.0), (5000, 98.0),
            (7000, 97.5), (9000, 97.0),
        ],
    }

    # ------------------------------------------------------------------
    # Compute model prediction for each case
    #
    # Each DST test cycle = 1 full rainflow cycle:
    #   DoD  = soc_start - soc_stop
    #   σ    = (soc_start + soc_stop) / 2
    #   S_T  = s_temp(T_cell_C)  [fixed at 21°C per paper Section V.B]
    #
    # fd_per_cycle = S_δ(DoD) × S_σ(σ) × S_T(T)
    #
    # This deliberately omits DST micro-cycles because those require the
    # proprietary DST signal to reproduce.  The paper's own reproduction
    # has 14% error from the same limitation.
    # ------------------------------------------------------------------
    N_MAX = 9000
    cycle_vec = np.arange(0, N_MAX + 1, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ax_data, ax_model = axes

    for (label, soc_s, soc_e), colour in zip(dst_cases, colours):
        dod     = soc_s - soc_e
        soc_m   = (soc_s + soc_e) / 2.0

        fd_per_cycle = float(fc_cycle(dod, soc_m, T_cell_C, p))
        fd_vec  = cycle_vec * fd_per_cycle
        cap_vec = xu_capacity_curve(fd_vec, p)

        # Right panel: model
        ax_model.plot(cycle_vec, cap_vec, color=colour, linewidth=1.8, label=label)

        # Left panel: digitised experimental data
        ref = fig5a_data.get(label, [])
        if ref:
            rx, ry = zip(*ref)
            ax_data.plot(rx, ry, color=colour, linewidth=1.2,
                         linestyle="--", alpha=0.6)
            ax_data.scatter(rx, ry, color=colour, s=22, zorder=4, label=label)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    for ax, title in zip(
        [ax_data, ax_model],
        ["(a) Digitised experimental data  [Fig. 5a, Xu et al. 2016]",
         "(b) Model reproduction  [this implementation, Fig. 5b equivalent]"],
    ):
        ax.set_xlim(0, N_MAX)
        ax.set_ylim(60, 107)
        ax.set_xlabel("Number of DST cycles", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=11)
        ax.axhline(80, color="gray", linestyle=":", linewidth=1.0)
        ax.text(200, 80.6, "80% EoL", fontsize=8, color="gray")

    ax_data.set_ylabel("1C capacity retention (%)", fontsize=11)
    ax_model.legend(fontsize=8.5, loc="lower left",
                    title="Start–Stop SoC @ 20°C room", title_fontsize=8)

    # Annotate key match from paper: 85-25 and 100-50 have same fd (paper text, p.7)
    ax_model.annotate(
        "85-25 ≈ 100-50 (same linearised\ndegradation rate, per paper p.7)",
        xy=(9000, 84), xytext=(5500, 78),
        fontsize=7.5, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

    fig.suptitle(
        "Validation: Xu et al. (2016) LMO Degradation Model — DST Cycle Test\n"
        f"T_cell = {T_cell_C}°C (paper: simulated 21°C).  "
        "Model: 1 full cycle per DST test (DoD = SoC_start − SoC_stop).\n"
        "Paper reports 14% error vs. data; residual gap here from omitted DST micro-cycles.",
        fontsize=9.5,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig



# =============================================================================
# Self-test — run DST validation directly
# =============================================================================

if __name__ == "__main__":
    print("Running DST validation plot...")
    validate_xu_dst(save_path="xu_validation_dst.png", show=True)
    print("Done. Check xu_validation_dst.png")