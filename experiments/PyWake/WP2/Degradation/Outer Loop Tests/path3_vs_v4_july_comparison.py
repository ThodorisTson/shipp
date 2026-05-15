"""
Path 3 vs v4 — July 2022 Comparison
=====================================

Extracts July from the saved v4 full-year LP dispatch, computes Shi
degradation on that slice, then compares with the Path 3 NLP results
for the same month.

This gives a direct apples-to-apples comparison:
  v4:     LP dispatch (ignores degradation) → compute fd post-hoc
  Path 3: NLP dispatch (minimises revenue − degradation cost)

Both use the same battery (300 MWh / 150 MW), same prices (DK1 2022),
same Shi polynomial, same SoC window (10–90%).

Usage:
    python path3_vs_v4_july_comparison.py

Author: Thodoris Tsonopoulos — MSc Thesis, TU Delft Wind Energy
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).parent
_DEGRAD_DIR  = _SCRIPT_DIR.parent
_RESULTS_DIR = _DEGRAD_DIR / "Results"
_PATH3_DIR   = _SCRIPT_DIR / "Results_Path3"

for p in [str(_SCRIPT_DIR), str(_DEGRAD_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from path3_cyipopt_prototype import (
    compute_f_deg,
    fit_shi_polynomial,
)
from degradation_xu import (
    ft_calendar,
    sei_capacity_loss,
    rainflow_cycle_counting,
    s_soc,
    s_temp,
    XU_LMO,
)

# ── Config ──────────────────────────────────────────────────────────────
E_CAP   = 300.0
P_CAP   = 150.0
SOC_MIN = 0.10
SOC_MAX = 0.90
B       = 150_000.0  # EUR/MWh replacement cost
T_CELL  = 25.0       # degC

# July boundaries (non-leap year)
JULY_START = (31+28+31+30+31+30) * 24  # h4344
JULY_END   = JULY_START + 31 * 24       # h5088
JULY_HOURS = JULY_END - JULY_START      # 744

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    print("=" * 72)
    print("Path 3 vs v4 — July 2022 Comparison")
    print(f"Timestamp: {timestamp}")
    print("=" * 72)

    # ── Load DK1 2022 prices (July slice) ───────────────────────────────
    import pandas as pd
    price_csv = _DEGRAD_DIR / "dk1_prices_2022.csv"
    df = pd.read_csv(price_csv)
    for col in ["price", "price [EUR/MWh]", "Price", "EUR/MWh", "DK1"]:
        if col in df.columns:
            prices_full = df[col].values.astype(float)
            break
    else:
        prices_full = df.select_dtypes(include=[np.number]).iloc[:, -1].values

    prices_july = prices_full[JULY_START:JULY_END]
    print(f"\n  July prices: mean={prices_july.mean():.1f}  "
          f"std={prices_july.std():.1f}  "
          f"range=[{prices_july.min():.1f}, {prices_july.max():.1f}] EUR/MWh")

    # ── Shi fit ─────────────────────────────────────────────────────────
    shi_fit = fit_shi_polynomial(SOC_MIN, SOC_MAX,
                                 source="v4-vs-Path3 comparison", verbose=False)

    # ══════════════════════════════════════════════════════════════════════
    # v4 baseline: extract July from full-year LP dispatch
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("v4 Architecture: LP dispatch (July slice from full-year)")
    print(f"{'─'*72}")

    v4_e_full = np.load(_RESULTS_DIR / "storage_e_fixed.npy")
    print(f"  Loaded storage_e_fixed.npy: shape={v4_e_full.shape}")

    # Slice July + one extra point for periodicity within the month
    v4_e_july = v4_e_full[JULY_START:JULY_END]
    # Append first point for rainflow periodicity
    v4_e_july = np.append(v4_e_july, v4_e_july[0])

    print(f"  July slice: hours {JULY_START}–{JULY_END}  "
          f"({JULY_HOURS} hours)")
    print(f"  SoC range: [{v4_e_july.min():.1f}, {v4_e_july.max():.1f}] MWh")

    # Revenue from v4 dispatch (reconstruct from SoC changes)
    delta_e = np.diff(v4_e_july[:JULY_HOURS+1])
    # Positive delta = charging, negative = discharging
    # Using v4 convention: eff_in=1.0, eff_out=rte_ac (~0.9025)
    # But for fair comparison, use same convention as Path 3
    v4_c = np.maximum(delta_e, 0) / 0.95   # charge power
    v4_d = np.maximum(-delta_e, 0) * 0.95   # discharge power
    v4_p = v4_d - v4_c
    v4_revenue = float(np.sum(prices_july * v4_p))

    # Shi degradation on v4 July dispatch
    f_v4_shi, cyc_v4 = compute_f_deg(v4_e_july, E_CAP, shi_fit, T_CELL)
    deg_cost_v4_shi = B * E_CAP * f_v4_shi

    # Calendar aging for July (31 days)
    sigma_mean_v4 = float(np.mean(v4_e_july)) / E_CAP
    t_july_sec = JULY_HOURS * 3600.0
    fd_cal_v4 = float(ft_calendar(t_july_sec, sigma_mean_v4, T_CELL))
    fd_total_v4 = f_v4_shi + fd_cal_v4
    deg_cost_v4_total = B * E_CAP * fd_total_v4
    cal_pct_v4 = 100 * fd_cal_v4 / fd_total_v4

    # EFC for v4
    efc_v4 = sum(c["dod"] * c["count"] for c in cyc_v4)

    print(f"\n  v4 Revenue (July):     {v4_revenue:>12,.2f} EUR")
    print(f"  v4 fd_cycle (Shi):     {f_v4_shi:.6e}")
    print(f"  v4 fd_calendar (Xu):   {fd_cal_v4:.6e}  ({cal_pct_v4:.1f}%)")
    print(f"  v4 fd_total:           {fd_total_v4:.6e}")
    print(f"  v4 Deg cost (cycle):   {deg_cost_v4_shi:>12,.2f} EUR")
    print(f"  v4 Deg cost (total):   {deg_cost_v4_total:>12,.2f} EUR")
    print(f"  v4 Cycles:             {len(cyc_v4)}")
    print(f"  v4 EFC:                {efc_v4:.1f}")

    # ══════════════════════════════════════════════════════════════════════
    # Path 3: load NLP results for July
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("Path 3 Architecture: NLP dispatch (degradation in objective)")
    print(f"{'─'*72}")

    # Find the most recent Path 3 July results
    p3_files = sorted(_PATH3_DIR.glob("*_m07_nlp_e.npy"))
    if not p3_files:
        print("  ERROR: No Path 3 July results found in Results_Path3/")
        print("  Run: python path3_wp2_runner.py --year 2022 --month 7")
        return

    p3_prefix = p3_files[-1].name.replace("_nlp_e.npy", "")
    print(f"  Loading: {p3_prefix}")

    p3_e = np.load(_PATH3_DIR / f"{p3_prefix}_nlp_e.npy")
    p3_c = np.load(_PATH3_DIR / f"{p3_prefix}_nlp_c.npy")
    p3_d = np.load(_PATH3_DIR / f"{p3_prefix}_nlp_d.npy")
    p3_p = p3_d - p3_c
    p3_revenue = float(np.sum(prices_july * p3_p))

    # Shi degradation on Path 3 dispatch
    f_p3_shi, cyc_p3 = compute_f_deg(p3_e, E_CAP, shi_fit, T_CELL)
    deg_cost_p3_shi = B * E_CAP * f_p3_shi

    # Calendar aging (same duration, different mean SoC)
    sigma_mean_p3 = float(np.mean(p3_e)) / E_CAP
    fd_cal_p3 = float(ft_calendar(t_july_sec, sigma_mean_p3, T_CELL))
    fd_total_p3 = f_p3_shi + fd_cal_p3
    deg_cost_p3_total = B * E_CAP * fd_total_p3
    cal_pct_p3 = 100 * fd_cal_p3 / fd_total_p3

    # EFC for Path 3
    efc_p3 = sum(c["dod"] * c["count"] for c in cyc_p3)

    print(f"\n  P3 Revenue (July):     {p3_revenue:>12,.2f} EUR")
    print(f"  P3 fd_cycle (Shi):     {f_p3_shi:.6e}")
    print(f"  P3 fd_calendar (Xu):   {fd_cal_p3:.6e}  ({cal_pct_p3:.1f}%)")
    print(f"  P3 fd_total:           {fd_total_p3:.6e}")
    print(f"  P3 Deg cost (cycle):   {deg_cost_p3_shi:>12,.2f} EUR")
    print(f"  P3 Deg cost (total):   {deg_cost_p3_total:>12,.2f} EUR")
    print(f"  P3 Cycles:             {len(cyc_p3)}")
    print(f"  P3 EFC:                {efc_p3:.1f}")

    # ══════════════════════════════════════════════════════════════════════
    # Also load Path 3's LP baseline for fair comparison
    # ══════════════════════════════════════════════════════════════════════
    p3_lp_files = sorted(_PATH3_DIR.glob("*_m07_lp_e.npy"))
    if p3_lp_files:
        p3_lp_e = np.load(p3_lp_files[-1])
        p3_lp_delta = np.diff(p3_lp_e[:JULY_HOURS+1])
        p3_lp_c = np.maximum(p3_lp_delta, 0) / 0.95
        p3_lp_d = np.maximum(-p3_lp_delta, 0) * 0.95
        p3_lp_p = p3_lp_d - p3_lp_c
        p3_lp_revenue = float(np.sum(prices_july * p3_lp_p))

        f_lp_shi, cyc_lp = compute_f_deg(p3_lp_e, E_CAP, shi_fit, T_CELL)
        deg_cost_lp_shi = B * E_CAP * f_lp_shi
        sigma_lp = float(np.mean(p3_lp_e)) / E_CAP
        fd_cal_lp = float(ft_calendar(t_july_sec, sigma_lp, T_CELL))
        fd_total_lp = f_lp_shi + fd_cal_lp
        deg_cost_lp_total = B * E_CAP * fd_total_lp
        efc_lp = sum(c["dod"] * c["count"] for c in cyc_lp)
        has_p3_lp = True
    else:
        has_p3_lp = False

    # ══════════════════════════════════════════════════════════════════════
    # Comparison table
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"COMPARISON — July 2022  (DK1, 300 MWh / 150 MW)")
    print(f"{'='*72}")

    if has_p3_lp:
        print(f"\n  {'Metric':<28s} {'v4 (LP)':>14s} {'P3 LP':>14s} {'P3 NLP':>14s} {'v4->NLP':>10s}")
        print(f"  {'─'*28} {'─'*14} {'─'*14} {'─'*14} {'─'*10}")
        print(f"  {'Revenue [EUR]':<28s} {v4_revenue:>14,.0f} {p3_lp_revenue:>14,.0f} "
              f"{p3_revenue:>14,.0f} {p3_revenue-v4_revenue:>+10,.0f}")
        print(f"  {'fd_cycle (Shi)':<28s} {f_v4_shi:>14.6e} {f_lp_shi:>14.6e} "
              f"{f_p3_shi:>14.6e}")
        print(f"  {'fd_calendar (Xu)':<28s} {fd_cal_v4:>14.6e} {fd_cal_lp:>14.6e} "
              f"{fd_cal_p3:>14.6e}")
        print(f"  {'fd_total':<28s} {fd_total_v4:>14.6e} {fd_total_lp:>14.6e} "
              f"{fd_total_p3:>14.6e}")
        print(f"  {'Deg cost cycle [EUR]':<28s} {deg_cost_v4_shi:>14,.0f} {deg_cost_lp_shi:>14,.0f} "
              f"{deg_cost_p3_shi:>14,.0f} {deg_cost_p3_shi-deg_cost_v4_shi:>+10,.0f}")
        print(f"  {'Deg cost total [EUR]':<28s} {deg_cost_v4_total:>14,.0f} {deg_cost_lp_total:>14,.0f} "
              f"{deg_cost_p3_total:>14,.0f} {deg_cost_p3_total-deg_cost_v4_total:>+10,.0f}")
        net_v4 = v4_revenue - deg_cost_v4_total
        net_lp = p3_lp_revenue - deg_cost_lp_total
        net_p3 = p3_revenue - deg_cost_p3_total
        print(f"  {'Net utility [EUR]':<28s} {net_v4:>14,.0f} {net_lp:>14,.0f} "
              f"{net_p3:>14,.0f} {net_p3-net_v4:>+10,.0f}")
        print(f"  {'Rainflow cycles':<28s} {len(cyc_v4):>14d} {len(cyc_lp):>14d} "
              f"{len(cyc_p3):>14d}")
        print(f"  {'EFC':<28s} {efc_v4:>14.1f} {efc_lp:>14.1f} "
              f"{efc_p3:>14.1f}")
        print(f"  {'Calendar %':<28s} {cal_pct_v4:>13.1f}% {100*fd_cal_lp/fd_total_lp:>13.1f}% "
              f"{cal_pct_p3:>13.1f}%")
    else:
        print(f"\n  {'Metric':<28s} {'v4 (LP)':>14s} {'P3 NLP':>14s} {'Delta':>10s}")
        print(f"  {'─'*28} {'─'*14} {'─'*14} {'─'*10}")
        print(f"  {'Revenue [EUR]':<28s} {v4_revenue:>14,.0f} {p3_revenue:>14,.0f} "
              f"{p3_revenue-v4_revenue:>+10,.0f}")

    # ── Key metrics ─────────────────────────────────────────────────────
    # Use P3 LP as fair baseline (same solver, same formulation)
    if has_p3_lp:
        sac = p3_lp_revenue - p3_revenue
        sav_cycle = deg_cost_lp_shi - deg_cost_p3_shi
        sav_total = deg_cost_lp_total - deg_cost_p3_total
        net_gain = (p3_revenue - deg_cost_p3_total) - (p3_lp_revenue - deg_cost_lp_total)

        print(f"\n  Revenue sacrifice (LP→NLP):         {sac:>+10,.0f} EUR")
        print(f"  Cycle degradation saving:            {sav_cycle:>+10,.0f} EUR")
        print(f"  Total degradation saving (w/ cal):   {sav_total:>+10,.0f} EUR")
        print(f"  Net utility improvement:             {net_gain:>+10,.0f} EUR")
        if sac > 0:
            print(f"  Cycle saving / sacrifice:            {sav_cycle/sac:.2f}x")
            print(f"  Total saving / sacrifice:            {sav_total/sac:.2f}x")

        # Degradation reduction percentages
        if f_lp_shi > 0:
            print(f"\n  fd_cycle reduction:    {100*(f_lp_shi-f_p3_shi)/f_lp_shi:.1f}%")
        if fd_total_lp > 0:
            print(f"  fd_total reduction:    {100*(fd_total_lp-fd_total_p3)/fd_total_lp:.1f}%")
        if efc_lp > 0:
            print(f"  EFC reduction:         {100*(efc_lp-efc_p3)/efc_lp:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # Comparison plot
    # ══════════════════════════════════════════════════════════════════════
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = _PATH3_DIR
        out_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
        fig.suptitle("v4 LP vs Path 3 NLP — July 2022 DK1  |  300 MWh / 150 MW",
                     fontsize=14, fontweight="bold")

        hours = np.arange(JULY_HOURS)

        # Panel 1: Prices
        ax = axes[0]
        ax.plot(hours, prices_july, color="#2166ac", linewidth=0.5, alpha=0.8)
        ax.set_ylabel("Price [EUR/MWh]", fontsize=11)
        ax.set_title("Day-Ahead Price (DK1 July 2022)", fontsize=12)

        # Panel 2: SoC comparison (v4 vs NLP)
        ax = axes[1]
        ax.plot(hours, v4_e_july[:JULY_HOURS], color="#2166ac", linewidth=0.6,
                label=f"v4 LP  (fd={fd_total_v4:.4e})", alpha=0.85)
        if has_p3_lp:
            ax.plot(hours, p3_lp_e[:JULY_HOURS], color="#666666", linewidth=0.5,
                    label=f"P3 LP  (fd={fd_total_lp:.4e})", alpha=0.5, linestyle="--")
        ax.plot(hours, p3_e[:JULY_HOURS], color="#b5351b", linewidth=0.6,
                label=f"P3 NLP (fd={fd_total_p3:.4e})", alpha=0.85)
        ax.axhline(SOC_MIN * E_CAP, color="gray", ls="--", lw=0.5, alpha=0.4)
        ax.axhline(SOC_MAX * E_CAP, color="gray", ls="--", lw=0.5, alpha=0.4)
        ax.set_ylabel("Stored Energy [MWh]", fontsize=11)
        ax.set_title("Battery SoC — v4 LP vs Path 3 NLP", fontsize=12)
        ax.legend(fontsize=9, loc="upper right")

        # Panel 3: Dispatch difference
        ax = axes[2]
        if has_p3_lp:
            diff = p3_p - (p3_lp_d - p3_lp_c)
            ax.fill_between(hours, diff, 0, where=diff > 0,
                            color="#2166ac", alpha=0.3, label="NLP discharges more")
            ax.fill_between(hours, diff, 0, where=diff < 0,
                            color="#b5351b", alpha=0.3, label="NLP charges more")
        ax.set_ylabel("$\\Delta$ Power [MW]", fontsize=11)
        ax.set_xlabel("Hour of July", fontsize=11)
        ax.set_title("Dispatch Difference (NLP - LP)", fontsize=12)
        ax.legend(fontsize=9)

        plt.tight_layout()
        png_path = out_dir / f"{timestamp}_v4_vs_path3_july2022.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"\n  Saved: {png_path.name}")

    except Exception as exc:
        print(f"\n  Plot failed: {exc}")
        import traceback; traceback.print_exc()

    # ── Save summary ────────────────────────────────────────────────────
    txt_path = _PATH3_DIR / f"{timestamp}_v4_vs_path3_july2022_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("v4 vs Path 3 — July 2022 Comparison\n")
        f.write("=" * 55 + "\n\n")
        if has_p3_lp:
            f.write(f"{'Metric':<28s} {'P3 LP':>12s} {'P3 NLP':>12s} {'Delta':>10s}\n")
            f.write(f"{'─'*28} {'─'*12} {'─'*12} {'─'*10}\n")
            f.write(f"{'Revenue [EUR]':<28s} {p3_lp_revenue:>12,.0f} {p3_revenue:>12,.0f} "
                    f"{p3_revenue-p3_lp_revenue:>+10,.0f}\n")
            f.write(f"{'fd_cycle':<28s} {f_lp_shi:>12.4e} {f_p3_shi:>12.4e}\n")
            f.write(f"{'fd_calendar':<28s} {fd_cal_lp:>12.4e} {fd_cal_p3:>12.4e}\n")
            f.write(f"{'fd_total':<28s} {fd_total_lp:>12.4e} {fd_total_p3:>12.4e}\n")
            f.write(f"{'Deg cost total [EUR]':<28s} {deg_cost_lp_total:>12,.0f} "
                    f"{deg_cost_p3_total:>12,.0f} "
                    f"{deg_cost_p3_total-deg_cost_lp_total:>+10,.0f}\n")
            f.write(f"{'Net utility [EUR]':<28s} {net_lp:>12,.0f} {net_p3:>12,.0f} "
                    f"{net_p3-net_lp:>+10,.0f}\n")
            f.write(f"{'Cycles':<28s} {len(cyc_lp):>12d} {len(cyc_p3):>12d}\n")
            f.write(f"{'EFC':<28s} {efc_lp:>12.1f} {efc_p3:>12.1f}\n\n")
            f.write(f"Revenue sacrifice:    {sac:>+10,.0f} EUR\n")
            f.write(f"Total deg saving:     {sav_total:>+10,.0f} EUR\n")
            f.write(f"Net improvement:      {net_gain:>+10,.0f} EUR\n")
            if sac > 0:
                f.write(f"Ratio:                {sav_total/sac:.2f}x\n")
    print(f"  Saved: {txt_path.name}")

    print(f"\n{'='*72}")
    print("Comparison complete.")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
