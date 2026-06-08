"""
LP Weekly Scan — Price & Dispatch Statistics for Slot Selection
===============================================================

Runs independent weekly LPs (168h each) for all 52 weeks of 2019 and 2022,
computes per-week price statistics and post-hoc degradation metrics, and
saves everything to a single CSV for analysis.

Usage:
    python lp_week_scan.py                   # both years
    python lp_week_scan.py --year 2019       # single year
    python lp_week_scan.py --year 2022

Output:
    Results_Path3/lp_week_scan_YYYYMMDD_HHMMSS.csv

Author: Thodoris Tsonopoulos -- MSc Thesis, TU Delft Wind Energy
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# -- Imports from path3_jenna (LP builder + degradation) -------------------
from path3_jenna import (
    build_lp_problem,
    compute_f_deg,
    load_dk1_prices,
    _SCRIPT_DIR,
)
from degradation_xu import fit_shi_polynomial


def scan_year(year: int, config: dict, shi_fit, results: list):
    """Run 52 independent weekly LPs and collect statistics."""
    prices_full = load_dk1_prices(year)
    E_CAP = config["e_cap"]
    B = config["replacement_cost"]
    w_deg = B * E_CAP
    WEEK_H = 168  # 7 days × 24 hours

    n_weeks = 8760 // WEEK_H  # 52 full weeks (8736h), last 24h ignored

    print(f"\n  Year {year}: {n_weeks} weeks × {WEEK_H}h")
    print(f"  {'Wk':>3s} {'h0':>5s} {'h1':>5s} {'Mean':>7s} {'Std':>7s} "
          f"{'Min':>7s} {'Max':>7s} {'Spread':>7s} {'Rev':>10s} "
          f"{'DegC':>9s} {'Net':>10s} {'Cyc':>4s} {'MeanDoD':>8s} {'LP_t':>5s}")
    print(f"  {'---' * 32}")

    for w in range(n_weeks):
        h0 = w * WEEK_H
        h1 = h0 + WEEK_H
        prices_w = prices_full[h0:h1]

        # Price statistics
        p_mean = float(np.mean(prices_w))
        p_std = float(np.std(prices_w))
        p_min = float(np.min(prices_w))
        p_max = float(np.max(prices_w))
        p_spread = p_max - p_min

        # Interquartile spread (robust volatility)
        p_q25, p_q75 = float(np.percentile(prices_w, 25)), float(np.percentile(prices_w, 75))
        p_iqr = p_q75 - p_q25

        # Count of negative-price hours
        n_negative = int(np.sum(prices_w < 0))

        # Run LP
        t0 = time.perf_counter()
        try:
            lp_mats, lp_data = build_lp_problem(prices_w, config)
            t_lp = time.perf_counter() - t0

            # Post-hoc degradation
            f_deg, cycles = compute_f_deg(lp_data["e"], E_CAP, shi_fit)
            deg_cost = w_deg * f_deg
            revenue = lp_data["revenue"]
            n_cycles = len(cycles)

            # Mean DoD of non-trivial cycles
            dods = [c["dod"] for c in cycles if c["dod"] > 0.01]
            mean_dod = float(np.mean(dods)) if dods else 0.0
            max_dod = float(np.max(dods)) if dods else 0.0

            # EFC (equivalent full cycles) = sum(count × dod) for all cycles
            efc = sum(c["count"] * c["dod"] for c in cycles)

            net = revenue - deg_cost
            status = "OK"

        except Exception as exc:
            t_lp = time.perf_counter() - t0
            revenue, deg_cost, net = 0.0, 0.0, 0.0
            n_cycles, mean_dod, max_dod, efc = 0, 0.0, 0.0, 0.0
            f_deg = 0.0
            status = str(exc)[:40]

        print(f"  {w+1:3d} {h0:5d} {h1:5d} {p_mean:7.1f} {p_std:7.1f} "
              f"{p_min:7.1f} {p_max:7.1f} {p_spread:7.1f} {revenue:10,.0f} "
              f"{deg_cost:9,.0f} {net:10,.0f} {n_cycles:4d} {mean_dod:8.3f} {t_lp:5.1f}")

        results.append(dict(
            year=year,
            week=w + 1,
            h_start=h0,
            h_end=h1,
            price_mean=round(p_mean, 2),
            price_std=round(p_std, 2),
            price_min=round(p_min, 2),
            price_max=round(p_max, 2),
            price_spread=round(p_spread, 2),
            price_iqr=round(p_iqr, 2),
            price_q25=round(p_q25, 2),
            price_q75=round(p_q75, 2),
            n_negative_hours=n_negative,
            revenue=round(revenue, 2),
            deg_cost=round(deg_cost, 2),
            f_deg=round(f_deg, 8),
            net_utility=round(net, 2),
            n_cycles=n_cycles,
            mean_dod=round(mean_dod, 4),
            max_dod=round(max_dod, 4),
            efc=round(efc, 4),
            deg_pct_of_rev=round(100 * deg_cost / revenue, 2) if revenue > 0 else 0.0,
            t_lp=round(t_lp, 2),
            status=status,
        ))


def main():
    parser = argparse.ArgumentParser(
        description="LP Weekly Scan -- Slot Selection Diagnostics")
    parser.add_argument("--year", type=int, default=None, choices=[2019, 2022],
                        help="Single year (default: both)")
    args = parser.parse_args()

    years = [args.year] if args.year else [2019, 2022]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = _SCRIPT_DIR / "Results_Path3"
    results_dir.mkdir(exist_ok=True)

    config = dict(
        e_cap=300.0, p_cap=150.0,
        soc_min=0.10, soc_max=0.90,
        eff_in=0.95, eff_out=0.95,
        dt=1.0,
        replacement_cost=150_000.0,
    )

    print("=" * 72)
    print(f"LP Weekly Scan  |  Years: {years}")
    print(f"Battery: {config['e_cap']} MWh / {config['p_cap']} MW  "
          f"SoC: [{config['soc_min']}, {config['soc_max']}]")
    print(f"Timestamp: {timestamp}")
    print("=" * 72)

    shi_fit = fit_shi_polynomial(
        soc_min=config["soc_min"], soc_max=config["soc_max"],
        source="LP week scan", verbose=True,
    )

    results = []
    t0 = time.perf_counter()

    for yr in years:
        scan_year(yr, config, shi_fit, results)

    t_total = time.perf_counter() - t0

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = results_dir / f"lp_week_scan_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  Saved: {csv_path}")
    print(f"  Total: {len(results)} weeks in {t_total:.1f}s ({t_total/60:.1f} min)")

    # Quick summary: top/bottom weeks by spread and by net utility
    for yr in years:
        dfy = df[df["year"] == yr].copy()
        print(f"\n  === {yr} Quick Summary ===")
        print(f"  Revenue range: {dfy['revenue'].min():,.0f} - {dfy['revenue'].max():,.0f}")
        print(f"  Spread range:  {dfy['price_spread'].min():.0f} - {dfy['price_spread'].max():.0f}")
        print(f"  Cycle range:   {dfy['n_cycles'].min()} - {dfy['n_cycles'].max()}")

        top3 = dfy.nlargest(3, "price_spread")[["week", "h_start", "price_spread", "n_cycles", "revenue"]]
        bot3 = dfy.nsmallest(3, "price_spread")[["week", "h_start", "price_spread", "n_cycles", "revenue"]]
        print(f"\n  Highest spread weeks:")
        print(top3.to_string(index=False))
        print(f"\n  Lowest spread weeks:")
        print(bot3.to_string(index=False))

    print(f"\n{'==='*24}")
    print(f"Done. Upload {csv_path.name} for slot selection analysis.")
    print(f"{'==='*24}")


if __name__ == "__main__":
    main()
