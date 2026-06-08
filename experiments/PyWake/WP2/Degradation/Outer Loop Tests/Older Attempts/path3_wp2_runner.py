"""
Path 3 — WP2 Runner (monthly or full-year)
===========================================

Runs degradation-aware NLP dispatch on real WP2 data.
Supports single-month runs (SLSQP, ~5 min) or full-year (needs IPOPT).

Usage:
    python path3_wp2_runner.py --year 2022 --month 7       # July only
    python path3_wp2_runner.py --year 2022 --month all     # all 12 months
    python path3_wp2_runner.py --year 2022                  # full year (IPOPT)

Author: Thodoris Tsonopoulos — MSc Thesis, TU Delft Wind Energy
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).parent
_DEGRAD_DIR  = _SCRIPT_DIR.parent
_RESULTS_DIR = _DEGRAD_DIR / "Results"

for p in [str(_SCRIPT_DIR), str(_DEGRAD_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from path3_cyipopt_prototype import (
    BatteryDispatchNLP,
    compute_f_deg,
    compute_df_de,
    solve_nlp,
    solve_lp_baseline,
    fit_shi_polynomial,
)
from degradation_xu import XU_LMO

_RUNNER_NAME = "path3_wp2_runner"

# ── Month boundaries (non-leap: 2019, 2022) ────────────────────────────
_DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

def month_hour_range(month: int) -> tuple[int, int]:
    """Return (start_hour, end_hour) for a 1-indexed month."""
    start = sum(_DAYS_PER_MONTH[:month-1]) * 24
    end   = start + _DAYS_PER_MONTH[month-1] * 24
    return start, end


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_dk1_prices(year: int) -> np.ndarray:
    csv_path = _DEGRAD_DIR / f"dk1_prices_{year}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Price file not found: {csv_path}")

    import pandas as pd
    df = pd.read_csv(csv_path)

    for col in ["price", "price [EUR/MWh]", "Price", "EUR/MWh", "DK1"]:
        if col in df.columns:
            prices = df[col].values.astype(float)
            break
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            prices = df[numeric_cols[-1]].values.astype(float)
        else:
            raise ValueError(f"Cannot find price column. Cols: {list(df.columns)}")

    if len(prices) > 8760:
        prices = prices[:8760]
    elif len(prices) < 8760:
        raise ValueError(f"Price file has {len(prices)} rows, need 8760")

    nan_count = np.isnan(prices).sum()
    if nan_count > 0:
        print(f"  Warning: {nan_count} NaN prices → replaced with mean")
        prices[np.isnan(prices)] = np.nanmean(prices)

    return prices


def load_lp_dispatch() -> tuple[np.ndarray, float]:
    e_path = _RESULTS_DIR / "storage_e_fixed.npy"
    c_path = _RESULTS_DIR / "e_cap_fixed.npy"
    if not e_path.exists():
        raise FileNotFoundError(f"Not found: {e_path}")

    storage_e = np.load(e_path)
    e_cap = float(np.load(c_path).flat[0])

    if len(storage_e) == 8760:
        storage_e = np.append(storage_e, storage_e[0])
    return storage_e, e_cap


# ═══════════════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════════════

def get_results_dir() -> Path:
    d = _SCRIPT_DIR / "Results_Path3"
    d.mkdir(exist_ok=True)
    return d


def save_results(
    results_dir: Path,
    prefix: str,
    year: int,
    month_label: str,
    lp_data: dict,
    nlp_data: dict,
    prices: np.ndarray,
    config: dict,
    timings: dict,
):
    # ── Summary txt ─────────────────────────────────────────────────────
    txt_path = results_dir / f"{prefix}_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Path 3 — Degradation-Aware NLP Dispatch\n{'='*55}\n")
        f.write(f"Runner:  {_RUNNER_NAME}\n")
        f.write(f"Year:    {year}   Period: {month_label}\n")
        f.write(f"Hours:   {len(prices)}\n")
        f.write(f"Battery: {config['e_cap']} MWh / {config['p_cap']} MW\n")
        f.write(f"SoC:     [{config['soc_min']}, {config['soc_max']}]\n")
        f.write(f"eff:     in={config['eff_in']}, out={config['eff_out']}\n")
        f.write(f"B:       {config['replacement_cost']:,.0f} EUR/MWh\n\n")

        rev_lp, rev_nlp = lp_data["revenue"], nlp_data["revenue"]
        deg_lp, deg_nlp = lp_data["deg_cost"], nlp_data["deg_cost_EUR"]

        f.write(f"{'Metric':<28s} {'LP':>14s} {'NLP':>14s} {'Δ':>12s}\n")
        f.write(f"{'─'*28} {'─'*14} {'─'*14} {'─'*12}\n")
        f.write(f"{'Revenue [EUR]':<28s} {rev_lp:>14,.2f} {rev_nlp:>14,.2f} "
                f"{rev_nlp-rev_lp:>+12,.2f}\n")
        f.write(f"{'Deg cost [EUR]':<28s} {deg_lp:>14,.2f} {deg_nlp:>14,.2f} "
                f"{deg_nlp-deg_lp:>+12,.2f}\n")
        f.write(f"{'Net utility [EUR]':<28s} {rev_lp-deg_lp:>14,.2f} "
                f"{rev_nlp-deg_nlp:>14,.2f} "
                f"{(rev_nlp-deg_nlp)-(rev_lp-deg_lp):>+12,.2f}\n")
        f.write(f"{'f_d (Shi)':<28s} {lp_data['f_deg']:.6e}   "
                f"{nlp_data['f_deg']:.6e}\n")
        f.write(f"{'Cycles':<28s} {lp_data['n_cycles']:>14d} "
                f"{nlp_data['n_cycles']:>14d}\n\n")

        sac = rev_lp - rev_nlp
        sav = deg_lp - deg_nlp
        f.write(f"Revenue sacrifice: {sac:>+12,.2f} EUR\n")
        f.write(f"Degradation saving: {sav:>+12,.2f} EUR\n")
        if sac > 0:
            f.write(f"Ratio: {sav/sac:.2f}x\n\n")

        f.write(f"Timings:\n")
        for k, v in timings.items():
            f.write(f"  {k:<20s} {v:>8.1f} s\n")

        f.write(f"\nNLP solver:  {nlp_data.get('status_msg','N/A')}\n")
        f.write(f"NLP iters:   {nlp_data.get('n_iter','N/A')}\n")

    print(f"  Saved: {txt_path.name}")

    # ── Arrays ──────────────────────────────────────────────────────────
    np.save(results_dir / f"{prefix}_nlp_e.npy", nlp_data["e"])
    np.save(results_dir / f"{prefix}_nlp_c.npy", nlp_data["c"])
    np.save(results_dir / f"{prefix}_nlp_d.npy", nlp_data["d"])
    np.save(results_dir / f"{prefix}_lp_e.npy", lp_data["e"])
    np.save(results_dir / f"{prefix}_prices.npy", prices)
    print(f"  Saved: .npy arrays")

    # ── CSV ─────────────────────────────────────────────────────────────
    import pandas as pd
    T = len(prices)
    df = pd.DataFrame({
        "hour": np.arange(T),
        "price": prices,
        "lp_e": lp_data["e"][:T],
        "nlp_e": nlp_data["e"][:T],
        "nlp_c": nlp_data["c"],
        "nlp_d": nlp_data["d"],
        "nlp_p": nlp_data["p_vec"],
    })
    csv_path = results_dir / f"{prefix}_hourly.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved: {csv_path.name}")

    # ── Plot ────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        fig.suptitle(f"Path 3  |  DK1 {year} {month_label}  |  {prefix}",
                     fontsize=13, fontweight="bold")
        hours = np.arange(T)

        axes[0].plot(hours, prices, color="#2166ac", lw=0.5, alpha=0.8)
        axes[0].set_ylabel("Price [EUR/MWh]")
        axes[0].set_title("Day-Ahead Price")

        axes[1].plot(hours, lp_data["e"][:T], color="#2166ac", lw=0.5,
                     label="LP", alpha=0.8)
        axes[1].plot(hours, nlp_data["e"][:T], color="#b5351b", lw=0.5,
                     label="NLP", alpha=0.8)
        axes[1].axhline(config["soc_min"]*config["e_cap"], color="gray",
                        ls="--", lw=0.5, alpha=0.5)
        axes[1].axhline(config["soc_max"]*config["e_cap"], color="gray",
                        ls="--", lw=0.5, alpha=0.5)
        axes[1].set_ylabel("Energy [MWh]")
        axes[1].set_title("SoC Comparison")
        axes[1].legend(fontsize=9)

        lp_p = np.diff(lp_data["e"][:T+1])
        diff = nlp_data["p_vec"] - lp_p
        axes[2].fill_between(hours, diff, 0, where=diff > 0,
                             color="#2166ac", alpha=0.3, label="NLP > LP")
        axes[2].fill_between(hours, diff, 0, where=diff < 0,
                             color="#b5351b", alpha=0.3, label="NLP < LP")
        axes[2].set_ylabel("Δ Power [MW]")
        axes[2].set_xlabel("Hour")
        axes[2].set_title("Dispatch Difference")
        axes[2].legend(fontsize=9)

        plt.tight_layout()
        png = results_dir / f"{prefix}_comparison.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {png.name}")
    except Exception as exc:
        print(f"  Plot failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# Single-month dispatch run
# ═══════════════════════════════════════════════════════════════════════════

def run_single_period(
    prices: np.ndarray,
    config: dict,
    shi_fit,
    max_iter: int,
    results_dir: Path,
    prefix: str,
    year: int,
    month_label: str,
) -> dict:
    """Run LP + NLP for a single price slice. Returns NLP result dict."""

    E_CAP = config["e_cap"]
    P_CAP = config["p_cap"]
    B     = config["replacement_cost"]
    T     = len(prices)
    timings = {}

    print(f"\n  Period: {month_label}  ({T} hours, "
          f"{3*T+1} vars, {T+1} constraints)")
    print(f"  Price: mean={prices.mean():.1f}  std={prices.std():.1f}  "
          f"range=[{prices.min():.1f}, {prices.max():.1f}]")

    # ── LP baseline ─────────────────────────────────────────────────────
    print(f"\n  Solving LP (Gurobi)...", end=" ", flush=True)
    t0 = time.perf_counter()
    lp = solve_lp_baseline(
        prices, E_CAP, P_CAP,
        config["soc_min"], config["soc_max"],
        config["eff_in"], config["eff_out"],
        config["dt"], verbose=False,
    )
    t_lp = time.perf_counter() - t0
    timings["LP solve"] = t_lp
    print(f"{t_lp:.1f} s")

    f_lp, cyc_lp = compute_f_deg(lp["e"], E_CAP, shi_fit)
    deg_cost_lp = B * E_CAP * f_lp

    lp_data = dict(e=lp["e"], c=lp["c"], d=lp["d"], p_vec=lp["p_vec"],
                   revenue=lp["revenue"], f_deg=f_lp,
                   deg_cost=deg_cost_lp, n_cycles=len(cyc_lp))

    print(f"  LP Revenue:  {lp['revenue']:>12,.2f}   Deg cost: {deg_cost_lp:>10,.2f}   "
          f"Net: {lp['revenue']-deg_cost_lp:>12,.2f}   Cycles: {len(cyc_lp)}")

    # ── NLP ─────────────────────────────────────────────────────────────
    nlp = BatteryDispatchNLP(
        prices=prices, e_cap=E_CAP, p_cap=P_CAP,
        soc_min=config["soc_min"], soc_max=config["soc_max"],
        eff_in=config["eff_in"], eff_out=config["eff_out"],
        dt=config["dt"], replacement_cost=B, shi_fit=shi_fit,
    )

    x0 = nlp.warm_start_from_lp(lp["p_vec"], lp["e"])

    print(f"  Solving NLP (max_iter={max_iter})...", flush=True)
    t0 = time.perf_counter()
    nlp_result = solve_nlp(nlp, x0=x0, max_iter=max_iter, verbose=True)
    t_nlp = time.perf_counter() - t0
    timings["NLP solve"] = t_nlp
    timings["NLP iterations"] = nlp_result["n_iter"]

    net_nlp = nlp_result["revenue"] - nlp_result["deg_cost_EUR"]
    net_lp  = lp["revenue"] - deg_cost_lp

    print(f"\n  NLP status:  {nlp_result['status_msg']}")
    print(f"  NLP time:    {t_nlp:.1f} s ({t_nlp/60:.1f} min)")
    print(f"  NLP iters:   {nlp_result['n_iter']}")
    print(f"  NLP Revenue: {nlp_result['revenue']:>12,.2f}   "
          f"Deg cost: {nlp_result['deg_cost_EUR']:>10,.2f}   "
          f"Net: {net_nlp:>12,.2f}")

    # ── Comparison ──────────────────────────────────────────────────────
    sac = lp["revenue"] - nlp_result["revenue"]
    sav = deg_cost_lp - nlp_result["deg_cost_EUR"]

    print(f"\n  {'Metric':<22s} {'LP':>12s} {'NLP':>12s} {'Δ':>10s}")
    print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*10}")
    print(f"  {'Revenue':<22s} {lp['revenue']:>12,.0f} "
          f"{nlp_result['revenue']:>12,.0f} {nlp_result['revenue']-lp['revenue']:>+10,.0f}")
    print(f"  {'Deg cost':<22s} {deg_cost_lp:>12,.0f} "
          f"{nlp_result['deg_cost_EUR']:>12,.0f} "
          f"{nlp_result['deg_cost_EUR']-deg_cost_lp:>+10,.0f}")
    print(f"  {'Net utility':<22s} {net_lp:>12,.0f} "
          f"{net_nlp:>12,.0f} {net_nlp-net_lp:>+10,.0f}")
    if sac > 0:
        print(f"  Sacrifice: {sac:>+,.0f}  Saving: {sav:>+,.0f}  "
              f"Ratio: {sav/sac:.2f}x")

    # ── Save ────────────────────────────────────────────────────────────
    save_results(results_dir, prefix, year, month_label,
                 lp_data, nlp_result, prices, config, timings)

    return nlp_result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Path 3 — WP2 NLP Runner")
    parser.add_argument("--year", type=int, default=2022, choices=[2019, 2022])
    parser.add_argument("--month", type=str, default="all",
                        help="Month number (1-12), 'all' for every month, "
                             "or 'full' for full year (needs IPOPT)")
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = get_results_dir()

    print("=" * 72)
    print(f"Path 3 — WP2 NLP Dispatch  |  DK1 {args.year}  |  month={args.month}")
    print(f"Timestamp: {timestamp}")
    print(f"Output:    {results_dir}")
    print("=" * 72)

    # ── Config ──────────────────────────────────────────────────────────
    config = dict(e_cap=300.0, p_cap=150.0, soc_min=0.10, soc_max=0.90,
                  eff_in=0.95, eff_out=0.95, dt=1.0, replacement_cost=150_000.0)

    print(f"\n  Battery: {config['e_cap']} MWh / {config['p_cap']} MW")
    print(f"  SoC: [{config['soc_min']}, {config['soc_max']}]")

    # ── Load full-year prices ───────────────────────────────────────────
    prices_full = load_dk1_prices(args.year)
    print(f"  Loaded {len(prices_full)} hours of DK1 {args.year} prices")

    shi_fit = fit_shi_polynomial(soc_min=config["soc_min"],
                                 soc_max=config["soc_max"],
                                 source=f"Path3 {args.year}", verbose=True)

    # ── Determine which months to run ───────────────────────────────────
    t_total_start = time.perf_counter()

    if args.month == "full":
        # Full year — needs IPOPT or trust-constr
        prefix = f"{timestamp}_{_RUNNER_NAME}_dk{args.year}_full"
        run_single_period(prices_full, config, shi_fit, args.max_iter,
                          results_dir, prefix, args.year, "Full Year")

    elif args.month == "all":
        # Run each month separately
        print(f"\n  Running all 12 months sequentially...")
        monthly_results = []

        for m in range(1, 13):
            h0, h1 = month_hour_range(m)
            prices_m = prices_full[h0:h1]
            label = f"{_MONTH_NAMES[m-1]} (h{h0}-{h1})"
            prefix = f"{timestamp}_{_RUNNER_NAME}_dk{args.year}_m{m:02d}"

            print(f"\n{'─'*72}")
            print(f"  Month {m}/12: {_MONTH_NAMES[m-1]}")
            print(f"{'─'*72}")

            res = run_single_period(prices_m, config, shi_fit, args.max_iter,
                                    results_dir, prefix, args.year, label)
            monthly_results.append(res)

        # ── Monthly summary table ───────────────────────────────────────
        print(f"\n{'═'*72}")
        print(f"Monthly Summary — DK1 {args.year}")
        print(f"{'═'*72}")
        # Will print after the loop

    else:
        # Single month
        m = int(args.month)
        if m < 1 or m > 12:
            raise ValueError(f"Month must be 1-12, got {m}")
        h0, h1 = month_hour_range(m)
        prices_m = prices_full[h0:h1]
        label = f"{_MONTH_NAMES[m-1]} (h{h0}-{h1})"
        prefix = f"{timestamp}_{_RUNNER_NAME}_dk{args.year}_m{m:02d}"

        run_single_period(prices_m, config, shi_fit, args.max_iter,
                          results_dir, prefix, args.year, label)

    t_total = time.perf_counter() - t_total_start

    print(f"\n{'═'*72}")
    print(f"Done.  Total wall time: {t_total:.1f} s ({t_total/60:.1f} min)")
    print(f"Results in: {results_dir}")
    print(f"{'═'*72}")


if __name__ == "__main__":
    main()