"""
Path 3 — Alpha Sweep: Pareto Front (Revenue vs Degradation)
============================================================

Sweeps degradation weight alpha from 0 (pure revenue LP) to 1 (full w_deg),
tracing the revenue-degradation tradeoff. Each NLP warm-starts from the
previous solution.

Imports LP building and degradation functions from path3_jenna.py.

Usage:
    python path3_alpha_sweep.py --year 2019 --month 7 --max-iter 50
    python path3_alpha_sweep.py --test --max-iter 50
    python path3_alpha_sweep.py --year 2019 --month 7 --solver slsqp --points 7

Author: Thodoris Tsonopoulos -- MSc Thesis, TU Delft Wind Energy
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, LinearConstraint

from path3_jenna import (
    build_lp_problem,
    compute_f_deg,
    make_nlp_functions,
    load_dk1_prices,
    month_hour_range,
    slot_hour_range,
    WEEK_SLOTS,
    _MONTH_NAMES,
    _RUNNER_NAME,
    _SCRIPT_DIR,
)
from degradation_xu import fit_shi_polynomial


def run_alpha_sweep(
    prices: np.ndarray,
    config: dict,
    shi_fit,
    max_iter: int,
    results_dir: Path,
    prefix: str,
    year: int,
    month_label: str,
    solver: str = "trust-constr",
    tr_radius: float = None,
    n_points: int = 11,
):
    E_CAP = config["e_cap"]
    B     = config["replacement_cost"]
    dt    = config["dt"]
    T     = len(prices)
    w_full = B * E_CAP #maximum degradation weight

    print(f"\n  Alpha sweep: {n_points} points, solver={solver}")
    print(f"  Period: {month_label}  ({T} hours)")

    # Build LP once
    lp_mats, lp_data = build_lp_problem(prices, config)

    f_lp, cyc_lp = compute_f_deg(lp_data["e"], E_CAP, shi_fit)
    deg_cost_lp = w_full * f_lp

    print(f"  LP: rev={lp_data['revenue']:,.0f}  deg={deg_cost_lp:,.0f}  "
          f"net={lp_data['revenue']-deg_cost_lp:,.0f}  cycles={len(cyc_lp)}")

    # Alpha grid: 0 (LP), then log-spaced from 0.001 to 1
    alphas = np.concatenate(([0.0], np.logspace(-3, 0, n_points - 1))) # gives points at 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0 (logarithmically spaced)
    alphas = np.unique(np.clip(alphas, 0, 1))  # Add alpha=0 at the front (= LP, no degradation penalty)

    results = []
    x_warm = lp_mats["x_lp"].copy() #start from the lp solution

    print(f"\n  {'alpha':>8s} {'Revenue':>12s} {'DegCost':>10s} "
          f"{'Net':>12s} {'Cyc':>5s} {'Iter':>5s} {'Time':>6s} {'Status':>10s}")
    print(f"  {'---'*25}")

    for alpha in alphas:
        w_eff = alpha * w_full

        if alpha == 0:
            rev, f_d, dc, nc = lp_data["revenue"], f_lp, deg_cost_lp, len(cyc_lp)
            ni, ts, st = 0, 0.0, "LP"
            x_warm = lp_mats["x_lp"].copy()
        else:
            obj_fn, grad_fn, state = make_nlp_functions(
                lp_mats["vec_obj"], T, E_CAP, shi_fit, w_eff,
            )

            t0 = time.perf_counter()

            if solver == "trust-constr":
                cst = [
                    LinearConstraint(lp_mats["mat_ineq"], ub=lp_mats["vec_ineq"]),
                    LinearConstraint(lp_mats["mat_eq"],
                                     lb=lp_mats["vec_eq"], ub=lp_mats["vec_eq"]),
                ]
                tc_opts = dict(sparse_jacobian=True, verbose=0, maxiter=max_iter)
                if tr_radius is not None:
                    tc_opts["initial_tr_radius"] = tr_radius

                res = minimize(obj_fn, x_warm, jac=grad_fn, constraints=cst,
                               bounds=lp_mats["bounds_list"],
                               method='trust-constr', tol=1e-6, options=tc_opts)

            elif solver == "slsqp":
                A_eq_d = lp_mats["mat_eq"].toarray()
                A_ub_d = lp_mats["mat_ineq"].toarray()
                cst = [
                    {"type": "eq",
                     "fun": lambda x: A_eq_d @ x - lp_mats["vec_eq"],
                     "jac": lambda x: A_eq_d},
                    {"type": "ineq",
                     "fun": lambda x: lp_mats["vec_ineq"] - A_ub_d @ x,
                     "jac": lambda x: -A_ub_d},
                ]
                res = minimize(obj_fn, x_warm, jac=grad_fn, constraints=cst,
                               bounds=lp_mats["bounds_list"],
                               method='SLSQP',
                               options=dict(maxiter=max_iter, disp=False, ftol=1e-9))

            ts = time.perf_counter() - t0
            nlp_p = res.x[0:T]
            nlp_e = res.x[lp_mats["e1_slice"]]
            rev = float(np.sum(prices[:T] * nlp_p * dt))
            f_d, cyc_n = compute_f_deg(nlp_e, E_CAP, shi_fit)
            dc = w_full * f_d       # always at full weight for fair comparison
            nc = len(cyc_n)
            ni = res.nit
            st = "OK" if res.success else "max_iter"
            x_warm = res.x.copy()

        net = rev - dc
        print(f"  {alpha:8.4f} {rev:>12,.0f} {dc:>10,.0f} "
              f"{net:>12,.0f} {nc:>5d} {ni:>5d} {ts:>6.1f} {st:>10s}")

        results.append(dict( # saves all results as JSON
            alpha=float(alpha), revenue=rev, deg_cost=dc,
            f_deg=f_d, net=net, n_cycles=nc,
            n_iter=ni, t_solve=ts, status=st,
        ))

    # Save JSON
    results_dir.mkdir(exist_ok=True)
    json_path = results_dir / f"{prefix}_alpha_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {json_path.name}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor("#f7f9fc")
        for ax in axes:
            ax.set_facecolor("#f7f9fc")
        fig.suptitle(f"Alpha Sweep  |  DK1 {year} {month_label}  |  {solver}",
                     fontsize=13, fontweight="bold")

        revs = [r["revenue"] for r in results]
        degs = [r["deg_cost"] for r in results]
        nets = [r["net"] for r in results]
        alps = [r["alpha"] for r in results]

        axes[0].scatter(revs, degs, c=alps, cmap="coolwarm", s=60, zorder=3)
        axes[0].plot(revs, degs, color="#2166ac", lw=1, alpha=0.5)
        axes[0].set_xlabel("Revenue [EUR]")
        axes[0].set_ylabel("Degradation Cost [EUR]")
        axes[0].set_title("Pareto Front", fontweight="bold")

        axes[1].plot(alps, nets, "o-", color="#2166ac", markersize=5)
        axes[1].set_xlabel("Alpha")
        axes[1].set_ylabel("Net Utility [EUR]")
        axes[1].set_title("Net Utility vs Weight", fontweight="bold")
        axes[1].set_xscale("symlog", linthresh=1e-3)

        sacs = [revs[0] - r for r in revs]
        savs = [degs[0] - d for d in degs]
        axes[2].scatter(sacs, savs, c=alps, cmap="coolwarm", s=60, zorder=3)
        if max(sacs) > 0:
            axes[2].plot([0, max(sacs)*1.1], [0, max(sacs)*1.1],
                         'k--', lw=0.8, alpha=0.4, label="1:1")
        axes[2].set_xlabel("Revenue Sacrifice [EUR]")
        axes[2].set_ylabel("Degradation Saving [EUR]")
        axes[2].set_title("Sacrifice vs Saving", fontweight="bold")
        axes[2].legend(fontsize=9)

        plt.tight_layout()
        png = results_dir / f"{prefix}_alpha_pareto.png"
        fig.savefig(png, dpi=150, bbox_inches="tight",
                    facecolor="#f7f9fc", edgecolor="none")
        plt.close(fig)
        print(f"  Saved: {png.name}")
    except Exception as exc:
        print(f"  Plot failed: {exc}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Path 3 Alpha Sweep -- Pareto Front")
    parser.add_argument("--year", type=int, default=2019, choices=[2019, 2022])
    parser.add_argument("--month", type=str, default="7")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--solver", type=str, default="trust-constr",
                        choices=["trust-constr", "slsqp"])
    parser.add_argument("--tr-radius", type=float, default=None)
    parser.add_argument("--points", type=int, default=11)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--start-hour", type=int, default=None,
                        help="Start hour (0-indexed) for arbitrary slice")
    parser.add_argument("--n-hours", type=int, default=168,
                        help="Number of hours (default: 168 = 1 week)")
    parser.add_argument("--slot", type=str, default=None,
                        choices=list(WEEK_SLOTS.keys()) + ["all"],
                        help="Predefined week slot (W1,W7,W13,W28,W34,W35), or 'all'")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = _SCRIPT_DIR / "Results_Path3"

    config = dict(
        e_cap=300.0, p_cap=150.0,
        soc_min=0.10, soc_max=0.90,
        eff_in=0.95, eff_out=0.95,
        dt=1.0,
        replacement_cost=150_000.0,
    )

    print("=" * 72)
    print(f"Alpha Sweep  |  {args.solver}  |  {args.points} points  |  max_iter={args.max_iter}")
    print("=" * 72)

    shi_fit = fit_shi_polynomial(
        soc_min=config["soc_min"], soc_max=config["soc_max"],
        source=f"Alpha sweep {args.year}", verbose=True,
    )

    if args.test:
        T = 120
        rng = np.random.RandomState(42)
        hours = np.arange(T) % 24
        prices = 30 + 20*np.sin(2*np.pi*(hours-6)/24) + rng.normal(0, 5, T)
        prices = np.maximum(prices, 0)
        run_alpha_sweep(prices, config, shi_fit, args.max_iter,
                        results_dir, f"{timestamp}_alpha_test", 0, f"Test ({T}h)",
                        solver=args.solver, tr_radius=args.tr_radius,
                        n_points=args.points)
        return

    prices_full = load_dk1_prices(args.year)

    if args.slot is not None:
        slots = list(WEEK_SLOTS.keys()) if args.slot == "all" else [args.slot]
        for sk in slots:
            h0, h1, slabel = slot_hour_range(sk)
            label = f"{sk} {slabel} (h{h0}-{h1})"
            pfx = f"{timestamp}_alpha_dk{args.year}_{sk}_{args.solver}"
            print(f"\n{'---'*24}")
            print(f"  Slot {sk}: {slabel}")
            print(f"{'---'*24}")
            run_alpha_sweep(prices_full[h0:h1], config, shi_fit, args.max_iter,
                            results_dir, pfx, args.year, label,
                            solver=args.solver, tr_radius=args.tr_radius,
                            n_points=args.points)

    elif args.start_hour is not None:
        h0 = args.start_hour
        h1 = min(h0 + args.n_hours, 8760)
        label = f"Custom h{h0}-{h1} ({h1-h0}h)"
        pfx = f"{timestamp}_alpha_dk{args.year}_h{h0}_{h1-h0}h_{args.solver}"
        run_alpha_sweep(prices_full[h0:h1], config, shi_fit, args.max_iter,
                        results_dir, pfx, args.year, label,
                        solver=args.solver, tr_radius=args.tr_radius,
                        n_points=args.points)

    else:
        m = int(args.month)
        h0, h1 = month_hour_range(m)
        label = f"{_MONTH_NAMES[m-1]} (h{h0}-{h1})"
        pfx = f"{timestamp}_alpha_dk{args.year}_m{m:02d}_{args.solver}"
        run_alpha_sweep(prices_full[h0:h1], config, shi_fit, args.max_iter,
                        results_dir, pfx, args.year, label,
                        solver=args.solver, tr_radius=args.tr_radius,
                        n_points=args.points)


if __name__ == "__main__":
    main()
