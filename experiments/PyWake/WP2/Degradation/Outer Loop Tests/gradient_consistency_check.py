"""
gradient_consistency_check.py

Checks the gradient handed to trust-constr against a central finite difference of the objective it is supposed to differentiate.

Both functions come from path3.py:
  compute_f_deg(e, e_cap, shi_fit)  -> Phi_shi(d_i) * S_sigma(s_i) * S_T * n_i
  compute_df_de(e, cycles, e_cap, shi_fit) -> retains the amplitude term only, holding S_sigma fixed per cycle

The expected result is a systematic, bounded discrepancy at SoC turning points:
larger than the analytic value at peaks (the two terms add) and smaller at troughs (they oppose), by roughly 0.44 * delta.

Run from the "Outer Loop Tests" folder, next to path3.py:
  python gradient_consistency_check.py
  python gradient_consistency_check.py --day D179 --which nlp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from path3 import compute_f_deg, compute_df_de          # noqa: E402
from degradation_xu import fit_shi_polynomial           # noqa: E402


def find_day(results_dir: Path, day: str):
    """Return (json_path, npz_path) for a day label such as 'D179'."""
    n = int("".join(c for c in day if c.isdigit()))
    for pat in (f"*_D{n:03d}_*_results.json", f"*_D{n}_*_results.json"):
        hits = sorted(results_dir.glob(pat))
        if hits:
            jp = hits[-1]
            npz = jp.with_name(jp.name.replace("_results.json", "_results.npz"))
            if npz.exists():
                return jp, npz
    raise FileNotFoundError(f"no results pair for {day} in {results_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None,
                    help="run folder; default is the latest under Results_Path3_AllDays")
    ap.add_argument("--day", default="D179")
    ap.add_argument("--which", default="nlp", choices=["lp", "nlp"],
                    help="which trajectory to differentiate around")
    ap.add_argument("--h-frac", type=float, default=1e-6,
                    help="perturbation as a fraction of pack size")
    args = ap.parse_args()

    if args.results:
        results_dir = Path(args.results)
    else:
        runs = sorted(HERE.glob("Results_Path3_AllDays/run_*"))
        if not runs:
            raise FileNotFoundError("no run folder found; pass --results")
        results_dir = runs[-1]
    print(f"run folder : {results_dir.name}")

    jp, npz = find_day(results_dir, args.day)
    summary = json.load(open(jp, encoding="utf-8"))["summary"]
    z = np.load(npz, allow_pickle=True)

    e_cap = float(summary["e_cap"])
    soc_min = float(summary.get("soc_min", 0.1))
    soc_max = float(summary.get("soc_max", 0.9))
    e = np.asarray(z[f"{args.which}_e"], float).copy()

    shi_fit = fit_shi_polynomial(soc_min=soc_min, soc_max=soc_max,
                                 source="grad check", verbose=False)

    print(f"day        : {args.day}  ({args.which} trajectory, {len(e)} points)")
    print(f"e_cap      : {e_cap:.1f} MWh   window [{soc_min}, {soc_max}]")

    f0, cycles = compute_f_deg(e, e_cap, shi_fit)
    g_an = compute_df_de(e, cycles, e_cap, shi_fit)
    print(f"f_deg      : {f0:.6e}   cycles counted: {len(cycles)}")

    h = args.h_frac * e_cap
    g_fd = np.zeros_like(e)
    for t in range(len(e)):
        ep, em = e.copy(), e.copy()
        ep[t] += h
        em[t] -= h
        fp, _ = compute_f_deg(ep, e_cap, shi_fit)
        fm, _ = compute_f_deg(em, e_cap, shi_fit)
        g_fd[t] = (fp - fm) / (2.0 * h)

    print()
    print(f"{'t':>3}  {'e/e_cap':>8}  {'analytic':>12}  {'central FD':>12}  "
          f"{'FD/analytic':>12}")
    print("-" * 56)
    ratios = []
    for t in range(len(e)):
        a, f = g_an[t], g_fd[t]
        if abs(a) < 1e-15 and abs(f) < 1e-15:
            continue
        r = f / a if abs(a) > 1e-15 else np.nan
        if np.isfinite(r):
            ratios.append(r)
        print(f"{t:>3}  {e[t] / e_cap:>8.4f}  {a:>12.4e}  {f:>12.4e}  {r:>12.3f}")

    ratios = np.asarray(ratios)
    if ratios.size:
        print()
        print(f"nonzero components : {ratios.size}")
        print(f"ratio min / median / max : "
              f"{ratios.min():.3f} / {np.median(ratios):.3f} / {ratios.max():.3f}")
        print(f"components within 1%% of 1.0 : "
              f"{int((np.abs(ratios - 1.0) < 0.01).sum())}")

    denom = max(np.linalg.norm(g_an), 1e-30)
    print(f"relative norm of the difference : "
          f"{np.linalg.norm(g_fd - g_an) / denom:.4f}")
    print()
    print("Expected if the omitted mean-SoC term is local and bounded:")
    print("  ratios above 1 at peaks, below 1 at troughs, spread roughly "
          "0.44 * delta,")
    print("  and a relative norm difference of order 0.1 to 0.3, not 10 or 100.")


if __name__ == "__main__":
    main()
