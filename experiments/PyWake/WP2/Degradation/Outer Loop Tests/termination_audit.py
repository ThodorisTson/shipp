"""
termination_audit.py

Cross-tabulates how the solver stopped against the convergence band, for the
365-day monolithic NLP run.

Reads the termination reason from summary.nlp_status in each per-day
*_results.json, and joins it to path3_summary_table.csv on the case name so
that the day set, the optimality values and the idle-day exclusion match the
numbers already reported in Section 4.5 exactly.

Run from the "Outer Loop Tests" folder:
    python termination_audit.py
    python termination_audit.py --results ".\\Results_Path3_AllDays\\run_20260706_010250"
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Days on which the battery stayed idle; excluded from the 356 active days.
IDLE_DAYS = {"D024", "D027", "D045", "D094", "D098",
             "D279", "D315", "D320", "D321"}

CONV, DIV = 1e-6, 1e-2

# scipy.optimize trust-constr status codes
STATUS_CODES = {
    0: "maxiter",
    1: "gtol",
    2: "xtol",
    3: "callback",
}


def classify_status(raw) -> str:
    """Map summary.nlp_status to gtol / xtol / maxiter / callback."""
    if isinstance(raw, bool):
        return "?"
    if isinstance(raw, int):
        return STATUS_CODES.get(raw, f"code {raw}")
    m = str(raw or "").lower()
    if "gtol" in m:
        return "gtol"
    if "xtol" in m:
        return "xtol"
    if "maxim" in m or "maxiter" in m:
        return "maxiter"
    if "callback" in m:
        return "callback"
    return f"unmapped: {str(raw)[:50]}"


def band(optimality: float) -> str:
    if optimality < CONV:
        return "Converged"
    if optimality < DIV:
        return "Stalled"
    return "Diverged"


def parse_case(fname: str) -> str:
    m = re.search(r"_dk\d{4}_([A-Za-z0-9]+)_", fname)
    return m.group(1) if m else "?"


def parse_ts(fname: str) -> str:
    m = re.match(r"(\d{8}_\d{6})_", fname)
    return m.group(1) if m else "0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None,
                    help="run folder; default is the latest under Results_Path3_AllDays")
    ap.add_argument("--csv", default=None,
                    help="path3_summary_table.csv; default is inside the run folder")
    args = ap.parse_args()

    if args.results:
        results_dir = Path(args.results)
    else:
        runs = sorted(HERE.glob("Results_Path3_AllDays/run_*"))
        if not runs:
            raise FileNotFoundError("no run folder found; pass --results")
        results_dir = runs[-1]
    csv_path = Path(args.csv) if args.csv else results_dir / "path3_summary_table.csv"

    print(f"run folder : {results_dir.name}")
    print(f"csv        : {csv_path.name}")

    # --- termination reason and iteration count per case, latest run wins ----
    status, iters_json, seen_ts = {}, {}, {}
    n_json = 0
    for jp in results_dir.glob("*_results.json"):
        try:
            s = json.load(open(jp, encoding="utf-8"))["summary"]
        except Exception:
            continue
        n_json += 1
        case, ts = parse_case(jp.name), parse_ts(jp.name)
        if case in seen_ts and ts <= seen_ts[case]:
            continue
        seen_ts[case] = ts
        status[case] = classify_status(s.get("nlp_status"))
        iters_json[case] = int(s.get("nlp_iters", -1))
    print(f"json files read: {n_json}, unique cases: {len(status)}\n")

    # --- join to the CSV -----------------------------------------------------
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    active = [r for r in rows if r["case"] not in IDLE_DAYS]

    missing = [r["case"] for r in active if r["case"] not in status]
    if missing:
        print(f"WARNING: no json for {len(missing)} active cases: "
              f"{missing[:8]}{' ...' if len(missing) > 8 else ''}\n")

    table = {b: Counter() for b in ("Converged", "Stalled", "Diverged")}
    mismatch = 0
    for r in active:
        c = r["case"]
        if c not in status:
            continue
        b = band(float(r["optimality"]))
        table[b][status[c]] += 1
        if iters_json[c] != int(r["iters"]):
            mismatch += 1

    if mismatch:
        print(f"WARNING: iteration count differs between json and csv on "
              f"{mismatch} cases; check you are reading the same run.\n")

    reasons = sorted({k for b in table.values() for k in b})
    width = max(len(x) for x in reasons + ["Converged"]) + 2

    print("Termination reason by convergence band, active days only")
    print(f"{'Band':<12}" + "".join(f"{x:>{width}}" for x in reasons) + f"{'Total':>8}")
    print("-" * (12 + width * len(reasons) + 8))
    for b in ("Converged", "Stalled", "Diverged"):
        tot = sum(table[b].values())
        print(f"{b:<12}" + "".join(f"{table[b][x]:>{width}}" for x in reasons)
              + f"{tot:>8}")

    nonconv = Counter()
    for b in ("Stalled", "Diverged"):
        nonconv.update(table[b])
    n_nc = sum(nonconv.values())
    print("-" * (12 + width * len(reasons) + 8))
    print(f"{'Not conv.':<12}" + "".join(f"{nonconv[x]:>{width}}" for x in reasons)
          + f"{n_nc:>8}")

    print()
    if n_nc:
        xtol = nonconv.get("xtol", 0)
        print(f"SENTENCE DATA: of the {n_nc} days that did not converge, "
              f"{xtol} stopped on step collapse (xtol) and "
              f"{nonconv.get('maxiter', 0)} at the iteration limit "
              f"({100 * xtol / n_nc:.0f}% step collapse).")
    print(f"               stalled: {sum(table['Stalled'].values())}, "
          f"of which xtol {table['Stalled'].get('xtol', 0)}")
    print(f"               diverged: {sum(table['Diverged'].values())}, "
          f"of which xtol {table['Diverged'].get('xtol', 0)}")


if __name__ == "__main__":
    main()
