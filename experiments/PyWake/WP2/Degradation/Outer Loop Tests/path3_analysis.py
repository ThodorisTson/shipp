"""
path3_analysis.py  —  Path 3 NLP results: summary table + convergence plots
=============================================================================
Replaces summarise_day_study.py AND plot_optimality_explosion.py.

Produces in one pass:
  1. A printed summary table (with solver config column, no ambiguous duplicates)
  2. optimality_trajectories.png  — log-scale KKT optimality per iteration
  3. final_optimality_scatter.png — final opt vs deg/rev, solver as line style

Usage:
    python path3_analysis.py --dir Results_Path3 --hours 24
    python path3_analysis.py --dir Results_Path3 --hours 24 --solver alpha0_1
    python path3_analysis.py --dir Results_Path3 --hours 168
    python path3_analysis.py --dir Results_Path3           # all horizons

Filters:
    --hours   INT    keep only runs with this many hours (24/168/336)
    --solver  STR    keep only runs whose solver tag contains this substring
                     e.g. --solver alpha0_1  or  --solver scaled

Author: Thodoris Tsonopoulos — MSc Thesis, TU Delft Wind Energy
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Aesthetics ────────────────────────────────────────────────────────────────
BG = "#f7f9fc"

# One colour per slot — consistent across all plots
SLOT_COLORS = {
    "D8":   "#b5351b",   # HIGH — red family
    "D238": "#d85a30",
    "D248": "#f09595",
    "D86":  "#2166ac",   # MID — blue family
    "D166": "#185FA5",
    "D309": "#85B7EB",
    "D43":  "#1b9e77",   # LOW — green family
    "D128": "#5DCAA5",
    "D197": "#9FE1CB",
}
DEFAULT_COLOR = "#888888"

# Regime classification (extend if needed)
REGIME = {
    "D8":   "HIGH", "D238": "HIGH", "D248": "HIGH",
    "D86":  "MID",  "D166": "MID",  "D309": "MID",
    "D43":  "LOW",  "D128": "LOW",  "D197": "LOW",
}
REGIME_ORDER = {"HIGH": 0, "MID": 1, "LOW": 2, "?": 3}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _extract_slot(period: str) -> str:
    """Extract slot key from the period string.

    Priority:
      1. Leading D<digits>  e.g. 'D43 Feb 12 ...'  → 'D43'
      2. Embedded W<digits> e.g. '... day 1 of W7'  → 'W7'
         (prefixed with year when available)
      3. Fallback to first 12 chars
    """
    m_day = re.match(r"(D\d+)", period.strip())
    if m_day:
        return m_day.group(1)
    m_week = re.search(r"\b(W\d+)\b", period)
    if m_week:
        return m_week.group(1)
    return period[:12]


def _solver_short(solver_str: str) -> str:
    """Condense 'trust-constr +alpha0.1' → 'α=0.1', etc."""
    s = solver_str.lower()
    m = re.search(r"alpha([0-9.]+)", s)
    if m:
        return f"α={m.group(1)}"
    if "hess" in s:
        return "FD-Hess"
    if "scaled" in s:
        return "scaled"
    if "slsqp" in s:
        return "SLSQP"
    return solver_str[:10]


def load_results(
    results_dir: Path,
    hours_filter: int | None = None,
    solver_filter: str | None = None,
) -> list[dict]:
    """Load all path3 result JSONs from ``results_dir``, return list of row dicts."""
    jsons = sorted(results_dir.glob("*path3*results.json"))
    if not jsons:
        jsons = sorted(results_dir.glob("**/*path3*results.json"))

    rows = []
    skipped = 0
    for jf in jsons:
        try:
            with open(jf, encoding="utf-8") as f:
                d = json.load(f)
            s = d["summary"]

            # hours filter
            if hours_filter is not None and s.get("hours", 0) != hours_filter:
                skipped += 1
                continue

            # solver filter (substring match on solver field)
            solver_str = s.get("solver", "")
            if solver_filter and solver_filter.lower() not in solver_str.lower():
                skipped += 1
                continue

            conv = d["convergence"]
            final_opt = s.get("final_optimality", conv[-1]["optimality"])
            slot = _extract_slot(s["period"])

            rows.append(dict(
                file=jf.name,
                slot=slot,
                regime=REGIME.get(slot, "?"),
                solver=solver_str,
                solver_short=_solver_short(solver_str),
                period=s["period"],
                hours=s["hours"],
                year=s["year"],
                lp_revenue=s["lp_revenue"],
                lp_deg_cost=s["lp_deg_cost"],
                deg_rev_pct=100 * s["lp_deg_cost"] / max(s["lp_revenue"], 1),
                revenue_sacrifice=s["revenue_sacrifice"],
                degradation_saving=s["degradation_saving"],
                ratio=s.get("ratio", 0.0),
                nlp_iters=s["nlp_iters"],
                final_optimality=final_opt,
                converged=final_opt < 1e-6,   # single consistent threshold
                nlp_status=s.get("nlp_status", ""),
                convergence=conv,
            ))
        except Exception as e:
            print(f"  Skipping {jf.name}: {e}")

    if skipped:
        print(f"  Filtered out {skipped} files")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_table(rows: list[dict]) -> None:
    if not rows:
        print("  No results.")
        return

    rows_s = sorted(
        rows,
        key=lambda r: (REGIME_ORDER.get(r["regime"], 3), -r["deg_rev_pct"], r["solver_short"]),
    )

    header = (
        f"  {'Slot':<6} {'Config':<10} {'Regime':<5} {'d/r%':>6}"
        f"  {'LP Rev':>9}  {'Sac':>8}  {'Sav':>8}  {'Ratio':>6}"
        f"  {'Iters':>5}  {'Opt':>10}  {'Conv'}"
    )
    sep = "  " + "─" * (len(header) - 2)
    print(f"\n{'═'*90}")
    print(f"  Path 3 results — {len(rows)} runs")
    print(f"{'═'*90}")
    print(header)

    prev_regime = None
    for r in rows_s:
        if r["regime"] != prev_regime:
            print(sep)
            prev_regime = r["regime"]

        conv_str = "YES ✓" if r["converged"] else "no"
        ratio_str = f"{r['ratio']:.2f}x" if r["revenue_sacrifice"] > 1 else "—"
        print(
            f"  {r['slot']:<6} {r['solver_short']:<10} {r['regime']:<5} {r['deg_rev_pct']:>5.1f}%"
            f"  {r['lp_revenue']:>9,.0f}"
            f"  {r['revenue_sacrifice']:>8,.0f}"
            f"  {r['degradation_saving']:>8,.0f}"
            f"  {ratio_str:>6}"
            f"  {r['nlp_iters']:>5}"
            f"  {r['final_optimality']:>10.2e}"
            f"  {conv_str}"
        )

    print(f"{'═'*90}")
    n_conv = sum(r["converged"] for r in rows)
    opts = [r["final_optimality"] for r in rows]
    print(f"  Converged: {n_conv}/{len(rows)}"
          f"   Min opt: {min(opts):.2e}"
          f"   Max opt: {max(opts):.2e}")
    print(f"{'═'*90}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TRAJECTORY PLOT
# ══════════════════════════════════════════════════════════════════════════════

def _solver_linestyle(solver_short: str):
    """Map solver config to (linestyle, linewidth, alpha)."""
    if solver_short.startswith("α=0.1"):
        return "-",  1.4, 0.95   # solid, prominent
    if solver_short == "scaled":
        return "--", 1.0, 0.75   # dashed
    if solver_short == "FD-Hess":
        return ":",  1.0, 0.75   # dotted
    if solver_short.startswith("α="):
        return "-.", 1.0, 0.70   # dash-dot for other alphas
    return "-",  0.8, 0.60       # fallback


def plot_trajectories(rows: list[dict], out_path: Path) -> None:
    rows_s = sorted(
        rows,
        key=lambda r: (REGIME_ORDER.get(r["regime"], 3), -r["deg_rev_pct"]),
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Track which slot+solver combos we've added to the legend
    legend_handles = {}

    for r in rows_s:
        conv  = r["convergence"]
        iters = [c["iter"] for c in conv]
        opts  = [c["optimality"] for c in conv]

        color = SLOT_COLORS.get(r["slot"], DEFAULT_COLOR)
        ls, lw, alpha_v = _solver_linestyle(r["solver_short"])

        legend_key = f"{r['slot']} ({r['solver_short']})"
        line, = ax.semilogy(
            iters, opts,
            color=color, lw=lw, ls=ls, alpha=alpha_v,
            label=legend_key,
        )
        legend_handles[legend_key] = line

        # Mark endpoint
        final_opt = opts[-1]
        if final_opt < 1e-6:
            ax.plot(iters[-1], final_opt, "*", color=color, ms=10, zorder=6)
            ax.annotate(
                f"{r['slot']}\n{final_opt:.1e}",
                xy=(iters[-1], final_opt),
                xytext=(10, 8), textcoords="offset points",
                fontsize=7.5, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=0.6),
            )
        elif final_opt > 1e6:
            ax.plot(iters[-1], min(final_opt, 1e13), "^", color=color, ms=7, zorder=6)

    # Reference lines
    ax.axhline(1e-6, ls="--", color="#777777", lw=0.9, zorder=0)
    ax.text(8, 4e-7, "KKT target 1e-6", fontsize=9, color="#777777")

    ax.set_ylabel("Optimality (KKT residual)", fontsize=12)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylim(1e-8, 1e14)
    ax.set_title(
        "Path 3 NLP — optimality trajectories by slot and solver config",
        fontsize=12, fontweight="bold", loc="left",
    )

    # Build clean legend: slot colours + solver line styles
    from matplotlib.lines import Line2D
    slot_handles = [
        Line2D([0],[0], color=SLOT_COLORS.get(s, DEFAULT_COLOR), lw=2,
               label=f"{s} ({REGIME.get(s,'?')})")
        for s in ["D8","D238","D248","D86","D166","D309","D43","D128","D197"]
        if any(r["slot"] == s for r in rows)
    ]
    style_handles = [
        Line2D([0],[0], color="#555", ls="-",  lw=1.4, label="α=0.1"),
        Line2D([0],[0], color="#555", ls="--", lw=1.0, label="scaled"),
        Line2D([0],[0], color="#555", ls=":",  lw=1.0, label="FD-Hess"),
    ]
    leg1 = ax.legend(handles=slot_handles,  loc="upper right",
                     fontsize=8, ncol=2, title="Slot", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="center right",
              fontsize=8, title="Config", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SCATTER PLOT  (final optimality vs deg/rev)
# ══════════════════════════════════════════════════════════════════════════════

SOLVER_MARKERS = {
    "α=0.1":   ("o", 9, 1.0),
    "scaled":  ("s", 8, 0.8),
    "FD-Hess": ("D", 8, 0.8),
}
DEFAULT_MARKER = ("^", 8, 0.7)


def plot_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for r in rows:
        color  = SLOT_COLORS.get(r["slot"], DEFAULT_COLOR)
        mk, ms, al = SOLVER_MARKERS.get(r["solver_short"], DEFAULT_MARKER)
        ax.semilogy(
            r["deg_rev_pct"], r["final_optimality"],
            marker=mk, color=color, ms=ms, alpha=al,
            ls="none", zorder=5,
        )
        ax.annotate(
            f"{r['slot']}\n{r['solver_short']}",
            xy=(r["deg_rev_pct"], r["final_optimality"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color=color,
        )

    ax.axhline(1e-6, ls="--", color="#777777", lw=0.9)
    ax.text(0.5, 4e-7, "KKT target 1e-6", fontsize=9, color="#777777")

    ax.set_xlabel("LP deg/rev ratio (%)", fontsize=12)
    ax.set_ylabel("Final optimality", fontsize=12)
    ax.set_title(
        "Final optimality vs LP degradation-to-revenue ratio",
        fontsize=12, fontweight="bold", loc="left",
    )

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker=mk, color="#555", ls="none", ms=ms,
               label=label, alpha=al)
        for label, (mk, ms, al) in SOLVER_MARKERS.items()
    ]
    ax.legend(handles=handles, title="Solver config", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Path 3 NLP analysis — summary table + convergence plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --dir Results_Path3 --hours 24                   # all daily runs
  %(prog)s --dir Results_Path3 --hours 24 --solver alpha0_1 # alpha=0.1 only
  %(prog)s --dir Results_Path3 --hours 24 --solver scaled   # scaled only
  %(prog)s --dir Results_Path3 --hours 168                  # weekly runs
""",
    )
    parser.add_argument("--dir",    type=str, default="Results_Path3")
    parser.add_argument("--hours",  type=int, default=None,
                        help="Filter by horizon length (24/168/336)")
    parser.add_argument("--solver", type=str, default=None,
                        help="Substring filter on solver tag")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for plots (default: same as --dir)")
    args = parser.parse_args()

    results_dir = Path(args.dir)
    if not results_dir.exists():
        results_dir = Path(__file__).parent / args.dir
    if not results_dir.exists():
        print(f"  Directory not found: {args.dir}")
        return

    out_dir = Path(args.out_dir) if args.out_dir else results_dir

    hours_tag  = f"_h{args.hours}"  if args.hours  else ""
    solver_tag = f"_{args.solver}"  if args.solver else ""

    rows = load_results(results_dir, args.hours, args.solver)
    print(f"  Loaded {len(rows)} runs from {results_dir}")

    if not rows:
        print("  Nothing to analyse — check --hours / --solver filters.")
        return

    print_table(rows)

    traj_path = out_dir / f"trajectories{hours_tag}{solver_tag}.png"
    plot_trajectories(rows, traj_path)

    scat_path = out_dir / f"scatter{hours_tag}{solver_tag}.png"
    plot_scatter(rows, scat_path)


if __name__ == "__main__":
    main()
