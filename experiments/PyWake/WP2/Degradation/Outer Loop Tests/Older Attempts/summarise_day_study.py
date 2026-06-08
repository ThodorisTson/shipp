"""
2022 Day Convergence Study — Summary Analysis
==============================================
Reads all path3 result JSONs from Results_Path3 and prints a ranked
comparison table plus a convergence-trajectory overlay plot.

Usage:
    python summarise_day_study.py                     # auto-find Results_Path3
    python summarise_day_study.py --dir Results_Path3
    python summarise_day_study.py --dir Results_Path3 --out summary.png

Author: Thodoris Tsonopoulos — MSc Thesis, TU Delft Wind Energy
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BG = "#f7f9fc"

# ── Regime classification ──────────────────────────────────────────────────
# Assign regime label from slot key embedded in filename.
_REGIME = {
    "D8":   "HIGH", "D238": "HIGH", "D248": "HIGH",
    "D86":  "MID",  "D166": "MID",  "D309": "MID",
    "D43":  "LOW",  "D128": "LOW",  "D197": "LOW",
}
_REGIME_COLOR = {"HIGH": "#b5351b", "MID": "#2166ac", "LOW": "#1b9e77"}
_REGIME_MARKER = {"HIGH": "^", "MID": "o", "LOW": "v"}


def _slot_from_path(p: Path) -> str:
    """Extract slot key (e.g. 'D43') from result filename."""
    for part in p.stem.split("_"):
        if part.startswith("D") and part[1:].isdigit():
            return part
    return "??"


def load_all(results_dir: Path, hours_filter: int | None = None) -> list[dict]:
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
            # Apply hours filter before building the full row
            if hours_filter is not None and s.get("hours", 0) != hours_filter:
                skipped += 1
                continue
            conv = d["convergence"]
            slot = _slot_from_path(jf)
            rows.append(dict(
                file=jf.name, slot=slot,
                regime=_REGIME.get(slot, "?"),
                period=s["period"],
                hours=s["hours"],
                year=s["year"],
                lp_revenue=s["lp_revenue"],
                lp_deg_cost=s["lp_deg_cost"],
                deg_rev_pct=100 * s["lp_deg_cost"] / max(s["lp_revenue"], 1),
                revenue_sacrifice=s["revenue_sacrifice"],
                degradation_saving=s["degradation_saving"],
                ratio=s.get("ratio", 0),
                nlp_iters=s["nlp_iters"],
                final_optimality=s.get("final_optimality", float("nan")),
                converged=s.get("converged", False),
                nlp_status=s.get("nlp_status", ""),
                obj_scale=s.get("obj_scale", 1.0),
                convergence=conv,
            ))
        except Exception as e:
            print(f"  Skipping {jf.name}: {e}")
    if skipped:
        print(f"  Filtered out {skipped} files with different horizon")
    return rows


def print_table(rows: list[dict]):
    if not rows:
        print("No results found.")
        return

    # Sort by regime order then by deg/rev descending
    order = {"HIGH": 0, "MID": 1, "LOW": 2, "?": 3}
    rows_s = sorted(rows, key=lambda r: (order[r["regime"]], -r["deg_rev_pct"]))

    header = (
        f"  {'Slot':<7s} {'Regime':<6s} {'Date':<12s} {'LP Rev':>10s} "
        f"{'deg/rev':>8s} {'Sac':>8s} {'Save':>8s} {'Ratio':>6s} "
        f"{'Iters':>6s} {'Opt':>10s} {'Conv':>6s}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print(f"\n{'='*90}")
    print(f"  2022 Day Convergence Study — {len(rows)} results")
    print(f"{'='*90}")
    print(header)
    print(sep)

    prev_regime = None
    for r in rows_s:
        if r["regime"] != prev_regime:
            if prev_regime is not None:
                print(sep)
            prev_regime = r["regime"]

        date_str = r["period"].split("(")[0].strip()[-10:]
        opt_str  = f"{r['final_optimality']:.2e}"
        conv_str = "YES **" if r["converged"] else "no"
        sac_str  = f"{r['revenue_sacrifice']:+,.0f}"
        sav_str  = f"{r['degradation_saving']:+,.0f}"
        ratio_str = f"{r['ratio']:.2f}x" if r['revenue_sacrifice'] > 0 else "—"

        print(
            f"  {r['slot']:<7s} {r['regime']:<6s} {date_str:<12s} "
            f"{r['lp_revenue']:>10,.0f} "
            f"{r['deg_rev_pct']:>7.1f}% "
            f"{sac_str:>8s} {sav_str:>8s} {ratio_str:>6s} "
            f"{r['nlp_iters']:>6d} {opt_str:>10s} {conv_str:>6s}"
        )

    print(f"{'='*90}")
    n_conv = sum(r["converged"] for r in rows)
    print(f"  Converged: {n_conv}/{len(rows)}  "
          f"  Min optimality: {min(r['final_optimality'] for r in rows):.2e}  "
          f"  Max optimality: {max(r['final_optimality'] for r in rows):.2e}")
    print(f"{'='*90}\n")


def plot_trajectories(rows: list[dict], out_path: Path):
    if not rows:
        return

    order = {"HIGH": 0, "MID": 1, "LOW": 2, "?": 3}
    rows_s = sorted(rows, key=lambda r: (order[r["regime"]], -r["deg_rev_pct"]))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=False,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.18},
    )
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    fig.suptitle(
        "2022 Day Convergence Study — Optimality trajectories",
        fontsize=14, fontweight="bold", y=0.97,
    )

    # Dash cycles per regime so lines are distinguishable
    _dashes = {
        "HIGH": [(None,None), (6,3), (3,3)],
        "MID":  [(None,None), (6,3), (3,3)],
        "LOW":  [(None,None), (6,3), (3,3)],
    }
    _regime_idx = {"HIGH": 0, "MID": 0, "LOW": 0}

    summary_x, summary_y, summary_c, summary_m, summary_label = [], [], [], [], []

    for r in rows_s:
        conv  = r["convergence"]
        iters = [c["iter"] for c in conv]
        opts  = [c["optimality"] for c in conv]

        regime = r["regime"]
        color  = _REGIME_COLOR.get(regime, "#888888")
        idx    = _regime_idx[regime]
        dash   = _dashes[regime][idx % 3]
        _regime_idx[regime] += 1

        lw = 1.8 if r["converged"] else 1.0
        ax1.semilogy(
            iters, opts,
            color=color, lw=lw,
            dashes=dash if dash[0] else [],
            label=f"{r['slot']} ({regime[:1]}) — opt={r['final_optimality']:.1e}",
            alpha=0.85,
        )

        # Mark final point
        marker = _REGIME_MARKER.get(regime, "o")
        ax1.plot(iters[-1], opts[-1], marker, color=color, ms=6, zorder=5)

        # Collect for summary scatter
        summary_x.append(r["deg_rev_pct"])
        summary_y.append(r["final_optimality"])
        summary_c.append(color)
        summary_m.append(marker)
        summary_label.append(r["slot"])

    ax1.axhline(1e-6, ls="--", color="#999999", lw=0.8, zorder=0)
    ax1.text(5, 2e-7, "convergence target (1e-6)", fontsize=9, color="#999999")
    ax1.set_ylabel("Optimality measure (KKT)", fontsize=12)
    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_title("Optimality per iteration — all 2022 days", fontsize=11,
                  fontweight="bold", loc="left")
    ax1.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.85)

    # ── Panel 2: final optimality vs deg/rev scatter ──────────────────────
    for x, y, c, m, lbl in zip(summary_x, summary_y, summary_c, summary_m, summary_label):
        ax2.semilogy(x, y, marker=m, color=c, ms=9, zorder=5, ls="none")
        ax2.annotate(lbl, (x, y), xytext=(4, 4),
                     textcoords="offset points", fontsize=8, color=c)

    ax2.axhline(1e-6, ls="--", color="#999999", lw=0.8)
    ax2.text(1, 2e-7, "1e-6 target", fontsize=9, color="#999999")

    # Legend patches for regimes
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker=_REGIME_MARKER[r], color=_REGIME_COLOR[r],
               ls="none", ms=9, label=r)
        for r in ["HIGH","MID","LOW"]
    ]
    ax2.legend(handles=handles, fontsize=9, loc="upper right")
    ax2.set_xlabel("LP deg/rev ratio (%)", fontsize=11)
    ax2.set_ylabel("Final optimality", fontsize=11)
    ax2.set_title("Final optimality vs deg/rev — diagnostic scatter",
                  fontsize=11, fontweight="bold", loc="left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --dir Results_Path3              # all results
  %(prog)s --dir Results_Path3 --hours 24   # daily only
  %(prog)s --dir Results_Path3 --hours 168  # weekly only
""")
    parser.add_argument("--dir", type=str, default="Results_Path3",
                        help="Folder containing *path3*results.json files")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG (default: <dir>/2022_day_study_summary.png)")
    parser.add_argument("--hours", type=int, default=None,
                        help="Only include runs with this horizon length "
                             "(e.g. 24=daily, 168=weekly). Default: include all.")
    args = parser.parse_args()

    results_dir = Path(args.dir)
    if not results_dir.exists():
        results_dir = Path(__file__).parent / args.dir
    if not results_dir.exists():
        print(f"Directory not found: {args.dir}")
        return

    rows = load_all(results_dir, hours_filter=args.hours)
    filter_tag = f" (hours={args.hours})" if args.hours else ""
    print(f"  Loaded {len(rows)} result files from {results_dir}{filter_tag}")

    print_table(rows)

    out_path = Path(args.out) if args.out else results_dir / "2022_day_study_summary.png"
    plot_trajectories(rows, out_path)


if __name__ == "__main__":
    main()
