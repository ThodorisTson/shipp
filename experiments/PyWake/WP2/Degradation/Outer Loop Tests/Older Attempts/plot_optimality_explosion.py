"""
Plot: Optimality Explosion Diagnostic
======================================
Reads all Path 3 NLP convergence JSONs and produces a 2-panel figure:
  Panel 1 — Optimality measure (log-scale) vs iteration
  Panel 2 — Objective value vs iteration

Highlights the iteration at which each case's optimality first exceeds 1e6
("explosion point"), annotating directly on the curve.

Usage:
    python plot_optimality_explosion.py                    # auto-detect JSONs in cwd
    python plot_optimality_explosion.py --dir Results_Path3  # specify folder

Author: Thodoris Tsonopoulos — MSc Thesis, TU Delft Wind Energy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Style ────────────────────────────────────────────────────────────────
BG       = "#f7f9fc"
FONT     = "DejaVu Sans"
PALETTE  = {
    "2019 W1":    "#2166ac",
    "2019 W28":   "#b5351b",
    "2019 W35":   "#6a3d9a",
    "2022 W7":    "#ff7f00",
    "2022 W13":   "#1b9e77",
    "2022 W34":   "#e7298a",
    "336h":       "#66a61e",
}
DASHES = {
    "2019 W1":  (None, None),
    "2019 W28": (6, 3),
    "2019 W35": (3, 3),
    "2022 W7":  (8, 4),
    "2022 W13": (None, None),
    "2022 W34": (4, 2),
    "336h":     (10, 5),
}


def _case_key(period: str, year: int) -> str:
    """Extract a short key like '2019 W1' or '336h' from the period string."""
    import re
    # Match week slot labels like "W1", "W7", "W13", etc.
    m = re.search(r'(W\d+)', period)
    if m:
        return f"{year} {m.group(1)}"
    if "336" in period:
        return "336h"
    return period[:15]


def load_convergence(json_path: Path) -> dict:
    """Load a single results JSON, return {key, period, hours, convergence}."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    s = data["summary"]
    key = _case_key(s["period"], s["year"])
    return dict(
        key=key,
        period=s["period"],
        year=s["year"],
        hours=s.get("hours", 0),
        convergence=data["convergence"],
    )


def find_explosion(conv: list[dict], threshold: float = 1e6) -> int | None:
    """Return first iteration where optimality exceeds threshold."""
    prev = conv[0]["optimality"]
    for c in conv:
        if c["optimality"] > threshold and prev < threshold:
            return c["iter"]
        prev = c["optimality"]
    return None


def plot(cases: list[dict], out_path: Path):
    """Produce the 2-panel diagnostic figure."""

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12},
    )
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    fig.suptitle(
        "Path 3 NLP convergence diagnostic — optimality explosion",
        fontsize=14, fontweight="bold", fontfamily=FONT, y=0.97,
    )

    # ── Panel 1: Optimality (log) ────────────────────────────────────────
    for case in cases:
        key  = case["key"]
        conv = case["convergence"]
        iters = [c["iter"] for c in conv]
        opts  = [c["optimality"] for c in conv]

        color = PALETTE.get(key, "#555555")
        dash  = DASHES.get(key, (None, None))
        lw    = 2.0 if key == "2022 W13" else 1.0

        ax1.semilogy(
            iters, opts, color=color, lw=lw,
            dashes=dash if dash[0] else [],
            label=f"{key}",
            alpha=0.9,
        )

        # Mark explosion point or convergence
        expl = find_explosion(conv)
        if expl is not None:
            # Find the data point closest to explosion iter
            idx = min(range(len(iters)), key=lambda i: abs(iters[i] - expl))
            ax1.plot(iters[idx], opts[idx], "o", color=color, ms=6, zorder=5)
            ax1.annotate(
                f"iter {expl}",
                xy=(iters[idx], opts[idx]),
                xytext=(12, 8), textcoords="offset points",
                fontsize=8, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.5),
            )
        elif opts[-1] < 1e-4:
            # Converged case — annotate endpoint
            ax1.plot(iters[-1], opts[-1], "*", color=color, ms=10, zorder=5)
            ax1.annotate(
                f"converged\n{opts[-1]:.1e}",
                xy=(iters[-1], opts[-1]),
                xytext=(15, 15), textcoords="offset points",
                fontsize=8, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )

    # Reference lines
    ax1.axhline(1e-6, ls="--", color="#888888", lw=0.8, zorder=0)
    ax1.text(5, 2e-7, "convergence target (1e-6)", fontsize=9,
             color="#888888", fontfamily=FONT)
    ax1.axhline(1e6, ls=":", color="#cccccc", lw=0.7, zorder=0)
    ax1.text(5, 2e6, "explosion threshold", fontsize=8,
             color="#bbbbbb", fontfamily=FONT)

    ax1.set_ylabel("Optimality measure (KKT)", fontsize=12, fontfamily=FONT)
    ax1.set_ylim(1e-8, 1e14)
    ax1.set_title("Optimality trajectories — all cases", fontsize=11,
                  fontweight="bold", fontfamily=FONT, loc="left")
    ax1.legend(fontsize=9, loc="upper left", ncol=2, framealpha=0.8)

    # ── Panel 2: Objective value ─────────────────────────────────────────
    for case in cases:
        key  = case["key"]
        conv = case["convergence"]
        iters = [c["iter"] for c in conv]
        objs  = [-c["obj"] for c in conv]  # flip sign: net utility

        color = PALETTE.get(key, "#555555")
        dash  = DASHES.get(key, (None, None))
        lw    = 2.0 if key == "2022 W13" else 1.0

        ax2.plot(
            iters, objs, color=color, lw=lw,
            dashes=dash if dash[0] else [],
            alpha=0.85,
        )

    ax2.set_ylabel("−Objective (net utility, EUR)", fontsize=12, fontfamily=FONT)
    ax2.set_xlabel("Iteration", fontsize=12, fontfamily=FONT)
    ax2.set_title("Objective trajectories — solver keeps improving despite explosion",
                  fontsize=11, fontweight="bold", fontfamily=FONT, loc="left")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e3:,.0f}k" if x < 1e6 else f"{x/1e6:,.1f}M"
    ))

    # ── Save ─────────────────────────────────────────────────────────────
    fig.savefig(
        out_path, dpi=150, bbox_inches="tight",
        facecolor=BG, edgecolor="none",
    )
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --dir Results_Path3                 # all results
  %(prog)s --dir Results_Path3 --hours 24      # daily only
  %(prog)s --dir Results_Path3 --hours 168     # weekly only
  %(prog)s --dir Results_Path3 --hours 336     # 2-week only
""")
    parser.add_argument("--dir", type=str, default=".",
                        help="Directory containing *_results.json files")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path (default: optimality_explosion.png in --dir)")
    parser.add_argument("--hours", type=int, default=None,
                        help="Only include runs with this horizon length "
                             "(e.g. 24=daily, 168=weekly, 336=2-week). "
                             "Default: include all.")
    args = parser.parse_args()

    search_dir = Path(args.dir)
    json_files = sorted(search_dir.glob("*_results.json"))
    if not json_files:
        # Try Results_Path3 subfolder
        search_dir = search_dir / "Results_Path3"
        json_files = sorted(search_dir.glob("*_results.json"))

    if not json_files:
        print("No *_results.json files found. Use --dir to specify location.")
        return

    cases = []
    skipped_hours = []
    for jf in json_files:
        try:
            c = load_convergence(jf)
            if args.hours is not None and c["hours"] != args.hours:
                skipped_hours.append(jf.name)
                continue
            cases.append(c)
        except Exception as e:
            print(f"  Skipping {jf.name}: {e}")

    filter_tag = f" (hours={args.hours})" if args.hours else ""
    print(f"Loaded {len(cases)} cases from {search_dir}{filter_tag}")
    if skipped_hours:
        print(f"  Filtered out {len(skipped_hours)} files with different horizon")

    if not cases:
        print("No cases match the filter. Check --hours value.")
        return

    for c in cases:
        expl = find_explosion(c["convergence"])
        n = len(c["convergence"])
        final_opt = c["convergence"][-1]["optimality"]
        tag = f"CONVERGED ({final_opt:.1e})" if final_opt < 1e-4 else f"EXPLODES at iter ~{expl} → {final_opt:.1e}"
        print(f"  {c['key']:<15s}  iters={n:>4d}  hours={c['hours']:>4d}  {tag}")

    hours_tag = f"_h{args.hours}" if args.hours else ""
    default_out = search_dir / f"optimality_explosion{hours_tag}.png"
    out_path = Path(args.out) if args.out else default_out
    plot(cases, out_path)


if __name__ == "__main__":
    main()
