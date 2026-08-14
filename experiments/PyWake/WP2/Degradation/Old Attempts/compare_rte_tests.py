"""
compare_rte_tests.py — Two-point RTE sensitivity comparison (0.877 vs 0.910).

Reads the npv_summary_*.csv files written by
run_battery_xu_shi_degradation_v5_6 into  Results/RTE Tests/ , pairs each price
scenario across the two AC round-trip values, and reports the deltas that decide
whether the efficiency assumption is critical:

    - headline battery NPV (npv_bat_multiyear)
    - final state of health
    - first replacement year
    - number of replacements and PV of replacements

Outputs
-------
    Results/RTE Tests/rte_comparison_summary.csv     numeric deltas, one row per scenario
    Results/RTE Tests/rte_npv_comparison.pdf         grouped bar figure (300 DPI, PDF)

Run from VS Code on Windows. Paths are anchored with Path(__file__).parent, so
place this file in the same Degradation folder as the run script.

This script does NOT run any simulation. It only reads existing outputs, so run
the four cases first (2 price years x 2 rte values) with RTE_AC_OVERRIDE set.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Paths (Windows-safe, anchored) ───────────────────────────────────────────
HERE     = Path(__file__).parent
RTE_DIR  = HERE / "Results" / "RTE Tests"
OUT_CSV  = RTE_DIR / "rte_comparison_summary.csv"
OUT_PDF  = RTE_DIR / "rte_npv_comparison.pdf"

# ── TU Delft palette ─────────────────────────────────────────────────────────
NAVY    = "#0C2340"   # baseline efficiency
FILLBLU = "#0076C2"   # DEA efficiency
DARKRED = "#A50034"
NEUTRAL = "#404040"

# Round-trip values under test (AC). Used only to label / order the pair.
RTE_BASELINE = 0.877
RTE_DEA      = 0.910
RTE_TOL      = 0.005   # bucket width for matching an rte_ac column to a label


def _first_replacement_year(cell) -> float:
    """Parse the stored replacement_years string ('[12]', '[]') to a first year."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return np.nan
    try:
        years = ast.literal_eval(str(cell))
    except (ValueError, SyntaxError):
        return np.nan
    return float(min(years)) if years else np.nan


def _dataset_from_label(label: str) -> str:
    """Pull the dk#### token out of the run_label."""
    m = re.search(r"dk\d{4}", str(label))
    return m.group(0) if m else "unknown"


def _rte_from_row(row) -> float:
    """rte_ac column if present (post provenance patch), else parse _rte### tag."""
    if "rte_ac" in row and pd.notna(row["rte_ac"]):
        return float(row["rte_ac"])
    m = re.search(r"_rte(\d{3})", str(row.get("run_label", "")))
    return int(m.group(1)) / 1000.0 if m else np.nan


def _load_summary() -> pd.DataFrame:
    files = sorted(RTE_DIR.glob("npv_summary_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No npv_summary_*.csv in {RTE_DIR}. Run the RTE cases first."
        )
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df["rte_ac_val"]    = df.apply(_rte_from_row, axis=1)
    df["dataset"]       = df["run_label"].apply(_dataset_from_label)
    df["first_repl_yr"] = df["replacement_years"].apply(_first_replacement_year)

    # If a scenario was run more than once, keep the most recent row.
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["dataset", "e_cap_MWh", "p_cap_MW", "soc_min", "soc_max", "rte_ac_val"],
        keep="last",
    )
    return df


def _label_for(rte: float) -> str:
    if abs(rte - RTE_BASELINE) <= RTE_TOL:
        return "baseline"
    if abs(rte - RTE_DEA) <= RTE_TOL:
        return "dea"
    return f"rte{int(round(rte * 1000)):03d}"


def build_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = ["dataset", "e_cap_MWh", "p_cap_MW", "soc_min", "soc_max"]
    for keys, g in df.groupby(group_keys):
        g = g.copy()
        g["role"] = g["rte_ac_val"].apply(_label_for)
        base = g[g["role"] == "baseline"]
        dea  = g[g["role"] == "dea"]
        if base.empty or dea.empty:
            print(f"  [skip] {dict(zip(group_keys, keys))}: "
                  f"have rte roles {sorted(g['role'].unique())} (need baseline + dea)")
            continue
        b = base.iloc[0]
        d = dea.iloc[0]

        npv_b = float(b["npv_bat_multiyear_MEUR"])
        npv_d = float(d["npv_bat_multiyear_MEUR"])
        pct   = 100.0 * (npv_d - npv_b) / abs(npv_b) if npv_b != 0 else np.nan

        rows.append({
            "dataset":              keys[0],
            "e_cap_MWh":            keys[1],
            "p_cap_MW":             keys[2],
            "soc_window":           f"{int(keys[3]*100)}-{int(keys[4]*100)}",
            "rte_baseline":         float(b["rte_ac_val"]),
            "rte_dea":              float(d["rte_ac_val"]),
            "npv_baseline_MEUR":    round(npv_b, 3),
            "npv_dea_MEUR":         round(npv_d, 3),
            "npv_delta_MEUR":       round(npv_d - npv_b, 3),
            "npv_delta_pct":        round(pct, 2),
            "soh_baseline_pct":     round(float(b["final_soh_pct"]), 2),
            "soh_dea_pct":          round(float(d["final_soh_pct"]), 2),
            "first_repl_baseline":  b["first_repl_yr"],
            "first_repl_dea":       d["first_repl_yr"],
            "n_repl_baseline":      int(b["n_replacements"]),
            "n_repl_dea":           int(d["n_replacements"]),
            "pv_repl_baseline_MEUR": round(float(b["pv_replacements_MEUR"]), 3),
            "pv_repl_dea_MEUR":      round(float(d["pv_replacements_MEUR"]), 3),
        })
    return pd.DataFrame(rows)


def _print_table(cmp: pd.DataFrame) -> None:
    if cmp.empty:
        print("\nNo complete baseline+DEA pairs found. Nothing to compare.")
        return
    print("\n" + "=" * 78)
    print("RTE SENSITIVITY  |  baseline (0.877) vs DEA (0.910)  |  headline NPV basis")
    print("=" * 78)
    for _, r in cmp.iterrows():
        print(f"\n{r['dataset']}  {int(r['e_cap_MWh'])} MWh / {int(r['p_cap_MW'])} MW  "
              f"SoC {r['soc_window']}%")
        print(f"  NPV       : {r['npv_baseline_MEUR']:>9.2f}  ->  "
              f"{r['npv_dea_MEUR']:>9.2f} MEUR   "
              f"(delta {r['npv_delta_MEUR']:+.2f} MEUR, {r['npv_delta_pct']:+.2f}%)")
        print(f"  Final SoH : {r['soh_baseline_pct']:>9.2f}  ->  "
              f"{r['soh_dea_pct']:>9.2f} %")
        print(f"  1st repl. : {r['first_repl_baseline']!s:>9}  ->  "
              f"{r['first_repl_dea']!s:>9}   "
              f"(n_repl {r['n_repl_baseline']} -> {r['n_repl_dea']})")
    print("\n" + "-" * 78)
    print("Read: if NPV delta and replacement year barely move, the 0.877 "
          "assumption\nis disclosed and shown non-critical. If they move, know it "
          "before the defense.")


def _make_figure(cmp: pd.DataFrame) -> None:
    if cmp.empty:
        print("  [figure skipped: no pairs]")
        return
    datasets = list(cmp["dataset"])
    npv_b = cmp["npv_baseline_MEUR"].to_numpy()
    npv_d = cmp["npv_dea_MEUR"].to_numpy()

    x = np.arange(len(datasets))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x - w / 2, npv_b, w, color=NAVY,    label="AC round-trip 0.877 (current)")
    ax.bar(x + w / 2, npv_d, w, color=FILLBLU, label="AC round-trip 0.910 (DEA)")

    for xi, (yb, yd) in enumerate(zip(npv_b, npv_d)):
        ax.annotate(f"{yb:.1f}", (xi - w / 2, yb), ha="center",
                    va="bottom" if yb >= 0 else "top", fontsize=8, color=NEUTRAL)
        ax.annotate(f"{yd:.1f}", (xi + w / 2, yd), ha="center",
                    va="bottom" if yd >= 0 else "top", fontsize=8, color=NEUTRAL)

    ax.axhline(0.0, color=NEUTRAL, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Battery NPV [MEUR]")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PDF, dpi=300)
    plt.close(fig)
    print(f"  Figure : {OUT_PDF.name}")


def main() -> None:
    df  = _load_summary()
    cmp = build_comparison(df)
    _print_table(cmp)
    if not cmp.empty:
        cmp.to_csv(OUT_CSV, index=False)
        print(f"\n  CSV    : {OUT_CSV.name}")
    _make_figure(cmp)


if __name__ == "__main__":
    main()
