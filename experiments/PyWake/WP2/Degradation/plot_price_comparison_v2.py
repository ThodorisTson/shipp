r"""
plot_price_comparison_v2.py
===========================
Produces two standalone thesis figures from DK1 2019 and 2022 price CSVs.

  fig_dk1_timeseries.pdf     -- hourly prices over the year, both regimes
  fig_dk1_ecdf_spread.pdf    -- ECDF of within-day price spread (arbitrage proxy)

CHANGES vs plot_price_comparison.py (green-light version)
---------------------------------------------------------
1. Colours now use the thesis_style semantic slots instead of hardcoded hexes.
   Two-line comparisons (2019 vs 2022) use PALETTE["primary"]/["secondary"]
   = navy / dark red, as thesis_style prescribes. The old blue/orange are the
   fill slots reserved for the cycle-vs-calendar stacked bars, so the previous
   version collided with those figures. 2022 = primary (focus year),
   2019 = secondary.
2. Time-series x-axis label changed from "Month of year  (-)" to "Month".
   Months are calendar categories, not a dimensionless quantity, so no unit.
3. Time-series plotting order swapped so 2022 is drawn first (underneath) and
   2019 second (on top). This makes the legend order match the ECDF (2022, 2019)
   and keeps the thinner 2019 line visible where the two series overlap.

In-figure legends are kept (minimal style, consistent with the rest of the
thesis). Captions in the .tex refer to the series by year, not colour.

Reproducible on Windows / VS Code: matplotlib + numpy + pandas, DejaVu font,
no LaTeX toolchain. Anchored on Path(__file__).parent.
"""
from __future__ import annotations
import sys
from pathlib import Path

# -- Path guard: find thesis_style.py in this dir or any parent ---------------
for _d in Path(__file__).resolve().parents:
    if (_d / "thesis_style.py").exists():
        sys.path.insert(0, str(_d))
        break
else:
    raise FileNotFoundError("thesis_style.py not found in any parent folder")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from thesis_style import apply_thesis_style, figsize, FS_ANNOT

P = apply_thesis_style(palette="brand", usetex=False)

# ===========================================================================
# CONFIGURATION
# ===========================================================================
HERE      = Path(__file__).parent
FILE_2019 = HERE / "dk1_prices_2019.csv"
FILE_2022 = HERE / "dk1_prices_2022.csv"
OUT_DIR   = HERE / "thesis_figs"

# Two-line comparison -> primary / secondary (navy / dark red) per thesis_style.
C_2022 = P["primary"]     # focus year (sizing sweep)
C_2019 = P["secondary"]   # low-volatility reference year

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_TICKS  = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016]


# ===========================================================================
# DATA HELPERS
# ===========================================================================
def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col, price_col = df.columns[0], df.columns[1]
    df = df.rename(columns={ts_col: "ts", price_col: "price"})
    df["ts"]   = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df         = df.dropna(subset=["ts"]).reset_index(drop=True)
    df["date"] = df["ts"].dt.date
    df["hour"] = np.arange(len(df), dtype=float)
    return df

def daily_spread(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date")["price"].agg(lambda s: s.max() - s.min())

def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(values.to_numpy())
    f = np.arange(1, v.size + 1) / v.size
    return v, f

def _save(fig, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  ok  {stem}.pdf")


# ===========================================================================
# FIG 1 -- HOURLY TIME SERIES (full text width)
# ===========================================================================
def plot_timeseries(d19: pd.DataFrame, d22: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.45))

    # 2022 first (underneath), 2019 second (on top, stays visible).
    ax.plot(d22["hour"], d22["price"],
            color=C_2022, lw=0.55, alpha=0.85, label="DK1 2022")
    ax.plot(d19["hour"], d19["price"],
            color=C_2019, lw=0.55, alpha=0.85, label="DK1 2019")

    ax.axhline(0, color=P["neutral"], lw=0.7, ls=(0, (4, 4)), alpha=0.6)
    ax.set_xticks(MONTH_TICKS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=FS_ANNOT)
    ax.set_xlim(0, 8760)
    ax.set_xlabel("Month")
    ax.set_ylabel("Day-ahead price  (EUR / MWh)")
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="upper left")

    _save(fig, "fig_dk1_timeseries", out_dir)


# ===========================================================================
# FIG 2 -- ECDF OF DAILY PRICE SPREAD (about 0.6 text width)
# ===========================================================================
def plot_ecdf_spread(ds19: pd.Series, ds22: pd.Series, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=figsize(0.58, aspect=0.88))

    v22, f22 = ecdf(ds22)
    v19, f19 = ecdf(ds19)

    ax.plot(v22, f22, color=C_2022, lw=1.5, label="DK1 2022")
    ax.plot(v19, f19, color=C_2019, lw=1.5, label="DK1 2019")

    # Median markers with value annotation.
    for ds, col in [(ds22, C_2022), (ds19, C_2019)]:
        med = float(np.median(ds))
        ax.axvline(med, color=col, lw=0.8, ls=(0, (2, 3)), alpha=0.80)
        ax.text(med + (v22.max() * 0.015), 0.06,
                f"{med:.0f} EUR/MWh",
                color=col, fontsize=FS_ANNOT, va="bottom", rotation=90)

    ax.set_xlabel("Daily price spread  (EUR / MWh)")
    ax.set_ylabel("Cumulative fraction of days  (-)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="lower right")

    _save(fig, "fig_dk1_ecdf_spread", out_dir)


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    print("Loading price data...")
    d19 = load_prices(FILE_2019)
    d22 = load_prices(FILE_2022)
    ds19 = daily_spread(d19)
    ds22 = daily_spread(d22)

    print("\n-- Caption values -- (copy into .tex)")
    for name, df, ds in [("2019", d19, ds19), ("2022", d22, ds22)]:
        p = df["price"]
        print(f"  DK1 {name}: mean={p.mean():.1f}  std={p.std():.1f}  "
              f"min={p.min():.1f}  max={p.max():.1f}  "
              f"pct_negative={(p < 0).mean()*100:.1f}%  "
              f"median_spread={ds.median():.1f}  "
              f"mean_spread={ds.mean():.1f}  EUR/MWh")

    print(f"\nProducing figures in: {OUT_DIR}")
    plot_timeseries(d19, d22, OUT_DIR)
    plot_ecdf_spread(ds19, ds22, OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
