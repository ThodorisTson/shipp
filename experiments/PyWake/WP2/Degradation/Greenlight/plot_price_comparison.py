r"""
plot_price_comparison.py  (patched to thesis_style)
====================================================
Produces two standalone thesis-ready figures from DK1 2019 and 2022 price CSVs.

  fig_dk1_timeseries.pdf     — raw hourly prices over the year, both regimes
  fig_dk1_ecdf_spread.pdf    — ECDF of within-day price spread (arbitrage proxy)

Each figure is designed at its natural width so both can be used independently
or combined side-by-side in LaTeX:
    \includegraphics[width=0.49\textwidth]{fig_dk1_timeseries}
    \includegraphics[width=0.49\textwidth]{fig_dk1_ecdf_spread}

CAPTION VALUES printed at runtime — copy into .tex.

COLOUR CONVENTION:
  DK1 2022: TUDELFT blue   (primary year; focus of sizing sweep)
  DK1 2019: TUDELFT orange (baseline / low-volatility reference)
"""
from __future__ import annotations
import sys
from pathlib import Path

# ── Path guard ────────────────────────────────────────────────────────────────
for _d in Path(__file__).resolve().parents:
    if (_d / "thesis_style.py").exists():
        sys.path.insert(0, str(_d))
        break
else:
    raise FileNotFoundError("thesis_style.py not found in any parent folder")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from thesis_style import (apply_thesis_style, figsize, TUDELFT,
                          FS_BASE, FS_ANNOT)

P = apply_thesis_style(palette="brand", usetex=False)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
HERE      = Path(__file__).parent
FILE_2019 = HERE / "dk1_prices_2019.csv"
FILE_2022 = HERE / "dk1_prices_2022.csv"
OUT_DIR   = HERE / "thesis_figs"

C_2022 = TUDELFT["blue"]     # primary year
C_2019 = TUDELFT["orange"]   # baseline year

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_TICKS  = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016]


# ═══════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════

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
    print(f"  ✓ {stem}.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 — RAW HOURLY TIME SERIES  (full textwidth)
# ═══════════════════════════════════════════════════════════════════════════

def plot_timeseries(d19: pd.DataFrame, d22: pd.DataFrame,
                    out_dir: Path) -> None:
    """
    CAPTION: "Hourly DK1 day-ahead electricity prices for 2019 and 2022.
    The 2022 series (blue) reflects the European energy crisis, with a mean of
    219 EUR/MWh and peak values exceeding 870 EUR/MWh; the 2019 series (orange)
    represents a low-volatility baseline with a mean of 38.5 EUR/MWh."
    """
    fig, ax = plt.subplots(figsize=figsize(1.0, aspect=0.45))

    ax.plot(d19["hour"], d19["price"],
            color=C_2019, lw=0.55, alpha=0.85, label="DK1 2019")
    ax.plot(d22["hour"], d22["price"],
            color=C_2022, lw=0.55, alpha=0.85, label="DK1 2022")

    ax.axhline(0, color=P["neutral"], lw=0.7, ls=(0, (4, 4)), alpha=0.6)
    ax.set_xticks(MONTH_TICKS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=FS_ANNOT)
    ax.set_xlim(0, 8760)
    ax.set_xlabel("Month of year  (–)")
    ax.set_ylabel("Day-ahead price  (EUR / MWh)")
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="upper left")

    _save(fig, "fig_dk1_timeseries", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 — ECDF OF DAILY PRICE SPREAD  (half textwidth, square)
# ═══════════════════════════════════════════════════════════════════════════

def plot_ecdf_spread(ds19: pd.Series, ds22: pd.Series,
                     out_dir: Path) -> None:
    """
    Empirical CDF of the within-day price spread (daily max minus daily min).
    The spread is the upper bound on single-cycle battery arbitrage revenue
    per MWh of capacity — it directly motivates why 2022 drives higher
    battery revenues than 2019 in the dispatch model.

    CAPTION: "Empirical CDF of the within-day price spread for DK1 2019 and
    2022. The median daily spread is 165 EUR/MWh in 2022 versus 24 EUR/MWh
    in 2019 -- a 7x difference that directly explains the higher arbitrage
    revenues modelled in the 2022 dispatch."
    """
    # Sized at 0.58 textwidth so it looks balanced next to the timeseries
    # when placed side-by-side in LaTeX at \includegraphics[width=0.49\textwidth]
    fig, ax = plt.subplots(figsize=figsize(0.58, aspect=0.88))

    v22, f22 = ecdf(ds22)
    v19, f19 = ecdf(ds19)

    ax.plot(v22, f22, color=C_2022, lw=1.5, label="DK1 2022")
    ax.plot(v19, f19, color=C_2019, lw=1.5, label="DK1 2019")

    # Median markers with value annotation
    for ds, col in [(ds22, C_2022), (ds19, C_2019)]:
        med = float(np.median(ds))
        ax.axvline(med, color=col, lw=0.8, ls=(0, (2, 3)), alpha=0.80)
        ax.text(med + (v22.max() * 0.015), 0.06,
                f"{med:.0f} EUR/MWh",
                color=col, fontsize=FS_ANNOT, va="bottom", rotation=90)

    ax.set_xlabel("Daily price spread  (EUR / MWh)")
    ax.set_ylabel("Cumulative fraction of days  (–)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="lower right")

    _save(fig, "fig_dk1_ecdf_spread", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Loading price data...")
    d19 = load_prices(FILE_2019)
    d22 = load_prices(FILE_2022)
    ds19 = daily_spread(d19)
    ds22 = daily_spread(d22)

    print("\n── Caption values ── (copy into .tex)")
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
