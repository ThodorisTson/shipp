"""
Replot the Test 1 frozen-dispatch slope figures from a verification_summary CSV.

The figures are redrawn without the in-figure equation legend and ratio
annotation. The two lines and the slope ratio are described in the thesis
caption instead. Fonts are enlarged because the panels are placed at half text
width.

The script reads every row of the most recent verification_summary_*.csv it can
find (searched recursively under this script's folder), so it regenerates one
figure per price set from the run's own numbers. No LP solve and no rainflow are
required: the finite-difference points and both slopes are columns in the CSV.

Run from VS Code or a terminal:
    python replot_test1_fd.py
Place this file in the Degradation folder, next to the Gradient_Verification
output tree, or set CSV_PATH below to an explicit file.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent

# Optional explicit CSV. Leave as None to auto-find the most recent summary.
CSV_PATH: Path | None = None

OUT_DIR = SCRIPT_DIR / "Gradient_Verification" / "test1_replot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Thesis style ------------------------------------------------------------
for _d in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
    if (_d / "thesis_style.py").exists():
        sys.path.insert(0, str(_d))
        break
try:
    from thesis_style import apply_thesis_style, figsize
    PALETTE = apply_thesis_style(palette="brand", usetex=False)
except Exception:                                   # pragma: no cover
    # TU Delft fallback (matches thesis_style brand palette if import fails).
    PALETTE = {"primary": "#0C2340", "secondary": "#A50034",
               "fill_a": "#0076C2", "neutral": "#404040"}
    def figsize(w, aspect=0.74):
        return (6.30 * w, 6.30 * w * aspect)

_TS_RE = re.compile(r"(\d{8}_\d{6})")


def find_summary_csv() -> Path:
    """Most recent verification_summary_*.csv under SCRIPT_DIR (recursive)."""
    if CSV_PATH is not None:
        return Path(CSV_PATH)
    cands = list(SCRIPT_DIR.rglob("verification_summary_*.csv"))
    if not cands:
        raise FileNotFoundError(
            "No verification_summary_*.csv found under "
            f"{SCRIPT_DIR}. Set CSV_PATH explicitly.")

    def _key(p: Path) -> str:
        m = _TS_RE.search(p.name)
        return m.group(1) if m else p.name

    return max(cands, key=_key)


def plot_fd_validation(fd_lo, fd_mid, fd_hi, eps_frac,
                       total_pred, total_complete, tag: str) -> Path:
    """Frozen-dispatch slope figure. Mirrors the run-script plot (no legend)."""
    pad = eps_frac * 0.4
    x   = np.linspace(1.0 - eps_frac - pad, 1.0 + eps_frac + pad, 80)
    y_full  = fd_mid + total_complete * (x - 1.0)
    y_naive = fd_mid + total_pred     * (x - 1.0)

    fig, ax = plt.subplots(figsize=figsize(0.74, aspect=0.74))
    ax.plot(x, y_full, "-", color=PALETTE["secondary"], lw=2.0, zorder=3)
    ax.plot(x, y_naive, ":", color=PALETTE["neutral"], lw=1.6, alpha=0.8, zorder=2)
    ax.plot([1.0 - eps_frac, 1.0, 1.0 + eps_frac], [fd_lo, fd_mid, fd_hi],
            "o", color=PALETTE["fill_a"], ms=8, zorder=5)

    ax.set_xlabel("SoC scale factor  (1.0 = nominal dispatch)")
    ax.set_ylabel(r"Annual degradation fraction  $f_d$  (-)")

    stem = OUT_DIR / f"test1_fd_{tag}"
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=300)
    plt.close(fig)
    return Path(f"{stem}.pdf")


def main() -> None:
    csv = find_summary_csv()
    print(f"Reading: {csv}")
    df = pd.read_csv(csv)
    need = ["priceset", "fd_lo", "fd_mid", "fd_hi", "eps_frac",
            "total_pred", "total_complete"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing columns: {missing}")

    for _, r in df.iterrows():
        if str(r.get("fd_consistency_status", "OK")) == "FAILED":
            continue
        out = plot_fd_validation(
            float(r["fd_lo"]), float(r["fd_mid"]), float(r["fd_hi"]),
            float(r["eps_frac"]), float(r["total_pred"]),
            float(r["total_complete"]), str(r["priceset"]),
        )
        ratio = float(r["fd_slope"]) / float(r["total_complete"])
        print(f"  {r['priceset']:8} full-slope ratio {ratio:.4f}  ->  {out.name}")

    print(f"\nFigures written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
