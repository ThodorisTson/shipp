r"""
verify_tables_dk.py
===================
Prints every metric that appears in the two DK results tables
(tab:eol_comparison and tab:degradation_yr1), read straight from the run's
CSV outputs, so you can diff them against the LaTeX by eye.

Reusable for BOTH years: point RUN_DIR / YEAR_TAG at the 2019 or 2022 outputs
and rerun. Depends only on pandas + standard library. Windows / VS Code safe:
all paths are anchored to this file's folder.

Sources used
  battery_degradation_results_*.csv : fd split, SoH yr1, EFC, rainflow count,
                                      mean depth, mean SoC, EoL 80/70/60
  battery_optimization_results_*.csv: single-year LP NPV (dispatch_fixed row)
  npv_summary_*.csv                 : multi-year NPV, PV replacements,
                                      replacement years, final SoH
"""
from __future__ import annotations
from pathlib import Path
import glob
import pandas as pd

HERE     = Path(__file__).parent
RUN_DIR  = HERE                      # folder holding the CSVs
YEAR_TAG = "dk2019"                  # "dk2019" or "dk2022"


def _one(pattern: str) -> Path:
    hits = sorted(glob.glob(str(RUN_DIR / pattern)))
    if not hits:
        raise FileNotFoundError(f"no file matching {pattern} in {RUN_DIR}")
    return Path(hits[-1])             # newest by name (timestamp-prefixed)


def main():
    deg = pd.read_csv(_one(f"battery_degradation_results_*{YEAR_TAG}*.csv")).iloc[0]
    opt = pd.read_csv(_one(f"battery_optimization_results_*{YEAR_TAG}*.csv"))
    npv = pd.read_csv(_one(f"npv_summary_*{YEAR_TAG}*.csv")).iloc[0]

    # single-year LP NPV = dispatch_fixed row of the optimization file
    disp = opt[opt["optimization_type"] == "dispatch_fixed"].iloc[0]

    cyc_pct = 100.0 - float(deg["fd_calendar_pct"])

    print(f"\n================  {YEAR_TAG.upper()}  ================\n")

    print("TABLE 1 — tab:eol_comparison")
    print("-" * 52)
    print(f"  EoL at SoH = 80%     : {deg['eol_80_yr']:.2f} yr")
    print(f"  EoL at SoH = 70%     : {deg['eol_70_yr']:.2f} yr")
    print(f"  Replacement year     : {npv['replacement_years']}")
    print(f"  Final SoH at year 20 : {npv['final_soh_pct']:.2f} %")
    print(f"  NPV (single-year LP) : {disp['npv_MEUR']:.1f} M EUR   (total, no deg)")
    print(f"  PV replacement cost  : {npv['pv_replacements_MEUR']:.2f} M EUR")
    print(f"  NPV (multi-year net) : {npv['npv_total_multiyear_MEUR']:.1f} M EUR")

    print("\nTABLE 2 — tab:degradation_yr1  (year 1)")
    print("-" * 52)
    print(f"  fd (total)           : {deg['fd_total']:.5f}")
    print(f"  fd_cycle             : {deg['fd_cycle']:.5f}  ({cyc_pct:.0f}%)")
    print(f"  fd_calendar          : {deg['fd_calendar']:.5f}  ({deg['fd_calendar_pct']:.0f}%)")
    print(f"  Year-1 SoH           : {deg['soh_pct']:.1f} %")
    print(f"  Equivalent full cyc. : {deg['total_cycles_efc']:.0f}")
    print(f"  n_rainflow_cycles    : {deg['n_rainflow_cycles']:.0f}   <-- confirm: half-cycles or full?")
    print(f"  mean depth (delta)   : {deg['mean_dod_pct']:.1f} %   <-- this is delta-bar, NOT the DoD design param")
    print(f"  mean SoC (sigma)     : {deg['mean_soc_pct']:.1f} %")

    print("\nSupport values")
    print("-" * 52)
    print(f"  NPV no-battery (wind): {opt[opt['no_battery']==True].iloc[0]['npv_MEUR']:.1f} M EUR")
    print(f"  npv_bat multi-year   : {npv['npv_bat_multiyear_MEUR']:.2f} M EUR")
    print(f"  npv_bat no-deg       : {npv['npv_bat_no_deg_MEUR']:.2f} M EUR")
    print(f"  npv_bat Plan B       : {npv['npv_bat_planB_MEUR']:.2f} M EUR")
    print(f"  eta_ac / eta_oneway  : {npv['rte_ac']:.4f} / {npv['eta_oneway']:.4f}")
    print()


if __name__ == "__main__":
    main()
