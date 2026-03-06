"""
WP2 Battery Degradation Analysis (v2)
====================================

This module analyzes battery degradation from SHIPP dispatch outputs.

Inputs (from SHIPP):
  - storage_p: battery power [MW] per timestep
  - storage_e: battery energy / SOC [MWh] per timestep

Outputs:
  - Equivalent full cycles (EFC)
  - DoD histogram
  - Rainflow cycle list (ASTM E1049 via `rainflow` package)
  - Simple linear capacity fade + degraded capacities

Notes
-----
- SHIPP convention is typically: storage_p > 0 means discharge, storage_p < 0 means charge.
  This is consistent with OpSchedule.get_power_partition() usage mentioned in the original file.
- By default dt_hours=1.0 (hourly). If you change the timestep later, pass dt_hours explicitly.

Keep this file importable by both baseline and degradation runners:
  from degradation_v_2 import analyze_degradation, plot_degradation_analysis, print_degradation_report
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import rainflow  # type: ignore
except Exception:  # pragma: no cover
    rainflow = None


# =============================================================================
# Cycle counting
# =============================================================================

def count_equivalent_full_cycles(
    storage_p: List[float],
    storage_e: List[float],
    e_cap: float,
    dt_hours: float = 1.0,
) -> float:
    """
    Equivalent full cycles (EFC) using discharged energy.

    EFC = (total discharged energy [MWh]) / (nominal energy capacity [MWh])

    Parameters
    ----------
    storage_p
        Battery power [MW] at each timestep (+ discharge, - charge).
    storage_e
        Battery energy [MWh] at each timestep (kept for signature stability; not used here).
    e_cap
        Nominal energy capacity [MWh].
    dt_hours
        Timestep duration in hours.

    Returns
    -------
    float
        Equivalent full cycles over the provided horizon.
    """
    if e_cap <= 0:
        return 0.0

    p = np.asarray(storage_p, dtype=float)
    discharged_MWh = float(np.sum(p[p > 0.0]) * dt_hours)
    return discharged_MWh / float(e_cap)


def calculate_dod_distribution(
    storage_e: List[float],
    e_cap: float,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Depth of discharge (DoD) histogram.

    DoD = (e_cap - SOC) / e_cap

    Returns
    -------
    (bin_centers, counts)
    """
    if e_cap <= 0:
        edges = np.linspace(0, 1, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, np.zeros(n_bins, dtype=int)

    soc = np.asarray(storage_e, dtype=float)
    dod = (float(e_cap) - soc) / float(e_cap)
    dod = np.clip(dod, 0.0, 1.0)

    counts, bin_edges = np.histogram(dod, bins=n_bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, counts


# =============================================================================
# Capacity fade models
# =============================================================================

def simple_linear_fade(
    cycles: float,
    nominal_cycles: int = 4000,
    eol_capacity: float = 0.80,
) -> float:
    """
    Linear capacity fade model.

    Capacity decreases linearly from 100% at 0 cycles to eol_capacity at nominal_cycles.

    Returns
    -------
    float
        Capacity retention in [eol_capacity, 1.0].
    """
    if nominal_cycles <= 0:
        return 1.0

    if cycles >= nominal_cycles:
        return float(eol_capacity)

    fade_rate = (1.0 - float(eol_capacity)) / float(nominal_cycles)
    capacity = 1.0 - fade_rate * float(cycles)
    return float(max(capacity, float(eol_capacity)))


# =============================================================================
# Rainflow counting
# =============================================================================

def rainflow_cycle_counting(storage_e: List[float]) -> List[Dict]:
    """
    ASTM E1049 rainflow counting via `rainflow` package.

    Returns a list of dicts with:
      - depth_MWh: cycle range (depth)
      - mean_MWh: mean SOC during cycle
      - count: 0.5 or 1.0 (half vs full cycle)
      - dod: normalized DoD (depth / max(storage_e))
      - i_start, i_end: indices of cycle endpoints (if provided by library)

    Raises
    ------
    ImportError
        If the `rainflow` package is not installed.
    """
    if rainflow is None:
        raise ImportError(
            "rainflow package is not available. Install it with: pip install rainflow"
        )

    e = np.asarray(storage_e, dtype=float)
    e_max = float(np.max(e)) if e.size else 0.0
    if e_max <= 0:
        return []

    cycles: List[Dict] = []
    for rng, mean, count, i_start, i_end in rainflow.extract_cycles(e):
        cycles.append(
            {
                "depth_MWh": float(rng),
                "mean_MWh": float(mean),
                "count": float(count),
                "dod": float(rng) / e_max,
                "i_start": int(i_start),
                "i_end": int(i_end),
            }
        )
    return cycles


# =============================================================================
# Main analysis
# =============================================================================

def analyze_degradation(
    storage_p: List[float],
    storage_e: List[float],
    e_cap_nominal: float,
    battery_params: Dict,
    dt_hours: float = 1.0,
    n_bins_dod: int = 10,
    enable_rainflow: bool = True,
) -> Dict:
    """
    Complete degradation analysis from SHIPP optimization results.

    Parameters
    ----------
    storage_p, storage_e
        SHIPP dispatch outputs.
    e_cap_nominal
        Nominal energy capacity [MWh].
    battery_params
        Battery parameters (from wp2_common). Expected keys:
          - n_full_load_cycles (optional, default 4000)
          - eol_capacity_fraction (optional, default 0.80)
          - power_capacity_W (required to compute degraded power)
    dt_hours
        Timestep duration [h]. For hourly series, dt_hours=1.
    n_bins_dod
        Number of bins in DoD histogram.
    enable_rainflow
        If True, compute rainflow cycles using `rainflow` package.

    Returns
    -------
    dict
        Degradation metrics.
    """
    total_cycles = count_equivalent_full_cycles(
        storage_p=storage_p,
        storage_e=storage_e,
        e_cap=e_cap_nominal,
        dt_hours=dt_hours,
    )

    nominal_cycles = int(battery_params.get("n_full_load_cycles", 4000))
    eol_capacity = float(battery_params.get("eol_capacity_fraction", 0.80))

    capacity_retention = simple_linear_fade(total_cycles, nominal_cycles, eol_capacity)
    capacity_fade_percent = (1.0 - capacity_retention) * 100.0
    soh = capacity_retention * 100.0

    dod_bins, dod_counts = calculate_dod_distribution(storage_e, e_cap_nominal, n_bins=n_bins_dod)

    cycle_depths = rainflow_cycle_counting(storage_e) if enable_rainflow else []

    if "power_capacity_W" not in battery_params:
        raise KeyError("battery_params must include 'power_capacity_W'")

    p_cap_nominal_MW = float(battery_params["power_capacity_W"]) / 1e6

    return {
        "total_cycles": float(total_cycles),
        "capacity_retention": float(capacity_retention),
        "capacity_fade_percent": float(capacity_fade_percent),
        "soh": float(soh),
        "dod_distribution": (dod_bins, dod_counts),
        "cycle_depth_distribution": cycle_depths,
        "e_cap_degraded": float(e_cap_nominal) * float(capacity_retention),
        "p_cap_degraded": p_cap_nominal_MW * float(capacity_retention),
        "meta": {
            "dt_hours": float(dt_hours),
            "nominal_cycles": nominal_cycles,
            "eol_capacity_fraction": float(eol_capacity),
            "enable_rainflow": bool(enable_rainflow),
        },
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_degradation_analysis(
    degradation: Dict,
    storage_e: List[float],
    time_vec: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
    verbose: bool = False,
):
    """
    Create degradation analysis plots.

    Parameters
    ----------
    degradation
        Output of analyze_degradation.
    storage_e
        SOC series [MWh].
    time_vec
        Time axis (typically days).
    save_path
        If given, saves the figure.
    show
        If True, calls plt.show().
    verbose
        If True, prints save message.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1) SOC time series
    ax = axes[0, 0]
    ax.plot(time_vec, storage_e, linewidth=1)
    ax.axhline(
        degradation["e_cap_degraded"],
        linestyle="--",
        label=f"Degraded capacity ({degradation['soh']:.1f}% SoH)",
    )
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("State of Charge [MWh]")
    ax.set_title("Battery State of Charge")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) DoD distribution
    ax = axes[0, 1]
    dod_bins, dod_counts = degradation["dod_distribution"]
    width = 0.08 if len(dod_bins) <= 1 else float(0.8 * (dod_bins[1] - dod_bins[0]))
    ax.bar(dod_bins, dod_counts, width=width, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Depth of Discharge")
    ax.set_ylabel("Frequency")
    ax.set_title("DoD Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    # 3) Capacity fade curve
    ax = axes[1, 0]
    meta = degradation.get("meta", {})
    nominal_cycles = int(meta.get("nominal_cycles", 4000))
    eol_capacity = float(meta.get("eol_capacity_fraction", 0.80))

    cycles_now = float(degradation["total_cycles"])
    cycles_hi = max(cycles_now * 1.5, 1.0)
    cycles_range = np.linspace(0, cycles_hi, 100)
    capacity_curve = [simple_linear_fade(c, nominal_cycles, eol_capacity) * 100 for c in cycles_range]

    ax.plot(cycles_range, capacity_curve, linewidth=2)
    ax.scatter([cycles_now], [float(degradation["soh"])], s=80, zorder=5, label="Current")
    ax.axhline(eol_capacity * 100, linestyle="--", alpha=0.5, label=f"EOL ({eol_capacity*100:.0f}%)")
    ax.set_xlabel("Equivalent Full Cycles")
    ax.set_ylabel("Capacity Retention [%]")
    ax.set_title("Capacity Fade Model (Linear)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4) Cycle depth distribution (rainflow)
    ax = axes[1, 1]
    cycle_list = degradation.get("cycle_depth_distribution", [])
    if cycle_list:
        depths = [c["depth_MWh"] for c in cycle_list]
        counts = [c["count"] for c in cycle_list]
        ax.hist(depths, bins=20, weights=counts, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Cycle Depth [MWh]")
    ax.set_ylabel("Cycle count")
    ax.set_title("Cycle Depth Distribution (Rainflow)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        if verbose:
            print(f"Saved degradation analysis: {save_path}")

    if show:
        plt.show()

    return fig


# =============================================================================
# Summary report
# =============================================================================

def print_degradation_report(degradation: Dict, period_days: float, enabled: bool = True) -> None:
    """
    Print a formatted degradation analysis report.
    Set enabled=False to suppress output.
    """
    if not enabled:
        return

    print("\n" + "=" * 72)
    print("BATTERY DEGRADATION ANALYSIS")
    print("=" * 72)

    cycles = float(degradation["total_cycles"])

    print(f"\nSimulation period: {period_days:.1f} days")
    print("\nCycle analysis:")
    print(f"  Equivalent full cycles: {cycles:.2f}")
    print(f"  Cycles per day: {cycles / period_days:.3f}")
    print(f"  Cycles per year: {cycles / period_days * 365:.1f}")

    print("\nCapacity degradation:")
    print(f"  State of Health (SoH): {float(degradation['soh']):.2f}%")
    print(f"  Capacity retention: {float(degradation['capacity_retention']):.4f}")
    print(f"  Capacity fade: {float(degradation['capacity_fade_percent']):.2f}%")
    print(f"  Degraded capacity: {float(degradation['e_cap_degraded']):.2f} MWh")
    print(f"  Degraded power: {float(degradation['p_cap_degraded']):.2f} MW")

    print("\nDepth of discharge:")
    dod_bins, dod_counts = degradation["dod_distribution"]
    if np.sum(dod_counts) > 0:
        mean_dod = float(np.average(dod_bins, weights=dod_counts))
        max_dod = float(np.max(dod_bins))
    else:
        mean_dod, max_dod = 0.0, 0.0
    print(f"  Mean DoD: {mean_dod:.2f}")
    print(f"  Max DoD (bin center): {max_dod:.2f}")

    meta = degradation.get("meta", {})
    if meta:
        print("\nMeta:")
        print(f"  dt_hours: {meta.get('dt_hours')}")
        print(f"  nominal_cycles: {meta.get('nominal_cycles')}")
        print(f"  eol_capacity_fraction: {meta.get('eol_capacity_fraction')}")
        print(f"  enable_rainflow: {meta.get('enable_rainflow')}")

    print("\n" + "=" * 72)
