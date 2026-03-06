"""
WP2 Battery Degradation — Shi et al. (2018) Subgradient Interface
=================================================================

Implements the per-timestep subgradient of the rainflow degradation cost,
following the algorithm of:

  Shi, Y. et al., "A Convex Cycle-based Degradation Model for Battery
  Energy Storage Planning and Operation," ACC 2018.

This module sits on top of degradation_xu.py. It does NOT replace the
Xu et al. physical model — that remains the validated ground truth for
all degradation reporting. This module provides the gradient signal that
connects the SHIPP dispatch LP to the outer sizing loop.

Architecture
------------
    degradation_subgradient.py   ← this file (gradient interface)
        ↓ imports
    degradation_xu.py            ← physical model, unchanged
        ↓ imports
    shipp kernel_pyomo.py        ← LP solver (also unchanged)

The chain rule connecting the two levels is (from Jenna, meeting 6):

    dDeg/dDoD = (∂Deg/∂E*) × (∂E*/∂DoD)
                 ↑                  ↑
           compute_subgradient   LP dual price
           [this file]           [kernel_pyomo.py, Gap D]

Public API
----------
    build_half_cycle_map(cycles, storage_e) -> Dict[str, np.ndarray]
    compute_subgradient(storage_e, cycles, e_cap, dt_hours,
                        battery_replacement_cost_per_MWh,
                        eff_in, eff_out, T_C, p) -> Dict

Notes on Φ_xu vs Φ_shi (two-function architecture)
------------------------------------------------------
Xu's LMO S_δ(δ) is the validated physical ground truth for degradation
reporting (fd, SoH, EoL curves). However it is only convex above
δ ≈ 0.1437 — Shi Theorem 1 requires global convexity of Φ.

A SoC-window floor does NOT fix this. SoC constraints bound the battery
*state*, not *cycle depth*. Even with soc_min = 0.40, a small arbitrage
cycle 0.40→0.45→0.40 has DoD = 0.05, which lies in the non-convex region.

Solution (methodological decision, not a bug fix):
  - Xu Φ  (S_δ via fc_cycle / fc_cycle_derivative):
        Used for ALL degradation REPORTING — fd, SoH, EoL, cycle stats.
        Validated against LMO DST data. Physical ground truth.
  - Shi Φ (phi_shi_prime_with_stress, polynomial k3·δ^k4):
        Used ONLY for GRADIENT computation in compute_subgradient().
        Fitted to Xu S_δ over [0.15, 0.95], k4 = 1.318 > 1 → globally convex.
        Satisfies Shi Theorem 1 everywhere, including small-DoD cycles.

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from degradation_xu import (
    XuModelParams,
    XU_LMO,
    ShiPolynomialFit,
    _DEFAULT_SHI_FIT,
    fit_shi_polynomial,
    load_soc_window_from_yaml,
    fc_cycle_derivative,          # Xu Phi' — reporting / comparison only
    phi_shi_prime_with_stress,    # Shi Phi' — gradient computation
)


# =============================================================================
# Gap B — Timestep-to-half-cycle attribution map
# =============================================================================

def build_half_cycle_map(
    cycles: List[Dict],
    storage_e: List[float],
) -> Dict[str, np.ndarray]:
    """Map every timestep to its owning rainflow half-cycle.

    Implements the T_{v_i} / T_{w_j} sets from Shi et al. (2018) Eqs.
    14-15, which partition the time horizon into charging and discharging
    half-cycles. This partition is required to evaluate the per-timestep
    subgradient of the rainflow degradation cost.

    Strategy — loop over cycles, not timesteps
    ------------------------------------------
    A yearly simulation has ~8760 timesteps but typically only ~100-400
    rainflow cycles. By iterating over cycles and using numpy slice
    assignment (array[i0:i1] = value), the inner work is a C-level
    vectorised write rather than a Python loop over each timestep.
    This makes the total Python-level iteration count ~200 instead of
    8760, and the slice writes themselves are O(1) in Python overhead.

    Junction handling
    -----------------
    Some timesteps sit at the boundary between two cycles (e.g. point E'
    in Shi et al. Fig. 2). These are resolved by assigning the timestep
    to the deeper cycle — consistent with Shi et al.'s footnote on
    junction subgradient non-uniqueness. This is achieved by sorting
    cycles by DoD ascending before slicing, so deeper cycles overwrite
    shallower ones at shared indices.

    Direction convention
    --------------------
    'CHC' (charging half-cycle): energy level is rising — battery is
          absorbing power. SHIPP convention: storage_p < 0.
    'DHC' (discharging half-cycle): energy level is falling — battery
          is delivering power. SHIPP convention: storage_p > 0.
    Direction is derived from np.diff(storage_e), which is fully
    vectorised with zero Python-level loop.

    Args:
        cycles:    Output of rainflow_cycle_counting() from
                   degradation_xu.py. Each dict must contain:
                   'i_start', 'i_end', 'dod', 'soc_mean', 'count'.
        storage_e: Battery energy [MWh] time-series — the same array
                   passed to rainflow_cycle_counting().

    Returns:
        Dict with numpy arrays of shape (n,) where n = len(storage_e):
            'cycle_owner'  int   — index into cycles list (-1 = unattributed)
            'dod_at_t'     float — DoD of the owning cycle (0 if unattributed)
            'soc_at_t'     float — mean SoC of the owning cycle (0 if unattributed)
            'direction'    int   — +1 = DHC (discharging), -1 = CHC (charging),
                                    0 = flat / unattributed
            'is_junction'  bool  — True where a timestep was overwritten by a
                                   deeper cycle (junction point)
    """
    e = np.asarray(storage_e, dtype=float)
    n = len(e)

    # Initialise all arrays — -1 / 0 means "not attributed to any cycle"
    cycle_owner = np.full(n, -1, dtype=np.int32)
    dod_at_t    = np.zeros(n, dtype=np.float64)
    soc_at_t    = np.zeros(n, dtype=np.float64)
    is_junction = np.zeros(n, dtype=bool)

    # Sort cycles by DoD ascending so deeper cycles overwrite at junctions
    sorted_cycles = sorted(enumerate(cycles), key=lambda x: x[1]["dod"])

    for ci, c in sorted_cycles:
        i0 = int(min(c["i_start"], c["i_end"]))
        i1 = int(max(c["i_start"], c["i_end"]))

        # Clip to valid array range
        i0 = max(i0, 0)
        i1 = min(i1, n - 1)

        if i0 > i1:
            continue

        # Mark junctions: timesteps already owned by a shallower cycle
        already_owned = cycle_owner[i0 : i1 + 1] != -1
        is_junction[i0 : i1 + 1] = np.where(already_owned, True,
                                              is_junction[i0 : i1 + 1])

        # Slice assignment — one C-level write per array, not a Python loop
        cycle_owner[i0 : i1 + 1] = ci
        dod_at_t[i0 : i1 + 1]    = c["dod"]
        soc_at_t[i0 : i1 + 1]    = c["soc_mean"]

    # --- Direction: fully vectorised, zero loops ----------------------------
    # np.diff produces e[t+1] - e[t] for every t simultaneously (shape n-1)
    # Append the last value repeated so output stays shape (n,)
    delta_e = np.diff(e, append=e[-1])

    # +1 = discharging (energy falling), -1 = charging (energy rising), 0 = flat
    direction = np.sign(-delta_e).astype(np.int8)

    # Unattributed timesteps get direction 0
    direction = np.where(cycle_owner == -1, np.int8(0), direction)

    return {
        "cycle_owner": cycle_owner,
        "dod_at_t":    dod_at_t,
        "soc_at_t":    soc_at_t,
        "direction":   direction,
        "is_junction": is_junction,
    }


# =============================================================================
# Gap C — Per-timestep subgradient (Shi et al. Eqs. 17-18)
# =============================================================================

def compute_subgradient(
    storage_e: List[float],
    cycles: List[Dict],
    # e_cap: float,
    dt_hours: float,
    battery_replacement_cost_per_MWh: float,
    eff_in:   float = 1.0,
    eff_out:  float = 0.85,
    T_C:      float = 25.0,
    p:        XuModelParams = XU_LMO,
    shi_fit:  "Optional[ShiPolynomialFit]" = None,
) -> Dict:
    """Per-timestep subgradient of the rainflow degradation cost.

    Implements Shi et al. (2018) Eqs. 17-18:

        Charging timestep  t ∈ T_{v_i}:
            ∂f/∂c_t = Φ'(v_i) · (B · τ · η_in) / 2

        Discharging timestep  t ∈ T_{w_j}:
            ∂f/∂d_t = Φ'(w_j) · (B · τ) / (2 · η_out)

    where Φ'(·) = fc_cycle_derivative(dod, soc_mean) from degradation_xu,
    B = battery replacement cost [currency/MWh], τ = dt_hours [h].

    This provides the (∂Deg/∂E*) term in Jenna's chain rule:
        dDeg/dDoD = (∂Deg/∂E*) × (∂E*/∂DoD)
    The second term is the LP dual price extracted from SHIPP (Gap D).

    All heavy computation is vectorised over the full time horizon in
    two numpy calls after the cycle attribution map is built.

    Args:
        storage_e:   Battery energy [MWh] from os.storage_e[0].data.
        cycles:      Output of rainflow_cycle_counting() from
                     degradation_xu.py.
        e_cap:       Nominal energy capacity [MWh].
        dt_hours:    Timestep duration [h] — matches SHIPP's dt.
        battery_replacement_cost_per_MWh: B in Shi et al. [currency/MWh].
                     Use e_cost from the SHIPP Storage object.
        eff_in:      Charging efficiency η_in (SHIPP: eff_in, default 1.0).
        eff_out:     Discharging efficiency η_out (SHIPP: eff_out,
                     default 0.85 matching WP2_Battery.yaml).
        T_C:      Cell temperature [°C] (default 25°C, S_T = 1.0).
        p:        Xu model parameters (default LMO, Table I).
        shi_fit:  ShiPolynomialFit from fit_shi_polynomial(soc_min, soc_max).
                  If None, uses the module-level default [0,1] fit.
                  Pass a window-specific fit for better gradient accuracy:
                    shi_fit = fit_shi_polynomial(0.10, 0.90, verbose=True)

    Returns:
        Dict with:
            'subgrad_charge'    np.ndarray (n,) — ∂f/∂c_t, charging cost
                                sensitivity. Zero at discharging timesteps.
            'subgrad_discharge' np.ndarray (n,) — ∂f/∂d_t, discharging cost
                                sensitivity. Zero at charging timesteps.
            'subgrad_combined'  np.ndarray (n,) — signed combined signal:
                                positive at discharge timesteps (deeper
                                discharge costs more), negative at charge
                                timesteps. Ready for use in outer loop
                                gradient update.
            'attribution'       Dict from build_half_cycle_map() — exposes
                                cycle ownership and direction per timestep
                                for inspection and debugging.
            'cycle_coverage'    float — fraction of timesteps attributed
                                to a rainflow cycle (should be close to 1.0
                                for a well-cycling battery).
            'n_cycles'          int   — number of rainflow cycles used.
    """
    n   = len(storage_e)
    B   = float(battery_replacement_cost_per_MWh)
    tau = float(dt_hours)

    # Resolve Shi polynomial fit — use provided window-specific fit or fall back
    # to the module-level default.  Printing the source here makes the provenance
    # visible in every run log without adding a separate verbose flag.
    _fit = shi_fit if shi_fit is not None else _DEFAULT_SHI_FIT

    # --- Step 1: build attribution map (loop over cycles, not timesteps) ---
    attribution = build_half_cycle_map(cycles, storage_e)
    dod_at_t  = attribution["dod_at_t"]    # shape (n,)
    soc_at_t  = attribution["soc_at_t"]    # shape (n,)
    direction = attribution["direction"]   # shape (n,) int8: +1 DHC / -1 CHC / 0

    # --- Step 2: Phi'(delta) vectorised over all timesteps in one call --------
    # KEY ARCHITECTURAL DECISION: use phi_shi_prime_with_stress (Shi polynomial)
    # NOT fc_cycle_derivative (Xu S_delta).
    #
    # Xu's S_delta is non-convex below delta ~0.1437. Shi's polynomial k3*delta^k4
    # is globally convex (k4 > 1), satisfying Theorem 1 for all cycle depths
    # including small arbitrage cycles that SoC bounds cannot exclude.
    # Xu S_delta remains in use for all degradation reporting (fd, SoH, EoL).
    phi_prime = phi_shi_prime_with_stress(
        dod_at_t, soc_at_t, T_C, _fit.k3, _fit.k4, p
    )  # shape (n,)

    # Zero out unattributed timesteps (cycle_owner == -1)
    attributed = attribution["cycle_owner"] != -1
    phi_prime  = np.where(attributed, phi_prime, 0.0)

    # --- Step 3: apply Shi Eqs. 17-18 in a single np.where call ------------
    is_charging    = direction == -1   # CHC: energy rising, power negative
    is_discharging = direction == +1   # DHC: energy falling, power positive

    # Eq. 17 — charging half-cycle contribution
    subgrad_c = np.where(
        is_charging,
        phi_prime * (B * tau * eff_in) / 2.0,
        0.0,
    )

    # Eq. 18 — discharging half-cycle contribution
    subgrad_d = np.where(
        is_discharging,
        phi_prime * (B * tau) / (2.0 * eff_out),
        0.0,
    )

    # Combined signed signal for outer loop:
    # discharging → positive (more discharge = deeper cycle = more cost)
    # charging    → negative (more charging = deeper cycle = more cost,
    #               but sign flipped because charging power is negative in SHIPP)
    subgrad_combined = subgrad_d - subgrad_c

    cycle_coverage = float(np.sum(attributed)) / max(n, 1)

    return {
        "subgrad_charge":    subgrad_c,
        "subgrad_discharge": subgrad_d,
        "subgrad_combined":  subgrad_combined,
        "attribution":       attribution,
        "cycle_coverage":    cycle_coverage,
        "n_cycles":          len(cycles),
        "shi_fit":           _fit,      # provenance: which polynomial was used
    }


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    import time
    from pathlib import Path as _Path
    from degradation_xu import rainflow_cycle_counting

    print("=" * 70)
    print("degradation_subgradient.py — Two-Phi Self-Test")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load SoC window from YAML (or default to [0, 1])
    # ------------------------------------------------------------------
    _yaml_candidates = [
        _Path(__file__).parent / "WP2_Battery.yaml",
        _Path(__file__).parent.parent / "WP2_Battery.yaml",
    ]
    _yaml_path = next((p for p in _yaml_candidates if p.exists()), None)

    if _yaml_path:
        yaml_soc_min, yaml_soc_max, yaml_src = load_soc_window_from_yaml(_yaml_path)
    else:
        yaml_soc_min, yaml_soc_max = 0.0, 1.0
        yaml_src = "default [0,1] — WP2_Battery.yaml not found"

    print(f"\n  YAML source : {yaml_src}")
    print(f"  soc_min={yaml_soc_min}  soc_max={yaml_soc_max}  "
          f"max_dod={yaml_soc_max - yaml_soc_min:.2f}")

    # ------------------------------------------------------------------
    # Fit both polynomials and print them for validation
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("Shi polynomial fits — default [0,1] vs YAML window")
    print(f"{'─'*70}")

    print("\n  Default fit  soc=[0.0, 1.0]:")
    default_fit = fit_shi_polynomial(
        soc_min=0.0, soc_max=1.0, source="default [0,1]", verbose=True
    )

    print(f"\n  YAML fit     soc=[{yaml_soc_min}, {yaml_soc_max}]:")
    yaml_fit = fit_shi_polynomial(
        soc_min=yaml_soc_min, soc_max=yaml_soc_max,
        source=yaml_src, verbose=True
    )

    # ------------------------------------------------------------------
    # Real data test (if available)
    # ------------------------------------------------------------------
    SCRIPT_DIR  = Path(__file__).parent
    RESULTS_DIR = SCRIPT_DIR / "Results"
    soc_path    = RESULTS_DIR / "storage_e_fixed.npy"
    e_cap_path  = RESULTS_DIR / "e_cap_fixed.npy"

    print(f"\n{'─'*70}")
    print("Real SHIPP data  (Results/storage_e_fixed.npy)")
    print(f"{'─'*70}")

    if not soc_path.exists():
        print(f"\n  Not found — skipping real-data section.")
        print("  To enable, add to run_battery_with_degradation_v2.py after _solve_shipp():")
        print("    np.save(RESULTS_DIR / 'storage_e_fixed.npy',")
        print("            np.array(os_fixed.storage_e[0].data))")
        print("    np.save(RESULTS_DIR / 'e_cap_fixed.npy',")
        print("            np.array([os_fixed.storage_list[0].e_cap]))")
        _run_real = False
    else:
        _run_real = True

    if _run_real:
        storage_e = np.load(soc_path)
        e_cap     = float(np.load(e_cap_path)[0])
        n_steps   = len(storage_e)

        print(f"\n  Loaded: {soc_path.name}")
        print(f"  Timesteps : {n_steps}  ({n_steps / 8760 * 365:.1f} days)")
        print(f"  e_cap     : {e_cap:.1f} MWh")
        print(f"  SoC range : {storage_e.min():.2f} - {storage_e.max():.2f} MWh")
        print(f"  Mean SoC  : {storage_e.mean():.2f} MWh "
              f"({storage_e.mean()/e_cap*100:.1f}%)")

        t0     = time.perf_counter()
        cycles = rainflow_cycle_counting(storage_e, e_cap)
        t1     = time.perf_counter()
        dods   = np.array([c["dod"]   for c in cycles])
        cnts   = np.array([c["count"] for c in cycles])
        n_shallow = int(np.sum(dods < 0.1437))

        print(f"\n  Rainflow ({(t1-t0)*1000:.1f} ms):  {len(cycles)} cycles")
        print(f"  DoD range     : {dods.min():.3f} - {dods.max():.3f}")
        print(f"  Mean DoD (wt) : {np.average(dods, weights=cnts):.3f}")
        print(f"  Shallow (<0.1437, Xu non-convex) : {n_shallow} "
              f"({100*n_shallow/max(len(cycles),1):.1f}%)")
        if n_shallow > 0:
            print(f"  -> Non-convex region IS accessed — Shi Phi required ✓")

        t0     = time.perf_counter()
        attr   = build_half_cycle_map(cycles, storage_e)
        t1     = time.perf_counter()
        n_attr = int(np.sum(attr["cycle_owner"] != -1))
        print(f"\n  Gap B  build_half_cycle_map  ({(t1-t0)*1000:.1f} ms):")
        print(f"    Attributed : {n_attr}/{n_steps}  ({100*n_attr/n_steps:.1f}%)")
        print(f"    Junctions  : {int(np.sum(attr['is_junction']))}")
        print(f"    CHC steps  : {int(np.sum(attr['direction']==-1))}")
        print(f"    DHC steps  : {int(np.sum(attr['direction']==+1))}")

        B = 150.0 * 1000.0  # capex_EUR_per_kWh * 1000 from WP2_Battery.yaml
        eff_in  = 1.0
        eff_out = 0.85

        # Run with BOTH fits and compare gradient magnitudes
        for _label, _fit in [("default [0,1]", default_fit),
                              (f"YAML   [{yaml_soc_min},{yaml_soc_max}]", yaml_fit)]:
            t0 = time.perf_counter()
            sg = compute_subgradient(
                storage_e, cycles,
                dt_hours=1.0,
                battery_replacement_cost_per_MWh=B,
                eff_in=eff_in, eff_out=eff_out,
                shi_fit=_fit,
            )
            t1 = time.perf_counter()

            active_d = sg["subgrad_discharge"][sg["subgrad_discharge"] > 0]
            active_c = sg["subgrad_charge"][sg["subgrad_charge"] > 0]
            ratio    = active_d.mean() / active_c.mean()
            expected = eff_in / eff_out

            print(f"\n  Gap C  compute_subgradient  ({(t1-t0)*1000:.1f} ms)  fit={_label}:")
            print(f"    Phi: k3={_fit.k3:.4e}  k4={_fit.k4:.4f}  "
                  f"R²={_fit.r2:.4f}  (source: {_fit.source})")
            print(f"    Coverage         : {sg['cycle_coverage']*100:.1f}%")
            print(f"    subgrad_discharge: max={active_d.max():.4e}  "
                  f"mean={active_d.mean():.4e}")
            print(f"    subgrad_charge   : max={active_c.max():.4e}  "
                  f"mean={active_c.mean():.4e}")
            print(f"    discharge/charge : {ratio:.4f}  "
                  f"(expected {expected:.4f} = 1/eta_out)")
            assert sg["cycle_coverage"] > 0.7
            assert np.all(sg["subgrad_discharge"] >= 0)
            assert np.all(sg["subgrad_charge"]    >= 0)
            assert abs(ratio - expected) < 0.02, \
                f"ratio {ratio:.4f} != {expected:.4f}"
            print(f"    Assertions passed ✓")

        # Confirm shi_fit provenance is returned in dict
        assert sg["shi_fit"] is yaml_fit
        print(f"\n  shi_fit provenance returned in result dict ✓")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("Summary")
    print(f"{'─'*70}")
    print(f"  Phi_xu  -> REPORTING (fd, SoH, EoL)  — Xu Eq.32")
    print(f"  Phi_shi -> GRADIENTS (Shi Eqs.17-18) — k3*delta^k4")
    print()
    print(f"  Default : {default_fit.summary()}")
    print(f"  YAML    : {yaml_fit.summary()}")
    print()
    print(f"  In run scripts:")
    print(f"    soc_min, soc_max, _ = load_soc_window_from_yaml('WP2_Battery.yaml')")
    print(f"    shi_fit = fit_shi_polynomial(soc_min, soc_max, verbose=True)")
    print(f"    sg = compute_subgradient(..., shi_fit=shi_fit)")
    print(f"\n  Next: Gap D — extract LP dual prices from kernel_pyomo.py")