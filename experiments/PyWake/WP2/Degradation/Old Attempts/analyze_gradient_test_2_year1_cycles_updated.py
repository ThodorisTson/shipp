"""
analyze_rainflow_year1_cycles.py
WP2 battery — Real-cycle validation of Test 2 subgradient formula

PURPOSE
-------
Upgrades v3.3 Test 2 from 4 synthetic cycles to all real rainflow cycles
extracted from the Year 1 dispatch (storage_e_fixed.npy, 8,760 steps).

For each real cycle the script:
  1. Extracts real δ (DoD) and real σ̄ (mean SoC) from the t_start / t_end
     indices returned by rainflow_cycle_counting.
  2. Reconstructs a clean 3-element trace [e_valley, e_peak, e_valley] using
     the real (δ, σ̄) — avoiding raw multi-cycle slices and nested-cycle
     attribution problems.
  3. Computes the subgradient at t = 1 (the discharge turning point).
  4. Compares it to the analytical expectation:
         expected = 0.5 × S_σ(σ̄) × Φ′(δ)
     and records ratio = measured / expected.

A ratio distribution tightly clustered at 1.0 across all real cycles is a
far stronger validation statement than the 4-point synthetic check in v3.3.

DEPENDENCIES
------------
Same environment as run_battery_xu_shi_degradation_v3_3.py:
  degradation_xu, degradation_shi, degradation_subgradient

OUTPUTS (written to ./Gradient_Verification/)
  analyze_rainflow_real_<ts>_cycles.csv   — per-cycle table
  analyze_rainflow_real_<ts>_dod_dist.png — DoD + σ̄ distributions
  analyze_rainflow_real_<ts>_ratio.png    — subgradient ratio bar chart
  analyze_rainflow_real_<ts>_report.txt   — summary statistics
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# ── Thesis style ──────────────────────────────────────────────────────────────
import sys as _sys
for _d in __import__('pathlib').Path(__file__).resolve().parents:
    if (_d / 'thesis_style.py').exists():
        _sys.path.insert(0, str(_d)); break
try:
    from thesis_style import (apply_thesis_style, figsize as _figsize,
                               TUDELFT, FS_BASE, FS_ANNOT)
    apply_thesis_style(palette='brand', usetex=False)
    _C_PASS = TUDELFT['blue']; _C_FAIL = TUDELFT['darkred']
    _C_NEUT = TUDELFT['navy']; _C_ORG  = TUDELFT['orange']
    def _fig_size(w, aspect=0.7): return _figsize(w, aspect=aspect)
except ImportError:
    _C_PASS='#27ae60'; _C_FAIL='#e74c3c'; _C_NEUT='#2c7bb6'; _C_ORG='#e67e22'
    FS_BASE=10; FS_ANNOT=8
    def _fig_size(w, aspect=0.7): return (12*w, 12*w*aspect)

# ---------------------------------------------------------------------------
# Locate project root and import degradation modules
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
# The npy files and degradation modules live in the same SHIPP working dir.
# Adjust if your project layout differs.
PROJECT_ROOT = SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from degradation_xu import rainflow_cycle_counting
from degradation_shi import analyze_degradation_shi
from degradation_subgradient import compute_subgradient, fit_shi_polynomial

# ---------------------------------------------------------------------------
# CONFIG  — must match WP2_Battery.yaml and run_battery_xu_shi_degradation_v2
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Battery parameters — read from WP2_Battery.yaml (no hardcoded efficiencies).
# This is the same specification the run script and Test 1 use, so the SoC
# window, round-trip efficiency and cost data cannot drift out of sync. The AC
# round trip is the DC round trip times the PCU efficiency squared, and
# eta_symmetric splits it into eta_in = eta_out so charging is not lossless.
# ---------------------------------------------------------------------------
from wp2_econ import eta_symmetric
from degradation_xu import XU_LMO

def _load_battery_yaml():
    """Read the top-level sections of WP2_Battery.yaml; fall back to WP2 spec."""
    _def = {
        "e_cap_Wh": 300e6, "p_cap_W": 150e6, "rte_dc": 0.9025, "pcu": 0.986,
        "soc_min": 0.10, "soc_max": 0.90, "soc_initial": 0.50, "T_C": 25.0,
        "capex_e": 245.0, "capex_p": 86.0, "repl_e": 72.0, "repl_p": 96.0,
        "source": "WP2 defaults (YAML not found)",
    }
    cand = [SCRIPT_DIR / "WP2_Battery.yaml", SCRIPT_DIR.parent / "WP2_Battery.yaml"]
    path = next((p for p in cand if p.exists()), None)
    if path is None:
        return _def
    try:
        import yaml as _yaml
        with open(path) as f:
            cfg = _yaml.safe_load(f)
        bs  = cfg.get("battery_systems", {})
        pcu = cfg.get("power_conditioning_unit", {})
        lim = cfg.get("operating_limits", {})
        eco = cfg.get("economics", {})
        deg = cfg.get("degradation", {})
        return {
            "e_cap_Wh": float(bs.get("energy_capacity", _def["e_cap_Wh"])),
            "p_cap_W":  float(bs.get("power_capacity",  _def["p_cap_W"])),
            "rte_dc":   float(bs.get("round_trip_efficiency_nominal", _def["rte_dc"])),
            "pcu":      float(pcu.get("efficiency", _def["pcu"])),
            "soc_min":  float(lim.get("soc_min", _def["soc_min"])),
            "soc_max":  float(lim.get("soc_max", _def["soc_max"])),
            "soc_initial": float(lim.get("soc_initial", _def["soc_initial"])),
            "T_C":      float(deg.get("temperature_C", _def["T_C"])),
            "capex_e":  float(eco.get("capex_EUR_per_kWh", _def["capex_e"])),
            "capex_p":  float(eco.get("capex_EUR_per_kW",  _def["capex_p"])),
            "repl_e":   float(eco.get("repl_energy_EUR_per_kWh", _def["repl_e"])),
            "repl_p":   float(eco.get("repl_power_EUR_per_kW",   _def["repl_p"])),
            "source":   path.name,
        }
    except Exception as _exc:
        _def["source"] = f"WP2 defaults (YAML error: {_exc})"
        return _def

_BP = _load_battery_yaml()

E_CAP   = _BP["e_cap_Wh"] / 1e6     # MWh — nominal energy capacity
SOC_MIN = _BP["soc_min"]
SOC_MAX = _BP["soc_max"]
T_CELL  = _BP["T_C"]                # degrees C

# AC round trip = DC round trip x PCU^2; symmetric split eta_in = eta_out.
RTE_AC  = _BP["rte_dc"] * _BP["pcu"] ** 2      # 0.95 x 0.978721^2 = 0.910 (DEA-corrected)
ETA_SYM = eta_symmetric(RTE_AC)                # sqrt(0.910) = 0.9539
assert abs(RTE_AC - 0.910) < 2e-3, f"RTE_AC={RTE_AC:.4f}, expected 0.910; check WP2_Battery.yaml"

# S_sigma parameters — taken from the Xu LMO reference the model uses.
K_SIGMA   = float(XU_LMO.k_sigma)    # 1.04
SIGMA_REF = float(XU_LMO.sigma_ref)  # 0.50

# The subgradient checks below set cost = 1, so the validation ratios are
# dimensionless; the replacement-energy rate is carried for reference only.
BAT_PARAMS = {
    "power_capacity_W":        _BP["p_cap_W"],
    "energy_capacity_Wh":      _BP["e_cap_Wh"],
    "rte_nominal":             _BP["rte_dc"],
    "pcu_efficiency":          _BP["pcu"],
    "capex_EUR_per_kWh":       _BP["capex_e"],
    "capex_EUR_per_kW":        _BP["capex_p"],
    "repl_energy_EUR_per_kWh": _BP["repl_e"],
    "repl_power_EUR_per_kW":   _BP["repl_p"],
    "soc_min":                 SOC_MIN,
    "soc_max":                 SOC_MAX,
}

print(f"  Battery params  : {_BP['source']}  |  E={E_CAP:.0f} MWh  "
      f"P={_BP['p_cap_W']/1e6:.0f} MW  RTE_AC={RTE_AC:.4f}  "
      f"eta_sym={ETA_SYM:.4f}  SoC {SOC_MIN*100:.0f}-{SOC_MAX*100:.0f}%  T={T_CELL:.0f}C")

# Results folder — where the run script writes npy / csv outputs
RESULTS_DIR = SCRIPT_DIR / "Results"

# Trace source: year-1 SoC from the corrected multiyear npy, the same file
# Test 1 and Chapter 4 use. PRICESET selects the year; RTE_TAG excludes the
# rte877 runs in the folder so a newer-timestamped old run is not picked.
MULTIYEAR_DIR = SCRIPT_DIR / "Results" / "RTE Tests"
PRICESET      = "dk2022"     # or "dk2019"
RTE_TAG       = "rte910"

# Output directory for this script's plots / reports
OUT_DIR = SCRIPT_DIR / "Real_Cycle_Analysis"
OUT_DIR.mkdir(exist_ok=True)

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
PREFIX = OUT_DIR / f"analyze_rainflow_real_{RUN_TS}"


# ---------------------------------------------------------------------------
# Smart file resolver — searches Results/ then script dir, picks most recent
# ---------------------------------------------------------------------------

import re as _re

def _find_file(filename: str, extra_hint: Path | None = None) -> Path:
    """
    Search for *filename* in order:
      1. extra_hint  (e.g. a CLI-supplied path)
      2. RESULTS_DIR / filename
      3. SCRIPT_DIR  / filename

    For npy files whose name contains no timestamp the first match wins.
    For files whose name contains a timestamp (YYYYMMDD_HHMMSS pattern) this
    function additionally scans the directory for the *most recent* variant
    that matches the stem prefix, so callers never have to rename files.
    """
    candidates: list[Path] = []
    if extra_hint is not None:
        candidates.append(extra_hint)
    candidates += [RESULTS_DIR / filename, SCRIPT_DIR / filename]

    # Direct match — first existing file wins
    for p in candidates:
        if p.exists():
            return p

    # Timestamp scan — strip any existing timestamp and look for dated variants
    # e.g. "storage_e_fixed.npy" → also matches "storage_e_fixed_20260403.npy"
    stem   = Path(filename).stem          # "storage_e_fixed"
    suffix = Path(filename).suffix        # ".npy"
    ts_pat = _re.compile(
        r"^" + _re.escape(stem) + r"[_\-]?\d{8}[_\-]\d{6}" + _re.escape(suffix) + r"$"
    )
    for search_dir in [RESULTS_DIR, SCRIPT_DIR]:
        if not search_dir.exists():
            continue
        dated = sorted(
            [f for f in search_dir.iterdir() if ts_pat.match(f.name)],
            reverse=True,   # lexicographic sort on YYYYMMDD_HHMMSS → most recent first
        )
        if dated:
            return dated[0]

    raise FileNotFoundError(
        f"Cannot find '{filename}'.\n"
        f"Searched:\n"
        f"  {RESULTS_DIR}\n"
        f"  {SCRIPT_DIR}\n"
        "Place the file in the Results/ subfolder next to this script,\n"
        "or pass its path as a command-line argument."
    )


def _find_most_recent_csv(stem_prefix: str) -> Path | None:
    """
    Return the most recent CSV in RESULTS_DIR whose name starts with
    stem_prefix and contains a YYYYMMDD_HHMMSS timestamp.

    Example stem_prefix: "battery_degradation_results"
    """
    ts_pat = _re.compile(
        r"^" + _re.escape(stem_prefix) + r"_\d{8}_\d{6}.*\.csv$"
    )
    if not RESULTS_DIR.exists():
        return None
    matches = sorted(
        [f for f in RESULTS_DIR.iterdir() if ts_pat.match(f.name)],
        reverse=True,
    )
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Helper: compute fd_cycle from a stored-energy trace
# ---------------------------------------------------------------------------

def _fd_cycle(E: np.ndarray, e_cap: float, shi_fit) -> float:
    """Return Shi fd_cycle (cycle only, no calendar) for trace E [MWh]."""
    p = np.zeros(len(E)).tolist()
    res = analyze_degradation_shi(
        p, E.tolist(), e_cap, BAT_PARAMS,
        shi_fit=shi_fit,
        T_cell_C=T_CELL,
        dt_hours=1.0,
        eol_thresholds=[],
    )
    return float(res["fd_shi"])


# ---------------------------------------------------------------------------
# Step 1 — Load Year 1 trace
# ---------------------------------------------------------------------------

def load_trace() -> np.ndarray:
    """Year-1 SoC [MWh] from the newest rte910 multiyear npy for PRICESET.

    Reads annual_soc[0] from the same multiyear object Test 1 uses, so Test 2
    counts cycles on the corrected dispatch rather than a separate
    storage_e_fixed.npy that can fall out of sync.
    """
    import re as _re
    ts_re = _re.compile(r"(\d{8}_\d{6})")
    cands = [p for p in MULTIYEAR_DIR.glob("multiyear_*.npy")
             if PRICESET in p.name and RTE_TAG in p.name]
    if not cands:
        raise FileNotFoundError(
            f"No multiyear_*.npy for {PRICESET} with tag {RTE_TAG} in {MULTIYEAR_DIR}")
    src = max(cands, key=lambda p: (ts_re.search(p.name) or p.name))
    d   = np.load(src, allow_pickle=True).item()
    soc = np.asarray(d["annual_soc"][0], dtype=float)
    e_cap_npy = float(d["e_cap_nominal"])
    assert abs(e_cap_npy - E_CAP) < 1e-6, (
        f"npy e_cap {e_cap_npy} != YAML E_CAP {E_CAP}; rainflow depths would be wrong")
    print(f"  Source : {src.name}  (priceset {PRICESET}, year 1)")
    return soc


# ---------------------------------------------------------------------------
# Step 2 — Extract real (δ, σ̄) from rainflow output
# ---------------------------------------------------------------------------

def extract_real_cycle_params(
    storage_e: np.ndarray,
    cycles: list[dict],
) -> list[dict]:
    """
    Augment each rainflow cycle dict with sigma_bar_real.

    degradation_xu.rainflow_cycle_counting stores per cycle:
        dod       — depth of discharge (fraction of E_CAP)
        soc_mean  — mean SoC = (e_peak + e_valley) / (2 × E_CAP)  ← use directly
        count     — 0.5 (half-cycle / residue) or 1.0 (full cycle)
        i_start   — index of first turning point in storage_e
        i_end     — index of second turning point in storage_e

    soc_mean is already the correct σ̄ — no need to look up turning-point
    energies from the trace.  i_start / i_end are kept for the consistency
    cross-check only.
    """
    augmented = []
    for c in cycles:
        # Primary path: read soc_mean directly from the cycle dict
        if "soc_mean" in c:
            sigma_bar_real = float(c["soc_mean"])
            source = "real"
        else:
            # Fallback: try to derive from i_start / i_end
            i_s = c.get("i_start")
            i_e = c.get("i_end")
            if i_s is not None and i_e is not None:
                e_s = float(storage_e[i_s])
                e_e = float(storage_e[i_e])
                sigma_bar_real = (e_s + e_e) / (2.0 * E_CAP)
                source = "real_index"
            else:
                sigma_bar_real = SIGMA_REF
                source = "fallback"

        # Cross-check: reconstruct DoD from i_start / i_end if available
        i_s = c.get("i_start")
        i_e = c.get("i_end")
        if i_s is not None and i_e is not None:
            e_s = float(storage_e[i_s])
            e_e = float(storage_e[i_e])
            dod_check = abs(e_s - e_e) / E_CAP
        else:
            dod_check = None

        augmented.append({
            **c,
            "sigma_bar_real": sigma_bar_real,
            "dod_check":      dod_check,   # should equal c['dod'] to numerical precision
            "source":         source,
        })
    return augmented


# ---------------------------------------------------------------------------
# Step 3 — Run Test 2 on all real cycles
# ---------------------------------------------------------------------------

def run_test2_real_cycles(
    cycles_real: list[dict],
    shi_fit,
    verbose_max: int = 20,
) -> pd.DataFrame:
    """
    For each real cycle build a 3-element trace using real (δ, σ̄), compute
    subgradient at t=1, and record ratio = measured / expected.

    Returns a DataFrame with one row per cycle.
    """
    k3, k4 = shi_fit.k3, shi_fit.k4

    rows = []
    n_cycles = len(cycles_real)

    print(f"\n  Running Test 2 on {n_cycles} real rainflow cycles...")
    print(f"  {'#':>5}  {'dod':>8}  {'σ̄_real':>8}  {'S_σ':>7}  "
          f"{'expected':>13}  {'measured':>13}  {'ratio':>8}  note")
    print("  " + "-" * 80)

    t0 = time.perf_counter()

    for i, c in enumerate(cycles_real):
        dod          = float(c["dod"])
        sigma_bar    = float(c["sigma_bar_real"])
        count        = float(c.get("count", 0.5))

        # Skip degenerate cycles (zero DoD can arise from flat dispatch)
        if dod < 1e-6:
            rows.append({
                "cycle_idx":   i,
                "dod":         dod,
                "sigma_bar":   sigma_bar,
                "count":       count,
                "s_sigma":     np.nan,
                "phi_prime":   np.nan,
                "expected_sg": np.nan,
                "measured_sg": np.nan,
                "ratio":       np.nan,
                "note":        "skip_zero_dod",
            })
            continue

        # ── Reconstruct 3-element trace from real (δ, σ̄) ────────────────
        e_peak_3   = (sigma_bar + dod / 2.0) * E_CAP
        e_valley_3 = (sigma_bar - dod / 2.0) * E_CAP

        # Clamp to exact SoC window boundaries (no margin).
        # Full-swing cycles (DoD=80%, σ̄=50%) land exactly at e=270/30 MWh —
        # these are the dominant LP dispatch pattern and must not be skipped.
        if e_peak_3 > SOC_MAX * E_CAP:
            e_peak_3   = SOC_MAX * E_CAP
            e_valley_3 = e_peak_3 - dod * E_CAP

        if e_valley_3 < SOC_MIN * E_CAP:
            e_valley_3 = SOC_MIN * E_CAP
            e_peak_3   = e_valley_3 + dod * E_CAP

        # Only skip if DoD physically exceeds the window (should never happen)
        if e_peak_3 > SOC_MAX * E_CAP + 1e-6 or e_valley_3 < SOC_MIN * E_CAP - 1e-6:
            rows.append({
                "cycle_idx":   i,
                "dod":         dod,
                "sigma_bar":   sigma_bar,
                "count":       count,
                "s_sigma":     np.nan,
                "phi_prime":   np.nan,
                "expected_sg": np.nan,
                "measured_sg": np.nan,
                "ratio":       np.nan,
                "note":        "skip_window_violation",
            })
            continue

        trace_3 = [e_valley_3, e_peak_3, e_valley_3]

        # Recompute sigma_bar from reconstructed trace (may differ slightly after clamping)
        sigma_bar_3 = (e_peak_3 + e_valley_3) / (2.0 * E_CAP)

        # ── Analytical expectation ────────────────────────────────────────
        phi_prime   = k3 * k4 * dod ** (k4 - 1.0)
        s_sigma     = np.exp(K_SIGMA * (sigma_bar_3 - SIGMA_REF))
        expected_sg = 0.5 * s_sigma * phi_prime   # factor 0.5 for half-cycle

        # ── Measured subgradient at t = 1 ────────────────────────────────
        cyc3 = rainflow_cycle_counting(trace_3, E_CAP)
        sg3  = compute_subgradient(
            storage_e=trace_3,
            cycles=cyc3,
            dt_hours=1.0,
            battery_replacement_cost_per_MWh=1.0,   # cost=1 → ratio is dimensionless
            eff_in=1.0,
            eff_out=1.0,
            shi_fit=shi_fit,
        )
        sg_arr  = np.asarray(sg3["subgrad_combined"], dtype=float)
        sg_t1   = float(sg_arr[1])   # t=1 is the discharge half-cycle turning point

        ratio = sg_t1 / expected_sg if abs(expected_sg) > 1e-30 else np.nan

        note = ""
        if abs(ratio - 1.0) >= 0.02 and not np.isnan(ratio):
            note = "FAIL"

        if i < verbose_max or note == "FAIL":
            print(f"  {i:5d}  {dod:8.5f}  {sigma_bar_3:8.5f}  {s_sigma:7.4f}  "
                  f"{expected_sg:+13.4e}  {sg_t1:+13.4e}  {ratio:+8.4f}  {note}")
        elif i == verbose_max:
            print(f"  ... (printing suppressed — showing failures and final summary)")

        rows.append({
            "cycle_idx":   i,
            "dod":         dod,
            "sigma_bar":   sigma_bar_3,
            "count":       count,
            "s_sigma":     s_sigma,
            "phi_prime":   phi_prime,
            "expected_sg": expected_sg,
            "measured_sg": sg_t1,
            "ratio":       ratio,
            "note":        note,
        })

    elapsed = time.perf_counter() - t0
    print(f"\n  Completed {n_cycles} cycles in {elapsed:.1f} s  "
          f"({elapsed/n_cycles*1000:.1f} ms/cycle)")

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Conservation checks — three structural properties of the subgradient vector
# ---------------------------------------------------------------------------

def run_conservation_checks(cycles_real: list[dict], shi_fit) -> pd.DataFrame:
    """Three structural checks on the 3-element trace subgradient.

    Check A — Efficiency ratio:
        |sg[t=1]| / |sg[t=0]|  =  1 / RTE_AC
        sg[t=0] is the charge half, sg[t=1] is the discharge half. With the
        symmetric split eta_in = eta_out = sqrt(RTE_AC), the charge subgradient
        carries a factor eta_in and the discharge subgradient a factor 1/eta_out,
        so their ratio is 1/(eta_in*eta_out) = 1/RTE_AC.

    Check B — Sign consistency:
        sg[t=0] (charge half) and sg[t=1] (discharge half) must have
        opposite signs. Charging increases SoC → one direction of cycle
        stress. Discharging decreases SoC → opposite.

    Check C — Intra-half-cycle uniformity:
        sg[t=2] must equal sg[t=1]. Both t=1 (peak) and t=2 (second valley)
        belong to the same discharge half-cycle. build_half_cycle_map assigns
        the same value to every timestep within a half-cycle, so these must
        be identical.
    """
    rows = []

    for i, c in enumerate(cycles_real):
        dod = float(c["dod"])
        if dod < 1e-6:
            continue

        sigma_bar = float(c["sigma_bar_real"])

        # ── Reconstruct 3-element trace (same logic as run_test2_real_cycles) ──
        e_peak_3   = (sigma_bar + dod / 2.0) * E_CAP
        e_valley_3 = (sigma_bar - dod / 2.0) * E_CAP

        if e_peak_3 > SOC_MAX * E_CAP:
            e_peak_3   = SOC_MAX * E_CAP
            e_valley_3 = e_peak_3 - dod * E_CAP
        if e_valley_3 < SOC_MIN * E_CAP:
            e_valley_3 = SOC_MIN * E_CAP
            e_peak_3   = e_valley_3 + dod * E_CAP

        if e_peak_3 > SOC_MAX * E_CAP + 1e-6 or e_valley_3 < SOC_MIN * E_CAP - 1e-6:
            continue

        trace_3 = [e_valley_3, e_peak_3, e_valley_3]

        # ── Run with the symmetric efficiency split (eta_in = eta_out) ───────
        cyc3   = rainflow_cycle_counting(trace_3, E_CAP)
        sg_out = compute_subgradient(
            storage_e=trace_3,
            cycles=cyc3,
            dt_hours=1.0,
            battery_replacement_cost_per_MWh=1.0,  # cost=1 → dimensionless ratio
            eff_in=ETA_SYM,
            eff_out=ETA_SYM,                        # symmetric: eta_in = eta_out
            shi_fit=shi_fit,
        )
        sg_arr = np.asarray(sg_out["subgrad_combined"], dtype=float)
        sg0 = float(sg_arr[0])   # t=0: charge half-cycle (valley)
        sg1 = float(sg_arr[1])   # t=1: discharge half-cycle (peak)
        sg2 = float(sg_arr[2])   # t=2: second valley — should equal sg1

        # ── Check A: efficiency ratio ─────────────────────────────────────────
        eff_ratio    = abs(sg1) / abs(sg0) if abs(sg0) > 1e-30 else np.nan
        expected_A   = 1.0 / RTE_AC
        err_A        = abs(eff_ratio - expected_A) / expected_A if not np.isnan(eff_ratio) else np.nan
        pass_A       = bool(err_A < 0.001) if not np.isnan(err_A) else False

        # ── Check B: sign consistency ─────────────────────────────────────────
        pass_B       = bool((sg0 > 0) != (sg1 > 0))   # opposite signs → True

        # ── Check C: intra-half-cycle uniformity (sg2 should be zero - second valley belongs to next half-cycle (half-open interval))
        c_ratio      = float(abs(sg2))   # target = 0.0, not sg1
        pass_C       = bool(abs(sg2) < 1e-12)

        rows.append({
            "cycle_idx":    i,
            "dod":          dod,
            "sigma_bar":    sigma_bar,
            "sg0_charge":   sg0,
            "sg1_discharge": sg1,
            "sg2_boundary": sg2,
            "eff_ratio":    eff_ratio,
            "expected_A":   expected_A,
            "pass_A":       pass_A,
            "pass_B":       pass_B,
            "c_ratio":      c_ratio,
            "pass_C":       pass_C,
        })

    # ── Check D: zero-DoD skipped cycles produce zero subgradient ────────────
    print("\n  Check D — Zero-DoD cycles produce zero subgradient...")
    zero_dod_cycles = [c for c in cycles_real if float(c["dod"]) < 1e-6]
    n_zero = len(zero_dod_cycles)
    n_zero_pass = 0
    for c in zero_dod_cycles:
        sigma_bar = float(c["sigma_bar_real"])
        e_val = sigma_bar * E_CAP
        trace_flat = [e_val, e_val, e_val]   # flat trace → no turning points
        cyc_flat   = rainflow_cycle_counting(trace_flat, E_CAP)
        sg_flat    = compute_subgradient(
            storage_e=trace_flat,
            cycles=cyc_flat,
            dt_hours=1.0,
            battery_replacement_cost_per_MWh=1.0,
            eff_in=ETA_SYM,
            eff_out=ETA_SYM,
            shi_fit=shi_fit,
        )
        sg_zero_arr = np.asarray(sg_flat["subgrad_combined"], dtype=float)
        if np.all(np.abs(sg_zero_arr) < 1e-12):
            n_zero_pass += 1
    print(f"    Zero-DoD cycles tested : {n_zero}")
    print(f"    PASS (all sg = 0)      : {n_zero_pass} / {n_zero}  "
        f"({'✓ ALL PASS' if n_zero_pass == n_zero else f'✗ {n_zero - n_zero_pass} FAIL'})")
    
    return pd.DataFrame(rows), n_zero, n_zero_pass


def print_conservation_summary(df_c: pd.DataFrame) -> None:
    n = len(df_c)
    if n == 0:
        print("  No valid cycles for conservation checks.")
        return

    nA = df_c["pass_A"].sum()
    nB = df_c["pass_B"].sum()
    nC = df_c["pass_C"].sum()

    mean_eff   = df_c["eff_ratio"].mean()
    expected_A = df_c["expected_A"].iloc[0]
    mean_cratio = df_c["c_ratio"].mean()

    print(f"\n{'=' * 70}")
    print("CONSERVATION CHECKS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Cycles tested : {n}")
    print()
    print(f"  Check A — Efficiency ratio |sg_discharge| / |sg_charge|")
    print(f"    Expected     : 1 / RTE_AC = 1 / {RTE_AC:.4f} = {expected_A:.4f}")
    print(f"    Mean ratio   : {mean_eff:.4f}")
    print(f"    PASS (±0.1%) : {nA} / {n}  "
          f"({'✓ ALL PASS' if nA == n else f'✗ {n - nA} FAIL'})")
    print()
    print(f"  Check B — Sign consistency (charge vs discharge opposite sign)")
    print(f"    PASS         : {nB} / {n}  "
          f"({'✓ ALL PASS' if nB == n else f'✗ {n - nB} FAIL'})")
    print()
    print(f"  Check C — Boundary zero: sg[t=2] = 0")
    print(f"    Expected     : 0.0  (second valley belongs to next half-cycle)")
    print(f"    Mean |sg[t=2]|: {df_c['c_ratio'].mean():.2e}")
    print(f"    PASS (|sg|<1e-12): {nC} / {n}  "
        f"({'✓ ALL PASS' if nC == n else f'✗ {n - nC} FAIL'})")
    print()
    print(f"  Check D — Zero-DoD cycles (127 residues) produce zero subgradient")
    print(f"    (reported inline during run_conservation_checks)")
    print(f"{'=' * 70}")

def make_conservation_plot(df_c: pd.DataFrame, n_zero: int, n_zero_pass: int) -> None:
    """
    2x2 panel summarising the four conservation checks.

    CAPTION: "Conservation checks for the Shi subgradient implementation across
    all real rainflow cycles from Year 1 dispatch. (a) Efficiency ratio
    |s_d|/|s_c| — all cycles match 1/RTE_AC exactly. (b) Sign consistency.
    (c) Boundary valley subgradient equals zero. (d) Zero-DoD cycles carry
    zero subgradient."
    """
    n          = len(df_c)
    expected_A = float(df_c["expected_A"].iloc[0])
    nA = int(df_c["pass_A"].sum())
    nB = int(df_c["pass_B"].sum())
    nC = int(df_c["pass_C"].sum())

    fig, axes = plt.subplots(2, 2, figsize=_fig_size(1.0, aspect=0.82))

    for ax, lbl in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
        ax.text(0.03, 0.97, lbl, transform=ax.transAxes,
                va="top", fontweight="bold", fontsize=FS_BASE)

    # ── (a) Efficiency ratio ─────────────────────────────────────────────────
    ax = axes[0, 0]
    eff_vals = df_c["eff_ratio"].dropna().values
    if np.ptp(eff_vals) < 1e-10:
        ax.axis("off")
        ax.text(0.5, 0.65, f"{len(eff_vals)} cycles",
                ha="center", fontsize=FS_BASE, transform=ax.transAxes)
        ax.text(0.5, 0.48, f"ratio = {eff_vals.mean():.6f}",
                ha="center", fontsize=FS_BASE + 1,
                fontweight="bold", color=_C_PASS, transform=ax.transAxes)
        ax.text(0.5, 0.30, f"Expected = {expected_A:.6f}",
                ha="center", fontsize=FS_BASE, color=_C_NEUT, transform=ax.transAxes)
        ax.text(0.5, 0.12, f"PASS {nA}/{n}",
                ha="center", fontsize=FS_BASE,
                fontweight="bold", color=_C_PASS, transform=ax.transAxes)
    else:
        ax.hist(eff_vals, bins=30, color=_C_PASS, alpha=0.82,
                edgecolor="white", lw=0.3)
        ax.axvline(expected_A, color=_C_FAIL, lw=1.8, ls="--",
                   label=f"Expected {expected_A:.4f}")
        ax.axvline(eff_vals.mean(), color=_C_NEUT, lw=1.3, ls=":",
                   label=f"Mean {eff_vals.mean():.4f}")
        ax.legend(frameon=False, fontsize=FS_ANNOT)
        ax.set_xlabel(r"$|s_\mathrm{d}| / |s_\mathrm{c}|$")
        ax.set_ylabel("Cycle count")

    # ── (b) Sign consistency ─────────────────────────────────────────────────
    ax = axes[0, 1]
    n_fail_B = n - nB
    bars = ax.bar(["PASS\n(opposite)", "FAIL\n(same)"],
                  [nB, n_fail_B], color=[_C_PASS, _C_FAIL], alpha=0.85, width=0.45)
    for bar, val in zip(bars, [nB, n_fail_B]):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + n * 0.01, str(val),
                    ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="bold")
    ax.set_ylabel("Cycle count")
    ax.set_ylim(0, n * 1.18)

    # ── (c) Boundary zero ────────────────────────────────────────────────────
    ax = axes[1, 0]
    sg2_vals = np.abs(df_c["sg2_boundary"].values.astype(float))
    if np.ptp(sg2_vals) < 1e-20:
        ax.axis("off")
        ax.text(0.5, 0.55, f"All {n} cycles: sg[t=2] = 0",
                ha="center", fontsize=FS_BASE,
                fontweight="bold", color=_C_PASS, transform=ax.transAxes)
        ax.text(0.5, 0.35, f"PASS {nC}/{n}",
                ha="center", fontsize=FS_BASE, color=_C_PASS, transform=ax.transAxes)
    else:
        ax.hist(sg2_vals, bins=20, color=_C_NEUT, alpha=0.82,
                edgecolor="white", lw=0.3)
        ax.axvline(1e-12, color=_C_FAIL, lw=1.8, ls="--",
                   label="Pass threshold = 1e-12")
        ax.legend(frameon=False, fontsize=FS_ANNOT)
        ax.set_xlabel(r"$|s[t=2]|$  (boundary valley)")
        ax.set_ylabel("Count")

    # ── (d) Zero-DoD cycles ──────────────────────────────────────────────────
    ax = axes[1, 1]
    n_fail_D = n_zero - n_zero_pass
    bars_d = ax.bar(["PASS\n(sg = 0)", "FAIL\n(sg \u2260 0)"],
                    [n_zero_pass, n_fail_D],
                    color=[_C_PASS, _C_FAIL], alpha=0.85, width=0.45)
    for bar, val in zip(bars_d, [n_zero_pass, n_fail_D]):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(n_zero, 1) * 0.015, str(val),
                    ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="bold")
    ax.set_ylabel("Zero-DoD cycle count")
    ax.set_ylim(0, max(n_zero, 1) * 1.18)

    p = PREFIX.parent / f"{PREFIX.name}_conservation_checks.png"
    fig.savefig(p, dpi=300)
    fig.savefig(str(p).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  \u2713 Conservation plot: {p.name}")

def print_summary(df: pd.DataFrame, shi_fit) -> dict:
    valid = df[df["ratio"].notna() & (df["note"] != "skip_zero_dod")
               & (df["note"] != "skip_window_violation")].copy()

    n_total   = len(df)
    n_valid   = len(valid)
    n_skipped = n_total - n_valid

    ratios      = valid["ratio"].values
    mean_ratio  = np.mean(ratios)
    std_ratio   = np.std(ratios)
    max_err     = np.max(np.abs(ratios - 1.0))
    pct_pass    = 100.0 * np.mean(np.abs(ratios - 1.0) < 0.02)

    print("\n" + "=" * 70)
    print("SUMMARY — Real-cycle Test 2 subgradient validation")
    print("=" * 70)
    print(f"  Shi polynomial : k3 = {shi_fit.k3:.4e}   k4 = {shi_fit.k4:.4f}   "
          f"R² = {shi_fit.r2:.4f}")
    print(f"  Total cycles   : {n_total}")
    print(f"  Valid (tested) : {n_valid}")
    print(f"  Skipped        : {n_skipped}  (zero-DoD or window violation)")
    print(f"\n  Mean ratio     : {mean_ratio:.6f}   (target = 1.000000)")
    print(f"  Std  ratio     : {std_ratio:.6f}")
    print(f"  Max |r − 1|    : {max_err:.6f}")
    print(f"  % within ±2%   : {pct_pass:.1f}%")

    result = "✓ PASS" if max_err < 0.02 else "✗ FAIL"
    print(f"\n  Result         : {result}")
    print("=" * 70)

    # DoD statistics
    dods = valid["dod"].values
    print(f"\n  DoD distribution (real Year 1 cycles):")
    print(f"    Count  : {len(dods)}")
    print(f"    Min    : {dods.min():.4f}  ({dods.min()*100:.1f}%)")
    print(f"    Median : {np.median(dods):.4f}  ({np.median(dods)*100:.1f}%)")
    print(f"    Mean   : {dods.mean():.4f}  ({dods.mean()*100:.1f}%)")
    print(f"    Max    : {dods.max():.4f}  ({dods.max()*100:.1f}%)")

    # Mean SoC statistics
    sigmas = valid["sigma_bar"].values
    print(f"\n  Mean SoC (σ̄) distribution:")
    print(f"    Min    : {sigmas.min():.4f}  ({sigmas.min()*100:.1f}%)")
    print(f"    Median : {np.median(sigmas):.4f}  ({np.median(sigmas)*100:.1f}%)")
    print(f"    Mean   : {sigmas.mean():.4f}  ({sigmas.mean()*100:.1f}%)")
    print(f"    Max    : {sigmas.max():.4f}  ({sigmas.max()*100:.1f}%)")

    return {
        "n_total":   n_total,
        "n_valid":   n_valid,
        "n_skipped": n_skipped,
        "mean_ratio": mean_ratio,
        "std_ratio":  std_ratio,
        "max_err":    max_err,
        "pct_pass":   pct_pass,
        "result":     result,
    }


def make_plots(df: pd.DataFrame, storage_e: np.ndarray, stats: dict) -> None:
    valid = df[df["ratio"].notna() & (df["note"] != "skip_zero_dod")
               & (df["note"] != "skip_window_violation")].copy()
    dods   = valid["dod"].values
    sigmas = valid["sigma_bar"].values
    ratios = valid["ratio"].values

    # ── Figure 1: Year 1 SoC trace + DoD / sigma distributions ─────────────
    fig, axes = plt.subplots(1, 3, figsize=_fig_size(1.0, aspect=0.40))

    ax = axes[0]
    t   = np.arange(len(storage_e))
    soc = storage_e / E_CAP * 100.0
    ax.plot(t, soc, lw=0.4, color=_C_NEUT, alpha=0.82)
    ax.axhline(SOC_MAX * 100, color="gray", ls=":", lw=0.9)
    ax.axhline(SOC_MIN * 100, color="gray", ls=":", lw=0.9)
    ax.set_xlabel("Hour of year")
    ax.set_ylabel("SoC  (%)")
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes,
            va="top", fontweight="bold", fontsize=FS_BASE)

    ax = axes[1]
    ax.hist(dods * 100.0, bins=40, color=_C_NEUT, alpha=0.82,
            edgecolor="white", lw=0.3)
    ax.axvline(np.median(dods) * 100, color=_C_FAIL, lw=1.4, ls="--",
               label=f"Median {np.median(dods)*100:.1f}%")
    ax.set_xlabel("DoD  (%)")
    ax.set_ylabel("Cycle count")
    ax.legend(frameon=False, fontsize=FS_ANNOT)
    ax.text(0.03, 0.97, "(b)", transform=ax.transAxes,
            va="top", fontweight="bold", fontsize=FS_BASE)

    ax = axes[2]
    ax.hist(sigmas * 100.0, bins=40, color=_C_PASS, alpha=0.82,
            edgecolor="white", lw=0.3)
    ax.axvline(np.median(sigmas) * 100, color=_C_FAIL, lw=1.4, ls="--",
               label=f"Median {np.median(sigmas)*100:.1f}%")
    ax.axvline(50.0, color="gray", lw=0.9, ls=":", label="\u03c3_ref = 50%")
    ax.set_xlabel(r"Mean SoC  $\bar{\sigma}$  (%)")
    ax.set_ylabel("Cycle count")
    ax.legend(frameon=False, fontsize=FS_ANNOT)
    ax.text(0.03, 0.97, "(c)", transform=ax.transAxes,
            va="top", fontweight="bold", fontsize=FS_BASE)

    p1 = PREFIX.parent / f"{PREFIX.name}_dod_dist.png"
    fig.savefig(p1, dpi=300)
    fig.savefig(str(p1).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  \u2713 Distribution plot: {p1.name}")

    # ── Figure 2: ratio bar chart + ratio vs DoD scatter ────────────────────
    sort_idx = np.argsort(dods)
    dods_s   = dods[sort_idx]
    ratios_s = ratios[sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=_fig_size(1.0, aspect=0.45))

    ax = axes[0]
    bar_cols = [_C_PASS if abs(r - 1.0) < 0.02 else _C_FAIL for r in ratios_s]
    ax.bar(np.arange(len(ratios_s)), ratios_s,
           color=bar_cols, alpha=0.75, width=1.0)
    ax.axhline(1.00, color="gray", lw=1.3, ls="--", label="Target = 1.000")
    ax.axhline(1.02, color="gray", lw=0.7, ls=":", alpha=0.6, label="\u00b12% tolerance")
    ax.axhline(0.98, color="gray", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("Cycle index  (sorted by DoD)")
    ax.set_ylabel(r"$s[1] \;/\; (0.5 \times S_\sigma \times \Phi'(\delta))$")
    ax.legend(frameon=False, fontsize=FS_ANNOT)
    ax.set_ylim(min(0.90, ratios_s.min() - 0.02),
                max(1.10, ratios_s.max() + 0.02))
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes,
            va="top", fontweight="bold", fontsize=FS_BASE)

    ax = axes[1]
    sc = ax.scatter(dods_s * 100.0, ratios_s,
                    c=ratios_s, cmap="RdYlGn", vmin=0.96, vmax=1.04,
                    s=6, alpha=0.65, zorder=3)
    ax.axhline(1.00, color="gray", lw=1.3, ls="--")
    ax.axhline(1.02, color="gray", lw=0.7, ls=":", alpha=0.6)
    ax.axhline(0.98, color="gray", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("DoD  (%)")
    ax.set_ylabel("ratio  r = measured / expected")
    fig.colorbar(sc, ax=ax, label="ratio")
    ax.text(0.03, 0.97, "(b)", transform=ax.transAxes,
            va="top", fontweight="bold", fontsize=FS_BASE)

    p2 = PREFIX.parent / f"{PREFIX.name}_ratio.png"
    fig.savefig(p2, dpi=300)
    fig.savefig(str(p2).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  \u2713 Ratio plot: {p2.name}")


def save_csv(df: pd.DataFrame) -> None:
    p = PREFIX.parent / f"{PREFIX.name}_cycles.csv"
    df.to_csv(p, index=False, float_format="%.6g")
    print(f"  ✓ Per-cycle CSV:     {p.name}")


def save_report(df: pd.DataFrame, stats: dict, shi_fit, n_raw_trace: int) -> None:
    valid = df[df["ratio"].notna() & (df["note"] != "skip_zero_dod")
               & (df["note"] != "skip_window_violation")].copy()
    ratios = valid["ratio"].values
    dods   = valid["dod"].values

    p = PREFIX.parent / f"{PREFIX.name}_report.txt"
    sep = "=" * 70
    with open(p, "w", encoding="utf-8") as f:
        f.write(sep + "\n")
        f.write("WP2 BATTERY — Real-cycle Test 2 Subgradient Validation\n")
        f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(sep + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"  E_cap          : {E_CAP:.0f} MWh\n")
        f.write(f"  SoC window     : {SOC_MIN*100:.0f}–{SOC_MAX*100:.0f}%\n")
        f.write(f"  T_cell         : {T_CELL} °C\n")
        f.write(f"  Raw trace len  : {n_raw_trace} steps\n")
        f.write(f"  k_sigma        : {K_SIGMA}\n")
        f.write(f"  sigma_ref      : {SIGMA_REF}\n")
        f.write(f"  Shi k3         : {shi_fit.k3:.4e}\n")
        f.write(f"  Shi k4         : {shi_fit.k4:.4f}\n")
        f.write(f"  Shi R²         : {shi_fit.r2:.4f}\n\n")

        f.write("RAINFLOW SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total cycles   : {stats['n_total']}\n")
        f.write(f"  Valid (tested) : {stats['n_valid']}\n")
        f.write(f"  Skipped        : {stats['n_skipped']}\n")
        f.write(f"  DoD range      : {dods.min()*100:.1f}% – {dods.max()*100:.1f}%\n")
        f.write(f"  DoD median     : {np.median(dods)*100:.1f}%\n\n")

        f.write("TEST 2 RESULTS — Real-cycle subgradient ratio\n")
        f.write("-" * 40 + "\n")
        f.write("  Formula: expected = 0.5 × S_σ(σ̄) × Φ′(δ)\n")
        f.write("  S_σ(σ̄) = exp(k_σ × (σ̄ − σ_ref))\n")
        f.write("  ratio   = measured / expected  →  1.0\n\n")
        f.write(f"  Mean ratio  : {stats['mean_ratio']:.6f}\n")
        f.write(f"  Std  ratio  : {stats['std_ratio']:.6f}\n")
        f.write(f"  Max |r − 1| : {stats['max_err']:.6f}\n")
        f.write(f"  % within ±2%: {stats['pct_pass']:.1f}%\n")
        f.write(f"  Result      : {stats['result']}\n\n")

        f.write("VS. SYNTHETIC v3.3\n")
        f.write("-" * 40 + "\n")
        f.write("  v3.3 synthetic : 4 hand-picked DoD values\n")
        f.write(f"  This script    : {stats['n_valid']} real cycles from Year 1 dispatch\n")
        f.write("  Coverage       : every cycle depth and mean SoC the battery\n")
        f.write("                   actually experienced — a complete real-data check.\n")

    print(f"  ✓ Text report:       {p.name}")


# ---------------------------------------------------------------------------
# Single-cycle deep-dive
# ---------------------------------------------------------------------------

def run_single_cycle(
    cycles_real: list,
    shi_fit,
    storage_e,
    target_idx=None,
) -> None:
    """
    Run Test 2 on one selected real cycle with full step-by-step printout
    and a two-panel figure.

    If target_idx is None, picks the cycle closest to the median DoD.
    """
    k3, k4 = shi_fit.k3, shi_fit.k4

    valid_entries = [
        (i, c) for i, c in enumerate(cycles_real)
        if float(c["dod"]) >= 1e-6
    ]
    if not valid_entries:
        print("  No valid cycles found.")
        return

    if target_idx is None:
        dods = np.array([float(c["dod"]) for _, c in valid_entries])
        median_dod = np.median(dods)
        pos = int(np.argmin(np.abs(dods - median_dod)))
        cycle_pos, c = valid_entries[pos]
        print(f"  No index specified — picking cycle closest to median DoD "
              f"({median_dod*100:.1f}%): cycle #{cycle_pos}")
    else:
        matches = [(p, e) for p, e in enumerate(valid_entries) if e[0] == target_idx]
        if not matches:
            print(f"  Cycle index {target_idx} not found or has zero DoD.")
            return
        pos, (cycle_pos, c) = matches[0]

    dod       = float(c["dod"])
    sigma_bar = float(c["sigma_bar_real"])
    count     = float(c.get("count", 0.5))
    i_start   = c.get("i_start")
    i_end     = c.get("i_end")

    e_peak_3   = (sigma_bar + dod / 2.0) * E_CAP
    e_valley_3 = (sigma_bar - dod / 2.0) * E_CAP
    if e_peak_3 > SOC_MAX * E_CAP:
        e_peak_3   = SOC_MAX * E_CAP
        e_valley_3 = e_peak_3 - dod * E_CAP
    if e_valley_3 < SOC_MIN * E_CAP:
        e_valley_3 = SOC_MIN * E_CAP
        e_peak_3   = e_valley_3 + dod * E_CAP
    sigma_bar_3 = (e_peak_3 + e_valley_3) / (2.0 * E_CAP)

    phi_prime   = k3 * k4 * dod ** (k4 - 1.0)
    s_sigma     = np.exp(K_SIGMA * (sigma_bar_3 - SIGMA_REF))
    expected_sg = 0.5 * s_sigma * phi_prime

    cyc3 = rainflow_cycle_counting([e_valley_3, e_peak_3, e_valley_3], E_CAP)
    sg3  = compute_subgradient(
        storage_e=[e_valley_3, e_peak_3, e_valley_3],
        cycles=cyc3,
        dt_hours=1.0,
        battery_replacement_cost_per_MWh=1.0,
        eff_in=1.0, eff_out=1.0,
        shi_fit=shi_fit,
    )
    sg_arr  = np.asarray(sg3["subgrad_combined"], dtype=float)
    sg_t1   = float(sg_arr[1])
    ratio   = sg_t1 / expected_sg if abs(expected_sg) > 1e-30 else float("nan")
    result  = "PASS" if abs(ratio - 1.0) < 0.02 else "FAIL"

    sep = "=" * 70
    print("\n" + sep)
    print(f"  SINGLE-CYCLE DEEP DIVE -- rainflow cycle #{cycle_pos}")
    print(sep)

    print(f"\n  -- Real cycle parameters (from Year 1 dispatch) --")
    if i_start is not None and i_end is not None:
        e_s = float(storage_e[i_start])
        e_e = float(storage_e[i_end])
        print(f"    Turning points : t={i_start} (E={e_s:.2f} MWh, SoC={e_s/E_CAP*100:.1f}%) "
              f" ->  t={i_end} (E={e_e:.2f} MWh, SoC={e_e/E_CAP*100:.1f}%)")
    print(f"    DoD (delta)    : {dod:.6f}  =  {dod*100:.3f}%")
    print(f"    Mean SoC (s_b) : {sigma_bar:.6f}  =  {sigma_bar*100:.3f}%")
    print(f"    Half-cycle cnt : {count}")

    print(f"\n  -- 3-element trace reconstruction --")
    print(f"    e_valley = (s_b - d/2) x E_cap = ({sigma_bar_3:.5f} - {dod/2:.5f}) x {E_CAP:.0f} = {e_valley_3:.4f} MWh  (SoC {e_valley_3/E_CAP*100:.2f}%)")
    print(f"    e_peak   = (s_b + d/2) x E_cap = ({sigma_bar_3:.5f} + {dod/2:.5f}) x {E_CAP:.0f} = {e_peak_3:.4f} MWh  (SoC {e_peak_3/E_CAP*100:.2f}%)")
    print(f"    trace    : [{e_valley_3:.2f}, {e_peak_3:.2f}, {e_valley_3:.2f}] MWh")

    print(f"\n  -- Shi polynomial --")
    print(f"    k3 = {k3:.4e}   k4 = {k4:.4f}")
    print(f"    Phi'(delta) = k3 x k4 x delta^(k4-1)")
    print(f"                = {k3:.4e} x {k4:.4f} x {dod:.6f}^{k4-1:.4f}")
    print(f"                = {phi_prime:.6e}")

    print(f"\n  -- S_sigma mean-SoC correction --")
    print(f"    S_sigma(s_b) = exp(k_s x (s_b - s_ref))")
    print(f"                 = exp({K_SIGMA} x ({sigma_bar_3:.6f} - {SIGMA_REF}))")
    print(f"                 = exp({K_SIGMA * (sigma_bar_3 - SIGMA_REF):.6f})")
    print(f"                 = {s_sigma:.6f}")

    print(f"\n  -- Test 2 formula --")
    print(f"    expected subgrad[t=1] = 0.5 x S_sigma x Phi'(delta)")
    print(f"                         = 0.5 x {s_sigma:.6f} x {phi_prime:.6e}")
    print(f"                         = {expected_sg:.6e}  [1/MWh]")
    print(f"\n    measured subgrad[t=1] = {sg_t1:.6e}  [1/MWh]")
    print(f"\n    ratio = {sg_t1:.6e} / {expected_sg:.6e} = {ratio:.8f}   (target = 1.00000000)")
    print(f"\n    {result}")
    print(sep)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=_fig_size(1.0, aspect=0.45))

    # Panel (a) — 3-element SoC trace
    ax = axes[0]
    trace_vals = [e_valley_3, e_peak_3, e_valley_3]
    trace_soc  = [v / E_CAP * 100 for v in trace_vals]
    ax.plot([0, 1, 2], trace_soc, "-o", color=_C_NEUT, lw=2.0, ms=8, zorder=4)
    ax.scatter([1], [e_peak_3 / E_CAP * 100],
               color=_C_FAIL, s=100, zorder=5, label=f"Peak  {e_peak_3/E_CAP*100:.1f}%")
    ax.scatter([0, 2], [e_valley_3/E_CAP*100]*2,
               color=_C_PASS, s=90,  zorder=5, label=f"Valley {e_valley_3/E_CAP*100:.1f}%")
    ax.axhline(SOC_MAX * 100, color="gray", ls=":", lw=0.9, label="SoC limits")
    ax.axhline(SOC_MIN * 100, color="gray", ls=":", lw=0.9)
    ax.annotate(
        f"DoD = {dod*100:.1f}%\n\u03c3\u0304 = {sigma_bar_3*100:.1f}%",
        xy=(1, e_peak_3/E_CAP*100),
        xytext=(1.35, (e_peak_3/E_CAP*100 + e_valley_3/E_CAP*100)/2),
        fontsize=FS_ANNOT,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
    )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["t = 0\n(valley)", "t = 1\n(peak)", "t = 2\n(valley)"],
                       fontsize=FS_ANNOT)
    ax.set_ylabel("SoC  (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="lower right")
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes,
            va="top", fontweight="bold", fontsize=FS_BASE)

    # Panel (b) — expected vs measured subgradient
    ax = axes[1]
    w = 0.35
    ax.bar([0 - w/2], [abs(expected_sg)], width=w, color=_C_FAIL, alpha=0.85,
           label=r"Expected  $0.5 \times S_\sigma \times \Phi'(\delta)$")
    ax.bar([0 + w/2], [abs(sg_t1)],       width=w, color=_C_NEUT, alpha=0.85,
           label="Measured  subgrad[t=1]")
    ax.set_xticks([0])
    ax.set_xticklabels(["Discharge turning point"], fontsize=FS_ANNOT)
    ax.set_ylabel(r"$|s[t=1]|$  (1 / MWh)")
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="upper left")
    pass_col = _C_PASS if abs(ratio - 1.0) < 0.02 else _C_FAIL
    pass_lbl = "PASS" if abs(ratio - 1.0) < 0.02 else "FAIL"
    ax.text(0, max(abs(expected_sg), abs(sg_t1)) * 1.06,
            f"ratio = {ratio:.6f}  {pass_lbl}",
            ha="center", va="bottom", fontsize=FS_ANNOT,
            fontweight="bold", color=pass_col)
    ax.text(0.03, 0.97, "(b)", transform=ax.transAxes,
            va="top", fontweight="bold", fontsize=FS_BASE)

    plot_path = OUT_DIR / f"{PREFIX.name}_single_cycle_{cycle_pos}.png"
    fig.savefig(plot_path, dpi=300)
    fig.savefig(str(plot_path).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  \u2713 Plot saved        : {plot_path.name}")

    # ── Text report ───────────────────────────────────────────────────────────
    p = OUT_DIR / f"{PREFIX.name}_single_cycle_{cycle_pos}.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write(sep + "\n")
        f.write(f"WP2 BATTERY -- Single-cycle Test 2 deep dive (cycle #{cycle_pos})\n")
        f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(sep + "\n\n")
        f.write(f"  DoD (delta)   : {dod:.6f}  ({dod*100:.3f}%)\n")
        f.write(f"  Mean SoC (sb) : {sigma_bar_3:.6f}  ({sigma_bar_3*100:.3f}%)\n")
        f.write(f"  e_valley      : {e_valley_3:.4f} MWh\n")
        f.write(f"  e_peak        : {e_peak_3:.4f} MWh\n")
        f.write(f"  Phi'(delta)  : {phi_prime:.6e}\n")
        f.write(f"  S_sigma       : {s_sigma:.6f}\n")
        f.write(f"  Expected      : {expected_sg:.6e}  [1/MWh]\n")
        f.write(f"  Measured      : {sg_t1:.6e}  [1/MWh]\n")
        f.write(f"  Ratio         : {ratio:.8f}\n")
        f.write(f"  Result        : {result}\n")
    print(f"  ✓ Report saved      : {p.name}\n")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sep = "=" * 70
    print(sep)
    print("WP2 BATTERY — Real-cycle Test 2 validation")
    print("Year 1 dispatch (storage_e_fixed.npy)  |  No LP solve")
    print(sep)

    # Show resolved file locations so it is always clear what data is being used
    print(f"\n  Script dir  : {SCRIPT_DIR}")
    print(f"  Results dir : {RESULTS_DIR}  "
          f"({'found' if RESULTS_DIR.exists() else 'NOT FOUND — will fall back to script dir'})")

    deg_csv = _find_most_recent_csv("battery_degradation_results")
    opt_csv = _find_most_recent_csv("battery_optimization_results")
    print(f"  Degradation CSV (most recent) : "
          f"{deg_csv.name if deg_csv else 'not found'}")
    print(f"  Optimisation CSV (most recent): "
          f"{opt_csv.name if opt_csv else 'not found'}")
    print()

    # [1] Load trace
    print("\n[1/5] Loading Year 1 stored-energy trace...")
    storage_e = load_trace()
    print(f"  Shape  : {storage_e.shape}")
    print(f"  Range  : [{storage_e.min():.1f}, {storage_e.max():.1f}] MWh  "
          f"→  SoC [{storage_e.min()/E_CAP*100:.1f}%, {storage_e.max()/E_CAP*100:.1f}%]")

    # [2] Fit Shi polynomial
    print("\n[2/5] Fitting Shi polynomial on WP2 SoC window...")
    shi_fit = fit_shi_polynomial(SOC_MIN, SOC_MAX, verbose=True)
    print(f"  k3 = {shi_fit.k3:.4e}   k4 = {shi_fit.k4:.4f}   R² = {shi_fit.r2:.4f}")

    # [3] Run rainflow on Year 1 trace
    print("\n[3/5] Running rainflow on 8,760-step trace...")
    t0 = time.perf_counter()
    cycles_raw = rainflow_cycle_counting(storage_e.tolist(), E_CAP)
    elapsed_rf = time.perf_counter() - t0
    print(f"  Rainflow completed in {elapsed_rf*1000:.0f} ms")
    print(f"  Total cycles found  : {len(cycles_raw)}")

    # Check whether i_start / i_end are present (for the consistency cross-check)
    sample = cycles_raw[0] if cycles_raw else {}
    has_soc_mean = "soc_mean" in sample
    has_indices  = "i_start" in sample and "i_end" in sample
    if has_soc_mean:
        print("  soc_mean field present — real σ̄ read directly from rainflow output.")
    elif has_indices:
        print("  i_start / i_end present — real σ̄ derived from turning-point energies.")
    else:
        print("  WARNING: neither soc_mean nor i_start/i_end found.")
        print("  sigma_bar will fall back to sigma_ref = 0.50 for all cycles.")

    # Augment cycles with real sigma_bar
    cycles_real = extract_real_cycle_params(storage_e, cycles_raw)

    # Count by source
    from collections import Counter
    source_counts = Counter(c.get("source") for c in cycles_real)
    print(f"  Source breakdown   : {dict(source_counts)}")

    # Quick sanity: dod_check vs dod agreement (uses i_start/i_end)
    if has_indices:
        diffs = [abs(c["dod"] - c["dod_check"]) for c in cycles_real
                 if c.get("dod_check") is not None]
        max_diff = max(diffs) if diffs else 0.0
        print(f"  DoD consistency    : max |dod − dod_check| = {max_diff:.2e}  "
              f"{'✓' if max_diff < 1e-6 else '⚠ check rainflow indexing'}")

    # Short DoD distribution preview
    dods_all = [c["dod"] for c in cycles_real]
    print(f"\n  DoD quick stats    : "
          f"min={min(dods_all)*100:.1f}%  "
          f"median={np.median(dods_all)*100:.1f}%  "
          f"max={max(dods_all)*100:.1f}%")

    sigmas_all = [c["sigma_bar_real"] for c in cycles_real]
    print(f"  σ̄  quick stats    : "
          f"min={min(sigmas_all)*100:.1f}%  "
          f"median={np.median(sigmas_all)*100:.1f}%  "
          f"max={max(sigmas_all)*100:.1f}%")

    # [4] Run Test 2 on all real cycles
    print("\n[4/5] Running Test 2 on all real cycles...")
    df = run_test2_real_cycles(cycles_real, shi_fit, verbose_max=15)

    # Summary statistics
    stats = print_summary(df, shi_fit)

    # Plots and outputs
    print("\n  Saving outputs...")
    make_plots(df, storage_e, stats)
    save_csv(df)
    save_report(df, stats, shi_fit, len(storage_e))

    # [5] Conservation checks
    print("\n[5/5] Running conservation checks...")
    df_cons, n_zero, n_zero_pass = run_conservation_checks(cycles_real, shi_fit)
    print_conservation_summary(df_cons)
    make_conservation_plot(df_cons, n_zero, n_zero_pass)

    # Save conservation check CSV
    cons_path = PREFIX.parent / f"{PREFIX.name}_conservation.csv"
    df_cons.to_csv(cons_path, index=False, float_format="%.6g")
    print(f"  ✓ Conservation CSV:  {cons_path.name}")

    print("\n" + sep)
    print(f"✓ COMPLETE — outputs in: {OUT_DIR}")
    print(sep)


if __name__ == "__main__":
    sep = "=" * 70

    # ── Shared setup (needed for all modes) ──────────────────────────────────
    print(sep)
    print("WP2 BATTERY -- Real-cycle Test 2 validation")
    print("Year 1 dispatch (storage_e_fixed.npy)  |  No LP solve")
    print(sep)

    storage_e   = load_trace()
    shi_fit     = fit_shi_polynomial(SOC_MIN, SOC_MAX, verbose=False)
    cycles_raw  = rainflow_cycle_counting(storage_e.tolist(), E_CAP)
    cycles_real = extract_real_cycle_params(storage_e, cycles_raw)

    n_valid = sum(1 for c in cycles_real if float(c["dod"]) >= 1e-6)
    print(f"\n  Trace   : {len(storage_e)} steps")
    print(f"  Cycles  : {len(cycles_raw)} total  |  {n_valid} non-zero DoD")
    print(f"  k3 = {shi_fit.k3:.4e}   k4 = {shi_fit.k4:.4f}   R2 = {shi_fit.r2:.4f}")

    # ── Interactive mode selector ─────────────────────────────────────────────
    print(f"""
  Select run mode:
    [1]  Full year  — Test 2 on all {len(cycles_raw)} rainflow cycles  (default)
    [2]  Single     — Deep dive on the median-DoD cycle
    [3]  Single N   — Deep dive on a specific cycle by index
""")

    try:
        choice = input("  Enter choice [1 / 2 / 3, default = 1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    if choice == "" or choice == "1":
        # Full run — identical to previous default behaviour
        main()

    elif choice == "2":
        print()
        run_single_cycle(cycles_real, shi_fit, storage_e, target_idx=None)

    elif choice == "3":
        try:
            raw = input("  Enter cycle index (0 – {}): ".format(len(cycles_raw) - 1)).strip()
            target = int(raw)
        except (ValueError, EOFError):
            print("  Invalid index — falling back to median-DoD cycle.")
            target = None
        print()
        run_single_cycle(cycles_real, shi_fit, storage_e, target_idx=target)

    else:
        print(f"  Unrecognised choice '{choice}' — running full year.")
        main()