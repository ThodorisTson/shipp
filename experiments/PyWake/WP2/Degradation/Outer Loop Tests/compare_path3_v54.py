"""
compare_path3_v54.py — daily LP-vs-NLP comparison: path3 (monolithic NLP) against v5.4's method (LP dispatch + post-process degradation), scored with v5.4's OWN
degradation functions so the only thing that differs is the dispatch.

WHAT IT DOES (default, no re-solve):
  For each of the 9 study days it reads path3's saved dispatches from the npz (lp_e/lp_p = degradation-blind LP = v5.4's dispatch; nlp_e/nlp_p = monolithic),
  scores BOTH with v5.4's _shi_with_calendar_correction (Shi cycling fd) and _xu_full_degradation (Xu cycling fd, cross-check), and produces per-day plots,
  a 3x3 SoC grid, rainflow-decomposition figures, and a CSV + LaTeX table.

  path3 is already an ISOLATED 24 h problem, so its LP IS the dispatch v5.4 would produce for that day — no re-run of the optimizer is needed.

OPTIONAL (--verify-lp, needs Gurobi + PyWake + shipp):
  Re-solves the isolated 24 h LP with v5.4's solve_lp_pyomo and reports whether it lands on the same revenue AND SoC as path3's HiGHS LP. This is a diagnostic
  (different LP formulations can return different equally-optimal SoC paths on a degenerate optimum), not a gate.

Calendar is OFF on both sides: path3 has no calendar term, and v5.4's fd_cycle is cycling-only, so the two measure the same thing.

Usage (from the Outer Loop Tests folder):
    python compare_path3_v54.py --day D166          # one day first, recommended
    python compare_path3_v54.py                     # all nine
    python compare_path3_v54.py --verify-lp         # also re-solve + compare LP
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── make sibling modules importable (mirrors path3 / v5.4) ───────────────────
_HERE = Path(__file__).parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── study days: slot -> (day_index, regime) ─────────────────────────────────
DAYS = {
    "D8":   (7,   "High"), "D238": (237, "High"), "D248": (247, "High"),
    "D86":  (85,  "Mid"),  "D166": (165, "Mid"),  "D309": (308, "Mid"),
    "D43":  (42,  "Low"),  "D128": (127, "Low"),  "D197": (196, "Low"),
}
REGIME_ORDER = {"High": 0, "Mid": 1, "Low": 2}

# ── fixed conventions (match path3 / v5.4) ──────────────────────────────────
T_CELL_C = 25.0
DT_H     = 1.0
EOL_THRESH = [0.80, 0.70, 0.60]
REPL_E   = 72_000.0          # EUR/MWh, degradation valuation rate
E_CAP    = 550.0             # MWh (comparison size; fresh battery, e_cap_eff = nominal)
P_CAP    = 175.0             # MW
SOC_MIN, SOC_MAX = 0.10, 0.90
P_MAX_GRID = 325.0           # MW (grid connection; fallback if YAML not loaded)


# ── helpers ──────────────────────────────────────────────────────────────────
def load_v54(v54_path: Path):
    """Import v5.4 by file path (its name has spaces/specials)."""
    spec = importlib.util.spec_from_file_location("v54mod", str(v54_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)              # may pull shipp/pywake on import
    return mod


def daily_revenue(price, storage_p, curtailed, wind, p_max, dt=DT_H):
    """Raw (un-annualised) daily revenue, two bases. marginal == arbitrage when
    there is no wind/curtailment."""
    price = np.asarray(price, float); sp = np.asarray(storage_p, float)
    cu = np.asarray(curtailed, float); wd = np.asarray(wind, float)
    arb  = dt * float(np.dot(price, sp))
    marg = dt * (float(np.dot(price, (wd - cu) + sp))
                 - float(np.dot(price, np.minimum(wd, p_max))))
    return arb, marg


def rainflow_cycles(storage_e, e_cap):
    """v5.x rainflow decomposition -> list of dicts with dod (delta), soc_mean, count."""
    from degradation_xu import rainflow_cycle_counting
    return rainflow_cycle_counting(np.asarray(storage_e, float), e_cap)


def efc_from_cycles(cycles):
    """Equivalent full cycles = sum(depth * count)."""
    return float(sum(c["dod"] * c["count"] for c in cycles))


def score_cycling(storage_p, storage_e, bat_params, shi_fit, v54):
    """Cycling-only fd via v5.4's OWN functions. Returns (fd_shi, fd_xu)."""
    shi = v54._shi_with_calendar_correction(
        storage_p, storage_e, E_CAP, bat_params, shi_fit,
        T_CELL_C, DT_H, EOL_THRESH,
    )
    xu = v54._xu_full_degradation(storage_p, storage_e, E_CAP, T_CELL_C, DT_H)
    return float(shi["fd_cycle"]), float(xu["fd_cycle"])


def find_runs(results_dir: Path, want_e, want_p, want_wind=True):
    """Map slot -> latest matching (json_path, npz_path) at the requested config."""
    out = {}
    for jp in results_dir.glob("*_results.json"):
        npz = jp.with_name(jp.name.replace("_results.json", "_results.npz"))
        if not npz.exists():
            continue
        try:
            s = json.load(open(jp, encoding="utf-8"))["summary"]
        except Exception:
            continue
        if bool(s.get("wind_mode", False)) != want_wind:
            continue
        if abs(float(s["e_cap"]) - want_e) > 1e-6 or abs(float(s["p_cap"]) - want_p) > 1e-6:
            continue
        # slot from filename: ..._dk2022_D166_...
        slot = None
        for d in DAYS:
            if f"_{d}_" in jp.name:
                slot = d; break
        if slot is None:
            continue
        ts = jp.name[:15]                      # YYYYMMDD_HHMMSS prefix
        if slot not in out or ts > out[slot][0]:
            out[slot] = (ts, jp, npz, s)
    return {k: v[1:] for k, v in out.items()}  # slot -> (json, npz, summary)


def npz_get(z, key, n, default=0.0):
    """Defensive npz access (older path3 npz may lack curtailment/wind)."""
    if key in z.files:
        return np.asarray(z[key], float)
    return np.full(n, default, float)


# ── per-day comparison ───────────────────────────────────────────────────────
def compare_day(slot, jpath, npath, summary, bat_params, shi_fit, v54, outdir):
    day_idx, regime = DAYS[slot]
    z = np.load(npath, allow_pickle=True)
    prices = np.asarray(z["prices"], float)
    n = len(prices)

    lp_e, lp_p = np.asarray(z["lp_e"], float), np.asarray(z["lp_p"], float)
    nlp_e, nlp_p = np.asarray(z["nlp_e"], float), np.asarray(z["nlp_p"], float)
    wind   = npz_get(z, "wind", n)
    lp_cu  = npz_get(z, "lp_curtailed", n)
    nlp_cu = npz_get(z, "nlp_curtailed", n)

    # revenue (daily, both bases)
    lp_arb, lp_marg   = daily_revenue(prices, lp_p, lp_cu, wind, P_MAX_GRID)
    nlp_arb, nlp_marg = daily_revenue(prices, nlp_p, nlp_cu, wind, P_MAX_GRID)

    # degradation via v5.4's own functions (cycling only)
    lp_shi, lp_xu   = score_cycling(lp_p, lp_e, bat_params, shi_fit, v54)
    nlp_shi, nlp_xu = score_cycling(nlp_p, nlp_e, bat_params, shi_fit, v54)

    # rainflow decomposition + EFC
    lp_cyc, nlp_cyc = rainflow_cycles(lp_e, E_CAP), rainflow_cycles(nlp_e, E_CAP)
    lp_efc, nlp_efc = efc_from_cycles(lp_cyc), efc_from_cycles(nlp_cyc)

    # deg cost @72k and net (use marginal as headline, Shi as headline metric)
    lp_deg  = lp_shi * REPL_E * E_CAP
    nlp_deg = nlp_shi * REPL_E * E_CAP
    lp_net, nlp_net = lp_marg - lp_deg, nlp_marg - nlp_deg

    # Corroboration: path3's OWN net (from its JSON summary), computed with
    # path3's fd and revenue. Should match the verified net above to within
    # rounding, since the degradation kernels are bit-identical. We report it so
    # "the two independent pipelines agree" is shown, not asserted.
    lp_net_p3   = float(summary.get("lp_net",  float("nan")))
    nlp_net_p3  = float(summary.get("nlp_net", float("nan")))
    net_gain_p3 = nlp_net_p3 - lp_net_p3
    net_gain_v  = nlp_net - lp_net
    corrob = bool(np.sign(net_gain_v) == np.sign(net_gain_p3)) if abs(net_gain_v) > 1e-6 else True

    row = dict(
        case=slot, regime=regime,
        lp_efc=lp_efc, nlp_efc=nlp_efc,
        lp_fd_shi=lp_shi, nlp_fd_shi=nlp_shi,
        lp_fd_xu=lp_xu,   nlp_fd_xu=nlp_xu,
        lp_rev_arb=lp_arb, nlp_rev_arb=nlp_arb,
        lp_rev_marg=lp_marg, nlp_rev_marg=nlp_marg,
        lp_deg=lp_deg, nlp_deg=nlp_deg,
        lp_net=lp_net, nlp_net=nlp_net,
        sacrifice=lp_marg - nlp_marg,
        saving=lp_deg - nlp_deg,
        net_gain=nlp_net - lp_net,
        net_gain_path3=net_gain_p3, corroborates=corrob,
        ratio=(lp_deg - nlp_deg) / (lp_marg - nlp_marg) if abs(lp_marg - nlp_marg) > 1e-9 else 0.0,
        optimality=float(summary.get("final_optimality", float("nan"))),
        nlp_cycles=len(nlp_cyc), lp_cycles=len(lp_cyc),
    )

    # per-day plot (price / SoC overlay / dispatch diff)
    _plot_day(slot, regime, prices, lp_e, nlp_e, lp_p, nlp_p, outdir)
    _plot_rainflow(slot, lp_cyc, nlp_cyc, outdir)
    return row, (lp_e, nlp_e)


def _plot_day(slot, regime, prices, lp_e, nlp_e, lp_p, nlp_p, outdir):
    h = np.arange(len(prices))
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(h, prices, color="#2166ac"); ax[0].set_ylabel("Price [EUR/MWh]")
    ax[0].set_title(f"{slot} ({regime}) | LP (v5.4) vs NLP (path3)")
    he = np.arange(len(lp_e))
    ax[1].plot(he, lp_e, label="LP (v5.4)", color="#2166ac")
    ax[1].plot(he, nlp_e, label="NLP (path3)", color="#b5351b")
    ax[1].set_ylabel("Stored energy [MWh]"); ax[1].legend()
    d = nlp_p - lp_p
    ax[2].fill_between(h, 0, np.where(d > 0, d, 0), color="#2166ac", alpha=0.5, label="NLP > LP")
    ax[2].fill_between(h, 0, np.where(d < 0, d, 0), color="#b5351b", alpha=0.5, label="NLP < LP")
    ax[2].set_ylabel("Delta power [MW]"); ax[2].set_xlabel("Hour"); ax[2].legend()
    fig.tight_layout()
    fig.savefig(outdir / f"day_{slot}.png", dpi=140); plt.close(fig)


def _plot_rainflow(slot, lp_cyc, nlp_cyc, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for cyc, color, lab in [(lp_cyc, "#2166ac", "LP (v5.4)"),
                            (nlp_cyc, "#b5351b", "NLP (path3)")]:
        if not cyc:
            continue
        dod = [c["dod"] for c in cyc]
        soc = [c["soc_mean"] for c in cyc]
        cnt = [200 * c["count"] for c in cyc]
        ax.scatter(soc, dod, s=cnt, color=color, alpha=0.5, edgecolors="k",
                   linewidths=0.4, label=lab)
    ax.set_xlabel("Cycle mean SoC"); ax.set_ylabel("Cycle depth (delta)")
    ax.set_title(f"{slot}: rainflow cycle structure (marker size ~ count)")
    ax.legend(); fig.tight_layout()
    fig.savefig(outdir / f"rainflow_{slot}.png", dpi=140); plt.close(fig)


def _plot_grid(soc_pairs, rows, outdir):
    """3x3 small-multiples of SoC overlays, tagged by outcome."""
    order = sorted(rows, key=lambda r: (REGIME_ORDER[r["regime"]], r["case"]))
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for ax, r in zip(axes.flat, order):
        lp_e, nlp_e = soc_pairs[r["case"]]
        he = np.arange(len(lp_e))
        ax.plot(he, lp_e, color="#2166ac", lw=1.2)
        ax.plot(he, nlp_e, color="#b5351b", lw=1.2)
        opt = r["optimality"]
        tag = ("diverged" if opt > 1e-2 else
               "worse" if r["net_gain"] < -0.5 else
               "improved" if r["net_gain"] > 0.5 else "inert")
        ax.set_title(f'{r["case"]} ({r["regime"]}) — {tag}', fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(order):]:
        ax.axis("off")
    fig.suptitle("Battery SoC: LP (blue, v5.4) vs NLP (red, path3), isolated daily",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outdir / "soc_grid_3x3.png", dpi=140); plt.close(fig)


# ── tables ───────────────────────────────────────────────────────────────────
def write_tables(rows, outdir):
    order = sorted(rows, key=lambda r: (REGIME_ORDER[r["regime"]], r["case"]))
    # console
    hdr = (f'{"Case":<6}{"Reg":<5}{"EFC LP/NLP":>13}{"fd_cyc LP":>11}{"fd_cyc NLP":>11}'
           f'{"Sacr":>8}{"Save":>8}{"Ratio":>7}{"NetGain":>9}{"Opt":>11}')
    print(hdr); print("-" * len(hdr))
    for r in order:
        print(f'{r["case"]:<6}{r["regime"]:<5}'
              f'{r["lp_efc"]:>6.2f}/{r["nlp_efc"]:<6.2f}'
              f'{r["lp_fd_shi"]:>11.3e}{r["nlp_fd_shi"]:>11.3e}'
              f'{r["sacrifice"]:>8.2f}{r["saving"]:>8.2f}{r["ratio"]:>7.2f}'
              f'{r["net_gain"]:>9.2f}{r["optimality"]:>11.2e}')
    # CSV
    import csv
    cols = ["case","regime","lp_efc","nlp_efc","lp_fd_shi","nlp_fd_shi",
            "lp_fd_xu","nlp_fd_xu","lp_rev_marg","nlp_rev_marg","lp_deg","nlp_deg",
            "lp_net","nlp_net","sacrifice","saving","ratio","net_gain",
            "net_gain_path3","corroborates",
            "optimality","lp_cycles","nlp_cycles"]
    with open(outdir / "compare_path3_v54.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in order:
            w.writerow(r)
    # LaTeX (thesis style: \hline + \noalign{\smallskip}, open sides, no booktabs)
    L = [r"\begin{tabular}{l l r r r r r r}",
         r"\hline\noalign{\smallskip}",
         r"Case & Regime & EFC (LP/NLP) & $f_d$ LP & $f_d$ NLP & Sacrifice & Saving & Net gain \\",
         r"\noalign{\smallskip}\hline\noalign{\smallskip}"]
    for r in order:
        L.append(f'{r["case"]} & {r["regime"]} & {r["lp_efc"]:.2f}/{r["nlp_efc"]:.2f} & '
                 f'{r["lp_fd_shi"]:.2e} & {r["nlp_fd_shi"]:.2e} & {r["sacrifice"]:.2f} & '
                 f'{r["saving"]:.2f} & {r["net_gain"]:.2f} \\\\')
    L += [r"\noalign{\smallskip}\hline", r"\end{tabular}"]
    (outdir / "compare_path3_v54.tex").write_text("\n".join(L), encoding="utf-8")


# ── optional LP verification (re-solve isolated day with solve_lp_pyomo) ─────
def verify_lp(slot, npath, setup, v54):
    """Re-solve the isolated 24 h LP with v5.4's Pyomo kernel and compare it to
    path3's HiGHS LP. Needs Gurobi + shipp.

    The comparison is on DISPATCH POWER, not SoC. The kernel's `check_losses`
    warning (Error ~1e2 on the charge side) is a known, harmless artifact of the
    relaxed-LP storage formulation: the SoC-balance inequality goes slack at
    objective-indifferent and SoC-bound steps, so the recorded SoC/loss trace
    does not match the power-implied one. That slack lives entirely in the SoC/
    loss accounting; revenue and dispatch power are exact. So power is the
    slack-clean quantity to verify on; SoC is reported descriptively only.

    Returns dict: d_rev (strict), max_dp (power agreement), max_dsoc (slack-affected).
    """
    import warnings
    from shipp.kernel_pyomo import solve_lp_pyomo
    from shipp.components import Storage, Production
    from shipp.timeseries import TimeSeries

    z = np.load(npath, allow_pickle=True)
    prices = np.asarray(z["prices"], float); n = len(prices)
    wind = npz_get(z, "wind", n)

    # Mirror v5.4's WORKING call exactly: reuse its own storage builder and its
    # N_YEARS / discount_rate / p_min constants. The earlier bug was n_year=1 ->
    # wp2_econ.discount_weights sums range(1,1) = EMPTY -> zero revenue weight ->
    # the LP is indifferent to dispatch and returns a degenerate frozen battery.
    bat    = setup["battery"]
    rte_ac = float(bat["rte_nominal"]) * float(bat["pcu_efficiency"]) ** 2   # 0.877
    e_cost = float(bat["capex_EUR_per_kWh"]) * 1000.0                        # 245_000
    stor      = v54._build_storage_year(E_CAP, P_CAP, rte_ac, e_cost, SOC_MIN, SOC_MAX)
    stor_null = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0, e_cost=0, p_cost=0)
    price_ts  = TimeSeries(prices.tolist(), DT_H)
    prod      = Production(TimeSeries(wind.tolist(), DT_H), p_cost=0.0)
    prod_null = Production(TimeSeries([0.0] * n, DT_H), p_cost=0.0)
    # The known relaxed-LP slack fires check_losses -> RuntimeWarning. Suppress that
    # one warning (the kernel's raw print() lines can't be suppressed and are expected).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        os_lp = solve_lp_pyomo(
            price_ts, prod, prod_null, stor, stor_null,
            v54.discount_rate, v54.N_YEARS, v54.p_min, P_MAX_GRID, n,
            v54.pyo_solver, fixed_cap=True, soc_max1=SOC_MAX,
            return_duals=True, e_start1=None,
        )

    p_pyo = np.asarray(os_lp.storage_p[0].data, float)
    e_pyo = np.asarray(os_lp.storage_e[0].data, float)
    lp_p  = np.asarray(z["lp_p"], float)
    lp_e  = np.asarray(z["lp_e"], float)

    # ── DIAGNOSTIC: dump raw arrays so the cause is measured, not guessed ──────
    # (sign flip -> corr<0; unit error -> ranges differ by ~1e6 or 0-1 vs MWh;
    #  wrong array/index -> e_pyo constant or |p|>P_CAP)
    mp0 = min(len(p_pyo), len(lp_p))
    corr = float(np.corrcoef(p_pyo[:mp0], lp_p[:mp0])[0, 1]) if mp0 > 1 else float("nan")
    rev_pyo   = float(np.dot(prices[:mp0], p_pyo[:mp0]))
    rev_path3 = float(np.dot(prices[:mp0], lp_p[:mp0]))
    print(f"      [diag] p_pyo[min,max]=[{p_pyo.min():.1f},{p_pyo.max():.1f}] (P_CAP={P_CAP}) "
          f"lp_p[min,max]=[{lp_p.min():.1f},{lp_p.max():.1f}] corr={corr:+.3f}")
    print(f"      [diag] rev_pyo={rev_pyo:,.0f}  rev_path3={rev_path3:,.0f}  "
          f"e_pyo[min,max,len]=[{e_pyo.min():.1f},{e_pyo.max():.1f},{len(e_pyo)}] "
          f"lp_e[min,max,len]=[{lp_e.min():.1f},{lp_e.max():.1f},{len(lp_e)}]")

    # PRIMARY: power is slack-clean. Align on common length (kernels differ on
    # whether the SoC carries the boundary point: 24 vs 25).
    mp = min(len(p_pyo), len(lp_p))
    d_rev  = float(np.dot(prices[:mp], p_pyo[:mp]) - np.dot(prices[:mp], lp_p[:mp]))
    max_dp = float(np.max(np.abs(p_pyo[:mp] - lp_p[:mp])))
    # DESCRIPTIVE: SoC is slack-affected; small scattered diffs are expected.
    ms = min(len(e_pyo), len(lp_e))
    max_dsoc = float(np.max(np.abs(e_pyo[:ms] - lp_e[:ms])))
    return dict(d_rev=d_rev, max_dp=max_dp, max_dsoc=max_dsoc)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="Results_Path3", help="path3 results folder")
    ap.add_argument("--v54", default=None, help="path to the v5.4 .py (auto-detected if omitted)")
    ap.add_argument("--out", default="Comparison_v54_path3")
    ap.add_argument("--day", default=None, help="single slot e.g. D166 (default: all 9)")
    ap.add_argument("--e-cap", type=float, default=E_CAP)
    ap.add_argument("--p-cap", type=float, default=P_CAP)
    ap.add_argument("--verify-lp", action="store_true",
                    help="also re-solve the isolated LP via solve_lp_pyomo (needs Gurobi/PyWake)")
    args = ap.parse_args()

    # Anchor relative paths to the SCRIPT location (_HERE), not the launch CWD,
    # so the tool behaves the same whether you run it from shipp\ or from inside
    # the Outer Loop Tests folder. Absolute paths (or --results / --out) still win.
    def _anchor(p):
        p = Path(p)
        return p if p.is_absolute() else (_HERE / p)

    results_dir = _anchor(args.results)
    # Each invocation writes into its own timestamped subfolder so a re-run (or a
    # half-failed run) can never overwrite a previous one. Everything downstream
    # uses `outdir`, so this one reassignment routes all PNGs + CSV + LaTeX inside.
    base_out = _anchor(args.out); base_out.mkdir(exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = base_out / f"run_{run_tag}"; outdir.mkdir(parents=True, exist_ok=True)
    print(f"  run folder: {outdir}")

    # --- Option A regime: reclassify each study day by its OWN daily price spread
    # (per-year tertiles), overriding the legacy hardcoded High/Mid/Low in DAYS.
    # Single source of truth: reuse aggregate_path3's classifier. Falls back to the
    # legacy labels if the price file / classifier is unavailable, so nothing breaks.
    global DAYS
    try:
        from aggregate_path3 import daily_spread_classifier
        _pcsv = _HERE.parent / "dk1_prices_2022.csv"
        if _pcsv.exists():
            _rod, (_q1, _q2), _sod, _nd = daily_spread_classifier(_pcsv, "maxmin")
            DAYS = {s: (idx, _rod(idx + 1)) for s, (idx, _old) in DAYS.items()}
            print(f"  [regime] 2022 daily maxmin tertile cuts = {_q1:.0f} / {_q2:.0f} EUR/MWh")
            print("  [regime] " + ", ".join(f"{s}:{r}" for s, (i, r) in DAYS.items()))
        else:
            print(f"  [regime] {_pcsv.name} not found beside parent dir - keeping legacy labels")
    except Exception as _e:
        print(f"  [regime] reclassification skipped ({type(_e).__name__}) - legacy labels")

    # locate + import v5.4
    if args.v54:
        v54_path = Path(args.v54)
    else:
        cand = []
        for base in (_HERE, _HERE.parent):
            cand += sorted(base.glob("run_battery_xu_shi_degradation_v5*.py"))
        if not cand:
            raise FileNotFoundError(
                "v5.4 .py not found in this folder or its parent; pass --v54 with the path")
        # prefer the 5.4 file specifically: a bare v5* glob also matches v5.2 etc.
        v54_only = [c for c in cand if "5.4" in c.name or "5_4" in c.name]
        if v54_only:
            v54_path = v54_only[0]
        else:
            v54_path = cand[0]
            print(f"  WARNING: no file matched '5.4'; using {v54_path.name}. "
                  f"Pass --v54 if this is wrong.")
        if len(cand) > 1:
            print(f"  ({len(cand)} v5* files found; picked {v54_path.name})")
        print(f"  auto-detected v5.4: {v54_path}")
        
    print(f"Importing v5.4 from {v54_path.name} ...")
    v54 = load_v54(v54_path)

    # bat_params + shi_fit (via the shared loader)
    from wp2_common import quick_setup
    from degradation_subgradient import fit_shi_polynomial
    HPP_YAML = next((p for p in (_HERE / "WP2_HPP.yaml", _HERE.parent / "WP2_HPP.yaml")
                     if p.exists()), None)
    if HPP_YAML is None:
        raise FileNotFoundError("WP2_HPP.yaml not found beside this script or its parent")
    setup = quick_setup(HPP_YAML, config={"interp_n": 2000}, verbose=False)
    bat_params = setup["battery"]
    shi_fit = fit_shi_polynomial(soc_min=SOC_MIN, soc_max=SOC_MAX, verbose=False)

    runs = find_runs(results_dir, args.e_cap, args.p_cap, want_wind=True)
    wanted = [args.day] if args.day else list(DAYS.keys())
    rows, soc_pairs = [], {}
    for slot in wanted:
        if slot not in runs:
            print(f"  [skip] {slot}: no wind {args.e_cap:.0f}/{args.p_cap:.0f} run in {results_dir}")
            continue
        jp, npz, summ = runs[slot]
        print(f"  scoring {slot} ...")
        row, soc = compare_day(slot, jp, npz, summ, bat_params, shi_fit, v54, outdir)
        if args.verify_lp:
            try:
                v = verify_lp(slot, npz, setup, v54)
                row.update(v)
                verdict = ("MATCH" if abs(v["d_rev"]) < 1.0 and v["max_dp"] < 1.0
                           else "REVENUE MISMATCH" if abs(v["d_rev"]) >= 1.0
                           else "DEGENERATE (same rev, diff dispatch)")
                print(f"      LP check: dRev={v['d_rev']:+.2f} EUR | "
                      f"max|dP|={v['max_dp']:.3f} MW | max|dSoC|={v['max_dsoc']:.2f} MWh (slack) "
                      f"-> {verdict}")
            except Exception as e:
                print(f"      LP check failed: {type(e).__name__}: {e}")
        rows.append(row); soc_pairs[slot] = soc

    if not rows:
        print("No matching runs found."); return
    if len(rows) == len(DAYS):
        _plot_grid(soc_pairs, rows, outdir)
    write_tables(rows, outdir)
    print(f"\nWrote tables + plots to {outdir.resolve()}  ({len(rows)} cases)")


if __name__ == "__main__":
    main()