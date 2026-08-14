"""
fetch_and_compare_era5.py

Re-download ERA5 wind data for the Task 50 onshore site directly from
Open-Meteo's ARCHIVE/ERA5 endpoint (model pinned to ERA5, NOT the
best-match historical endpoint that blends sources), then compare the
fresh download against:
  (a) your current working file  (suspected older / blended-source data)
  (b) the Task 50 GitHub file    (the reference target)

Goal: decide whether a clean ERA5 pull lands closer to Task 50 than your
current file does. If it does, switch to the fresh pull (or to theirs).

Runs in VS Code on Windows. Requires: requests, numpy, matplotlib, pyyaml.
    pip install requests numpy matplotlib pyyaml
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import re
import requests
import numpy as np
import yaml
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------
LAT, LON = 56.20, 8.54
START, END = "2022-01-01", "2022-12-31"          # archive endpoint uses dates
HEIGHTS_WANTED = [10, 80, 86, 90, 100, 120]      # ask broadly; see what comes back
TZ = "UTC"                                        # pull in UTC; align later

HERE = Path(__file__).parent
MINE_PATH   = HERE / "wind_resource_2022hourly_referenceHPP.yaml"
THEIRS_PATH = HERE / "Wind_Resource_2022_IEA_Task_Force_Github_.yaml"
RAW_OUT     = HERE / "openmeteo_era5_raw_2022.json"
H_REF, H_HUB = 86.0, 90.0

# ----------------------------------------------------------------------
# 1. DOWNLOAD -- ERA5 ARCHIVE ENDPOINT, MODEL PINNED
#    This is the reproducible reanalysis endpoint. The forecast/best-match
#    endpoint is what silently blends ERA5 with other models -- do NOT use it.
# ----------------------------------------------------------------------
def fetch_era5():
    hourly_vars = []
    for h in HEIGHTS_WANTED:
        hourly_vars.append(f"wind_speed_{h}m")
        hourly_vars.append(f"wind_direction_{h}m")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": START, "end_date": END,
        "hourly": ",".join(hourly_vars),
        "models": "era5",            # PIN to ERA5 reanalysis explicitly
        "wind_speed_unit": "ms",
        "timezone": TZ,
    }
    print("Requesting heights:", HEIGHTS_WANTED)
    r = requests.get(url, params=params, timeout=120)
    print("HTTP", r.status_code, "->", r.url)
    r.raise_for_status()
    data = r.json()
    RAW_OUT.write_text(json.dumps(data, indent=2))
    print(f"raw response saved -> {RAW_OUT}  (keep this as your download audit trail)")
    return data

# ----------------------------------------------------------------------
# 2. PARSE the API response; report which heights actually came back
# ----------------------------------------------------------------------
def parse_api(data):
    h = data.get("hourly", {})
    t = np.array([datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
                  for s in h["time"]])
    got = {}
    for key, vals in h.items():
        m = re.match(r"wind_speed_(\d+)m", key)
        if m and any(v is not None for v in vals):
            got[int(m.group(1))] = np.array([np.nan if v is None else v
                                             for v in vals], float)
    print("Heights the API actually returned (non-empty):", sorted(got))
    if not got:
        raise SystemExit("No wind-speed levels returned. Check variable names "
                         "against current Open-Meteo docs; ERA5 native is 10 & 100 m.")
    return t, got

# ----------------------------------------------------------------------
# 3. ROBUST TIMESTAMP PARSER for the YAML files (handles '+01:00Z')
# ----------------------------------------------------------------------
_OFFSET = re.compile(r"[+-]\d{2}:\d{2}")
def to_utc(s):
    s = s.strip()
    if _OFFSET.search(s) and s.endswith("Z"):
        s = s[:-1]
    elif s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)

def load_yaml(path):
    wr = yaml.safe_load(open(path))["wind_resource"]
    t = np.array([to_utc(s) for s in wr["time"]])
    ws = np.asarray(wr["wind_speed"], float)
    return t, ws

# ----------------------------------------------------------------------
# 4. ALIGN ON UTC + COMPARE
# ----------------------------------------------------------------------
def compare(name, t_a, ws_a, t_b, ws_b):
    ma = {t: i for i, t in enumerate(t_a)}
    mb = {t: i for i, t in enumerate(t_b)}
    common = sorted(set(ma) & set(mb))
    if not common:
        print(f"[{name}] NO overlapping UTC hours -- check timezone handling.")
        return None
    a = ws_a[[ma[t] for t in common]]
    b = ws_b[[mb[t] for t in common]]
    v = (a > 0.5) & np.isfinite(a) & np.isfinite(b)
    r = np.corrcoef(a[v], b[v])[0, 1]
    rmse = np.sqrt(np.mean((a[v] - b[v])**2))
    ratio = b[v] / a[v]
    print(f"[{name}] n={v.sum()}  mean_a={a[v].mean():.3f}  mean_b={b[v].mean():.3f}  "
          f"r={r:.4f}  RMSE={rmse:.3f}  ratio mean={ratio.mean():.4f} std={ratio.std():.4f}")
    return dict(a=a[v], b=b[v], r=r, rmse=rmse)

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
data = fetch_era5()
t_api, got = parse_api(data)

# pick the fresh-download level to use for the hub comparison
if 90 in got:
    t_fresh, ws_fresh, lvl = t_api, got[90], 90
elif 86 in got:
    t_fresh, ws_fresh, lvl = t_api, got[86], 86
    print("NOTE: 90 m not served; using 86 m fresh pull (shear factor ~1.009 to hub).")
else:
    lvl = sorted(got)[-1]
    t_fresh, ws_fresh = t_api, got[lvl]
    print(f"NOTE: neither 86 nor 90 m served; using nearest level {lvl} m.")

t_mine, ws_mine     = load_yaml(MINE_PATH)
t_theirs, ws_theirs = load_yaml(THEIRS_PATH)

print(f"\nfresh ERA5 pull annual mean ({lvl} m): {np.nanmean(ws_fresh):.3f} m/s")
print(f"your file annual mean (86 m)        : {np.nanmean(ws_mine):.3f} m/s")
print(f"Task 50 file annual mean            : {np.nanmean(ws_theirs):.3f} m/s\n")

print("--- pairwise comparisons (b/a) ---")
r1 = compare("fresh_vs_theirs", t_fresh, ws_fresh, t_theirs, ws_theirs)
r2 = compare("mine_vs_theirs ", t_mine,  ws_mine,  t_theirs, ws_theirs)
r3 = compare("fresh_vs_mine  ", t_fresh, ws_fresh, t_mine,   ws_mine)

# ----------------------------------------------------------------------
# VERDICT
# ----------------------------------------------------------------------
if r1 and r2:
    print("\n=== VERDICT ===")
    if r1["r"] > r2["r"] and r1["rmse"] < r2["rmse"]:
        print(f"Fresh ERA5 pull is CLOSER to Task 50 than your current file "
              f"(r {r1['r']:.3f} vs {r2['r']:.3f}, RMSE {r1['rmse']:.3f} vs {r2['rmse']:.3f}).")
        print("-> Your current file is the stale/blended one. Switch to the fresh pull or to theirs.")
    else:
        print(f"Fresh ERA5 pull is NOT closer (r {r1['r']:.3f} vs {r2['r']:.3f}).")
        print("-> Task 50's series likely differs from clean ERA5 too; their data is its own product.")

# ----------------------------------------------------------------------
# FIGURE
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
if r1:
    hi = max(r1["a"].max(), r1["b"].max())
    ax[0].scatter(r1["a"], r1["b"], s=4, alpha=0.25, color="#2166ac", edgecolors="none")
    ax[0].plot([0, hi], [0, hi], "k", lw=1)
    ax[0].set_title(f"fresh ERA5 ({lvl} m) vs Task 50   r={r1['r']:.3f}")
    ax[0].set_xlabel(f"fresh ERA5 {lvl} m [m/s]"); ax[0].set_ylabel("Task 50 [m/s]")
if r2:
    hi = max(r2["a"].max(), r2["b"].max())
    ax[1].scatter(r2["a"], r2["b"], s=4, alpha=0.25, color="#b5351b", edgecolors="none")
    ax[1].plot([0, hi], [0, hi], "k", lw=1)
    ax[1].set_title(f"your current file vs Task 50   r={r2['r']:.3f}")
    ax[1].set_xlabel("your 86 m file [m/s]"); ax[1].set_ylabel("Task 50 [m/s]")
fig.tight_layout()
fig.savefig("era5_refetch_comparison.png", dpi=150)
print("\nsaved -> era5_refetch_comparison.png")