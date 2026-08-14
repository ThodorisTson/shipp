"""
Scale the downloaded Open-Meteo ERA5 100 m series down to 90 m hub height
(power-law), write it as a Task-50-style YAML, and compare it against:
  (a) the Task 50 GitHub 90 m series
  (b) your current 86 m file, itself scaled UP to 90 m for a like-for-like test.
"""
from pathlib import Path
from datetime import datetime, timezone
import json, re
import numpy as np
import yaml

HERE = Path(".")
RAW   = HERE/"openmeteo_era5_raw_2022.json"
MINE  = HERE/"wind_resource_2022hourly_referenceHPP.yaml"        # 86 m Open-Meteo
THEIRS= HERE/"Wind_Resource_2022_IEA_Task_Force_Github_.yaml"    # Task 50, 90 m
OUT   = HERE/"wind_resource_2022_era5_90m_openmeteo.yaml"        # deliverable 1
ALPHA = 0.20                          # terrain shear exponent (consistent with your 86->90 use)

# ---------- load raw JSON ----------
d = json.load(open(RAW))
h = d["hourly"]
t_api = np.array([datetime.fromisoformat(s).replace(tzinfo=timezone.utc) for s in h["time"]])
ws10  = np.array(h["wind_speed_10m"], float)
ws100 = np.array(h["wind_speed_100m"], float)
wd100 = np.array(h["wind_direction_100m"], float)

# ---------- data-derived shear, for sanity ----------
m = (ws10>0.5)&(ws100>0.5)
alpha_data = np.log(ws100[m]/ws10[m]) / np.log(100/10)
print(f"shear exponent: assumed alpha={ALPHA:.3f}   data-derived median alpha={np.median(alpha_data):.3f} "
      f"(mean {alpha_data.mean():.3f})")

# ---------- scale 100 m -> 90 m ----------
f_100_90 = (90/100)**ALPHA
ws90 = ws100 * f_100_90
print(f"100->90 m factor (alpha={ALPHA}) = {f_100_90:.5f}")
print(f"ERA5 means: 10m={ws10.mean():.3f}  100m={ws100.mean():.3f}  ->90m={ws90.mean():.3f} m/s")

# ---------- DELIVERABLE 1: write YAML in Task-50 structure ----------
def col(a): return [round(float(x),4) for x in a]
out = {"name":"Wind resource ERA5 90m (Open-Meteo archive, models=era5; 100m power-law alpha=0.20 to 90m)",
       "wind_resource":{
          "time":[s+"Z" for s in h["time"]],          # UTC, mark explicitly
          "turbulence_intensity":{"data":[0.1]*len(ws90)},
          "wind_direction": col(wd100),                # direction ~height-invariant; use 100m
          "wind_speed": col(ws90)}}
yaml.safe_dump(out, open(OUT,"w"), default_flow_style=False, sort_keys=False, width=80)
print(f"\nDELIVERABLE 1 written -> {OUT.name}  ({OUT.stat().st_size/1024:.0f} KB)")

# ---------- timestamp parser for the YAML files ----------
_OFF=re.compile(r"[+-]\d{2}:\d{2}")
def to_utc(s):
    s=s.strip()
    if _OFF.search(s) and s.endswith("Z"): s=s[:-1]
    elif s.endswith("Z"): s=s[:-1]+"+00:00"
    dt=datetime.fromisoformat(s)
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
def load_yaml(p):
    wr=yaml.safe_load(open(p))["wind_resource"]
    return (np.array([to_utc(s) for s in wr["time"]]),
            np.asarray(wr["wind_speed"],float))

t_mine, ws_mine   = load_yaml(MINE)      # 86 m
t_their, ws_their = load_yaml(THEIRS)    # 90 m

# scale YOUR 86 m UP to 90 m for like-for-like
ws_mine90 = ws_mine * (90/86)**ALPHA

def compare(name, ta, a, tb, b):
    ma={t:i for i,t in enumerate(ta)}; mb={t:i for i,t in enumerate(tb)}
    common=sorted(set(ma)&set(mb))
    A=a[[ma[t] for t in common]]; B=b[[mb[t] for t in common]]
    v=(A>0.5)&(B>0.5)&np.isfinite(A)&np.isfinite(B)
    r=np.corrcoef(A[v],B[v])[0,1]; rmse=np.sqrt(np.mean((A[v]-B[v])**2))
    ratio=B[v]/A[v]
    print(f"[{name}] n={v.sum()}  mean_a={A[v].mean():.3f}  mean_b={B[v].mean():.3f}  "
          f"r={r:.4f}  RMSE={rmse:.3f}  ratio mean={ratio.mean():.4f} std={ratio.std():.4f}")
    return dict(A=A[v],B=B[v],r=r,rmse=rmse)

print("\n=== DELIVERABLE 2: fresh ERA5 90m  vs  Task 50 90m ===")
r_et = compare("era5_90  vs theirs", t_api, ws90, t_their, ws_their)
print("\n=== DELIVERABLE 3: your file (scaled 86->90)  vs  Task 50 90m ===")
r_mt = compare("mine_90  vs theirs", t_mine, ws_mine90, t_their, ws_their)
print("\n--- bonus: fresh ERA5 90m vs your file (86->90) ---")
r_em = compare("era5_90  vs mine90", t_api, ws90, t_mine, ws_mine90)

print("\n=== VERDICT ===")
print(f"fresh ERA5->90m  vs Task50 : r={r_et['r']:.3f}  RMSE={r_et['rmse']:.3f}  mean {r_et['A'].mean():.2f} vs {r_et['B'].mean():.2f}")
print(f"your 86->90m     vs Task50 : r={r_mt['r']:.3f}  RMSE={r_mt['rmse']:.3f}  mean {r_mt['A'].mean():.2f} vs {r_mt['B'].mean():.2f}")
better = "FRESH ERA5 pull" if (r_et['r']>r_mt['r'] and r_et['rmse']<r_mt['rmse']) else "YOUR CURRENT FILE"
print(f"-> closer to Task 50: {better}")

np.save("_cache.npy", dict(et=r_et, mt=r_mt, em=r_em, ws90=ws90, ws100=ws100), allow_pickle=True)
