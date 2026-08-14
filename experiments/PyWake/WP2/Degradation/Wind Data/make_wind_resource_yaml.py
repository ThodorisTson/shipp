"""
make_wind_resource_yaml.py
--------------------------
Build a hub-height wind-resource YAML for the WP2 HPP from a raw Open-Meteo
ERA5 archive download (models=era5).

Why this script exists
----------------------
Open-Meteo's ERA5 archive serves only 10 m and 100 m at this site (80/86/90/120 m
come back empty). This script:
  1. reads the raw JSON,
  2. derives the site shear exponent alpha from the 10 m and 100 m levels
     (reported, for transparency; you can override it),
  3. scales the 100 m wind speed to hub height (default 90 m) by the power law,
  4. writes a YAML in the Task-50 structure with an explicit constant TI,
     already at hub height so no further shear scaling is needed downstream.

All physical choices are parameters at the top. Nothing is hard-coded silently.
Reproducible in VS Code on Windows: numpy + pyyaml only.

Usage:
    python make_wind_resource_yaml.py
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import yaml

# ----------------------------------------------------------------------
# CONFIG  -- the only knobs. Edit here.
# ----------------------------------------------------------------------
HERE        = Path(__file__).parent if "__file__" in globals() else Path(".")
RAW_JSON    = HERE / "openmeteo_era5_raw_2022.json"
OUT_YAML    = HERE / "wind_resource_2022_era5_90m.yaml"

HUB_HEIGHT  = 90.0          # target hub height [m]
SRC_HEIGHT  = 100.0         # source level present in the ERA5 archive [m]
ALPHA       = 0.17          # shear exponent: site-derived (see printout). Set None to auto-use data alpha.
TI_CONST    = 0.14          # constant turbulence intensity (IEC Class B, onshore)
ROUND_WS    = 4             # decimals for wind speed
ROUND_WD    = 2             # decimals for direction

# ----------------------------------------------------------------------
# 1. LOAD RAW JSON
# ----------------------------------------------------------------------
d = json.load(open(RAW_JSON))
lat, lon, elev = d.get("latitude"), d.get("longitude"), d.get("elevation")
h = d["hourly"]
time_iso = h["time"]                                   # 'YYYY-MM-DDTHH:MM' (UTC, timezone=UTC was requested)
ws10  = np.array(h["wind_speed_10m"],  float)
ws100 = np.array(h["wind_speed_100m"], float)
wd100 = np.array(h["wind_direction_100m"], float)
n = len(time_iso)
print(f"raw JSON: site grid cell {lat} N, {lon} E (elev {elev} m), {n} hours")

# ----------------------------------------------------------------------
# 2. SITE SHEAR FROM DATA (10 m & 100 m), for transparency
# ----------------------------------------------------------------------
m = (ws10 > 0.5) & (ws100 > 0.5)
alpha_data = np.median(np.log(ws100[m] / ws10[m]) / np.log(100.0 / 10.0))
alpha_use = ALPHA if ALPHA is not None else float(round(alpha_data, 3))
print(f"shear: data-derived alpha (median) = {alpha_data:.4f}   |   using alpha = {alpha_use}")

# ----------------------------------------------------------------------
# 3. SCALE SRC_HEIGHT -> HUB_HEIGHT (power law)
# ----------------------------------------------------------------------
factor = (HUB_HEIGHT / SRC_HEIGHT) ** alpha_use
ws_hub = ws100 * factor
print(f"scaling {SRC_HEIGHT:.0f}->{HUB_HEIGHT:.0f} m : factor = {factor:.5f}")
print(f"means [m/s]: 10m={ws10.mean():.3f}  100m={ws100.mean():.3f}  {HUB_HEIGHT:.0f}m={ws_hub.mean():.3f}")

# ----------------------------------------------------------------------
# 4. WRITE YAML (Task-50 structure, hub height, explicit constant TI)
# ----------------------------------------------------------------------
def col(a, nd): return [round(float(x), nd) for x in a]

doc = {
    "name": (f"WP2 Wind Resource ERA5 {HUB_HEIGHT:.0f}m | Open-Meteo archive models=era5 | "
             f"src {SRC_HEIGHT:.0f}m power-law alpha={alpha_use} | TI={TI_CONST} | "
             f"grid {lat}N {lon}E | year 2022"),
    "wind_resource": {
        # mark times explicitly UTC (the raw pull used timezone=UTC)
        "time": [s + "Z" if not s.endswith("Z") else s for s in time_iso],
        "turbulence_intensity": {"data": [TI_CONST] * n},
        "wind_direction": col(wd100, ROUND_WD),     # direction ~ height-invariant; 100 m used
        "wind_speed":     col(ws_hub, ROUND_WS),    # at hub height
    },
}
yaml.safe_dump(doc, open(OUT_YAML, "w"), default_flow_style=False, sort_keys=False, width=88)
print(f"\nwritten -> {OUT_YAML.name}  ({OUT_YAML.stat().st_size/1024:.0f} KB)")
print("NOTE: this file is already at hub height. Set h_ref = 90.0 in WP2_Wind_Resource.yaml")
print("      so the loader's height scaling becomes a no-op (90 -> 90).")
