"""
verify_windload.py  -- confirm the loader produces a 90 m hub-height series
with NO unintended height scaling, before the full rerun.
Run from the folder that holds wp2_common.py and the YAMLs.
"""
from pathlib import Path
import numpy as np
import wp2_common as wp2          # adjust import if wp2_common is elsewhere on your path

HERE = Path(__file__).parent

# Load the master HPP exactly as your pipeline does. Adjust this one line to
# however you normally load the top-level YAML (e.g. wp2.load_hpp(...)).
hpp = wp2.load_yaml(HERE / "WP2_HPP.yaml")        # <-- use your real entry point / filename

wr = wp2.load_wind_resource(hpp, verbose=True)

print("\n--- height sanity ---")
print(f"h_ref in config      : {wr['h_ref']:.1f} m")
print(f"shear_alpha in config: {wr['shear_alpha']}")
A = wp2.scale_weibull_to_hub(wr['weibull_A'], wr['h_ref'], 90.0, wr['shear_alpha'])
factor = (90.0 / wr['h_ref']) ** wr['shear_alpha']
print(f"Weibull scale factor (h_ref->90): {factor:.6f}   (MUST be 1.000000)")
print(f"mean Weibull A @ hub : {np.average(A, weights=wr['frequencies']):.3f} m/s")

# The decisive check: the raw hourly series the dispatch will actually use.
ts = hpp['site']['energy_resource']['time_series']['wind_resource']
ws = np.asarray(ts['wind_speed'], float)
ti = np.asarray(ts['turbulence_intensity']['data'], float)
print(f"\nhourly series mean   : {ws.mean():.3f} m/s   (EXPECT ~8.19)")
print(f"hourly TI unique     : {np.unique(ti)}   (EXPECT [0.14])")
print("PASS" if abs(factor-1) < 1e-9 and abs(ws.mean()-8.19) < 0.05 and np.allclose(ti,0.14)
      else "CHECK -- something did not land as expected")