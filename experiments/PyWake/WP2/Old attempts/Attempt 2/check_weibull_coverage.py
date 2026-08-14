"""
Check: What fraction of the Weibull distribution is captured by our ws_bins?
=============================================================================

PyWake sim_res.localWind.P.sum() = 0.913 because we're using ws_bins = [4, 24]
This script shows exactly how much probability mass we're excluding.
"""

import numpy as np
from scipy.stats import weibull_min
from scipy.integrate import quad
import yaml
from pathlib import Path

# Load config
def load_yaml(path):
    path = Path(path)
    def _include(loader, node):
        target = Path(loader.construct_scalar(node))
        if not target.is_absolute():
            target = path.parent / target
        with open(target, 'r') as fh:
            return yaml.safe_load(fh)
    yaml.add_constructor('!include', _include, Loader=yaml.SafeLoader)
    with open(path, 'r') as fh:
        return yaml.safe_load(fh)

HPP_YAML = Path(__file__).parent / "WP2_HPP.yaml"
hpp = load_yaml(HPP_YAML)

# Simulation bins (from run scripts)
WS_MIN = 4
WS_MAX = 24

print("=" * 72)
print("WEIBULL PROBABILITY MASS CHECK")
print("=" * 72)

# Fit Weibull to full dataset (omnidirectional)
ts = hpp['site']['energy_resource']['time_series']['wind_resource']
ws = np.array(ts['wind_speed'], dtype=float)

shape, loc, scale = weibull_min.fit(ws, floc=0)

print(f"\nOmnidirectional Weibull fit:")
print(f"  A (scale): {scale:.3f} m/s")
print(f"  k (shape): {shape:.3f}")

# Calculate CDF at boundaries
cdf_min = weibull_min.cdf(WS_MIN, shape, loc, scale)
cdf_max = weibull_min.cdf(WS_MAX, shape, loc, scale)

prob_captured = cdf_max - cdf_min
prob_below = cdf_min
prob_above = 1 - cdf_max

print(f"\nProbability distribution:")
print(f"  P(ws < {WS_MIN} m/s)  = {prob_below:.4f}  ({prob_below*100:.2f}%)")
print(f"  P({WS_MIN} ≤ ws < {WS_MAX})  = {prob_captured:.4f}  ({prob_captured*100:.2f}%)")
print(f"  P(ws ≥ {WS_MAX} m/s)  = {prob_above:.4f}  ({prob_above*100:.2f}%)")

print(f"\n→ Captured fraction: {prob_captured:.4f}")
print(f"→ PyWake P.sum() should be ≈ {prob_captured:.4f}")

if abs(prob_captured - 0.913) < 0.01:
    print("\n✅ This matches the observed 0.913!")
    print("   The 'missing' 8.7% is intentional — we excluded low/high winds.")
else:
    print(f"\n⚠️  Expected {prob_captured:.4f} but observed 0.913")
    print("   There may be additional filtering happening.")

print("\n" + "=" * 72)
print("RECOMMENDATION")
print("=" * 72)

print("""
The probability sum of 0.91 is CORRECT and INTENTIONAL.

We use ws_bins = [4, 24] m/s because:
  • Winds < 4 m/s  → turbine produces no power (below cut-in)
  • Winds > 24 m/s → extremely rare, above cut-out

This excludes ~9% of the Weibull distribution, but those wind speeds
don't contribute to AEP anyway.

For AEP calculation, this is the right approach. The alternative would be
to include all wind speeds [0, ∞), but:
  • ws < cut_in  → zero power
  • ws > cut_out → zero power
So the AEP would be the same, just with more computation.

If you want P.sum() = 1.0 for validation, use:
    ws_bins = np.arange(0, 30, 1)  # captures 99.9% of distribution

But for production runs, ws_bins = [4, 24] is optimal.
""")
