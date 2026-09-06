"""Test applying the physical manhole rim ceiling (2.55m) and outfall basin stage alignment.
"""
import requests, numpy as np

CITIES = [
    'hsr', 'bellandur', 'whitefield', 'ecity', 'koramangala',
    'tokyo', 'hongkong', 'singapore', 'london', 'paris',
    'nyc', 'chicago', 'berlin', 'bangkok', 'mumbai', 'delhi'
]

print("=" * 110)
print("EVALUATING PURE NEURAL + PHYSICAL RIM CEILING (2.55m) ACROSS ALL 16 CITIES:")
print("=" * 110)
print(f"{'City':<15s} | {'Top Hotspots':<12s} | {'MATCH':<8s} | {'OVER':<8s} | {'UNDER':<8s} | {'Live Match Rate %'}")
print("-" * 110)

all_rates = []
for city in CITIES:
    resp = requests.post("http://127.0.0.1:5000/api/predict", json={'region': city, 'rainfall_intensity': 50.0, 'duration_min': 60.0}).json()
    r_nodes = resp['risk_nodes']
    
    n_match = 0
    n_over = 0
    n_under = 0
    for n in r_nodes:
        # Pure neural depth with physical 2.55m rim ceiling
        pred_p = min(n['gnn_depth'], 2.55)
        swmm_d = n['swmm_depth']
        diff = pred_p - swmm_d
        if abs(diff) < 0.15:
            n_match += 1
        elif diff > 0:
            n_over += 1
        else:
            n_under += 1
            
    rate = (n_match / max(1, len(r_nodes))) * 100.0
    all_rates.append(rate)
    print(f"{city:<15s} | {len(r_nodes):<12d} | {n_match:<8d} | {n_over:<8d} | {n_under:<8d} | {rate:<15.1f}%")

print("=" * 110)
print(f"Mean Live Match Rate across all 16 cities: {np.mean(all_rates):.1f}%")
