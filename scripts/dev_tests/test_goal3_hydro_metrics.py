"""Test Goal 3 Hydrodynamic Field Metrics:
- Nash-Sutcliffe Efficiency (NSE) >= 0.82 - 0.88
- Flooded-only MAE (MAE_hazard) <= 8.5 cm
- Volumetric Mass Continuity Error <= 5.0%
"""
import requests, numpy as np

regs = ['hsr', 'bellandur', 'whitefield', 'ecity', 'koramangala', 'tokyo', 'hongkong', 'singapore', 'london', 'paris', 'nyc', 'chicago', 'berlin', 'bangkok', 'mumbai', 'delhi']

print("Evaluating Hydrodynamic Field Metrics across all 16 cities:")
all_y_true = []
all_y_pred = []
all_areas = []

for reg in regs:
    r = requests.post('http://127.0.0.1:5000/api/predict', json={'region': reg, 'rainfall_mmhr': 50.0, 'duration_min': 60.0}).json()
    nodes = r['nodes']
    s = np.array([n['swmm_depth'] for n in nodes], dtype=np.float64)
    g = np.array([n['gnn_depth'] for n in nodes], dtype=np.float64)
    
    # Hydraulic catchment area proxy per node (road cell footprint ~500 m2)
    areas = np.full(len(nodes), 500.0, dtype=np.float64)
    
    all_y_true.append(s)
    all_y_pred.append(g)
    all_areas.append(areas)
    
y_t = np.concatenate(all_y_true)
y_p = np.concatenate(all_y_pred)
a = np.concatenate(all_areas)

# 1. Flooded Mask (y_swmm >= 0.15m)
haz_mask = (y_t >= 0.15)
y_t_haz = y_t[haz_mask]
y_p_haz = y_p[haz_mask]

# Flooded-only MAE
mae_hazard_cm = np.mean(np.abs(y_t_haz - y_p_haz)) * 100.0

# Nash-Sutcliffe Efficiency (NSE) on flooded nodes
numerator = np.sum((y_t_haz - y_p_haz) ** 2)
denominator = np.sum((y_t_haz - np.mean(y_t_haz)) ** 2)
nse = 1.0 - (numerator / denominator)

# Volumetric Mass Continuity Error (%)
vol_true = np.sum(y_t * a)
vol_pred = np.sum(y_p * a)
mass_error_pct = abs(vol_pred - vol_true) / vol_true * 100.0

print("-" * 70)
print(f"Global Nash-Sutcliffe Efficiency (NSE): {nse:.4f} (Target: >= 0.82 - 0.88)")
print(f"Flooded-Only MAE (MAE_hazard):         {mae_hazard_cm:.2f} cm (Target: <= 8.5 cm)")
print(f"Volumetric Mass Continuity Error:       {mass_error_pct:.2f}% (Target: <= 5.0%)")
print("-" * 70)
