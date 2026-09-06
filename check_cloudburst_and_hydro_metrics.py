"""Verify Goal 2 (Cloudburst Stress Test) and Goal 3 (Hydrodynamic Field Metrics) from live API.
"""
import requests, numpy as np

regs = ['hsr', 'bellandur', 'whitefield', 'ecity', 'koramangala', 'tokyo', 'hongkong', 'singapore', 'london', 'paris', 'nyc', 'chicago', 'berlin', 'bangkok', 'mumbai', 'delhi']

print("=" * 80)
print("1. EVALUATING 100 MM/HR CLOUDBURST STRESS TEST (GOAL 2):")
print("=" * 80)
tot_nodes = 0
tot_swmm_fl = 0
tot_gnn_fl = 0
tot_tp = 0
tot_fp = 0
tot_fn = 0
maes_100 = []

for reg in regs:
    r = requests.post('http://127.0.0.1:5000/api/predict', json={'region': reg, 'rainfall_mmhr': 100.0, 'duration_min': 60.0}).json()
    nodes = r['nodes']
    s = np.array([n['swmm_depth'] for n in nodes])
    g = np.array([n['gnn_depth'] for n in nodes])
    tot_nodes += len(nodes)
    swmm_fl = np.sum(s >= 0.15)
    gnn_fl = np.sum(g >= 0.15)
    tot_swmm_fl += swmm_fl
    tot_gnn_fl += gnn_fl
    tot_tp += np.sum((s >= 0.15) & (g >= 0.15))
    tot_fp += np.sum((s < 0.15) & (g >= 0.15))
    tot_fn += np.sum((s >= 0.15) & (g < 0.15))
    maes_100.append(np.mean(np.abs(g - s)) * 100.0)

rec_100 = tot_tp / (tot_tp + tot_fn) * 100.0
prec_100 = tot_tp / (tot_tp + tot_fp) * 100.0
f1_100 = 2 * prec_100 * rec_100 / (prec_100 + rec_100)
mae_100 = np.mean(maes_100)
fl_pct_100 = tot_swmm_fl / tot_nodes * 100.0

print(f"Total Network Nodes:        {tot_nodes:,}")
print(f"SWMM Flooded Nodes (100mm): {tot_swmm_fl:,} ({fl_pct_100:.1f}%) [Target: 28.0% - 38.0%]")
print(f"GNN Flooded Nodes (100mm):  {tot_gnn_fl:,}")
print(f"Cloudburst True Recall:     {rec_100:.1f}% [Target: >= 88.0%]")
print(f"Cloudburst Precision:       {prec_100:.1f}%")
print(f"Cloudburst F1-Score:        {f1_100/100:.3f} ({f1_100:.1f}%) [Target: >= 0.810]")
print(f"Cloudburst MAE:             {mae_100:.2f} cm [Target: 5.5 - 7.2 cm]")

print("\n" + "=" * 80)
print("2. EVALUATING HYDRODYNAMIC FIELD METRICS (GOAL 3):")
print("=" * 80)
# Evaluate on 50 mm/hr
all_s = []
all_g = []
for reg in regs:
    r = requests.post('http://127.0.0.1:5000/api/predict', json={'region': reg, 'rainfall_mmhr': 50.0, 'duration_min': 60.0}).json()
    nodes = r['nodes']
    all_s.append(np.array([n['swmm_depth'] for n in nodes]))
    all_g.append(np.array([n['gnn_depth'] for n in nodes]))

y_s = np.concatenate(all_s)
y_g = np.concatenate(all_g)

# Flooded-only MAE (y_swmm >= 0.15m)
haz_mask = (y_s >= 0.15)
mae_hazard_cm = np.mean(np.abs(y_s[haz_mask] - y_g[haz_mask])) * 100.0

# Nash-Sutcliffe Efficiency (NSE) on non-zero flooded nodes
num = np.sum((y_s[haz_mask] - y_g[haz_mask]) ** 2)
den = np.sum((y_s[haz_mask] - np.mean(y_s[haz_mask])) ** 2)
nse_haz = 1.0 - (num / den)

# Global Network NSE (all nodes)
nse_glob = 1.0 - (np.sum((y_s - y_g)**2) / np.sum((y_s - np.mean(y_s))**2))

# Volumetric Mass Continuity Error Proxy
vol_s = np.sum(y_s * 500.0)
vol_g = np.sum(y_g * 500.0)
mass_err = abs(vol_g - vol_s) / vol_s * 100.0

print(f"Flooded-Only MAE (MAE_hazard):         {mae_hazard_cm:.2f} cm [Target: <= 8.5 cm]")
print(f"Nash-Sutcliffe Efficiency (Flooded):    {nse_haz:.4f}")
print(f"Nash-Sutcliffe Efficiency (Catchment):  {nse_glob:.4f} [Target: 0.82 - 0.88]")
print(f"Volumetric Mass Continuity Error:       {mass_err:.2f}% [Target: <= 5.0%]")
