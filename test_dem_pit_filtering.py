"""Test realistic urban DEM pit filtering on HSR Layout.
"""
import sys, numpy as np

sys.stdout.reconfigure(line_buffering=True)

from app import REGION_CACHE, PRODUCTION_PREDICTOR

hsr = REGION_CACHE['hsr']
pyg = hsr['pyg_data'].clone()
node_list = hsr['node_list']
node_pos = hsr['node_pos']

x_np = pyg.x.cpu().numpy().copy()
rel_drop = x_np[:, 0]
log_area = x_np[:, 10]
dep_d = x_np[:, 16]
sink_d = x_np[:, 23]

# Compute true contributing area in m^2 (normalized by 10,000 in app.py)
acc_area_m2 = np.expm1(log_area) * 10000.0

# In urban hydrology, a local road segment without large catchment area (Area < 25,000 m^2)
# cannot physically store more than 15 cm of water because it overflows into the road gutter
realistic_dep_d = np.where(
    acc_area_m2 < 25000.0,
    np.minimum(dep_d, 0.15),
    dep_d
)

# Corrected sink depth: requires valley location (rel_drop >= 0.50) and large drainage area or lake outfall
x_np[:, 16] = realistic_dep_d
x_np[:, 23] = np.where(
    (rel_drop >= 0.50) & (realistic_dep_d >= 0.05) & (acc_area_m2 >= 5000.0),
    realistic_dep_d,
    0.0
)

import torch
pyg.x = torch.tensor(x_np, dtype=torch.float32, device=pyg.x.device)

preds, raw, probs = PRODUCTION_PREDICTOR.predict(pyg, 50.0, 60.0)

y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])

top_idx = np.argsort(preds)[::-1]

print("=" * 115)
print("APP PREDICTION WITH URBAN PIT FILTERING @ 50 mm/hr (TOP 15 NODES):")
print("=" * 115)
print(f"{'Rank':<5s} | {'Node ID':<14s} | {'GNN Depth':<12s} | {'SWMM Depth':<12s} | {'Elev':<8s} | {'AccArea(m2)':<12s} | {'Diff (cm)'}")
print("-" * 115)

for r, idx in enumerate(top_idx[:15], 1):
    nid = node_list[idx]
    p = preds[idx]
    swmm = y_swmm[idx]
    diff = (p - swmm) * 100.0
    elev = node_pos[nid]['elevation']
    aa = acc_area_m2[idx]
    print(f"{r:<5d} | {str(nid):<14s} | {p:<12.4f} | {swmm:<12.4f} | {elev:<8.2f} | {aa:<12.1f} | {diff:<+10.1f}")

swmm_crit = (y_swmm >= 0.30)
gnn_crit = (preds >= 0.30)
tp = int(np.sum(swmm_crit & gnn_crit))
fp = int(np.sum(~swmm_crit & gnn_crit))
shallow_fp = int(np.sum((y_swmm <= 0.08) & (preds >= 0.15)))
mae_cm = np.mean(np.abs(preds - y_swmm)) * 100.0
pct_30 = np.mean(np.abs(preds - y_swmm) <= 0.30) * 100.0

print(f"\n--- OVERALL BENCHMARK @ 50 mm/hr ---")
print(f"  Critical TP: {tp} | Critical FP: {fp}")
print(f"  Shallow FP (p >= 15cm on SWMM <= 8cm): {shallow_fp} / {int(np.sum(y_swmm <= 0.08))}")
print(f"  Depth MAE: {mae_cm:.2f} cm | % <= 30cm: {pct_30:.1f}%")
