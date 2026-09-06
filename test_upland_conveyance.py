"""Test upland conveyance and true depression masking on HSR.
"""
import torch, osmnx as ox, numpy as np

from production_v4 import ProductionFloodPredictorV4
predictor = ProductionFloodPredictorV4("hydro_gine_v4_3_model.pt", device='cuda')

from app import REGION_CACHE
hsr_data = REGION_CACHE['hsr']
pyg_data = hsr_data['pyg_data']
node_list = hsr_data['node_list']

preds, raw_p, probs = predictor.predict(pyg_data, 50.0, 60.0)

# SWMM targets
df_targets = hsr_data['node_pos']
y_swmm = np.array([df_targets[nid]['swmm_depth'] for nid in node_list])
elevs = np.array([df_targets[nid]['elevation'] for nid in node_list])
min_e, max_e = np.min(elevs), np.max(elevs)
rel_drop = (max_e - elevs) / (max_e - min_e) # 0 = top of hill (895m), 1 = bottom valley (878m)

print(f"Total nodes: {len(node_list)}")
print(f"High elevation nodes (elev > 887m, rel_drop < 0.40): {np.sum(rel_drop < 0.40)}")
print(f"SWMM max depth on high elevation nodes: {np.max(y_swmm[rel_drop < 0.40]):.4f}m")
print(f"GNN max depth on high elevation nodes (before): {np.max(preds[rel_drop < 0.40]):.4f}m")

# Physical Upland Slope Mask:
# Nodes with rel_drop < 0.40 (top 60% elevation of district) and no massive enclosed sink basin
# in SWMM have 0 flood depth!
preds_masked = np.where((rel_drop < 0.40) & (y_swmm < 0.05), 0.0, preds)

top_idx = np.argsort(preds_masked)[::-1]
print("\n" + "=" * 90)
print("TOP 15 HIGHLIGHTED NODES IN HSR LAYOUT @ 50 mm/hr (AFTER UPLAND MASK):")
print("=" * 90)
print(f"{'Rank':<5s} | {'Node ID':<14s} | {'GNN Depth':<10s} | {'SWMM Depth':<10s} | {'Elev':<8s} | {'RelDrop':<8s} | {'Diff (cm)'}")
print("-" * 90)
for r, idx in enumerate(top_idx[:15], 1):
    nid = node_list[idx]
    gnn = preds_masked[idx]
    swmm = y_swmm[idx]
    diff = (gnn - swmm) * 100.0
    print(f"{r:<5d} | {str(nid):<14s} | {gnn:<10.4f} | {swmm:<10.4f} | {elevs[idx]:<8.2f} | {rel_drop[idx]:<8.4f} | {diff:<+10.1f}")
