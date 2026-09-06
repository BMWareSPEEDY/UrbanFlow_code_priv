"""Inspect the 9 mismatches in HSR Layout top 30 risk hotspots.
"""
from app import REGION_CACHE, PRODUCTION_PREDICTOR
import numpy as np

r_data = REGION_CACHE['hsr']
g = r_data['pyg_data']
node_list = r_data['node_list']
node_pos = r_data['node_pos']
y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])

preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(g, 50.0, 60.0)

risk_mask = (preds > 0.08)
risk_indices = np.where(risk_mask)[0]
sorted_order = np.argsort(-preds[risk_indices])
top_30 = risk_indices[sorted_order[:30]]

print(f"{'Rank':<5s} | {'Node ID':<12s} | {'GNN (m)':<8s} | {'SWMM (m)':<8s} | {'Diff (cm)':<10s} | {'Status':<8s} | {'Prob':<6s} | {'Elev':<6s} | {'Acc':<6s} | {'Dep':<6s} | {'ConvDef'}")
print("-" * 105)

matches = 0
for rank, idx in enumerate(top_30, 1):
    nid = node_list[idx]
    p = preds[idx]
    s = y_swmm[idx]
    diff = p - s
    diff_cm = diff * 100.0
    status = 'MATCH' if abs(diff) < 0.15 else ('OVER' if diff > 0 else 'UNDER')
    if status == 'MATCH':
        matches += 1
    prob = probs[idx]
    elev = node_pos[nid]['elevation']
    acc = g.x[idx, 5].item()
    dep = g.x[idx, 16].item()
    cdef = g.x[idx, 30].item()
    
    flag = "   " if status == 'MATCH' else ("[O]" if status == 'OVER' else "[U]")
    print(f"{flag} {rank:<3d} | {str(nid):<12s} | {p:<8.3f} | {s:<8.3f} | {diff_cm:<+10.1f} | {status:<8s} | {prob:<6.2f} | {elev:<6.1f} | {acc:<6.2f} | {dep:<6.2f} | {cdef:.2f}")

print("-" * 105)
print(f"HSR Live Match Rate: {matches}/30 = {matches/30*100:.1f}%")
