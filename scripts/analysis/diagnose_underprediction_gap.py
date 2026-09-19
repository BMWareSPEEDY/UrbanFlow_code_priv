"""Diagnose why predictions are underpredicting SWMM depths in international cities.
"""
import sys, numpy as np, torch

sys.stdout.reconfigure(line_buffering=True)
from app import REGION_CACHE, PRODUCTION_PREDICTOR

print("=" * 115)
print("DIAGNOSIS OF UNDERPREDICTION GAP IN INTERNATIONAL CITIES (@ 50 mm/hr):")
print("=" * 115)

for city in ['london', 'paris', 'tokyo', 'hongkong', 'hsr']:
    r_data = REGION_CACHE[city]
    preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(r_data['pyg_data'], 50.0, 60.0)
    node_list = r_data['node_list']
    node_pos = r_data['node_pos']
    
    y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])
    
    # Check all nodes where swmm_depth > 0.08m
    flood_mask = (y_swmm > 0.08)
    p_flood = preds[flood_mask]
    s_flood = y_swmm[flood_mask]
    raw_flood = raw_p[flood_mask]
    prob_flood = probs[flood_mask]
    
    mean_swmm = np.mean(s_flood)
    mean_pred = np.mean(p_flood)
    mean_raw = np.mean(raw_flood)
    mean_prob = np.mean(prob_flood)
    
    # What percentage of swmm_depth is predicted?
    ratio = (mean_pred / max(1e-4, mean_swmm)) * 100.0
    raw_ratio = (mean_raw / max(1e-4, mean_swmm)) * 100.0
    
    print(f"City: {city:<12s} | Flooded Nodes: {len(s_flood):5d}")
    print(f"  Mean SWMM Depth: {mean_swmm:.4f} m ({mean_swmm*100:.1f} cm)")
    print(f"  Mean GNN Depth : {mean_pred:.4f} m ({mean_pred*100:.1f} cm) -> Ratio: {ratio:.1f}%")
    print(f"  Mean Raw GNN   : {mean_raw:.4f} m ({mean_raw*100:.1f} cm) -> Raw Ratio: {raw_ratio:.1f}%")
    print(f"  Mean Prob      : {mean_prob:.4f}")
    print("-" * 115)
