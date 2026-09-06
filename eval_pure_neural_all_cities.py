"""Evaluate pure neural predictor across all 16 cities.
"""
import torch, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
from app import REGION_CACHE, PRODUCTION_PREDICTOR

print("=" * 115)
print("PURE NEURAL GNN EVALUATION (TOP 30 RISK NODES PER CITY @ 50 mm/hr):")
print("=" * 115)
print(f"{'City':<15s} | {'Total Nodes':<12s} | {'Top Hotspots':<12s} | {'MATCH':<8s} | {'OVER':<8s} | {'UNDER':<8s} | {'Live Match Rate %'}")
print("-" * 115)

tot_nodes = 0
tot_risk = 0
tot_match = 0
tot_over = 0
tot_under = 0
all_rates = []

for r_key, r_data in REGION_CACHE.items():
    preds, _, probs = PRODUCTION_PREDICTOR.predict(r_data['pyg_data'], 50.0, 60.0)
    node_list = r_data['node_list']
    node_pos = r_data['node_pos']
    y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])
    
    # Risk nodes with pred > 0.08
    risk_mask = (preds > 0.08)
    risk_indices = np.where(risk_mask)[0]
    
    # Sort risk nodes descending by predicted depth (matching UI behavior)
    sorted_order = np.argsort(-preds[risk_indices])
    top_30_indices = risk_indices[sorted_order[:30]]
    
    n_match = 0
    n_over = 0
    n_under = 0
    for idx in top_30_indices:
        p = preds[idx]
        s = y_swmm[idx]
        diff = p - s
        if abs(diff) < 0.15:
            n_match += 1
        elif diff > 0:
            n_over += 1
        else:
            n_under += 1
            
    n_hotspots = len(top_30_indices)
    rate = (n_match / max(1, n_hotspots)) * 100.0
    all_rates.append(rate)
    
    tot_nodes += len(node_list)
    tot_risk += n_hotspots
    tot_match += n_match
    tot_over += n_over
    tot_under += n_under
    
    print(f"{r_key:<15s} | {len(node_list):<12d} | {n_hotspots:<12d} | {n_match:<8d} | {n_over:<8d} | {n_under:<8d} | {rate:<15.1f}%")

print("=" * 115)
overall = (tot_match / max(1, tot_risk)) * 100.0
print(f"{'TOTAL':<15s} | {tot_nodes:<12d} | {tot_risk:<12d} | {tot_match:<8d} | {tot_over:<8d} | {tot_under:<8d} | {overall:<15.1f}%")
print(f"\nMean Live Match Rate across all 16 cities: {np.mean(all_rates):.1f}%")
