"""Evaluate hydro_gine_v5_bangalore_opt.pt across all 16 cities.
"""
from app import REGION_CACHE
from production_v4 import ProductionFloodPredictorV4
import numpy as np

predictor = ProductionFloodPredictorV4("hydro_gine_v5_bangalore_opt.pt")

print("=" * 115)
print("EVALUATION OF BANGALORE-OPTIMIZED GNN ACROSS ALL 16 CITIES (@ 50 mm/hr):")
print("=" * 115)
print(f"{'City':<15s} | {'Total Nodes':<12s} | {'Risk Nodes':<12s} | {'MATCH':<8s} | {'OVER':<8s} | {'UNDER':<8s} | {'Live Match Rate %'}")
print("-" * 115)

all_rates = []
for r_key, r_data in REGION_CACHE.items():
    preds, raw_p, probs = predictor.predict(r_data['pyg_data'], 50.0, 60.0)
    node_list = r_data['node_list']
    node_pos = r_data['node_pos']
    y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])
    
    risk_mask = (preds > 0.08)
    risk_indices = np.where(risk_mask)[0]
    sorted_order = np.argsort(-preds[risk_indices])
    top_30 = risk_indices[sorted_order[:30]]
    
    diff = preds[top_30] - y_swmm[top_30]
    n_match = np.sum(np.abs(diff) < 0.15)
    n_over = np.sum(diff >= 0.15)
    n_under = np.sum(diff <= -0.15)
    
    rate = (n_match / max(1, len(top_30))) * 100.0
    all_rates.append(rate)
    print(f"{r_key:<15s} | {len(node_list):<12d} | {len(top_30):<12d} | {n_match:<8d} | {n_over:<8d} | {n_under:<8d} | {rate:<15.1f}%")

print("=" * 115)
print(f"Mean Live Match Rate across all 16 cities: {np.mean(all_rates):.1f}%")
