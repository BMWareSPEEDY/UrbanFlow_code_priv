"""Test higher threshold for choked fill across all 16 cities.
"""
from app import REGION_CACHE, PRODUCTION_PREDICTOR
import numpy as np

print("=" * 115)
print("TESTING TARGETED DUAL-DRAINAGE SURCHARGE HEAD ACROSS ALL 16 CITIES (@ 50 mm/hr):")
print("=" * 115)
print(f"{'City':<15s} | {'Total Nodes':<12s} | {'Risk Nodes':<12s} | {'MATCH':<8s} | {'OVER':<8s} | {'UNDER':<8s} | {'Live Match Rate %'}")
print("-" * 115)

all_rates = []
for r_key, r_data in REGION_CACHE.items():
    preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(r_data['pyg_data'], 50.0, 60.0)
    node_list = r_data['node_list']
    node_pos = r_data['node_pos']
    y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])
    
    g = r_data['pyg_data']
    x_raw = g.x.cpu().numpy()
    dep_d = x_raw[:, 16]
    conv_def = x_raw[:, 30]
    accum_s = x_raw[:, 5]
    
    # Re-evaluate with targeted severe choked fill:
    is_severe_choked_fill = (probs >= 0.70) & (conv_def >= 2.5) & (accum_s >= 2.0) & (dep_d >= 1.5)
    fill_head = np.minimum(3.0, np.minimum(dep_d, accum_s * 0.40))
    p_refined = np.where(is_severe_choked_fill, np.maximum(preds, fill_head), preds)
    p_refined = np.minimum(p_refined, 3.0)
    
    risk_mask = (p_refined > 0.08)
    risk_indices = np.where(risk_mask)[0]
    sorted_order = np.argsort(-p_refined[risk_indices])
    top_30 = risk_indices[sorted_order[:30]]
    
    diff = p_refined[top_30] - y_swmm[top_30]
    n_match = np.sum(np.abs(diff) < 0.15)
    n_over = np.sum(diff >= 0.15)
    n_under = np.sum(diff <= -0.15)
    
    rate = (n_match / max(1, len(top_30))) * 100.0
    all_rates.append(rate)
    print(f"{r_key:<15s} | {len(node_list):<12d} | {len(top_30):<12d} | {n_match:<8d} | {n_over:<8d} | {n_under:<8d} | {rate:<15.1f}%")

print("=" * 115)
print(f"Mean Live Match Rate across all 16 cities: {np.mean(all_rates):.1f}%")
