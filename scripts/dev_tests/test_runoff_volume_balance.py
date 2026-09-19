"""Test runoff volume condition for dual drainage filling.
"""
import torch, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
from app import REGION_CACHE, PRODUCTION_PREDICTOR

print("=" * 110)
print("EVALUATING RUNOFF VOLUME MASS BALANCE ON DUAL DRAINAGE SURCHARGE:")
print("=" * 110)
print(f"{'City':<15s} | {'Total Nodes':<12s} | {'Risk Nodes':<12s} | {'MATCH':<8s} | {'OVER':<8s} | {'UNDER':<8s} | {'Live Match Rate %'}")
print("-" * 110)

tot_nodes = 0
tot_risk = 0
tot_match = 0
tot_over = 0
tot_under = 0
rates = []

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
    
    # Dual drainage depression only fills if upstream volume exceeds storage
    # accum_s >= 1.5 AND conv_def >= 1.2 AND probs >= 0.65
    is_choked_fill = (probs >= 0.65) & (conv_def >= 1.2) & (accum_s >= 1.5) & (dep_d >= 0.3)
    fill_head = np.minimum(2.55, np.minimum(dep_d, accum_s * 0.40))
    p_aligned = np.where(is_choked_fill, np.maximum(preds, fill_head), preds)
    p_aligned = np.minimum(p_aligned, 2.55)
    
    risk_mask = (p_aligned > 0.08)
    diff = p_aligned[risk_mask] - y_swmm[risk_mask]
    n_risk = np.sum(risk_mask)
    n_match = np.sum(np.abs(diff) < 0.15)
    n_over = np.sum(diff >= 0.15)
    n_under = np.sum(diff <= -0.15)
    
    rate = (n_match / max(1, n_risk)) * 100.0
    rates.append(rate)
    
    tot_nodes += len(node_list)
    tot_risk += n_risk
    tot_match += n_match
    tot_over += n_over
    tot_under += n_under
    
    print(f"{r_key:<15s} | {len(node_list):<12d} | {n_risk:<12d} | {n_match:<8d} | {n_over:<8d} | {n_under:<8d} | {rate:<15.1f}%")

print("=" * 110)
overall = (tot_match / max(1, tot_risk)) * 100.0
print(f"{'TOTAL':<15s} | {tot_nodes:<12d} | {tot_risk:<12d} | {tot_match:<8d} | {tot_over:<8d} | {tot_under:<8d} | {overall:<15.1f}%")
print(f"Mean Live Match Rate across 16 cities: {np.mean(rates):.1f}%")
