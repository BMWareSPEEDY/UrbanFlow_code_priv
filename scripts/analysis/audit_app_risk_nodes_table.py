"""Audit the exact UI table that the user sees across all 16 cities.
"""
import sys, numpy as np

sys.stdout.reconfigure(line_buffering=True)
from app import REGION_CACHE, PRODUCTION_PREDICTOR

print("=" * 115)
print("EXACT UI RISK NODES TABLE AUDIT ACROSS ALL 16 CITIES (@ 50 mm/hr):")
print("=" * 115)
print(f"{'City':<15s} | {'Total Nodes':<12s} | {'Risk Nodes':<12s} | {'MATCH':<8s} | {'OVER (FP)':<12s} | {'UNDER (FN)':<12s} | {'Accuracy %'}")
print("-" * 115)

tot_nodes = 0
tot_risk = 0
tot_match = 0
tot_over = 0
tot_under = 0

for r_key, r_data in REGION_CACHE.items():
    preds, _, _ = PRODUCTION_PREDICTOR.predict(r_data['pyg_data'], 50.0, 60.0)
    node_list = r_data['node_list']
    node_pos = r_data['node_pos']
    
    n_risk = 0
    n_match = 0
    n_over = 0
    n_under = 0
    
    for idx, nid in enumerate(node_list):
        p_d = round(float(preds[idx]), 4)
        s_d = round(float(node_pos[nid]['swmm_depth']), 4)
        
        if p_d > 0.08 or s_d > 0.08:
            n_risk += 1
            diff = p_d - s_d
            if abs(diff) < 0.15:
                n_match += 1
            elif diff > 0:
                n_over += 1
            else:
                n_under += 1
                
    tot_nodes += len(node_list)
    tot_risk += n_risk
    tot_match += n_match
    tot_over += n_over
    tot_under += n_under
    
    acc = (n_match / max(1, n_risk)) * 100.0
    print(f"{r_key:<15s} | {len(node_list):<12d} | {n_risk:<12d} | {n_match:<8d} | {n_over:<12d} | {n_under:<12d} | {acc:<9.1f}%")

print("=" * 115)
print(f"{'TOTAL':<15s} | {tot_nodes:<12d} | {tot_risk:<12d} | {tot_match:<8d} | {tot_over:<12d} | {tot_under:<12d} | {(tot_match/max(1,tot_risk))*100.0:<9.1f}%")
