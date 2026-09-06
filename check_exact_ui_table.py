"""Check exact UI risk table as displayed in app.py.
"""
import sys, numpy as np
from app import REGION_CACHE, PRODUCTION_PREDICTOR

sys.stdout.reconfigure(line_buffering=True)

print("=" * 115)
print("EXACT UI TABLE AS RENDERED BY APP.PY (if pred_depth > 0.08):")
print("=" * 115)
print(f"{'City':<15s} | {'Total Nodes':<12s} | {'UI Risk Nodes':<14s} | {'MATCH':<8s} | {'OVER':<8s} | {'UNDER':<8s} | {'Live Match Rate %'}")
print("-" * 115)

tot_nodes = 0
tot_risk = 0
tot_match = 0
tot_over = 0
tot_under = 0

for r_key, r_data in REGION_CACHE.items():
    preds, _, probs = PRODUCTION_PREDICTOR.predict(r_data['pyg_data'], 50.0, 60.0)
    node_list = r_data['node_list']
    node_pos = r_data['node_pos']
    
    n_risk = 0
    n_match = 0
    n_over = 0
    n_under = 0
    
    for idx, nid in enumerate(node_list):
        p_d = round(float(preds[idx]), 4)
        s_d = round(float(node_pos[nid]['swmm_depth']), 4)
        
        # EXACT APP.PY CONDITION (line 794):
        if p_d > 0.08:
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
    
    rate = (n_match / max(1, n_risk)) * 100.0
    print(f"{r_key:<15s} | {len(node_list):<12d} | {n_risk:<14d} | {n_match:<8d} | {n_over:<8d} | {n_under:<8d} | {rate:<15.1f}%")

print("=" * 115)
overall = (tot_match / max(1, tot_risk)) * 100.0
print(f"{'TOTAL':<15s} | {tot_nodes:<12d} | {tot_risk:<14d} | {tot_match:<8d} | {tot_over:<8d} | {tot_under:<8d} | {overall:<15.1f}%")
