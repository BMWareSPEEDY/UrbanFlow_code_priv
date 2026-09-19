"""Diagnose the remaining mismatches (|pred - swmm| >= 0.15m) across cities.
"""
import sys, numpy as np, torch

sys.stdout.reconfigure(line_buffering=True)
from app import REGION_CACHE, PRODUCTION_PREDICTOR

print("=" * 115)
print("ANALYZING RESIDUAL ERROR DISTRIBUTION FOR RISK NODES (|diff| >= 0.15m):")
print("=" * 115)

for city in ['hsr', 'hongkong', 'london', 'paris', 'tokyo']:
    r_data = REGION_CACHE[city]
    preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(r_data['pyg_data'], 50.0, 60.0)
    node_list = r_data['node_list']
    node_pos = r_data['node_pos']
    y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])
    
    risk_mask = (preds > 0.08) | (y_swmm > 0.08)
    p_risk = preds[risk_mask]
    s_risk = y_swmm[risk_mask]
    diff = p_risk - s_risk
    abs_diff = np.abs(diff)
    
    mismatch_mask = (abs_diff >= 0.15)
    n_mismatch = np.sum(mismatch_mask)
    n_total_risk = len(p_risk)
    
    over_mismatch = np.sum(diff >= 0.15)
    under_mismatch = np.sum(diff <= -0.15)
    
    # Check how many are close to the 15cm boundary: e.g. [0.15, 0.20], [0.20, 0.30], > 0.30
    c_15_20 = np.sum((abs_diff >= 0.15) & (abs_diff < 0.20))
    c_20_30 = np.sum((abs_diff >= 0.20) & (abs_diff < 0.30))
    c_gt_30 = np.sum(abs_diff >= 0.30)
    
    print(f"City: {city:<12s} | Risk Nodes: {n_total_risk:5d} | Mismatches: {n_mismatch:5d} ({n_mismatch/n_total_risk*100:.1f}%)")
    print(f"  OVER (Pred >> SWMM) : {over_mismatch:5d}")
    print(f"  UNDER (SWMM >> Pred): {under_mismatch:5d}")
    print(f"  Diff in [15, 20cm)  : {c_15_20:5d}")
    print(f"  Diff in [20, 30cm)  : {c_20_30:5d}")
    print(f"  Diff >= 30cm        : {c_gt_30:5d}")
    print("-" * 115)
