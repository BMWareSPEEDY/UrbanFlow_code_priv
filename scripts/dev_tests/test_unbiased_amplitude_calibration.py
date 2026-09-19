"""Test unbiased hydrodynamic amplitude calibration across all 16 cities.
"""
import sys, numpy as np, torch

sys.stdout.reconfigure(line_buffering=True)
from app import REGION_CACHE, PRODUCTION_PREDICTOR

print("=" * 115)
print("TESTING HYDRODYNAMIC AMPLITUDE CALIBRATION ACROSS ALL 16 CITIES (@ 50 mm/hr):")
print("=" * 115)
print(f"{'City':<15s} | {'Risk Nodes':<12s} | {'MATCH':<8s} | {'OVER (FP)':<12s} | {'UNDER (FN)':<12s} | {'Old Match%':<12s} | {'New Match%':<12s} | {'Gain'}")
print("-" * 115)

old_match_rates = {
    'hsr': 53.6, 'bellandur': 46.2, 'whitefield': 55.0, 'ecity': 50.1,
    'koramangala': 40.7, 'tokyo': 56.3, 'hongkong': 40.5, 'singapore': 56.3,
    'london': 48.2, 'paris': 43.1, 'nyc': 50.1, 'chicago': 68.7,
    'berlin': 63.5, 'bangkok': 62.7, 'mumbai': 45.6, 'delhi': 44.5
}

tot_risk = 0
tot_match = 0
tot_over = 0
tot_under = 0

for r_key, r_data in REGION_CACHE.items():
    preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(r_data['pyg_data'], 50.0, 60.0)
    node_list = r_data['node_list']
    node_pos = r_data['node_pos']
    
    # Apply smooth hydrodynamic amplitude scaling to recover true water depth
    amp_scale = 1.0 + 1.25 / (1.0 + np.exp(-12.0 * (probs - 0.40)))
    cal_preds = preds * amp_scale
    
    n_risk = 0
    n_match = 0
    n_over = 0
    n_under = 0
    
    for idx, nid in enumerate(node_list):
        p_d = round(float(cal_preds[idx]), 4)
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
                
    tot_risk += n_risk
    tot_match += n_match
    tot_over += n_over
    tot_under += n_under
    
    new_acc = (n_match / max(1, n_risk)) * 100.0
    old_acc = old_match_rates.get(r_key, 50.0)
    gain = new_acc - old_acc
    print(f"{r_key:<15s} | {n_risk:<12d} | {n_match:<8d} | {n_over:<12d} | {n_under:<12d} | {old_acc:<11.1f}% | {new_acc:<11.1f}% | {gain:+6.1f}%")

print("=" * 115)
overall_new = (tot_match / max(1, tot_risk)) * 100.0
print(f"{'TOTAL':<15s} | {tot_risk:<12d} | {tot_match:<8d} | {tot_over:<12d} | {tot_under:<12d} | 52.4%       | {overall_new:<11.1f}% | {overall_new - 52.4:+6.1f}%")
