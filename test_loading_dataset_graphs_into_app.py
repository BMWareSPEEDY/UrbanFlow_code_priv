"""Test loading true dataset graphs into app REGION_CACHE.
Verify the Live Match Rate, OVER (FP), and UNDER (FN) across all 16 cities!
"""
import torch, sys, numpy as np
from app import REGIONS, PRODUCTION_PREDICTOR

sys.stdout.reconfigure(line_buffering=True)

dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)

# Build a lookup of 50 mm/hr graphs from expanded_master_physics_dataset
region_graphs_50 = {}
for g in dl:
    r = getattr(g, 'region', '') or getattr(g, 'city', '')
    if r and abs(g.rain_intensity - 50.0) < 1.0 and r not in region_graphs_50:
        region_graphs_50[r] = g

print("=" * 115)
print("LIVE MATCH RATE & ERROR AUDIT USING AUTHENTIC DATASET GRAPHS (@ 50 mm/hr):")
print("=" * 115)
print(f"{'City':<15s} | {'Total':<8s} | {'Risk Nodes':<12s} | {'MATCH':<8s} | {'OVER (FP)':<12s} | {'UNDER (FN)':<12s} | {'Match Rate %'}")
print("-" * 115)

tot_nodes = 0
tot_risk = 0
tot_match = 0
tot_over = 0
tot_under = 0

for r_key in REGIONS.keys():
    if r_key not in region_graphs_50:
        continue
    g = region_graphs_50[r_key]
    preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(g, 50.0, 60.0)
    y_swmm = g.y.cpu().numpy().ravel()
    
    n_risk = 0
    n_match = 0
    n_over = 0
    n_under = 0
    
    for idx in range(len(y_swmm)):
        p_d = round(float(preds[idx]), 4)
        s_d = round(float(y_swmm[idx]), 4)
        
        if p_d > 0.08 or s_d > 0.08:
            n_risk += 1
            diff = p_d - s_d
            if abs(diff) < 0.15:
                n_match += 1
            elif diff > 0:
                n_over += 1
            else:
                n_under += 1
                
    tot_nodes += len(y_swmm)
    tot_risk += n_risk
    tot_match += n_match
    tot_over += n_over
    tot_under += n_under
    
    rate = (n_match / max(1, n_risk)) * 100.0
    print(f"{r_key:<15s} | {len(y_swmm):<8d} | {n_risk:<12d} | {n_match:<8d} | {n_over:<12d} | {n_under:<12d} | {rate:<11.1f}%")

print("=" * 115)
overall_rate = (tot_match / max(1, tot_risk)) * 100.0
print(f"{'TOTAL':<15s} | {tot_nodes:<8d} | {tot_risk:<12d} | {tot_match:<8d} | {tot_over:<12d} | {tot_under:<12d} | {overall_rate:<11.1f}%")
