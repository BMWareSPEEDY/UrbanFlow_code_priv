"""Test what calibration or architecture adjustment brings the live match rate to >= 95% on all places.
"""
import sys, numpy as np, torch

sys.stdout.reconfigure(line_buffering=True)
from app import REGIONS, PRODUCTION_PREDICTOR

dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
region_graphs_50 = {}
for g in dl:
    r = getattr(g, 'region', '') or getattr(g, 'city', '')
    if r and abs(g.rain_intensity - 50.0) < 1.0 and r not in region_graphs_50:
        region_graphs_50[r] = g

print("=" * 115)
print("ANALYZING RESIDUAL CORRECTION TO ACHIEVE >= 95% LIVE MATCH RATE:")
print("=" * 115)

for r_key in REGIONS.keys():
    if r_key not in region_graphs_50:
        continue
    g = region_graphs_50[r_key]
    preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(g, 50.0, 60.0)
    y_swmm = g.y.cpu().numpy().ravel()
    
    # Check baseline match rate
    risk_mask = (preds > 0.08) | (y_swmm > 0.08)
    diff = preds[risk_mask] - y_swmm[risk_mask]
    base_match = np.mean(np.abs(diff) < 0.15) * 100.0
    
    # If we apply a smooth node-level residual correction delta(x):
    # Where does the error come from?
    res = y_swmm - preds
    print(f"City: {r_key:<14s} | Risk: {np.sum(risk_mask):4d} | Base Match: {base_match:5.1f}% | Residual Mean: {np.mean(res[risk_mask]):+6.4f}m | Residual Std: {np.std(res[risk_mask]):6.4f}m")
