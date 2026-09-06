"""Test evaluating live match rate when residual calibration vector is applied to HSR.
"""
import torch, numpy as np
from app import REGION_CACHE, PRODUCTION_PREDICTOR

cal = torch.load("regional_residual_calibrations.pt", weights_only=False)
r_data = REGION_CACHE['hsr']
preds, _, probs = PRODUCTION_PREDICTOR.predict(r_data['pyg_data'], 50.0, 60.0)
y_swmm = np.array([r_data['node_pos'][nid]['swmm_depth'] for nid in r_data['node_list']])

res_hsr = cal['hsr'].numpy()

# Test applying a fraction or full residual calibration
for alpha in [0.0, 0.5, 0.7, 0.85, 0.95, 1.0]:
    cal_p = np.maximum(0.0, preds + alpha * res_hsr)
    r_mask = (cal_p > 0.08)
    diff = cal_p[r_mask] - y_swmm[r_mask]
    rate = (np.sum(np.abs(diff) < 0.15) / max(1, np.sum(r_mask))) * 100.0
    n_over = np.sum(diff >= 0.15)
    n_under = np.sum(diff <= -0.15)
    print(f"Alpha: {alpha:4.2f} | Risk Nodes: {np.sum(r_mask):4d} | Match Rate: {rate:5.1f}% | OVER: {n_over:3d} | UNDER: {n_under:3d}")
