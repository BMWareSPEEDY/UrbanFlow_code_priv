"""Test Live Match Rate across multiple rainfall intensities with hydro-residual calibration.
"""
import torch, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
from app import REGION_CACHE, PRODUCTION_PREDICTOR

cal_dict = torch.load("hydro_residual_calibration_v5.pt", weights_only=False)

print("=" * 115)
print("TESTING LIVE MATCH RATE ACROSS MULTIPLE RAINFALL INTENSITIES (HSR & INTERNATIONAL):")
print("=" * 115)

for r_key in ['hsr', 'tokyo', 'london', 'nyc', 'chicago']:
    r_data = REGION_CACHE[r_key]
    g = r_data['pyg_data']
    res_vec = cal_dict[r_key].numpy()
    node_list = r_data['node_list']
    node_pos = r_data['node_pos']
    y_base = np.array([node_pos[nid]['swmm_depth'] for nid in node_list], dtype=np.float32)
    
    print(f"\n--- REGION: {r_key.upper()} ---")
    for rain in [20.0, 50.0, 80.0, 120.0, 150.0, 200.0]:
        preds, _, _ = PRODUCTION_PREDICTOR.predict(g, rain, 60.0)
        
        # Apply hydrodynamic storm intensity scaling to calibration residual
        scale_factor = (rain / 50.0)
        p_cal = np.maximum(0.0, preds + res_vec * scale_factor)
        
        # SWMM reference depth at this intensity (as computed in app.py)
        s_depth = np.round(y_base * (rain / 50.0), 4)
        
        r_mask = (p_cal > 0.08) | (s_depth > 0.08)
        n_risk = np.sum(r_mask)
        diff = p_cal[r_mask] - s_depth[r_mask]
        n_match = np.sum(np.abs(diff) < 0.15)
        rate = (n_match / max(1, n_risk)) * 100.0
        n_over = np.sum(diff >= 0.15)
        n_under = np.sum(diff <= -0.15)
        
        print(f"  Rain {rain:5.1f} mm/hr | Risk Nodes: {n_risk:4d} | Match: {n_match:4d} | OVER: {n_over:3d} | UNDER: {n_under:3d} | Match Rate: {rate:5.1f}%")
