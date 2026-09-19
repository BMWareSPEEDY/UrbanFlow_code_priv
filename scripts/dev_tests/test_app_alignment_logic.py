"""Test hydraulic post-processing logic for app.py across all 16 cities.
"""
from app import REGION_CACHE, PRODUCTION_PREDICTOR
import numpy as np

print("=" * 110)
print("TESTING POST-INFERENCE HYDRAULIC HOTSPOT CALIBRATION ACROSS ALL 16 CITIES:")
print("=" * 110)

def test_hotspot_alignment_in_app():
    all_rates = {}
    tot_h = 0
    tot_m = 0
    cat_recalls = {}
    
    for r_key, r_data in REGION_CACHE.items():
        g = r_data['pyg_data']
        node_list = r_data['node_list']
        node_pos = r_data['node_pos']
        y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list], dtype=np.float32)
        
        preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(g, 50.0, 60.0)
        
        x_raw = g.x.cpu().numpy()
        rel_drop = x_raw[:, 0]
        slope = np.abs(x_raw[:, 2])
        out_d = x_raw[:, 4]
        accum_s = x_raw[:, 5]
        dep_d = x_raw[:, 16]
        conv_def = x_raw[:, 30]
        dist_outlet = x_raw[:, 22] if x_raw.shape[1] > 22 else np.full(len(preds), 1.0)
        
        p_final = np.copy(preds)
        
        # 1. Ridge Clamping: Steep slopes with low accumulation cannot pool water
        is_ridge = (slope > 0.025) & (accum_s < 0.8) & (dep_d < 0.05)
        p_final = np.where(is_ridge, 0.0, p_final)
        
        # 2. Terminal Basin Head Alignment:
        # Outfall nodes terminating into retention basins/lakes (out_d <= 1 or dist_outlet < 0.05)
        # where conduit is surcharging and depression is deep:
        is_terminal = ((out_d <= 1) | (dist_outlet < 0.05)) & (probs >= 0.70)
        
        # In deep sags, water surface elevation levels up to the depression storage head
        # We align the top predicted depths using the local storage depth:
        head_boost = np.where(is_terminal & (dep_d > 0.5), np.minimum(2.55, dep_d * 0.85), p_final)
        p_final = np.maximum(p_final, head_boost)
        p_final = np.minimum(p_final, 2.55)
        
        # Select top 30 risk hotspots by predicted depth
        risk_idx = np.where(p_final > 0.08)[0]
        top30 = risk_idx[np.argsort(-p_final[risk_idx])[:30]]
        
        diff = p_final[top30] - y_swmm[top30]
        matches = np.sum(np.abs(diff) < 0.15)
        rate = (matches / len(top30)) * 100.0
        all_rates[r_key] = rate
        tot_h += len(top30)
        tot_m += matches
        
        swmm_haz = (y_swmm[top30] >= 0.15)
        pred_haz = (p_final[top30] >= 0.15)
        cat_rec = np.sum(swmm_haz & pred_haz) / max(1, np.sum(swmm_haz)) * 100.0
        cat_recalls[r_key] = cat_rec
        
        print(f"  {r_key:<15s}: {matches:2d}/30 ({rate:5.1f}%) | Cat Recall: {cat_rec:5.1f}%")
        
    print("-" * 110)
    print(f"Global Hotspot Match Rate: {tot_m}/{tot_h} ({tot_m/tot_h*100:.1f}%) | Min: {min(all_rates.values()):.1f}%")

test_hotspot_alignment_in_app()
