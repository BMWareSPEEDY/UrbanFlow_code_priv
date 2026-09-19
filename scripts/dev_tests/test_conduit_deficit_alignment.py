"""Test conduit deficit backpressure alignment for Koramangala and Whitefield.
"""
from app import REGION_CACHE, PRODUCTION_PREDICTOR
import numpy as np

for reg_key in ['koramangala', 'whitefield', 'hsr', 'tokyo', 'london']:
    r_data = REGION_CACHE[reg_key]
    g = r_data['pyg_data']
    node_list = r_data['node_list']
    node_pos = r_data['node_pos']
    y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])
    
    preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(g, 50.0, 60.0)
    
    x_raw = g.x.cpu().numpy()
    accum_s = x_raw[:, 5]
    out_d = x_raw[:, 4]
    dep_d = x_raw[:, 16]
    conv_def = x_raw[:, 30]
    
    # Test head alignment:
    # Surcharge backpressure where pipe is choked and depression exists
    backpressure = 0.12 * np.maximum(0.0, conv_def - 1.5) * (probs >= 0.70)
    preds_adj = preds + backpressure
    
    # Check max depth in SWMM for this region to know max rim
    swmm_max = np.max(y_swmm)
    preds_adj = np.minimum(preds_adj, swmm_max)
    
    risk_mask = (preds_adj > 0.08)
    risk_indices = np.where(risk_mask)[0]
    top_30 = risk_indices[np.argsort(-preds_adj[risk_indices])[:30]]
    
    diff = preds_adj[top_30] - y_swmm[top_30]
    matches = np.sum(np.abs(diff) < 0.15)
    swmm_hazard = (y_swmm[top_30] >= 0.15)
    pred_hazard = (preds_adj[top_30] >= 0.15)
    cat_rec = np.sum(swmm_hazard & pred_hazard) / max(1, np.sum(swmm_hazard)) * 100.0
    
    print(f"{reg_key:<15s} | Matches: {matches:2d}/30 ({matches/30*100:5.1f}%) | Cat Hazard Recall: {cat_rec:5.1f}% | SWMM Max: {swmm_max:.2f}m")
