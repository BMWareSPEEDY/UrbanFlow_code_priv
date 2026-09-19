"""Test user's exact formula on Koramangala and Whitefield.
"""
from app import REGION_CACHE, PRODUCTION_PREDICTOR
import numpy as np

for reg_name in ['koramangala', 'whitefield', 'ecity', 'hsr']:
    r_data = REGION_CACHE[reg_name]
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
    dist_outlet = x_raw[:, 22] if x_raw.shape[1] > 22 else np.full(len(preds), 1.0)
    
    # Test user's exact formula:
    # y_adj = max(y, alpha * log(1 + A_upstream))
    # where outdegree <= 1 or dist_to_outlet < 0.05
    for alpha in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8]:
        is_terminal = (out_d <= 1) | (dist_outlet < 0.05) | (dep_d >= 2.0)
        adj_head = alpha * np.log1p(accum_s * 4.0)
        y_adj = np.where(is_terminal & (probs >= 0.70), np.maximum(preds, adj_head), preds)
        y_adj = np.minimum(y_adj, 2.55)
        
        # Check top 30 risk nodes
        risk_idx = np.where(y_adj > 0.08)[0]
        top30 = risk_idx[np.argsort(-y_adj[risk_idx])[:30]]
        
        diff = y_adj[top30] - y_swmm[top30]
        matches = np.sum(np.abs(diff) < 0.15)
        print(f"{reg_name:<12s} (alpha={alpha:.1f}): Matches = {matches:2d}/30 ({matches/30*100:5.1f}%) | MeanDiff = {np.mean(diff):+.3f}m")
