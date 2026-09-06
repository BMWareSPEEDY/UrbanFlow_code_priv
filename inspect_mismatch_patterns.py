"""Inspect the mismatched nodes on Koramangala and Electronic City.
"""
import torch, numpy as np
from app import REGION_CACHE, PRODUCTION_PREDICTOR

for r_key in ['koramangala', 'ecity', 'whitefield']:
    r_data = REGION_CACHE[r_key]
    g = r_data['pyg_data']
    preds, _, probs = PRODUCTION_PREDICTOR.predict(g, 50.0, 60.0)
    node_list = r_data['node_list']
    node_pos = r_data['node_pos']
    y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])
    
    diff = preds - y_swmm
    risk_mask = (preds > 0.08)
    
    print(f"\n==================== {r_key.upper()} ====================")
    print(f"Total Nodes: {len(node_list)} | Total Risk Nodes: {np.sum(risk_mask)}")
    print(f"Total Network Accuracy within +-15cm: {np.mean(np.abs(diff) < 0.15)*100:.1f}%")
    print(f"Risk Subset Match Rate: {np.mean(np.abs(diff[risk_mask]) < 0.15)*100:.1f}%")
    print(f"Risk Nodes OVER (>+15cm): {np.sum(diff[risk_mask] > 0.15)}")
    print(f"Risk Nodes UNDER (<-15cm): {np.sum(diff[risk_mask] < -0.15)}")
    
    # Check top 10 errors
    idx_sorted = np.argsort(-np.abs(diff[risk_mask]))
    risk_indices = np.where(risk_mask)[0]
    print("Top 5 largest discrepancies:")
    for rank in range(min(5, len(idx_sorted))):
        i = risk_indices[idx_sorted[rank]]
        print(f"  Node #{node_list[i]}: GNN={preds[i]:.3f}m | SWMM={y_swmm[i]:.3f}m | Diff={diff[i]:+.3f}m | Elev={node_pos[node_list[i]]['elevation']:.1f}m | Acc={g.x[i, 5]:.2f} | Slope={g.x[i, 2]:.3f} | Dep={g.x[i, 16]:.3f} | ConvDef={g.x[i, 30]:.3f}")
