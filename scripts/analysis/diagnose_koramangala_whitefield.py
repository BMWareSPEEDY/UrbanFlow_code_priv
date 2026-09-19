"""Diagnose Koramangala and Whitefield top 30 SWMM hotspots.
"""
from app import REGION_CACHE, PRODUCTION_PREDICTOR
import numpy as np

for reg_key in ['koramangala', 'whitefield', 'hsr']:
    r_data = REGION_CACHE[reg_key]
    g = r_data['pyg_data']
    node_list = r_data['node_list']
    node_pos = r_data['node_pos']
    y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])
    
    preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(g, 50.0, 60.0)
    
    # Sort by SWMM depth descending (The genuine ground truth worst flood hotspots)
    top_swmm_idx = np.argsort(-y_swmm)[:30]
    
    # Sort by GNN prediction descending (What the model currently ranks as top 30)
    top_gnn_idx = np.argsort(-preds)[:30]
    
    print("=" * 80)
    print(f"REGION: {reg_key.upper()} (Total nodes: {len(node_list)})")
    print("=" * 80)
    print(f"Top 30 SWMM Floods: min={y_swmm[top_swmm_idx[-1]]:.2f}m, max={y_swmm[top_swmm_idx[0]]:.2f}m, mean={np.mean(y_swmm[top_swmm_idx]):.2f}m")
    print(f"Top 30 GNN Preds:   min={preds[top_gnn_idx[-1]]:.2f}m, max={preds[top_gnn_idx[0]]:.2f}m, mean={np.mean(preds[top_gnn_idx]):.2f}m")
    
    # Check GNN accuracy on the top 30 SWMM hotspots
    diff_on_swmm_hotspots = preds[top_swmm_idx] - y_swmm[top_swmm_idx]
    matches_swmm = np.sum(np.abs(diff_on_swmm_hotspots) < 0.15)
    print(f"GNN Match Rate on TOP 30 SWMM HOTSPOTS: {matches_swmm}/30 ({matches_swmm/30*100:.1f}%)")
    
    # Check GNN accuracy on the top 30 GNN predicted hotspots
    diff_on_gnn_hotspots = preds[top_gnn_idx] - y_swmm[top_gnn_idx]
    matches_gnn = np.sum(np.abs(diff_on_gnn_hotspots) < 0.15)
    print(f"GNN Match Rate on TOP 30 GNN PREDICTED HOTSPOTS: {matches_gnn}/30 ({matches_gnn/30*100:.1f}%)")
    
    print("\nSample top 5 SWMM nodes:")
    for idx in top_swmm_idx[:5]:
        nid = node_list[idx]
        print(f"  Node {nid}: SWMM={y_swmm[idx]:.3f}m | GNN={preds[idx]:.3f}m | Prob={probs[idx]:.2f} | OutDeg={g.x[idx, 4]:.0f} | Acc={g.x[idx, 5]:.2f} | Dep={g.x[idx, 16]:.2f} | ConvDef={g.x[idx, 30]:.2f}")
