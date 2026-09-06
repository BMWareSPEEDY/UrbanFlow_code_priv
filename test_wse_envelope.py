"""Test Hydrostatic WSE Inundation Envelope on HSR and international cities.
"""
import torch, numpy as np, sys, osmnx as ox

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from production_v4 import ProductionFloodPredictorV4

def apply_hydrostatic_wse_envelope(G, preds, x_np):
    nodes_list = list(G.nodes())
    node_to_idx = {nid: idx for idx, nid in enumerate(nodes_list)}
    elevs = np.array([float(G.nodes[nid].get('elevation', 880.0)) for nid in nodes_list])
    
    dep_d = x_np[:, 16]
    sink_d = x_np[:, 23]
    rel_drop = x_np[:, 0]
    out_d = x_np[:, 4]
    
    # 1. Compute Water Surface Elevation (WSE) for every node:
    # WSE = ground elevation + predicted depth
    wse = elevs + preds
    
    # 2. For every node that is NOT a genuine valley sink bowl:
    # (i.e. dep_depth < 0.05m or rel_drop < 0.50m)
    # The water depth CANNOT exceed the backwater inundation level from any adjacent node
    preds_corrected = preds.copy()
    
    for nid, idx in node_to_idx.items():
        # If it is a true enclosed valley sink bowl, it holds its own pooled water
        if dep_d[idx] >= 0.06 and sink_d[idx] >= 0.05 and rel_drop[idx] >= 0.50:
            continue
            
        # For non-sink sloped/flat nodes, find maximum incoming backwater surface level from neighbors
        nbrs = list(G.neighbors(nid))
        if not nbrs:
            if dep_d[idx] < 0.03:
                preds_corrected[idx] = 0.0
            continue
            
        nbr_indices = [node_to_idx[n] for n in nbrs if n in node_to_idx]
        if not nbr_indices:
            continue
            
        # Max water surface elevation among all adjacent neighbors
        max_nbr_wse = max(wse[n_idx] for n_idx in nbr_indices)
        
        # Physical maximum water depth node `idx` can hold from neighbor backwater
        allowed_backwater_depth = max(0.0, max_nbr_wse - elevs[idx])
        
        # If the node's ground is above the neighbor's water surface, depth is 0
        if allowed_backwater_depth < 0.03 and dep_d[idx] < 0.03:
            preds_corrected[idx] = 0.0
        else:
            preds_corrected[idx] = min(preds_corrected[idx], allowed_backwater_depth)
            
    return preds_corrected

def test_wse_envelope_hsr():
    predictor = ProductionFloodPredictorV4("hydro_gine_v5_model.pt", device=device)
    dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
    G = ox.load_graphml("bengaluru_complete_graph.graphml")
    
    g_50 = [g for g in dl if getattr(g, 'region', '') == 'hsr' and abs(g.rain_intensity - 50.0) < 1e-3][0]
    y_true = g_50.y.cpu().numpy().ravel()
    preds_raw, _, probs = predictor.predict(g_50, 50.0, 60.0)
    
    preds_wse = apply_hydrostatic_wse_envelope(G, preds_raw, g_50.x.cpu().numpy())
    
    # Evaluate BEFORE vs AFTER
    for name, p in [("Raw GNN v5.0", preds_raw), ("With Hydrostatic WSE Envelope", preds_wse)]:
        swmm_crit = (y_true >= 0.30)
        gnn_crit = (p >= 0.30)
        tp = int(np.sum(swmm_crit & gnn_crit))
        fp = int(np.sum(~swmm_crit & gnn_crit))
        swmm_shallow = (y_true <= 0.08)
        shallow_fp = int(np.sum(swmm_shallow & (p >= 0.15)))
        mae_cm = np.mean(np.abs(p - y_true)) * 100.0
        pct_30 = np.mean(np.abs(p - y_true) <= 0.30) * 100.0
        
        print(f"\n--- {name} @ 50 mm/hr ---")
        print(f"  Critical TP: {tp}/91 ({tp/91*100:.1f}%) | Critical FP: {fp}")
        print(f"  Shallow FP (p >= 15cm on SWMM <= 8cm): {shallow_fp} / {int(np.sum(swmm_shallow))} ({shallow_fp/np.sum(swmm_shallow)*100:.2f}%)")
        print(f"  Depth MAE: {mae_cm:.2f} cm | % <= 30cm: {pct_30:.1f}%")

if __name__ == '__main__':
    test_wse_envelope_hsr()
