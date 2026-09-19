"""Diagnose Adjacent Node Spillover and WSE violations in HSR and international cities.
"""
import os, sys, torch, numpy as np, osmnx as ox

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from production_v4 import ProductionFloodPredictorV4

def diagnose_hsr_spillover():
    print("=" * 90)
    print("DIAGNOSING ADJACENT NODE SPILLOVER ON HSR LAYOUT")
    print("=" * 90)
    
    predictor = ProductionFloodPredictorV4("hydro_gine_v5_model.pt", device=device)
    dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
    
    g_50 = [g for g in dl if getattr(g, 'region', '') == 'hsr' and abs(g.rain_intensity - 50.0) < 1e-3][0]
    y_true = g_50.y.cpu().numpy().ravel()
    preds, raw, probs = predictor.predict(g_50, 50.0, 60.0)
    
    # Load networkx graph for elevations and coordinates
    G = ox.load_graphml("bengaluru_complete_graph.graphml")
    nodes_list = list(G.nodes())
    node_to_idx = {nid: idx for idx, nid in enumerate(nodes_list)}
    
    elevs = np.array([float(G.nodes[nid].get('elevation', 880.0)) for nid in nodes_list])
    
    # Find all edges (u, v) where one node is predicted flooded (>0.15m) and the other is predicted flooded (>0.15m)
    # but SWMM shows only one of them is actually flooded!
    spillover_cases = []
    
    for u, v in G.edges():
        if u not in node_to_idx or v not in node_to_idx:
            continue
        idx_u = node_to_idx[u]
        idx_v = node_to_idx[v]
        
        swmm_u, swmm_v = y_true[idx_u], y_true[idx_v]
        pred_u, pred_v = preds[idx_u], preds[idx_v]
        elev_u, elev_v = elevs[idx_u], elevs[idx_v]
        
        # Case where u is true flood in SWMM (e.g. >0.15m), but v is dry (<=0.05m) in SWMM,
        # yet the model predicts >0.15m on v!
        if (swmm_u >= 0.15 and swmm_v <= 0.05 and pred_v >= 0.15):
            elev_diff = elev_v - elev_u  # positive if v is uphill from u
            wse_u = elev_u + pred_u
            allowed_depth_v = max(0.0, wse_u - elev_v)
            spillover_cases.append({
                'sink_nid': u, 'sink_elev': elev_u, 'sink_swmm': swmm_u, 'sink_pred': pred_u,
                'dry_nid': v, 'dry_elev': elev_v, 'dry_swmm': swmm_v, 'dry_pred': pred_v,
                'elev_diff': elev_diff, 'wse_u': wse_u, 'allowed_depth_v': allowed_depth_v
            })
            
    print(f"Total HSR Nodes: {len(nodes_list)}")
    print(f"Found {len(spillover_cases)} edge instances where a dry neighbor is falsely inflated by an adjacent flooded node!")
    
    print("\nTop 15 Worst Spillover Edges:")
    print(f"{'Sink ID':<12s} | {'Sink El':<8s} | {'SWMM':<6s} | {'Pred':<6s} || {'Dry Nbr ID':<12s} | {'Dry El':<8s} | {'SWMM':<6s} | {'Pred':<6s} || {'El Diff':<8s} | {'Allowed'}")
    print("-" * 115)
    for c in spillover_cases[:15]:
        print(f"{str(c['sink_nid']):<12s} | {c['sink_elev']:<8.2f} | {c['sink_swmm']:<6.2f} | {c['sink_pred']:<6.2f} || "
              f"{str(c['dry_nid']):<12s} | {c['dry_elev']:<8.2f} | {c['dry_swmm']:<6.2f} | {c['dry_pred']:<6.2f} || "
              f"{c['elev_diff']:<+8.2f} | {c['allowed_depth_v']:<6.2f}")

if __name__ == '__main__':
    diagnose_hsr_spillover()
