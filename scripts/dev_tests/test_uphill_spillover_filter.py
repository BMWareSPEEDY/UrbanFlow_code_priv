"""Test Uphill Conveyance Spillover Filter on 3-node clusters in HSR.
"""
import torch, numpy as np, sys, osmnx as ox

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from test_app_feature_computation import compute_true_graph_features
from production_v4 import ProductionFloodPredictorV4

def test_spillover_filter():
    G = ox.load_graphml("bengaluru_complete_graph.graphml")
    data_hsr = compute_true_graph_features(G, 50.0, 60.0)
    predictor = ProductionFloodPredictorV4("hydro_gine_v4_3_model.pt", device=device)
    
    pred, raw, prob = predictor.predict(data_hsr, 50.0, 60.0)
    
    nodes_list = list(G.nodes())
    node_to_idx = {nid: idx for idx, nid in enumerate(nodes_list)}
    elevs = np.array([float(G.nodes[nid].get('elevation', 880.0)) for nid in nodes_list])
    
    # Check 1-hop downhill slope
    has_downhill_escape = np.zeros(len(nodes_list), dtype=bool)
    max_downhill_drop = np.zeros(len(nodes_list), dtype=np.float32)
    
    for u, v, k, data in G.edges(keys=True, data=True):
        idx_u = node_to_idx[u]
        idx_v = node_to_idx[v]
        drop = elevs[idx_u] - elevs[idx_v]
        if drop > 0.15: # u is uphill from v by >15cm
            has_downhill_escape[idx_u] = True
            max_downhill_drop[idx_u] = max(max_downhill_drop[idx_u], drop)
            
    x_np = data_hsr.x.cpu().numpy()
    dep_depth = x_np[:, 16]
    sink_depth = x_np[:, 23]
    
    # Nodes on uphill sloped road with zero depression and positive downhill drop
    is_uphill_free_drain = has_downhill_escape & (dep_depth < 0.03) & (sink_depth < 0.03)
    
    print(f"Total HSR Nodes: {len(nodes_list)}")
    print(f"Nodes with active downhill gravity escape (>15cm drop & dep_depth=0): {np.sum(is_uphill_free_drain)}")
    
    pred_cleaned = np.where(is_uphill_free_drain & (prob < 0.80), 0.0, pred)
    
    print(f"\nBefore Spillover Filter @ 50mm/hr:")
    print(f"  Flooded Nodes (>15cm): {np.sum(pred > 0.15)}")
    print(f"  Watch Nodes (5-15cm):  {np.sum((pred > 0.05) & (pred <= 0.15))}")
    print(f"  Safe Nodes (<=5cm):    {np.sum(pred <= 0.05)}")
    
    print(f"\nAfter Spillover Filter @ 50mm/hr:")
    print(f"  Flooded Nodes (>15cm): {np.sum(pred_cleaned > 0.15)}")
    print(f"  Watch Nodes (5-15cm):  {np.sum((pred_cleaned > 0.05) & (pred_cleaned <= 0.15))}")
    print(f"  Safe Nodes (<=5cm):    {np.sum(pred_cleaned <= 0.05)}")

if __name__ == '__main__':
    test_spillover_filter()
