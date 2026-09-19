"""Test vectorized WSE envelope calculation using PyG edge_index.
"""
import torch, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vectorized_wse_envelope(edge_index, preds, rel_drop, dep_d, sink_d, elev_range=17.0):
    num_nodes = len(preds)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    
    # Ground elevation for every node: elev_i = - rel_drop_i * elev_range
    elevs = - rel_drop * elev_range
    wse = elevs + preds
    
    # For every edge (u -> v):
    # backwater from u to v: backwater_v = max(0, wse[u] - elevs[v])
    backwater_depth = np.maximum(0.0, wse[src] - elevs[dst])
    
    # Compute maximum allowed backwater for every node from all incoming/outgoing edges
    max_allowed_backwater = np.zeros(num_nodes, dtype=np.float32)
    np.maximum.at(max_allowed_backwater, dst, backwater_depth)
    np.maximum.at(max_allowed_backwater, src, np.maximum(0.0, wse[dst] - elevs[src]))
    
    # For nodes that are NOT genuine deep enclosed valley sinks (dep_d < 0.05 or rel_drop < 0.50)
    is_true_sink = (dep_d >= 0.06) & (sink_d >= 0.05) & (rel_drop >= 0.50)
    
    corrected_preds = preds.copy()
    
    # Sloped/flat non-sink nodes are bounded by the maximum backwater from neighbors
    non_sink_mask = ~is_true_sink
    corrected_preds[non_sink_mask] = np.minimum(
        preds[non_sink_mask],
        np.maximum(0.0, max_allowed_backwater[non_sink_mask])
    )
    
    # If the ground is completely above neighbor water surfaces, zero it out
    zero_out = non_sink_mask & (max_allowed_backwater < 0.02) & (dep_d < 0.03)
    corrected_preds[zero_out] = 0.0
    
    return corrected_preds

def test():
    from app import REGION_CACHE
    from production_v4 import ProductionFloodPredictorV4
    
    predictor = ProductionFloodPredictorV4("hydro_gine_v5_model.pt", device=device)
    hsr_data = REGION_CACHE['hsr']
    pyg = hsr_data['pyg_data']
    node_list = hsr_data['node_list']
    node_pos = hsr_data['node_pos']
    
    preds, raw, probs = predictor.predict(pyg, 50.0, 60.0)
    x_np = pyg.x.cpu().numpy()
    
    preds_wse = vectorized_wse_envelope(
        pyg.edge_index, preds, x_np[:, 0], x_np[:, 16], x_np[:, 23]
    )
    
    y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])
    
    print("=" * 115)
    print("TOP 15 HIGHLIGHTED NODES (WITH VECTORIZED WSE ENVELOPE):")
    print("=" * 115)
    print(f"{'Rank':<5s} | {'Node ID':<14s} | {'GNN Depth (m)':<16s} | {'SWMM Depth (m)':<16s} | {'Elev':<8s} | {'DepD':<8s} | {'Diff (cm)'}")
    print("-" * 115)
    
    top_idx = np.argsort(preds_wse)[::-1]
    for r, idx in enumerate(top_idx[:15], 1):
        nid = node_list[idx]
        p = preds_wse[idx]
        swmm = y_swmm[idx]
        diff = (p - swmm) * 100.0
        elev = node_pos[nid]['elevation']
        depd = x_np[idx, 16]
        print(f"{r:<5d} | {str(nid):<14s} | {p:<16.4f} | {swmm:<16.4f} | {elev:<8.2f} | {depd:<8.4f} | {diff:<+10.1f}")

if __name__ == '__main__':
    test()
