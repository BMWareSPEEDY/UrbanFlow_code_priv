"""Test fixed sink depth and WSE envelope in app.py.
"""
import torch, numpy as np, sys, osmnx as ox

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_fixed_sinks():
    from app import REGION_CACHE
    from production_v4 import ProductionFloodPredictorV4
    from test_vectorized_wse import vectorized_wse_envelope
    
    predictor = ProductionFloodPredictorV4("hydro_gine_v5_model.pt", device=device)
    hsr_data = REGION_CACHE['hsr']
    pyg = hsr_data['pyg_data']
    node_list = hsr_data['node_list']
    node_pos = hsr_data['node_pos']
    
    x_np = pyg.x.cpu().numpy().copy()
    rel_drop = x_np[:, 0]
    dep_d = x_np[:, 16]
    
    # Corrected sink depth: only non-zero in low-lying valley bottoms (rel_drop >= 0.50)
    x_np[:, 23] = np.where((rel_drop >= 0.50) & (dep_d >= 0.05), dep_d, 0.0)
    
    pyg_fixed = pyg.clone()
    pyg_fixed.x = torch.tensor(x_np, dtype=torch.float32, device=device)
    
    preds, raw, probs = predictor.predict(pyg_fixed, 50.0, 60.0)
    
    preds_wse = vectorized_wse_envelope(
        pyg_fixed.edge_index, preds, rel_drop, x_np[:, 16], x_np[:, 23]
    )
    
    y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])
    
    print("=" * 115)
    print("TOP 15 HIGHLIGHTED NODES (WITH VALLEY-GATED SINK + WSE ENVELOPE):")
    print("=" * 115)
    print(f"{'Rank':<5s} | {'Node ID':<14s} | {'GNN Depth (m)':<16s} | {'SWMM Depth (m)':<16s} | {'Elev':<8s} | {'RelDrop':<8s} | {'Diff (cm)'}")
    print("-" * 115)
    
    top_idx = np.argsort(preds_wse)[::-1]
    for r, idx in enumerate(top_idx[:15], 1):
        nid = node_list[idx]
        p = preds_wse[idx]
        swmm = y_swmm[idx]
        diff = (p - swmm) * 100.0
        elev = node_pos[nid]['elevation']
        rd = rel_drop[idx]
        print(f"{r:<5d} | {str(nid):<14s} | {p:<16.4f} | {swmm:<16.4f} | {elev:<8.2f} | {rd:<8.4f} | {diff:<+10.1f}")

if __name__ == '__main__':
    test_fixed_sinks()
