"""Test local catchment elevation normalization.
"""
import torch, numpy as np, sys, osmnx as ox

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_local_elev():
    from app import REGION_CACHE
    from production_v4 import ProductionFloodPredictorV4
    from test_vectorized_wse import vectorized_wse_envelope
    
    predictor = ProductionFloodPredictorV4("hydro_gine_v5_model.pt", device=device)
    hsr_data = REGION_CACHE['hsr']
    pyg = hsr_data['pyg_data']
    node_list = hsr_data['node_list']
    node_pos = hsr_data['node_pos']
    
    # Compute true local catchment elevation range
    elevs = np.array([node_pos[nid]['elevation'] for nid in node_list])
    min_el, max_el = np.min(elevs), np.max(elevs)
    el_range = max(1.0, max_el - min_el)
    
    local_rel_drop = (max_el - elevs) / el_range
    
    x_np = pyg.x.cpu().numpy().copy()
    dep_d = x_np[:, 16]
    
    # 0: local_rel_drop
    x_np[:, 0] = local_rel_drop
    # 23: sink_depth (strictly only for valley bottoms in the local catchment)
    x_np[:, 23] = np.where((local_rel_drop >= 0.50) & (dep_d >= 0.05), dep_d, 0.0)
    
    pyg_local = pyg.clone()
    pyg_local.x = torch.tensor(x_np, dtype=torch.float32, device=device)
    
    preds, raw, probs = predictor.predict(pyg_local, 50.0, 60.0)
    preds_wse = vectorized_wse_envelope(
        pyg_local.edge_index, preds, local_rel_drop, dep_d, x_np[:, 23], elev_range=el_range
    )
    
    y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])
    
    print("=" * 115)
    print("TOP 15 HIGHLIGHTED NODES (WITH LOCAL CATCHMENT ELEVATION + WSE ENVELOPE):")
    print("=" * 115)
    print(f"{'Rank':<5s} | {'Node ID':<14s} | {'GNN Depth (m)':<16s} | {'SWMM Depth (m)':<16s} | {'Elev':<8s} | {'LocRelDrop':<10s} | {'Diff (cm)'}")
    print("-" * 115)
    
    top_idx = np.argsort(preds_wse)[::-1]
    for r, idx in enumerate(top_idx[:15], 1):
        nid = node_list[idx]
        p = preds_wse[idx]
        swmm = y_swmm[idx]
        diff = (p - swmm) * 100.0
        elev = elevs[idx]
        lrd = local_rel_drop[idx]
        print(f"{r:<5d} | {str(nid):<14s} | {p:<16.4f} | {swmm:<16.4f} | {elev:<8.2f} | {lrd:<10.4f} | {diff:<+10.1f}")

if __name__ == '__main__':
    test_local_elev()
