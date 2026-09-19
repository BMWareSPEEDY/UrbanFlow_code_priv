"""Test production_v4 with integrated Hydrostatic WSE Inundation Envelope.
"""
import torch, numpy as np, sys, osmnx as ox

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_envelope_integration():
    from app import REGION_CACHE, init_app_data
    from production_v4 import ProductionFloodPredictorV4
    
    predictor = ProductionFloodPredictorV4("hydro_gine_v5_model.pt", device=device)
    hsr_data = REGION_CACHE['hsr']
    
    preds, raw, probs = predictor.predict(hsr_data['pyg_data'], 50.0, 60.0)
    
    # Check top risk nodes
    node_list = hsr_data['node_list']
    node_pos = hsr_data['node_pos']
    
    top_idx = np.argsort(preds)[::-1]
    
    print("=" * 115)
    print("TOP 15 HIGHLIGHTED NODES IN HSR LAYOUT @ 50 mm/hr (WITH INTEGRATED WSE ENVELOPE):")
    print("=" * 115)
    print(f"{'Rank':<5s} | {'Node ID':<14s} | {'GNN Depth (m)':<16s} | {'SWMM Depth (m)':<16s} | {'Elev':<8s} | {'DepD':<8s} | {'Diff (cm)'}")
    print("-" * 115)
    
    for r, idx in enumerate(top_idx[:15], 1):
        nid = node_list[idx]
        p = preds[idx]
        swmm = node_pos[nid]['swmm_depth']
        diff = (p - swmm) * 100.0
        elev = node_pos[nid]['elevation']
        depd = hsr_data['pyg_data'].x[idx, 16].item()
        print(f"{r:<5d} | {str(nid):<14s} | {p:<16.4f} | {swmm:<16.4f} | {elev:<8.2f} | {depd:<8.4f} | {diff:<+10.1f}")

if __name__ == '__main__':
    test_envelope_integration()
