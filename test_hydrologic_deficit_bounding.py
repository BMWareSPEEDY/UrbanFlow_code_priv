"""Test physical hydrologic inflow deficit bounding on HSR.
"""
import osmnx as ox, numpy as np, torch, sys

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_deficit_bounding():
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
    in_deg = x_np[:, 3]
    out_deg = x_np[:, 4]
    log_area = x_np[:, 10]
    dep_d = x_np[:, 16]
    sink_d = x_np[:, 23]
    total_rain_mm = 50.0
    
    # Compute real upstream catchment area in m^2
    acc_area = np.expm1(log_area)
    # Total rain volume falling on this catchment in m^3
    rain_vol_m3 = acc_area * (total_rain_mm / 1000.0)
    
    # Standard street inlet drain capacity over 1 hour = 0.05 m^3/s * 3600s = 180 m^3
    inlet_capacity_m3 = 180.0 * np.maximum(1.0, out_deg)
    
    # Net excess water volume (m^3)
    excess_volume_m3 = np.maximum(0.0, rain_vol_m3 - inlet_capacity_m3)
    
    # Physical maximum depth from net volume: V / pond_area (assuming min pond area 300 m^2)
    volumetric_depth_cap = np.where(
        excess_volume_m3 <= 0.0,
        0.02, # 100% drained by storm sewer catch basins
        np.clip(excess_volume_m3 / 400.0, 0.02, 3.0)
    )
    
    # Corrected sink depth using volumetric deficit
    x_np[:, 23] = np.where(
        (rel_drop >= 0.50) & (dep_d >= 0.05) & (excess_volume_m3 > 0.0),
        np.minimum(dep_d, volumetric_depth_cap),
        0.0
    )
    
    pyg_bounded = pyg.clone()
    pyg_bounded.x = torch.tensor(x_np, dtype=torch.float32, device=device)
    
    preds, raw, probs = predictor.predict(pyg_bounded, 50.0, 60.0)
    
    # Apply volumetric ceiling and WSE envelope
    preds_bounded = np.minimum(preds, volumetric_depth_cap)
    preds_final = vectorized_wse_envelope(
        pyg_bounded.edge_index, preds_bounded, rel_drop, dep_d, x_np[:, 23]
    )
    
    y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])
    
    print("=" * 115)
    print("TOP 15 HIGHLIGHTED NODES (WITH VOLUMETRIC DEFICIT CEILING + WSE ENVELOPE):")
    print("=" * 115)
    print(f"{'Rank':<5s} | {'Node ID':<14s} | {'GNN Depth (m)':<16s} | {'SWMM Depth (m)':<16s} | {'Elev':<8s} | {'AccArea':<10s} | {'Diff (cm)'}")
    print("-" * 115)
    
    top_idx = np.argsort(preds_final)[::-1]
    for r, idx in enumerate(top_idx[:15], 1):
        nid = node_list[idx]
        p = preds_final[idx]
        swmm = y_swmm[idx]
        diff = (p - swmm) * 100.0
        elev = node_pos[nid]['elevation']
        aa = acc_area[idx]
        print(f"{r:<5d} | {str(nid):<14s} | {p:<16.4f} | {swmm:<16.4f} | {elev:<8.2f} | {aa:<10.1f} | {diff:<+10.1f}")
        
    swmm_crit = (y_swmm >= 0.30)
    gnn_crit = (preds_final >= 0.30)
    tp = int(np.sum(swmm_crit & gnn_crit))
    fp = int(np.sum(~swmm_crit & gnn_crit))
    shallow_fp = int(np.sum((y_swmm <= 0.08) & (preds_final >= 0.15)))
    mae_cm = np.mean(np.abs(preds_final - y_swmm)) * 100.0
    pct_30 = np.mean(np.abs(preds_final - y_swmm) <= 0.30) * 100.0
    
    print(f"\n--- OVERALL BENCHMARK @ 50 mm/hr ---")
    print(f"  Critical TP: {tp} | Critical FP: {fp}")
    print(f"  Shallow FP (p >= 15cm on SWMM <= 8cm): {shallow_fp} / {int(np.sum(y_swmm <= 0.08))}")
    print(f"  Depth MAE: {mae_cm:.2f} cm | % <= 30cm: {pct_30:.1f}%")

if __name__ == '__main__':
    test_deficit_bounding()
