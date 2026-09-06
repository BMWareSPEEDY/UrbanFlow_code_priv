"""Test calibrated pipeline that achieves high recall (low FN) while preserving low FP.
"""
import torch, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from train_hydro_gine_v5_0 import HydroGINE_v5

def test_calibrated_pipeline():
    ck = torch.load("hydro_gine_v5_model.pt", map_location=device, weights_only=False)
    model = HydroGINE_v5(in_c=ck['in_c'], edge_c=2, hidden=ck.get('hidden', 128), n_layers=ck.get('n_layers', 6)).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    
    x_mean = ck['x_mean'].to(device)
    x_std = ck['x_std'].to(device)
    e_mean = ck['e_mean'].to(device)
    e_std = ck['e_std'].to(device)
    yl_mean = float(ck['yl_mean'])
    yl_std = float(ck['yl_std'])
    
    dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
    hsr_50 = [g for g in dl if getattr(g, 'region', '') == 'hsr' and abs(g.rain_intensity - 50.0) < 1e-3][0].to(device)
    
    y_true = hsr_50.y.cpu().numpy().ravel()
    gx = (hsr_50.x - x_mean) / x_std
    gea = (hsr_50.edge_attr - e_mean) / e_std
    
    with torch.no_grad():
        c_l, d_o = model(gx, hsr_50.edge_index, gea)
        p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
        p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0).cpu().numpy().ravel()
        
    x_raw = hsr_50.x.cpu().numpy()
    rel_drop = x_raw[:, 0]
    in_d = x_raw[:, 3]
    out_d = x_raw[:, 4]
    accum_s = x_raw[:, 5]
    sag_idx = x_raw[:, 8]
    log_area = x_raw[:, 10]
    dep_d = x_raw[:, 16]
    sink_d = x_raw[:, 23]
    total_r = x_raw[:, 27]
    conv_def = x_raw[:, 30]
    
    # 1. Physical Classification of Regimes:
    # A. True Valley Sinks & Surcharging Bottlenecks
    is_choked_surcharge = (conv_def >= 0.7) | (accum_s >= 1.5) | ((dep_d >= 0.20) & (out_d <= in_d))
    is_deep_sink = (sink_d >= 0.08) & (rel_drop >= 0.40)
    is_valley_depression = (dep_d >= 0.05) & (rel_drop >= 0.40)
    is_convergent_sag = (in_d > out_d) | (sag_idx >= 0.03)
    
    # B. Upland Free Drainage (Hill ridges where water physically cannot accumulate)
    is_ridge_crest = (rel_drop < 0.25) & (sink_d < 0.03) & (dep_d < 0.03)
    is_free_drain_slope = (sink_d < 0.02) & (dep_d < 0.02) & (out_d >= 2) & (~is_choked_surcharge)
    
    # 2. Continuous Probability Gating (Smooth sigmoid gate rather than hard threshold)
    # Target baseline threshold tau:
    tau = np.where(
        is_deep_sink | is_valley_depression,
        0.15,
        np.where(
            is_choked_surcharge | is_convergent_sag,
            0.25,
            np.where(
                is_ridge_crest | is_free_drain_slope,
                0.75,
                0.35
            )
        )
    )
    
    # Smooth confidence gate:
    conf_gate = 1.0 / (1.0 + np.exp(-12.0 * (p_prob - tau)))
    
    # 3. Dynamic Mass Headroom Bounds
    mass_bound = np.where(
        is_deep_sink | is_valley_depression | is_choked_surcharge,
        3.0,
        np.where(
            is_ridge_crest,
            0.02,
            np.where(
                is_free_drain_slope,
                0.04 if total_r[0] <= 50.0 else 0.10,
                np.where(p_prob >= 0.65, np.maximum(0.40, sink_d * 2.0 + 0.20), np.maximum(0.15, sink_d * 1.5 + 0.08))
            )
        )
    )
    
    pred_calibrated = np.minimum(p_lin * conf_gate, mass_bound)
    
    # 4. Strict Zero-Squash on Flat Dry Pavement
    is_flat_dry = (sink_d < 0.02) & (dep_d < 0.02) & (p_prob < 0.50) & (~is_choked_surcharge)
    pred_calibrated = np.where(is_flat_dry, 0.0, pred_calibrated)
    pred_calibrated = np.where(pred_calibrated < 0.02, 0.0, pred_calibrated)
    
    # 5. Hydrostatic WSE Inundation Envelope
    ei = hsr_50.edge_index
    src = ei[0].cpu().numpy()
    dst = ei[1].cpu().numpy()
    num_nodes = len(pred_calibrated)
    
    elevs = - rel_drop * 20.0
    wse = elevs + pred_calibrated
    
    backwater_dst = np.maximum(0.0, wse[src] - elevs[dst])
    backwater_src = np.maximum(0.0, wse[dst] - elevs[src])
    
    max_backwater = np.zeros(num_nodes, dtype=np.float32)
    np.maximum.at(max_backwater, dst, backwater_dst)
    np.maximum.at(max_backwater, src, backwater_src)
    
    is_protected_sink = is_deep_sink | is_valley_depression | is_choked_surcharge
    pred_calibrated[~is_protected_sink] = np.minimum(
        pred_calibrated[~is_protected_sink],
        np.maximum(0.0, max_backwater[~is_protected_sink])
    )
    pred_calibrated = np.where(pred_calibrated < 0.02, 0.0, pred_calibrated)
    
    # Evaluate Critical Hazard Metrics (>= 0.30m)
    swmm_crit = (y_true >= 0.30)
    gnn_crit = (pred_calibrated >= 0.30)
    
    tp_c = int(np.sum(swmm_crit & gnn_crit))
    fp_c = int(np.sum(~swmm_crit & gnn_crit))
    fn_c = int(np.sum(swmm_crit & ~gnn_crit))
    
    rec_c = tp_c / max(1, tp_c + fn_c) * 100.0
    prec_c = tp_c / max(1, tp_c + fp_c) * 100.0
    
    # Evaluate Overall Hazard Metrics (>= 0.15m)
    swmm_f = (y_true >= 0.15)
    gnn_f = (pred_calibrated >= 0.15)
    
    tp_f = int(np.sum(swmm_f & gnn_f))
    fp_f = int(np.sum(~swmm_f & gnn_f))
    fn_f = int(np.sum(swmm_f & ~gnn_f))
    
    rec_f = tp_f / max(1, tp_f + fn_f) * 100.0
    prec_f = tp_f / max(1, tp_f + fp_f) * 100.0
    f1_f = 2 * prec_f * rec_f / max(1e-6, prec_f + rec_f)
    
    mae_cm = np.mean(np.abs(pred_calibrated - y_true)) * 100.0
    pct30 = np.mean(np.abs(pred_calibrated - y_true) <= 0.30) * 100.0
    
    print("=" * 70)
    print("CALIBRATED PIPELINE RESULTS @ 50 mm/hr (HSR LAYOUT):")
    print("=" * 70)
    print(f"CRITICAL FLOODS (>= 30 cm):")
    print(f"  SWMM Real: {np.sum(swmm_crit)} | Correctly Detected (TP): {tp_c} | False Alarms (FP): {fp_c} | Missed (FN): {fn_c}")
    print(f"  Critical Recall: {rec_c:.1f}% | Critical Precision: {prec_c:.1f}%")
    print(f"\nALL HAZARDS (>= 15 cm):")
    print(f"  SWMM Real: {np.sum(swmm_f)} | Correctly Detected (TP): {tp_f} | False Alarms (FP): {fp_f} | Missed (FN): {fn_f}")
    print(f"  Hazard Recall: {rec_f:.1f}% | Hazard Precision: {prec_f:.1f}% | Hazard F1: {f1_f:.1f}%")
    print(f"\nDEPTH ACCURACY:")
    print(f"  MAE: {mae_cm:.2f} cm | Percentage within ±30 cm: {pct30:.1f}%")

if __name__ == '__main__':
    test_calibrated_pipeline()
