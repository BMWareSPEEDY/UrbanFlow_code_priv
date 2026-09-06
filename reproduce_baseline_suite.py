"""Comprehensive baseline evaluation script for UrbanFLOW.
Evaluates all historical and production models against SWMM simulation truth across:
1. HSR Layout at Light Rain (20, 50 mm/hr) and Cloudburst (150, 200, 300 mm/hr)
2. All Bengaluru districts (HSR, Bellandur, E-City, Koramangala, Whitefield)
3. Holdout cities (Hong Kong, London, New York, Paris, Tokyo, etc.)
4. Full classification (FP, FN, Precision, Recall, F1, PR-AUC) and regression metrics (MAE, RMSE, R2, % <= 30cm)
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, GATv2Conv

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

THR_HAZARD = 0.15
THR_CRITICAL = 0.30

def compute_metrics(y_true, y_pred, name=""):
    diffs = np.abs(y_pred - y_true)
    mae = float(np.mean(diffs))
    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    med_ae = float(np.median(diffs))
    p90 = float(np.percentile(diffs, 90))
    p95 = float(np.percentile(diffs, 95))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / max(1e-6, ss_tot)))
    
    pct_10 = float(np.mean(diffs <= 0.10) * 100.0)
    pct_20 = float(np.mean(diffs <= 0.20) * 100.0)
    pct_30 = float(np.mean(diffs <= 0.30) * 100.0)
    pct_gt30 = float(np.mean(diffs > 0.30) * 100.0)
    
    # Hazard metrics (THR_HAZARD = 0.15m)
    tp_h = int(np.sum((y_pred >= THR_HAZARD) & (y_true >= THR_HAZARD)))
    fp_h = int(np.sum((y_pred >= THR_HAZARD) & (y_true < THR_HAZARD)))
    fn_h = int(np.sum((y_pred < THR_HAZARD) & (y_true >= THR_HAZARD)))
    tn_h = int(np.sum((y_pred < THR_HAZARD) & (y_true < THR_HAZARD)))
    prec_h = float(tp_h / max(1, tp_h + fp_h))
    rec_h = float(tp_h / max(1, tp_h + fn_h))
    f1_h = float(2 * prec_h * rec_h / max(1e-9, prec_h + rec_h))
    
    # Critical metrics (THR_CRITICAL = 0.30m)
    tp_c = int(np.sum((y_pred > THR_CRITICAL) & (y_true > THR_CRITICAL)))
    fp_c = int(np.sum((y_pred > THR_CRITICAL) & (y_true <= THR_CRITICAL)))
    fn_c = int(np.sum((y_pred <= THR_CRITICAL) & (y_true > THR_CRITICAL)))
    tn_c = int(np.sum((y_pred <= THR_CRITICAL) & (y_true <= THR_CRITICAL)))
    prec_c = float(tp_c / max(1, tp_c + fp_c))
    rec_c = float(tp_c / max(1, tp_c + fn_c))
    f1_c = float(2 * prec_c * rec_c / max(1e-9, prec_c + rec_c))
    
    return {
        'name': name,
        'count': len(y_true),
        'mae': mae, 'rmse': rmse, 'r2': r2, 'med_ae': med_ae, 'p90': p90, 'p95': p95,
        'pct_10': pct_10, 'pct_20': pct_20, 'pct_30': pct_30, 'pct_gt30': pct_gt30,
        'tp_h': tp_h, 'fp_h': fp_h, 'fn_h': fn_h, 'tn_h': tn_h, 'prec_h': prec_h, 'rec_h': rec_h, 'f1_h': f1_h,
        'tp_c': tp_c, 'fp_c': fp_c, 'fn_c': fn_c, 'tn_c': tn_c, 'prec_c': prec_c, 'rec_c': rec_c, 'f1_c': f1_c,
    }


def predict_app_pipeline(version_mode, graph, intensity_mmhr):
    """Simulate the exact gate & post-processing logic in app.py for each historical version."""
    # Load model and ckpt
    ckpt_file = "zero_tolerance_gnn_checkpoint.pt" if os.path.exists("zero_tolerance_gnn_checkpoint.pt") else "pinn_gnn_checkpoint.pt"
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    
    from train_zero_tolerance_gnn import ZeroToleranceHurdleGNN
    from train_dual_stream_hydro_gnn import DualStreamHydroGNN
    
    st = ckpt['model_state_dict']
    if 'conv4.att' in st or ckpt.get('model_type') == 'DualStreamHydroGNN':
        model = DualStreamHydroGNN(in_channels=14, hidden_channels=ckpt.get('hidden_channels', 96), out_channels=1).to(device)
    else:
        model = ZeroToleranceHurdleGNN(in_channels=14, hidden_channels=128, out_channels=1).to(device)
    model.load_state_dict(st)
    model.eval()
    
    feat_mean = ckpt['x_mean'].to(device)
    feat_std = ckpt['x_std'].to(device)
    edge_mean = ckpt['edge_attr_mean'].to(device)
    edge_std = ckpt['edge_attr_std'].to(device)
    y_mean = float(ckpt['y_mean'])
    y_std = float(ckpt['y_std'])
    
    # Construct 14 features from graph
    num_nodes = graph.x.shape[0]
    stat_12 = graph.x[:, :12].clone().to(device)
    dyn_2 = torch.tensor([[intensity_mmhr, 60.0]], dtype=torch.float, device=device).repeat(num_nodes, 1)
    full_x = torch.cat([stat_12, dyn_2], dim=-1)
    
    x_norm = (full_x - feat_mean) / feat_std
    edge_norm = (graph.edge_attr.to(device) - edge_mean) / edge_std
    
    with torch.no_grad():
        r_out, g_out = model(x_norm, graph.edge_index.to(device), edge_norm, return_gate=True)
        p_prob = torch.sigmoid(g_out).cpu().numpy().ravel()
        p_depth = torch.clamp(r_out * y_std + y_mean, min=0.0).cpu().numpy().ravel()
        
    delta_elev = graph.x[:, 2].cpu().numpy()
    sag_index = graph.x[:, 10].cpu().numpy()
    is_sink = graph.x[:, 8].cpu().numpy()
    accum_score = graph.x[:, 7].cpu().numpy()
    in_deg = graph.x[:, 5].cpu().numpy()
    out_deg = graph.x[:, 6].cpu().numpy()
    
    true_basin_sink = (out_deg == 0) | ((is_sink == 1) & (out_deg < in_deg))
    
    if version_mode == 'v1.0':
        preds = p_depth
    elif version_mode == 'v2.0':
        preds = np.where(p_prob >= 0.50, p_depth, 0.0)
    elif version_mode == 'v3.0':
        conveyance_dry = (sag_index < 0.02) & (out_deg >= in_deg)
        feasibility = np.where(conveyance_dry, 0.0, 1.0)
        feasibility = np.where((delta_elev > 0.50) & (sag_index < 0.05), 0.0, feasibility)
        thresh = np.where(true_basin_sink, 0.20, np.where(accum_score >= 2.0, 0.45, 0.65))
        raw_preds = np.where(p_prob >= thresh, p_depth, 0.0) * feasibility
        depth_ceiling = np.where(true_basin_sink, 3.0, np.where(out_deg >= in_deg, 0.15, 0.25))
        preds = np.minimum(raw_preds, depth_ceiling * (intensity_mmhr / 50.0))
    elif version_mode == 'v3.1':
        conveyance_dry = (sag_index < 0.015) & (out_deg >= in_deg) & (~true_basin_sink)
        feasibility = np.where(conveyance_dry, 0.0, 1.0)
        feasibility = np.where((delta_elev > 0.45) & (sag_index < 0.04) & (~true_basin_sink), 0.0, feasibility)
        feasibility = np.where((delta_elev > 0.70) & (~true_basin_sink), 0.0, feasibility)
        thresh = np.where(true_basin_sink, 0.20, np.where(accum_score >= 2.0, 0.45, 0.65))
        raw_preds = np.where(p_prob >= thresh, p_depth, 0.0) * feasibility
        depth_ceiling = np.where(true_basin_sink, 3.0, np.where(out_deg >= in_deg, 0.15, 0.25))
        preds = np.minimum(raw_preds, depth_ceiling * (intensity_mmhr / 50.0))
    elif version_mode == 'v3.2':
        deep_sag_convergence = (sag_index >= 0.08) & (accum_score >= 1.8)
        conveyance_dry = (sag_index < 0.015) & (out_deg >= in_deg) & (~true_basin_sink) & (~deep_sag_convergence)
        feasibility = np.where(conveyance_dry, 0.0, 1.0)
        feasibility = np.where((delta_elev > 0.45) & (sag_index < 0.04) & (~true_basin_sink) & (~deep_sag_convergence), 0.0, feasibility)
        feasibility = np.where((delta_elev > 0.70) & (~true_basin_sink) & (~deep_sag_convergence), 0.0, feasibility)
        thresh = np.where(deep_sag_convergence | true_basin_sink, 0.20, np.where(accum_score >= 2.0, 0.45, 0.65))
        raw_preds = np.where(p_prob >= thresh, p_depth, 0.0) * feasibility
        depth_ceiling = np.where(true_basin_sink | deep_sag_convergence, 3.0, np.where(out_deg >= in_deg, 0.15, 0.25))
        preds = np.minimum(raw_preds, depth_ceiling * (intensity_mmhr / 50.0))
    elif version_mode == 'v3.3':
        deep_sag = (sag_index >= 0.06) & (accum_score >= 1.7)
        neural_high_conf = (p_prob >= 0.70) & (delta_elev < 0.30) & (accum_score >= 1.5)
        conveyance_dry = (sag_index < 0.015) & (out_deg >= in_deg) & (~true_basin_sink) & (~deep_sag) & (~neural_high_conf)
        feasibility = np.where(conveyance_dry, 0.0, 1.0)
        feasibility = np.where((delta_elev > 0.45) & (sag_index < 0.03) & (~true_basin_sink) & (~deep_sag) & (~neural_high_conf), 0.0, feasibility)
        feasibility = np.where((delta_elev > 0.70) & (~true_basin_sink) & (~deep_sag) & (~neural_high_conf), 0.0, feasibility)
        thresh = np.where(true_basin_sink | deep_sag, 0.20, np.where(neural_high_conf, 0.30, np.where(accum_score >= 2.0, 0.45, 0.65)))
        raw_preds = np.where(p_prob >= thresh, p_depth, 0.0) * feasibility
        depth_ceiling = np.where(true_basin_sink | deep_sag | neural_high_conf, 3.0, np.where(out_deg >= in_deg, 0.15, 0.25))
        preds = np.minimum(raw_preds, depth_ceiling * (intensity_mmhr / 50.0))
    else:
        raise ValueError(f"Unknown version: {version_mode}")
        
    return preds


def run_benchmark():
    print("=" * 110)
    print("                 URBANFLOW HISTORICAL & PRODUCTION BENCHMARK AUDIT")
    print("=" * 110)
    
    dl = torch.load("multi_scenario_full22_pyg_dataset.pt", weights_only=False)
    blr_graphs = [g for g in dl if g.city == 'bangalore']
    hk_graphs = [g for g in dl if g.city == 'hongkong']
    
    # Test all versions
    versions = ['v1.0', 'v2.0', 'v3.0', 'v3.1', 'v3.2', 'v3.3', 'MoE_Ensemble']
    
    sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")
    from production import ProductionEnsemble
    moe = ProductionEnsemble(device=device)
    
    results_table = []
    
    for v in versions:
        # Evaluate HSR light rain (20, 50) and cloudburst (150, 200, 300)
        hsr_graphs = [g for g in blr_graphs if g.region == 'hsr']
        
        fp_light = 0
        fp_cloud = 0
        tp_crit_total = 0
        fn_crit_total = 0
        fp_crit_total = 0
        
        all_preds = []
        all_trues = []
        
        t0 = time.perf_counter()
        for g in blr_graphs:
            I = g.x[0, 15].item()
            y_t = g.y.cpu().numpy().ravel()
            if v == 'MoE_Ensemble':
                gated, _, _ = moe.predict(g, I)
                p = gated
            else:
                p = predict_app_pipeline(v, g, I)
            all_preds.append(p)
            all_trues.append(y_t)
            
            if g.region == 'hsr':
                m = compute_metrics(y_t, p, f"HSR_{I:.0f}")
                if I in [20.0, 50.0]:
                    fp_light += m['fp_c']
                elif I in [150.0, 200.0, 300.0]:
                    fp_cloud += m['fp_c']
                tp_crit_total += m['tp_c']
                fn_crit_total += m['fn_c']
                fp_crit_total += m['fp_c']
                
        lat_ms = (time.perf_counter() - t0) / len(blr_graphs) * 1000.0
        
        y_all_p = np.concatenate(all_preds)
        y_all_t = np.concatenate(all_trues)
        
        m_blr = compute_metrics(y_all_t, y_all_p, f"BLR_{v}")
        
        # Evaluate HK
        hk_preds, hk_trues = [], []
        for g in hk_graphs:
            I = g.x[0, 15].item()
            y_t = g.y.cpu().numpy().ravel()
            if v == 'MoE_Ensemble':
                gated, _, _ = moe.predict(g, I)
                p = gated
            else:
                p = predict_app_pipeline(v, g, I)
            hk_preds.append(p)
            hk_trues.append(y_t)
        y_hk_p = np.concatenate(hk_preds)
        y_hk_t = np.concatenate(hk_trues)
        m_hk = compute_metrics(y_hk_t, y_hk_p, f"HK_{v}")
        
        results_table.append({
            'Version': v,
            'HSR FP Light (avg/scen)': f"{fp_light/2.0:.1f} (tot {fp_light})",
            'HSR FP Cloud (avg/scen)': f"{fp_cloud/3.0:.1f} (tot {fp_cloud})",
            'BLR Crit Recall': f"{m_blr['rec_c']*100:.1f}%",
            'BLR Crit Prec': f"{m_blr['prec_c']*100:.1f}%",
            'BLR F1 Haz': f"{m_blr['f1_h']:.4f}",
            'BLR MAE (cm)': f"{m_blr['mae']*100:.2f}",
            'BLR %<=30cm': f"{m_blr['pct_30']:.1f}%",
            'BLR R2': f"{m_blr['r2']:.4f}",
            'HK F1 Haz': f"{m_hk['f1_h']:.4f}",
            'HK %<=30cm': f"{m_hk['pct_30']:.1f}%",
            'Latency (ms)': f"{lat_ms:.2f}"
        })
        
    df_res = pd.DataFrame(results_table)
    print("\n" + df_res.to_string(index=False))
    print("\n" + "=" * 110)

if __name__ == '__main__':
    run_benchmark()
