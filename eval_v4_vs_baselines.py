"""Compare HydroGINE-v4.0 against v3.3 production and MoE baseline on the full benchmark suite.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

THR_HAZARD = 0.15
THR_CRITICAL = 0.30

# Import HydroGINE_v4 from train_hydro_gine_v4
from train_hydro_gine_v4 import HydroGINE_v4
from reproduce_baseline_suite import compute_metrics, predict_app_pipeline

def eval_full():
    print("=" * 110)
    print("           EVALUATION: HydroGINE-v4.0 vs v3.3 PRODUCTION vs MoE BASELINE")
    print("=" * 110)
    
    # Load dataset
    dl_phys = torch.load("multi_scenario_physics_pyg_dataset.pt", weights_only=False)
    dl_raw = torch.load("multi_scenario_full22_pyg_dataset.pt", weights_only=False)
    
    blr_phys = [g for g in dl_phys if g.city == 'bangalore']
    hk_phys = [g for g in dl_phys if g.city == 'hongkong']
    
    blr_raw = [g for g in dl_raw if g.city == 'bangalore']
    hk_raw = [g for g in dl_raw if g.city == 'hongkong']
    
    # Load v4 model
    ck_v4 = torch.load("hydro_gine_v4_model.pt", map_location=device, weights_only=False)
    model_v4 = HydroGINE_v4(in_c=ck_v4['in_c'], edge_c=2, hidden=ck_v4['hidden'], n_layers=ck_v4['n_layers']).to(device)
    model_v4.load_state_dict(ck_v4['model'])
    model_v4.eval()
    
    x_mean = ck_v4['x_mean'].to(device)
    x_std = ck_v4['x_std'].to(device)
    e_mean = ck_v4['e_mean'].to(device)
    e_std = ck_v4['e_std'].to(device)
    yl_mean = ck_v4['yl_mean'].to(device)
    yl_std = ck_v4['yl_std'].to(device)
    
    # Load MoE
    sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")
    from production import ProductionEnsemble
    moe = ProductionEnsemble(device=device)
    
    # Predictions collection
    preds = {'v3.3': [], 'MoE': [], 'v4.0_raw': [], 'v4.0_gated': []}
    trues = []
    
    # HSR breakdown
    hsr_records = []
    
    intensities = [20.0, 50.0, 80.0, 120.0, 150.0, 200.0, 250.0, 300.0]
    
    print("\n--- HSR LAYOUT SCENARIO COMPARISON ---")
    print(f"{'Rain':<6s} | {'SWMM Crit':<10s} | {'v3.3 (Crit/FP/MAE)':<22s} | {'MoE (Crit/FP/MAE)':<22s} | {'v4.0 (Crit/FP/MAE)':<22s}")
    print("-" * 90)
    
    for I in intensities:
        # Find HSR graph
        g_raw = [g for g in blr_raw if g.region == 'hsr' and abs(g.x[0, 15].item() - I) < 1e-3][0]
        g_phy = [g for g in blr_phys if g.region == 'hsr' and abs(g.x[0, 13].item() - I) < 1e-3][0]
        
        y_true = g_phy.y.cpu().numpy().ravel()
        swmm_crit = int(np.sum(y_true > THR_CRITICAL))
        
        # v3.3
        p_v33 = predict_app_pipeline('v3.3', g_raw, I)
        m_v33 = compute_metrics(y_true, p_v33)
        
        # MoE
        gated_moe, _, _ = moe.predict(g_raw, I)
        m_moe = compute_metrics(y_true, gated_moe)
        
        # v4.0
        with torch.no_grad():
            gx = (g_phy.x.to(device) - x_mean) / x_std
            gea = (g_phy.edge_attr.to(device) - e_mean) / e_std
            c_l, d_o = model_v4(gx, g_phy.edge_index.to(device), gea)
            p_v4_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0).cpu().numpy().ravel()
            p_v4_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
            
            # Physics-grounded soft thresholding:
            # If hazard prob < 0.35, depth is suppressed unless local sink_depth > 0.30m
            sink_depth = g_phy.x[:, 23].cpu().numpy()
            p_v4_gated = np.where((p_v4_prob < 0.35) & (sink_depth < 0.20), 0.0, p_v4_lin)
            
        m_v4 = compute_metrics(y_true, p_v4_gated)
        
        str_v33 = f"{m_v33['tp_c']+m_v33['fp_c']:3d} / FP={m_v33['fp_c']:3d} / {m_v33['mae']*100:4.1f}cm"
        str_moe = f"{m_moe['tp_c']+m_moe['fp_c']:3d} / FP={m_moe['fp_c']:3d} / {m_moe['mae']*100:4.1f}cm"
        str_v4  = f"{m_v4['tp_c']+m_v4['fp_c']:3d} / FP={m_v4['fp_c']:3d} / {m_v4['mae']*100:4.1f}cm"
        
        print(f"{I:<6.0f} | {swmm_crit:<10d} | {str_v33:<22s} | {str_moe:<22s} | {str_v4:<22s}")
        
    print("\n" + "=" * 110)
    print("--- OVERALL BENGALURU & HONG KONG BENCHMARK SUMMARY ---")
    print("=" * 110)
    
    # Bangalore Overall
    blr_v33, blr_moe, blr_v4, blr_trues = [], [], [], []
    for g_raw, g_phy in zip(blr_raw, blr_phys):
        I = g_raw.x[0, 15].item()
        y_true = g_phy.y.cpu().numpy().ravel()
        blr_trues.append(y_true)
        blr_v33.append(predict_app_pipeline('v3.3', g_raw, I))
        gated_moe, _, _ = moe.predict(g_raw, I)
        blr_moe.append(gated_moe)
        
        with torch.no_grad():
            gx = (g_phy.x.to(device) - x_mean) / x_std
            gea = (g_phy.edge_attr.to(device) - e_mean) / e_std
            c_l, d_o = model_v4(gx, g_phy.edge_index.to(device), gea)
            p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0).cpu().numpy().ravel()
            p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
            sink_depth = g_phy.x[:, 23].cpu().numpy()
            p_gated = np.where((p_prob < 0.35) & (sink_depth < 0.20), 0.0, p_lin)
            blr_v4.append(p_gated)
            
    yt_b = np.concatenate(blr_trues)
    mb_v33 = compute_metrics(yt_b, np.concatenate(blr_v33))
    mb_moe = compute_metrics(yt_b, np.concatenate(blr_moe))
    mb_v4  = compute_metrics(yt_b, np.concatenate(blr_v4))
    
    # Hong Kong Overall
    hk_v33, hk_moe, hk_v4, hk_trues = [], [], [], []
    for g_raw, g_phy in zip(hk_raw, hk_phys):
        I = g_raw.x[0, 15].item()
        y_true = g_phy.y.cpu().numpy().ravel()
        hk_trues.append(y_true)
        hk_v33.append(predict_app_pipeline('v3.3', g_raw, I))
        gated_moe, _, _ = moe.predict(g_raw, I)
        hk_moe.append(gated_moe)
        
        with torch.no_grad():
            gx = (g_phy.x.to(device) - x_mean) / x_std
            gea = (g_phy.edge_attr.to(device) - e_mean) / e_std
            c_l, d_o = model_v4(gx, g_phy.edge_index.to(device), gea)
            p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0).cpu().numpy().ravel()
            p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
            sink_depth = g_phy.x[:, 23].cpu().numpy()
            p_gated = np.where((p_prob < 0.35) & (sink_depth < 0.20), 0.0, p_lin)
            hk_v4.append(p_gated)
            
    yt_h = np.concatenate(hk_trues)
    mh_v33 = compute_metrics(yt_h, np.concatenate(hk_v33))
    mh_moe = compute_metrics(yt_h, np.concatenate(hk_moe))
    mh_v4  = compute_metrics(yt_h, np.concatenate(hk_v4))
    
    summary = [
        {
            'Model': 'v3.3 Production',
            'BLR Crit Recall': f"{mb_v33['rec_c']*100:.1f}%",
            'BLR F1 Haz': f"{mb_v33['f1_h']:.4f}",
            'BLR MAE (cm)': f"{mb_v33['mae']*100:.2f}",
            'BLR %<=30cm': f"{mb_v33['pct_30']:.1f}%",
            'BLR R2': f"{mb_v33['r2']:.4f}",
            'HK F1 Haz': f"{mh_v33['f1_h']:.4f}",
            'HK %<=30cm': f"{mh_v33['pct_30']:.1f}%",
            'Coordinates Used': 'Yes (Leaked)'
        },
        {
            'Model': 'MoE Baseline (v6)',
            'BLR Crit Recall': f"{mb_moe['rec_c']*100:.1f}%",
            'BLR F1 Haz': f"{mb_moe['f1_h']:.4f}",
            'BLR MAE (cm)': f"{mb_moe['mae']*100:.2f}",
            'BLR %<=30cm': f"{mb_moe['pct_30']:.1f}%",
            'BLR R2': f"{mb_moe['r2']:.4f}",
            'HK F1 Haz': f"{mh_moe['f1_h']:.4f}",
            'HK %<=30cm': f"{mh_moe['pct_30']:.1f}%",
            'Coordinates Used': 'Yes (Leaked)'
        },
        {
            'Model': 'HydroGINE-v4.0 (Ours)',
            'BLR Crit Recall': f"{mb_v4['rec_c']*100:.1f}%",
            'BLR F1 Haz': f"{mb_v4['f1_h']:.4f}",
            'BLR MAE (cm)': f"{mb_v4['mae']*100:.2f}",
            'BLR %<=30cm': f"{mb_v4['pct_30']:.1f}%",
            'BLR R2': f"{mb_v4['r2']:.4f}",
            'HK F1 Haz': f"{mh_v4['f1_h']:.4f}",
            'HK %<=30cm': f"{mh_v4['pct_30']:.1f}%",
            'Coordinates Used': 'STRICTLY ZERO'
        }
    ]
    df_sum = pd.DataFrame(summary)
    print(df_sum.to_string(index=False))

if __name__ == '__main__':
    eval_full()
