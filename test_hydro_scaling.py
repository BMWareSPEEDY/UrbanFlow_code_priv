"""Test Hydrodynamic Mass Scaling across all intensities in HSR Layout and Bangalore.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

THR_HAZARD = 0.15
THR_CRITICAL = 0.30

from train_hydro_gine_v4 import HydroGINE_v4
from reproduce_baseline_suite import compute_metrics

def test_scaling():
    print("=" * 110)
    print("      TESTING HYDRODYNAMIC MASS SCALING (Zero-Leakage GNN)")
    print("=" * 110)
    
    dl_phys = torch.load("multi_scenario_physics_pyg_dataset.pt", weights_only=False)
    blr_phys = [g for g in dl_phys if g.city == 'bangalore']
    hk_phys = [g for g in dl_phys if g.city == 'hongkong']
    
    ck = torch.load("hydro_gine_v4_model.pt", map_location=device, weights_only=False)
    model = HydroGINE_v4(in_c=ck['in_c'], edge_c=2, hidden=ck['hidden'], n_layers=ck['n_layers']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    
    x_mean = ck['x_mean'].to(device)
    x_std = ck['x_std'].to(device)
    e_mean = ck['e_mean'].to(device)
    e_std = ck['e_std'].to(device)
    yl_mean = ck['yl_mean'].to(device)
    yl_std = ck['yl_std'].to(device)
    
    def predict_scaled(g):
        with torch.no_grad():
            gx = (g.x.to(device) - x_mean) / x_std
            gea = (g.edge_attr.to(device) - e_mean) / e_std
            c_l, d_o = model(gx, g.edge_index.to(device), gea)
            p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0, max=3.0).cpu().numpy().ravel()
            p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
            
        x_raw = g.x.cpu().numpy()
        in_d = x_raw[:, 3]
        out_d = x_raw[:, 4]
        log_imp = x_raw[:, 11]
        intensity = x_raw[:, 13]
        duration = x_raw[:, 14]
        dep_d = x_raw[:, 16]
        path_cap = x_raw[:, 18]
        sink_d = x_raw[:, 23]
        total_r = x_raw[:, 27]
        conv_def = x_raw[:, 30]
        
        is_true_sink = (out_d == 0) | (sink_d >= 0.15) | (x_raw[:, 29] >= 1.2)
        is_deep_sag = (x_raw[:, 8] >= 0.05) & (x_raw[:, 5] >= 1.6) & (dep_d >= 0.40)
        is_escape = (out_d >= in_d) & (sink_d < 0.03) & (dep_d < 0.20) & (~is_true_sink) & (~is_deep_sag)
        
        # Adaptive confidence threshold
        tau = np.where(is_true_sink | is_deep_sag, 0.12, np.where(is_escape, 0.52, 0.32))
        
        # Smooth neural confidence modulation
        conf = 1.0 / (1.0 + np.exp(-12.0 * (p_prob - tau)))
        raw_gated = p_lin * conf
        
        # Hydrodynamic Physical Mass Ceiling:
        # Depth is strictly bounded by the maximum volume of water delivered to the catchment
        rain_ratio = total_r / 50.0 # 0.4 at 20mm/hr, 1.0 at 50mm/hr, 3.0 at 150mm/hr (60min), 3.0 at 300mm/hr (30min)
        
        hydro_ceiling = np.where(
            is_true_sink | is_deep_sag,
            3.0,
            np.where(
                is_escape,
                np.clip(0.12 * rain_ratio * (1.0 + 0.2 * conv_def) + 0.10 * dep_d, 0.02, 0.26),
                np.clip(0.24 * rain_ratio * (1.0 + 0.3 * conv_def) + 0.25 * dep_d, 0.05, 0.85)
            )
        )
        
        pred_final = np.minimum(raw_gated, hydro_ceiling)
        return pred_final

    print("\n--- HSR LAYOUT SCENARIO BY SCENARIO ---")
    print(f"{'Rain (mm/hr)':<12s} | {'SWMM Crit':<10s} | {'Pred Crit':<10s} | {'TP Crit':<8s} | {'FP Crit':<8s} | {'FN Crit':<8s} | {'Crit Rec':<10s} | {'Crit Prec':<10s} | {'MAE (cm)':<10s} | {'%<=30cm'}")
    print("-" * 115)
    
    total_fp_light = 0
    total_fp_cloud = 0
    
    for I in [20.0, 50.0, 80.0, 120.0, 150.0, 200.0, 250.0, 300.0]:
        g = [gg for gg in blr_phys if gg.region == 'hsr' and abs(gg.x[0, 13].item() - I) < 1e-3][0]
        y_true = g.y.cpu().numpy().ravel()
        p = predict_scaled(g)
        m = compute_metrics(y_true, p)
        swmm_crit = int(np.sum(y_true > THR_CRITICAL))
        pred_crit = int(np.sum(p > THR_CRITICAL))
        
        if I in [20.0, 50.0]:
            total_fp_light += m['fp_c']
        elif I in [150.0, 200.0, 300.0]:
            total_fp_cloud += m['fp_c']
            
        print(f"{I:<12.0f} | {swmm_crit:<10d} | {pred_crit:<10d} | {m['tp_c']:<8d} | {m['fp_c']:<8d} | {m['fn_c']:<8d} | {m['rec_c']*100:<9.1f}% | {m['prec_c']*100:<9.1f}% | {m['mae']*100:<9.2f} | {m['pct_30']:<9.1f}%")

    print("-" * 115)
    print(f"HSR FP Light Rain (20+50 mm/hr): Avg = {total_fp_light/2.0:.1f} (Total = {total_fp_light})")
    print(f"HSR FP Cloudburst (150+200+300 mm/hr): Avg = {total_fp_cloud/3.0:.1f} (Total = {total_fp_cloud})")
    
    print("\n" + "=" * 110)
    print("--- BENGALURU OVERALL & HONG KONG ---")
    print("=" * 110)
    
    blr_preds, blr_trues = [], []
    for g in blr_phys:
        blr_preds.append(predict_scaled(g))
        blr_trues.append(g.y.cpu().numpy().ravel())
    y_bp = np.concatenate(blr_preds)
    y_bt = np.concatenate(blr_trues)
    mb = compute_metrics(y_bt, y_bp)
    
    hk_preds, hk_trues = [], []
    for g in hk_phys:
        hk_preds.append(predict_scaled(g))
        hk_trues.append(g.y.cpu().numpy().ravel())
    y_hp = np.concatenate(hk_preds)
    y_ht = np.concatenate(hk_trues)
    mh = compute_metrics(y_ht, y_hp)
    
    print(f"BENGALURU POOLED (40 graphs, 55,160 node-evaluations):")
    print(f"  SWMM Critical: {np.sum(y_bt > 0.30)} | Pred Critical: {np.sum(y_bp > 0.30)} | TP: {mb['tp_c']} | FP: {mb['fp_c']} | FN: {mb['fn_c']}")
    print(f"  Critical Recall: {mb['rec_c']*100:.1f}% | Critical Precision: {mb['prec_c']*100:.1f}% | Hazard F1: {mb['f1_h']:.4f}")
    print(f"  MAE: {mb['mae']*100:.2f} cm | RMSE: {mb['rmse']*100:.2f} cm | R2: {mb['r2']:.4f} | % <= 30cm: {mb['pct_30']:.1f}%")
    
    print(f"\nHONG KONG ZERO-SHOT (8 graphs, 30,784 node-evaluations):")
    print(f"  SWMM Critical: {np.sum(y_ht > 0.30)} | Pred Critical: {np.sum(y_hp > 0.30)} | TP: {mh['tp_c']} | FP: {mh['fp_c']} | FN: {mh['fn_c']}")
    print(f"  Critical Recall: {mh['rec_c']*100:.1f}% | Critical Precision: {mh['prec_c']*100:.1f}% | Hazard F1: {mh['f1_h']:.4f}")
    print(f"  MAE: {mh['mae']*100:.2f} cm | RMSE: {mh['rmse']*100:.2f} cm | R2: {mh['r2']:.4f} | % <= 30cm: {mh['pct_30']:.1f}%")

if __name__ == '__main__':
    test_scaling()
