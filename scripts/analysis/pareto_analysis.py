"""Pareto trade-off analysis: False Criticals vs Critical Recall across HSR and Bangalore.
Quantifies the exact mathematical frontier between precision and recall under zero-leakage physics.
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

def pareto_curve():
    print("=" * 105)
    print("             PARETO FRONTIER: FALSE CRITICALS vs CRITICAL RECALL")
    print("=" * 105)
    
    dl_phys = torch.load("multi_scenario_physics_pyg_dataset.pt", weights_only=False)
    blr_phys = [g for g in dl_phys if g.city == 'bangalore']
    
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
    
    # Precompute raw predictions and true labels for all Bangalore graphs
    blr_data = []
    for g in blr_phys:
        with torch.no_grad():
            gx = (g.x.to(device) - x_mean) / x_std
            gea = (g.edge_attr.to(device) - e_mean) / e_std
            c_l, d_o = model(gx, g.edge_index.to(device), gea)
            p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0, max=3.0).cpu().numpy().ravel()
            p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
            
        x_raw = g.x.cpu().numpy()
        y_true = g.y.cpu().numpy().ravel()
        blr_data.append({
            'region': g.region,
            'intensity': g.x[0, 13].item(),
            'p_lin': p_lin,
            'p_prob': p_prob,
            'y_true': y_true,
            'x_raw': x_raw
        })
        
    print(f"{'Operating Point / Tau':<25s} | {'HSR FP Light':<14s} | {'HSR FP Cloud':<14s} | {'BLR Rec':<10s} | {'BLR Prec':<10s} | {'BLR F1':<10s} | {'BLR MAE':<10s} | {'%<=30cm'}")
    print("-" * 115)
    
    # Sweep over baseline tau thresholds from 0.05 to 0.80
    for tau_base in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]:
        hsr_fp_light, hsr_fp_cloud = 0, 0
        all_p, all_t = [], []
        
        for item in blr_data:
            p_lin = item['p_lin']
            p_prob = item['p_prob']
            y_t = item['y_true']
            x_raw = item['x_raw']
            I = item['intensity']
            reg = item['region']
            
            in_d = x_raw[:, 3]
            out_d = x_raw[:, 4]
            dep_d = x_raw[:, 16]
            sink_d = x_raw[:, 23]
            total_r = x_raw[:, 27]
            conv_def = x_raw[:, 30]
            
            is_deep_bowl = (sink_d >= 0.28) & ((out_d <= 1) | (sink_d >= 0.50))
            is_sink = (out_d == 0) | (sink_d >= 0.12) | (x_raw[:, 29] >= 1.2)
            is_deep_sag = (x_raw[:, 8] >= 0.05) & (x_raw[:, 5] >= 1.6) & (dep_d >= 0.40)
            is_escape = (out_d >= in_d) & (sink_d < 0.03) & (dep_d < 0.20) & (~is_sink) & (~is_deep_sag)
            
            tau = np.where(is_deep_bowl, max(0.05, tau_base - 0.20),
                  np.where(is_sink | is_deep_sag, max(0.08, tau_base - 0.12),
                  np.where(is_escape, min(0.85, tau_base + 0.20), tau_base)))
                  
            conf = 1.0 / (1.0 + np.exp(-12.0 * (p_prob - tau)))
            raw_gated = p_lin * conf
            
            rain_scale = total_r / 50.0
            light_cap = np.where(
                is_deep_bowl,
                np.clip(0.30 + 0.30 * (total_r / 20.0) * sink_d, 0.30, 0.65),
                np.clip(0.12 + 0.12 * (total_r / 20.0) * (1.0 + 0.2 * x_raw[:, 11]), 0.04, 0.24)
            )
            mod_cap = np.where(is_deep_bowl | is_deep_sag, 1.50, np.where(is_escape, 0.25, 0.65))
            heavy_cap = np.where(is_sink | is_deep_sag, 3.0, np.where(is_escape, np.clip(0.15 * rain_scale * (1.0 + 0.2 * conv_def), 0.05, 0.28), 3.0))
            hydro_ceiling = np.where(total_r <= 25.0, light_cap, np.where(total_r <= 60.0, mod_cap, heavy_cap))
            
            pred = np.minimum(raw_gated, hydro_ceiling)
            
            all_p.append(pred)
            all_t.append(y_t)
            
            if reg == 'hsr':
                m = compute_metrics(y_t, pred)
                if I in [20.0, 50.0]:
                    hsr_fp_light += m['fp_c']
                elif I in [150.0, 200.0, 300.0]:
                    hsr_fp_cloud += m['fp_c']
                    
        y_all_p = np.concatenate(all_p)
        y_all_t = np.concatenate(all_t)
        mb = compute_metrics(y_all_t, y_all_p)
        
        name_str = f"Threshold tau_base = {tau_base:.2f}"
        print(f"{name_str:<25s} | {hsr_fp_light/2.0:4.1f} (tot {hsr_fp_light:2d})   | {hsr_fp_cloud/3.0:4.1f} (tot {hsr_fp_cloud:2d})   | {mb['rec_c']*100:6.1f}%   | {mb['prec_c']*100:6.1f}%   | {mb['f1_h']:7.4f}  | {mb['mae']*100:5.2f}cm   | {mb['pct_30']:5.1f}%")

if __name__ == '__main__':
    pareto_curve()
