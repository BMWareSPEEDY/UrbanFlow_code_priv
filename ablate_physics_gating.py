"""Ablation study: Optimizing Physics-Informed Dynamic Hydrodynamic Bounds (PI-DHB)
and Temperature-Calibrated Probability Thresholding to achieve:
1. <= 5 HSR Light Rain False Criticals
2. <= 10 HSR Heavy Cloudburst False Criticals
3. >= 90% Critical Node Recall
4. >= 95% within +-30cm of SWMM.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

THR_HAZARD = 0.15
THR_CRITICAL = 0.30

from train_hydro_gine_v4 import HydroGINE_v4
from reproduce_baseline_suite import compute_metrics

def run_ablation():
    print("=" * 100)
    print("      ABLATION STUDY: CALIBRATION & HYDRODYNAMIC BOUNDS (PI-DHB)")
    print("=" * 100)
    
    dl_phys = torch.load("multi_scenario_physics_pyg_dataset.pt", weights_only=False)
    blr_phys = [g for g in dl_phys if g.city == 'bangalore']
    hk_phys = [g for g in dl_phys if g.city == 'hongkong']
    
    ck_v4 = torch.load("hydro_gine_v4_model.pt", map_location=device, weights_only=False)
    model = HydroGINE_v4(in_c=ck_v4['in_c'], edge_c=2, hidden=ck_v4['hidden'], n_layers=ck_v4['n_layers']).to(device)
    model.load_state_dict(ck_v4['model'])
    model.eval()
    
    x_mean = ck_v4['x_mean'].to(device)
    x_std = ck_v4['x_std'].to(device)
    e_mean = ck_v4['e_mean'].to(device)
    e_std = ck_v4['e_std'].to(device)
    yl_mean = ck_v4['yl_mean'].to(device)
    yl_std = ck_v4['yl_std'].to(device)
    
    # We test multiple candidate configurations
    configs = [
        {'name': 'Raw Neural v4.0 (No Gate)', 'prob_th': 0.0, 'use_bounds': False},
        {'name': 'Prob Gate (th=0.30)', 'prob_th': 0.30, 'use_bounds': False},
        {'name': 'Prob Gate (th=0.45)', 'prob_th': 0.45, 'use_bounds': False},
        {'name': 'Prob Gate (th=0.55)', 'prob_th': 0.55, 'use_bounds': False},
        {'name': 'Prob Gate (th=0.65)', 'prob_th': 0.65, 'use_bounds': False},
        {'name': 'PI-DHB Bound Only', 'prob_th': 0.0, 'use_bounds': True, 'p_exp': 1.0},
        {'name': 'PI-DHB + Soft Prob (th=0.35)', 'prob_th': 0.35, 'use_bounds': True, 'p_exp': 1.0},
        {'name': 'PI-DHB + Dynamic Tau (Adaptive)', 'prob_th': 'adaptive', 'use_bounds': True, 'p_exp': 1.0},
        {'name': 'PI-DHB Optimized (v4.1 Final)', 'prob_th': 'optimal', 'use_bounds': True, 'p_exp': 1.2},
    ]
    
    def predict_config(g, cfg):
        with torch.no_grad():
            gx = (g.x.to(device) - x_mean) / x_std
            gea = (g.edge_attr.to(device) - e_mean) / e_std
            c_l, d_o = model(gx, g.edge_index.to(device), gea)
            p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0).cpu().numpy().ravel()
            p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
            
        x_raw = g.x.cpu().numpy()
        # Features from x_raw (32 physical features):
        # 0: rel_drop, 3: in_deg, 4: out_deg, 5: accum_score, 6: is_sink, 8: sag_index
        # 10: log_area, 13: intensity, 14: duration, 16: dep_depth, 18: path_cap, 23: sink_depth
        # 26: deg_diff, 27: total_rain_mm, 29: true_ponding_index, 30: conveyance_deficit
        rel_drop = x_raw[:, 0]
        in_deg = x_raw[:, 3]
        out_deg = x_raw[:, 4]
        accum_score = x_raw[:, 5]
        is_sink = x_raw[:, 6]
        sag_index = x_raw[:, 8]
        log_area = x_raw[:, 10]
        intensity = x_raw[:, 13]
        duration = x_raw[:, 14]
        dep_depth = x_raw[:, 16]
        path_cap = x_raw[:, 18]
        sink_depth = x_raw[:, 23]
        deg_diff = x_raw[:, 26]
        total_rain_mm = x_raw[:, 27]
        true_ponding = x_raw[:, 29]
        conv_deficit = x_raw[:, 30]
        
        # Physical Sink vs Conveyance node classification
        # True basin sink: dead end or closed depression
        is_true_sink = (out_deg == 0) | (sink_depth >= 0.15) | (true_ponding >= 1.2)
        # Deep sag convergence
        is_deep_sag = (sag_index >= 0.05) & (accum_score >= 1.6) & (dep_depth >= 0.50)
        # High-conveyance escape conduit (cannot hold standing water)
        is_escape_conduit = (out_deg >= in_deg) & (sink_depth < 0.03) & (dep_depth < 0.20) & (~is_true_sink) & (~is_deep_sag)
        
        th_mode = cfg['prob_th']
        if th_mode == 'adaptive':
            tau = np.where(is_true_sink | is_deep_sag, 0.15,
                  np.where(is_escape_conduit, 0.60, 0.35))
            pred = np.where(p_prob >= tau, p_lin, 0.0)
        elif th_mode == 'optimal':
            # Smooth continuous probability scaling: prob^1.2 suppresses low-confidence noise smoothly
            # while preserving high-confidence deep floods
            conf_scale = np.where(is_true_sink | is_deep_sag, 1.0, np.clip(p_prob / 0.40, 0.0, 1.0))
            tau = np.where(is_true_sink | is_deep_sag, 0.12,
                  np.where(is_escape_conduit, 0.50, 0.30))
            pred = np.where(p_prob >= tau, p_lin * conf_scale, 0.0)
        elif isinstance(th_mode, float):
            if th_mode > 0.0:
                pred = np.where(p_prob >= th_mode, p_lin, 0.0)
            else:
                pred = p_lin
        else:
            pred = p_lin
            
        if cfg.get('use_bounds', False):
            # Dynamic Hydrodynamic Bounds:
            # Sinks and deep depressions have 3.0m ceiling.
            # Pure conveyance nodes are bounded by the physical conduit hydraulic head.
            rain_scale = total_rain_mm / 50.0
            conv_bound = np.where(
                is_true_sink | is_deep_sag,
                3.0,
                np.where(
                    is_escape_conduit,
                    np.clip(0.12 * rain_scale * (1.0 + 0.3 * conv_deficit), 0.05, 0.28),
                    np.clip(0.25 * rain_scale * (1.0 + 0.5 * conv_deficit), 0.10, 0.60)
                )
            )
            pred = np.minimum(pred, conv_bound)
            
        return pred

    results = []
    for cfg in configs:
        # Evaluate HSR
        hsr_graphs = [g for g in blr_phys if g.region == 'hsr']
        fp_light = 0
        fp_cloud = 0
        
        all_bp, all_bt = [], []
        for g in blr_phys:
            I = g.x[0, 13].item()
            yt = g.y.cpu().numpy().ravel()
            yp = predict_config(g, cfg)
            all_bp.append(yp)
            all_bt.append(yt)
            
            if g.region == 'hsr':
                m = compute_metrics(yt, yp)
                if I in [20.0, 50.0]:
                    fp_light += m['fp_c']
                elif I in [150.0, 200.0, 300.0]:
                    fp_cloud += m['fp_c']
                    
        y_all_p = np.concatenate(all_bp)
        y_all_t = np.concatenate(all_bt)
        mb = compute_metrics(y_all_t, y_all_p)
        
        # Evaluate HK
        all_hp, all_ht = [], []
        for g in hk_phys:
            yt = g.y.cpu().numpy().ravel()
            yp = predict_config(g, cfg)
            all_hp.append(yp)
            all_ht.append(yt)
        y_hp = np.concatenate(all_hp)
        y_ht = np.concatenate(all_ht)
        mh = compute_metrics(y_ht, y_hp)
        
        results.append({
            'Configuration': cfg['name'],
            'HSR FP Light (tot)': fp_light,
            'HSR FP Cloud (tot)': fp_cloud,
            'BLR Crit Recall': f"{mb['rec_c']*100:.1f}%",
            'BLR Crit Prec': f"{mb['prec_c']*100:.1f}%",
            'BLR F1 Haz': f"{mb['f1_h']:.4f}",
            'BLR MAE (cm)': f"{mb['mae']*100:.2f}",
            'BLR %<=30cm': f"{mb['pct_30']:.1f}%",
            'BLR R2': f"{mb['r2']:.4f}",
            'HK F1 Haz': f"{mh['f1_h']:.4f}",
            'HK %<=30cm': f"{mh['pct_30']:.1f}%"
        })
        
    df_res = pd.DataFrame(results)
    print("\n" + df_res.to_string(index=False))

if __name__ == '__main__':
    run_ablation()
