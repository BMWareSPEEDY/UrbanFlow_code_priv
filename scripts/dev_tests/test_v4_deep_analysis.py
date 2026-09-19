"""Deep analysis and optimization of HydroGINE-v4.0.
Evaluates continuous sigmoid transition gating vs hard thresholding across all 5 Bengaluru districts and international cities.
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

def analyze():
    print("=" * 105)
    print("        DEEP EVALUATION: HydroGINE-v4.0 CONTINUOUS HYDRAULIC GATING")
    print("=" * 105)
    
    dl_phys = torch.load("multi_scenario_physics_pyg_dataset.pt", weights_only=False)
    blr_phys = [g for g in dl_phys if g.city == 'bangalore']
    hk_phys = [g for g in dl_phys if g.city == 'hongkong']
    
    # Also load international holdouts
    cities_test = ['london', 'newyork', 'paris', 'tokyo', 'singapore']
    intl_phys = {c: [g for g in dl_phys if g.city == c] for c in cities_test}
    
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
    
    def predict_neural_gate(g, gate_sharpness=8.0, tau_sink=0.18, tau_conv=0.45, tau_gen=0.30):
        with torch.no_grad():
            gx = (g.x.to(device) - x_mean) / x_std
            gea = (g.edge_attr.to(device) - e_mean) / e_std
            c_l, d_o = model(gx, g.edge_index.to(device), gea)
            p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0, max=3.0).cpu().numpy().ravel()
            p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
            
        x_raw = g.x.cpu().numpy()
        out_d = x_raw[:, 4]
        in_d = x_raw[:, 3]
        sink_d = x_raw[:, 23]
        dep_d = x_raw[:, 16]
        total_r = x_raw[:, 27]
        conv_def = x_raw[:, 30]
        
        is_true_sink = (out_d == 0) | (sink_d >= 0.10)
        is_esc = (out_d >= in_d) & (sink_d < 0.03) & (dep_d < 0.20) & (~is_true_sink)
        
        # Physical dynamic threshold
        tau = np.where(is_true_sink, tau_sink, np.where(is_esc, tau_conv, tau_gen))
        
        # Smooth continuous sigmoid confidence transition:
        # Avoids abrupt discontinuous hard zeroing, allowing smooth gradient from 0 to 1
        conf_gate = 1.0 / (1.0 + np.exp(-gate_sharpness * (p_prob - tau)))
        
        # Hydrodynamic physical ceiling:
        rain_scale = total_r / 50.0
        conv_bound = np.where(
            is_true_sink,
            3.0,
            np.where(is_esc, np.clip(0.15 * rain_scale * (1.0 + 0.3 * conv_def), 0.05, 0.30), 3.0)
        )
        
        pred = np.minimum(p_lin * conf_gate, conv_bound)
        return pred

    # Grid search over tau_sink, tau_conv, tau_gen and gate_sharpness
    gate_params = [
        (8.0, 0.18, 0.45, 0.30, "Smooth Gating (Default)"),
        (10.0, 0.15, 0.50, 0.32, "Smooth Gating (Sharp Conv)"),
        (12.0, 0.12, 0.55, 0.35, "Smooth Gating (High Precision)"),
        (15.0, 0.10, 0.58, 0.38, "Smooth Gating (Ultra Suppression)"),
    ]
    
    print("\n--- PARAMETER SEARCH: BENGALURU OVERALL & HSR METRICS ---")
    print(f"{'Config':<32s} | {'HSR FP Light':<14s} | {'HSR FP Cloud':<14s} | {'BLR Rec':<10s} | {'BLR Prec':<10s} | {'BLR F1':<10s} | {'BLR MAE':<10s} | {'%<=30cm'}")
    print("-" * 115)
    
    best_cfg = None
    best_f1 = 0
    
    for (k, ts, tc, tg, desc) in gate_params:
        # HSR Light (20, 50) and Cloud (150, 200, 300)
        hsr_fp_light, hsr_fp_cloud = 0, 0
        all_bp, all_bt = [], []
        
        for g in blr_phys:
            I = g.x[0, 13].item()
            yt = g.y.cpu().numpy().ravel()
            yp = predict_neural_gate(g, gate_sharpness=k, tau_sink=ts, tau_conv=tc, tau_gen=tg)
            all_bp.append(yp)
            all_bt.append(yt)
            
            if g.region == 'hsr':
                m = compute_metrics(yt, yp)
                if I in [20.0, 50.0]:
                    hsr_fp_light += m['fp_c']
                elif I in [150.0, 200.0, 300.0]:
                    hsr_fp_cloud += m['fp_c']
                    
        y_bp = np.concatenate(all_bp)
        y_bt = np.concatenate(all_bt)
        mb = compute_metrics(y_bt, y_bp)
        
        print(f"{desc:<32s} | {hsr_fp_light/2.0:4.1f} (tot {hsr_fp_light:2d})   | {hsr_fp_cloud/3.0:4.1f} (tot {hsr_fp_cloud:2d})   | {mb['rec_c']*100:6.1f}%   | {mb['prec_c']*100:6.1f}%   | {mb['f1_h']:7.4f}  | {mb['mae']*100:5.2f}cm   | {mb['pct_30']:5.1f}%")

    print("\n" + "=" * 105)
    print("--- DETAILED BREAKDOWN PER BENGALURU DISTRICT (Selected Optimal Config) ---")
    print("=" * 105)
    
    opt_k, opt_ts, opt_tc, opt_tg = 12.0, 0.12, 0.55, 0.35
    for reg in ['hsr', 'bellandur', 'ecity', 'koramangala', 'whitefield']:
        r_graphs = [g for g in blr_phys if g.region == reg]
        r_preds, r_trues = [], []
        for g in r_graphs:
            yt = g.y.cpu().numpy().ravel()
            yp = predict_neural_gate(g, gate_sharpness=opt_k, tau_sink=opt_ts, tau_conv=opt_tc, tau_gen=opt_tg)
            r_preds.append(yp)
            r_trues.append(yt)
        y_rp = np.concatenate(r_preds)
        y_rt = np.concatenate(r_trues)
        mr = compute_metrics(y_rt, y_rp)
        swmm_crit = int(np.sum(y_rt > THR_CRITICAL))
        print(f"{reg:<14s}: SWMM Crit={swmm_crit:4d} | TP={mr['tp_c']:4d} | FP={mr['fp_c']:4d} | FN={mr['fn_c']:4d} | Rec={mr['rec_c']*100:5.1f}% | Prec={mr['prec_c']*100:5.1f}% | F1={mr['f1_h']:.4f} | MAE={mr['mae']*100:5.2f}cm | %<=30cm={mr['pct_30']:5.1f}%")

    print("\n" + "=" * 105)
    print("--- CROSS-CITY ZERO-SHOT GENERALIZATION ON UNSEEN INTERNATIONAL CITIES ---")
    print("=" * 105)
    for c in ['hongkong', 'london', 'newyork', 'paris', 'tokyo', 'singapore']:
        c_graphs = [g for g in dl_phys if g.city == c]
        if not c_graphs:
            continue
        c_preds, c_trues = [], []
        for g in c_graphs:
            yt = g.y.cpu().numpy().ravel()
            yp = predict_neural_gate(g, gate_sharpness=opt_k, tau_sink=opt_ts, tau_conv=opt_tc, tau_gen=opt_tg)
            c_preds.append(yp)
            c_trues.append(yt)
        y_cp = np.concatenate(c_preds)
        y_ct = np.concatenate(c_trues)
        mc = compute_metrics(y_ct, y_cp)
        swmm_crit = int(np.sum(y_ct > THR_CRITICAL))
        print(f"{c.upper():<14s}: Nodes={len(y_ct):6d} | SWMM Crit={swmm_crit:4d} | TP={mc['tp_c']:4d} | FP={mc['fp_c']:4d} | FN={mc['fn_c']:4d} | Rec={mc['rec_c']*100:5.1f}% | Prec={mc['prec_c']*100:5.1f}% | F1={mc['f1_h']:.4f} | MAE={mc['mae']*100:5.2f}cm | %<=30cm={mc['pct_30']:5.1f}%")

if __name__ == '__main__':
    analyze()
