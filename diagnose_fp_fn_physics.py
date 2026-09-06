"""Deep physical error analysis script for UrbanFLOW.
Investigates the hydraulic and topological mechanisms behind False Positives and False Negatives in HSR Layout and Bangalore.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")
from production import ProductionEnsemble

THR_HAZARD = 0.15
THR_CRITICAL = 0.30

def diagnose():
    print("=" * 90)
    print("               DEEP PHYSICAL ERROR ANALYSIS: FP & FN MECHANISMS")
    print("=" * 90)
    
    dl = torch.load("multi_scenario_full22_pyg_dataset.pt", weights_only=False)
    ens = ProductionEnsemble(device=device)
    
    blr_graphs = [g for g in dl if g.city == 'bangalore']
    
    # Analyze HSR at Light Rain (50 mm/hr) and Cloudburst (200 mm/hr)
    feat_names = [
        "rel_x (LEAK)", "rel_y (LEAK)", "rel_drop", "imp", "manning_n", "in_deg", "out_deg",
        "accum_score", "is_sink", "max_in_grade", "sag_index", "hyd_capacity",
        "log_area", "log_imp_area", "dist_frac", "intensity", "duration",
        "elev_std2", "dep_depth", "surcharge", "path_cap", "path_hops",
        "dist_outlet", "elev_above_outlet", "slope_outlet_ratio", "sink_depth",
        "inlet_cap", "surcharge_ratio"
    ]
    
    for I in [20.0, 50.0, 150.0, 200.0, 300.0]:
        g_hsr = [g for g in blr_graphs if g.region == 'hsr' and abs(g.x[0, 15].item() - I) < 1e-3][0]
        gated, raw, prob = ens.predict(g_hsr, I)
        y_true = g_hsr.y.cpu().numpy().ravel()
        x_raw = g_hsr.x.cpu().numpy()
        
        is_tp = (gated > THR_CRITICAL) & (y_true > THR_CRITICAL)
        is_fp = (gated > THR_CRITICAL) & (y_true <= THR_CRITICAL)
        is_fn = (gated <= THR_CRITICAL) & (y_true > THR_CRITICAL)
        is_tn = (gated <= THR_CRITICAL) & (y_true <= THR_CRITICAL)
        
        print(f"\n>>> HSR Layout @ Rain Intensity = {I:.0f} mm/hr (Total Nodes = {len(y_true)}) <<<")
        print(f"  TP: {np.sum(is_tp):4d} | FP: {np.sum(is_fp):4d} | FN: {np.sum(is_fn):4d} | TN: {np.sum(is_tn):4d}")
        print(f"  SWMM Critical: {np.sum(y_true > THR_CRITICAL):4d} | Model Critical: {np.sum(gated > THR_CRITICAL):4d}")
        
        if np.sum(is_fp) > 0:
            print("\n  [FALSE POSITIVE PROFILE vs TRUE NEGATIVE PROFILE]")
            print(f"  {'Feature':<24s} {'FP Mean':<12s} {'TN Mean':<12s} {'TP Mean':<12s} {'Physical Interpretation'}")
            print("  " + "-" * 80)
            
            key_feats = [2, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27]
            for fi in key_feats:
                fp_m = x_raw[is_fp, fi].mean()
                tn_m = x_raw[is_tn, fi].mean()
                tp_m = x_raw[is_tp, fi].mean() if np.sum(is_tp) > 0 else 0.0
                
                comment = ""
                if fi == 10:  # sag_index
                    comment = "FP has elevated sag index (looks like depression)"
                elif fi == 20: # path_cap
                    comment = "FP has positive conduit capacity (SWMM conveys flow away!)"
                elif fi == 6: # out_deg
                    comment = "FP has out_deg >= in_deg (water drains downstream)"
                elif fi == 18: # dep_depth
                    comment = "Local geometric depression depth"
                elif fi == 7: # accum_score
                    comment = "Flow accumulation score"
                    
                print(f"  {feat_names[fi]:<24s} {fp_m:<12.4f} {tn_m:<12.4f} {tp_m:<12.4f} {comment}")

    print("\n" + "=" * 90)
    print("                ROOT CAUSE ANALYSIS SUMMARY")
    print("=" * 90)

if __name__ == '__main__':
    diagnose()
