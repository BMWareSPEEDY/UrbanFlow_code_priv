"""Definitive multi-model benchmark report comparing all historical versions against HydroGINE-v4.0.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from reproduce_baseline_suite import compute_metrics, predict_app_pipeline
from production_v4 import ProductionFloodPredictorV4

def run_benchmark():
    print("=" * 125)
    print("                      URBANFLOW DEFINITIVE MULTI-MODEL BENCHMARK AUDIT")
    print("=" * 125)
    
    # 1. Load multi-scenario full22 dataset (for v1-v3.3 and MoE)
    dl_22 = torch.load("multi_scenario_full22_pyg_dataset.pt", weights_only=False)
    blr_22 = [g for g in dl_22 if g.city == 'bangalore']
    hk_22 = [g for g in dl_22 if g.city == 'hongkong']
    
    # 2. Load zero-leakage physics dataset (for HydroGINE-v4.0)
    dl_phys = torch.load("multi_scenario_physics_pyg_dataset.pt", weights_only=False)
    blr_phys = [g for g in dl_phys if g.city == 'bangalore']
    hk_phys = [g for g in dl_phys if g.city == 'hongkong']
    
    # Load models
    sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")
    from production import ProductionEnsemble
    moe = ProductionEnsemble(device=device)
    v4 = ProductionFloodPredictorV4("hydro_gine_v4_model.pt", device=device)
    
    def eval_model(name, pred_fn, is_phys=False):
        blr_graphs = blr_phys if is_phys else blr_22
        hk_graphs = hk_phys if is_phys else hk_22
        
        # A. HSR Light (20, 50 mm/hr) and Cloudburst (150, 200, 300 mm/hr)
        hsr_fp_light, hsr_fp_cloud = 0, 0
        for g in blr_graphs:
            I = g.x[0, 13].item() if is_phys else g.x[0, 15].item()
            if g.region == 'hsr':
                yt = g.y.cpu().numpy().ravel()
                yp = pred_fn(g, I)
                m = compute_metrics(yt, yp)
                if I in [20.0, 50.0]:
                    hsr_fp_light += m['fp_c']
                elif I in [150.0, 200.0, 300.0]:
                    hsr_fp_cloud += m['fp_c']
                    
        hsr_fp_light_avg = hsr_fp_light / 2.0
        hsr_fp_cloud_avg = hsr_fp_cloud / 3.0
        
        # B. Pooled Bengaluru
        blr_preds, blr_trues = [], []
        t0 = time.perf_counter()
        for g in blr_graphs:
            I = g.x[0, 13].item() if is_phys else g.x[0, 15].item()
            blr_preds.append(pred_fn(g, I))
            blr_trues.append(g.y.cpu().numpy().ravel())
        t_infer = (time.perf_counter() - t0) * 1000.0 / len(blr_graphs)
        
        y_bp = np.concatenate(blr_preds)
        y_bt = np.concatenate(blr_trues)
        mb = compute_metrics(y_bt, y_bp)
        
        # C. Hong Kong Zero-Shot
        hk_preds, hk_trues = [], []
        for g in hk_graphs:
            I = g.x[0, 13].item() if is_phys else g.x[0, 15].item()
            hk_preds.append(pred_fn(g, I))
            hk_trues.append(g.y.cpu().numpy().ravel())
        y_hp = np.concatenate(hk_preds)
        y_ht = np.concatenate(hk_trues)
        mh = compute_metrics(y_ht, y_hp)
        
        return {
            'Version': name,
            'HSR FP Light (avg)': hsr_fp_light_avg,
            'HSR FP Cloud (avg)': hsr_fp_cloud_avg,
            'BLR Crit Recall': f"{mb['rec_c']*100:.1f}%",
            'BLR Crit Prec': f"{mb['prec_c']*100:.1f}%",
            'BLR Hazard F1': f"{mb['f1_h']:.4f}",
            'BLR MAE (cm)': f"{mb['mae']*100:.2f}",
            'BLR %<=30cm': f"{mb['pct_30']:5.1f}%",
            'BLR R2': f"{mb['r2']:.4f}",
            'HK Hazard F1': f"{mh['f1_h']:.4f}",
            'HK %<=30cm': f"{mh['pct_30']:5.1f}%",
            'Latency (ms)': f"{t_infer:.2f}"
        }

    def pred_moe(g, I):
        gated, _, _ = moe.predict(g, [I])
        return gated

    def pred_v4(g, I):
        gated, _, _ = v4.predict(g, I)
        return gated

    model_configs = [
        ("v1.0 Baseline", lambda g, I: predict_app_pipeline('v1.0', g, I), False),
        ("v2.0 Dual-Stream", lambda g, I: predict_app_pipeline('v2.0', g, I), False),
        ("v3.0 Adaptive Hydro Ceiling", lambda g, I: predict_app_pipeline('v3.0', g, I), False),
        ("v3.1 Hydro Basin Sink", lambda g, I: predict_app_pipeline('v3.1', g, I), False),
        ("v3.2 Precision Deep Sag", lambda g, I: predict_app_pipeline('v3.2', g, I), False),
        ("v3.3 Production Baseline", lambda g, I: predict_app_pipeline('v3.3', g, I), False),
        ("MoE v6 (iter14+iter17)", pred_moe, False),
        ("HydroGINE-v4.0 (Ours Zero-Leakage)", pred_v4, True),
    ]
    
    rows = []
    for name, fn, is_p in model_configs:
        print(f"Evaluating {name}...")
        res = eval_model(name, fn, is_phys=is_p)
        rows.append(res)
        
    df_res = pd.DataFrame(rows)
    print("\n" + "=" * 125)
    print("                               SUMMARY COMPARISON TABLE")
    print("=" * 125)
    print(df_res.to_string(index=False))
    print("=" * 125)
    
    df_res.to_csv("definitive_benchmark_summary.csv", index=False)
    print("Saved results to definitive_benchmark_summary.csv")

if __name__ == '__main__':
    run_benchmark()
