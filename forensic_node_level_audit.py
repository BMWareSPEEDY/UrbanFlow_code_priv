"""Forensic Node-Level Diagnostic: Inspect every False Positive and False Negative.
Breaks down predictions node-by-node against SWMM ground truth for all rainfall intensities in HSR Layout and Bengaluru.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from production_v4 import ProductionFloodPredictorV4

THR_HAZARD = 0.15
THR_CRITICAL = 0.30

def forensic_audit():
    print("=" * 110)
    print("                  FORENSIC NODE-LEVEL DIAGNOSTIC AUDIT")
    print("=" * 110)
    
    predictor = ProductionFloodPredictorV4("hydro_gine_v4_model.pt", device=device)
    dl = torch.load("multi_scenario_physics_pyg_dataset.pt", weights_only=False)
    
    # Let's inspect HSR Layout across all intensities: 20, 50, 80, 120, 150, 200, 250, 300 mm/hr
    hsr_graphs = [g for g in dl if g.region == 'hsr']
    hsr_graphs.sort(key=lambda g: g.x[0, 13].item())
    
    print("\n" + "=" * 110)
    print("1. HSR LAYOUT: SCENARIO-BY-SCENARIO ACCURACY & ERROR BREAKDOWN")
    print("=" * 110)
    print(f"{'Rain (mm/hr)':<12s} | {'SWMM Crit':<10s} | {'GNN Crit':<10s} | {'TP Crit':<8s} | {'FP Crit':<8s} | {'FN Crit':<8s} | {'SWMM Haz':<10s} | {'GNN Haz':<10s} | {'FP Haz':<8s} | {'FN Haz':<8s} | {'MAE (cm)':<8s} | {'%<=30cm'}")
    print("-" * 125)
    
    for g in hsr_graphs:
        I = g.x[0, 13].item()
        dur = g.x[0, 14].item()
        y_true = g.y.cpu().numpy().ravel()
        gated, raw, prob = predictor.predict(g, I, dur)
        
        swmm_crit = np.sum(y_true > THR_CRITICAL)
        gnn_crit = np.sum(gated > THR_CRITICAL)
        tp_c = np.sum((gated > THR_CRITICAL) & (y_true > THR_CRITICAL))
        fp_c = np.sum((gated > THR_CRITICAL) & (y_true <= THR_CRITICAL))
        fn_c = np.sum((gated <= THR_CRITICAL) & (y_true > THR_CRITICAL))
        
        swmm_haz = np.sum(y_true >= THR_HAZARD)
        gnn_haz = np.sum(gated >= THR_HAZARD)
        fp_h = np.sum((gated >= THR_HAZARD) & (y_true < THR_HAZARD))
        fn_h = np.sum((gated < THR_HAZARD) & (y_true >= THR_HAZARD))
        
        mae = np.mean(np.abs(gated - y_true)) * 100.0
        pct_30 = np.mean(np.abs(gated - y_true) <= 0.30) * 100.0
        
        print(f"{I:<12.0f} | {swmm_crit:<10d} | {gnn_crit:<10d} | {tp_c:<8d} | {fp_c:<8d} | {fn_c:<8d} | {swmm_haz:<10d} | {gnn_haz:<10d} | {fp_h:<8d} | {fn_h:<8d} | {mae:<8.2f} | {pct_30:<6.1f}%")

    # Let's inspect the specific False Positives at 20 mm/hr and 50 mm/hr (Light Rain)
    g_20 = [g for g in hsr_graphs if abs(g.x[0, 13].item() - 20.0) < 1e-3][0]
    y_20 = g_20.y.cpu().numpy().ravel()
    gated_20, raw_20, prob_20 = predictor.predict(g_20, 20.0, 60.0)
    
    fp_20_idx = np.where((gated_20 > THR_CRITICAL) & (y_20 <= THR_CRITICAL))[0]
    print(f"\n--- HSR @ 20 mm/hr: Exact False Positive Nodes (Count = {len(fp_20_idx)}) ---")
    print(f"{'Node Idx':<10s} | {'GNN Depth':<12s} | {'GNN Prob':<10s} | {'SWMM Depth':<12s} | {'Sink Depth':<12s} | {'Dep Depth':<12s} | {'In/Out Deg':<12s} | {'Accum Score'}")
    print("-" * 100)
    for idx in fp_20_idx[:15]:
        sd = g_20.x[idx, 23].item()
        dd = g_20.x[idx, 16].item()
        ind = g_20.x[idx, 3].item()
        outd = g_20.x[idx, 4].item()
        accum = g_20.x[idx, 5].item()
        print(f"{idx:<10d} | {gated_20[idx]:<12.4f} | {prob_20[idx]:<10.4f} | {y_20[idx]:<12.4f} | {sd:<12.4f} | {dd:<12.4f} | {ind:.0f} / {outd:.0f}        | {accum:.4f}")

    # Let's inspect False Negatives at 200 mm/hr (Heavy Cloudburst)
    g_200 = [g for g in hsr_graphs if abs(g.x[0, 13].item() - 200.0) < 1e-3][0]
    y_200 = g_200.y.cpu().numpy().ravel()
    gated_200, raw_200, prob_200 = predictor.predict(g_200, 200.0, 60.0)
    
    fn_200_idx = np.where((gated_200 <= THR_CRITICAL) & (y_200 > THR_CRITICAL))[0]
    print(f"\n--- HSR @ 200 mm/hr: Exact False Negative Nodes (Count = {len(fn_200_idx)}) ---")
    print(f"{'Node Idx':<10s} | {'GNN Depth':<12s} | {'GNN Prob':<10s} | {'SWMM Depth':<12s} | {'Sink Depth':<12s} | {'Dep Depth':<12s} | {'In/Out Deg':<12s} | {'Accum Score'}")
    print("-" * 100)
    for idx in fn_200_idx[:15]:
        sd = g_200.x[idx, 23].item()
        dd = g_200.x[idx, 16].item()
        ind = g_200.x[idx, 3].item()
        outd = g_200.x[idx, 4].item()
        accum = g_200.x[idx, 5].item()
        print(f"{idx:<10d} | {gated_200[idx]:<12.4f} | {prob_200[idx]:<10.4f} | {y_200[idx]:<12.4f} | {sd:<12.4f} | {dd:<12.4f} | {ind:.0f} / {outd:.0f}        | {accum:.4f}")

if __name__ == '__main__':
    forensic_audit()
