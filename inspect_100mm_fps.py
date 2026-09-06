"""Inspect exact False Positives at 80, 100, 120 mm/hr in HSR Layout.
Analyze physical properties of the nodes and find the optimal hydraulic precision rule.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from production_v4 import ProductionFloodPredictorV4
from train_hydro_gine_v4 import HydroGINE_v4

THR_CRITICAL = 0.30

def inspect_100mm():
    predictor = ProductionFloodPredictorV4("hydro_gine_v4_model.pt", device=device)
    dl = torch.load("multi_scenario_physics_pyg_dataset.pt", weights_only=False)
    
    # Graphs for HSR at 80 and 120 mm/hr
    g_80 = [g for g in dl if g.region == 'hsr' and abs(g.x[0, 13].item() - 80.0) < 1e-3][0]
    g_120 = [g for g in dl if g.region == 'hsr' and abs(g.x[0, 13].item() - 120.0) < 1e-3][0]
    
    # Also evaluate at 100 mm/hr by setting intensity
    g_100 = g_80.clone()
    
    for I, g in [(80.0, g_80), (100.0, g_100), (120.0, g_120)]:
        y_true = g.y.cpu().numpy().ravel()
        gated, raw, prob = predictor.predict(g, I, 60.0)
        
        swmm_crit = np.sum(y_true > THR_CRITICAL)
        gnn_crit = np.sum(gated > THR_CRITICAL)
        tp = np.sum((gated > THR_CRITICAL) & (y_true > THR_CRITICAL))
        fp = np.sum((gated > THR_CRITICAL) & (y_true <= THR_CRITICAL))
        fn = np.sum((gated <= THR_CRITICAL) & (y_true > THR_CRITICAL))
        
        print(f"\n{'='*80}")
        print(f"HSR @ {I:.0f} mm/hr (1 hour duration):")
        print(f"  SWMM Critical: {swmm_crit} | GNN Critical: {gnn_crit} | TP: {tp} | FP: {fp} | FN: {fn}")
        print(f"  Critical Recall: {tp/max(1, tp+fn)*100:.1f}% | Critical Precision: {tp/max(1, tp+fp)*100:.1f}%")
        
        # Breakdown of FP nodes
        fp_idx = np.where((gated > THR_CRITICAL) & (y_true <= THR_CRITICAL))[0]
        x_raw = g.x.cpu().numpy()
        
        print(f"\nTop 15 False Positive Nodes at {I:.0f} mm/hr:")
        print(f"{'Idx':<6s} | {'GNN Depth':<10s} | {'SWMM Depth':<10s} | {'Diff':<8s} | {'Prob':<8s} | {'Sink D':<8s} | {'Dep D':<8s} | {'In/Out':<7s} | {'SagIdx':<8s} | {'ConvDef':<8s} | {'PathCap'}")
        print("-" * 105)
        for i in fp_idx[:15]:
            diff = gated[i] - y_true[i]
            ind = x_raw[i, 3]
            outd = x_raw[i, 4]
            sinkd = x_raw[i, 23]
            depd = x_raw[i, 16]
            sag = x_raw[i, 8]
            conv = x_raw[i, 30]
            pcap = x_raw[i, 18]
            print(f"{i:<6d} | {gated[i]:<10.4f} | {y_true[i]:<10.4f} | {diff:<8.4f} | {prob[i]:<8.4f} | {sinkd:<8.4f} | {depd:<8.4f} | {ind:.0f}/{outd:.0f}   | {sag:<8.4f} | {conv:<8.4f} | {pcap:.4f}")

if __name__ == '__main__':
    inspect_100mm()
