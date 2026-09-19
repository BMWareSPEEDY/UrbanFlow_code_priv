"""Inspect shallow false alarms / advisory nodes in HSR @ 50mm/hr.
Find all nodes where SWMM is 0-1 cm but model predicts ~10cm.
"""
import os, sys, torch, numpy as np

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from production_v4 import ProductionFloodPredictorV4
from train_hydro_gine_v4 import HydroGINE_v4

def inspect_shallow_advisory():
    dl = torch.load("multi_scenario_physics_pyg_dataset.pt", weights_only=False)
    g_50 = [g for g in dl if g.region == 'hsr' and abs(g.x[0, 13].item() - 50.0) < 1e-3][0]
    y_true = g_50.y.cpu().numpy().ravel()
    
    predictor = ProductionFloodPredictorV4("hydro_gine_v4_model.pt", device=device)
    gated, raw, prob = predictor.predict(g_50, 50.0, 60.0)
    
    x_raw = g_50.x.cpu().numpy()
    
    # Let's find nodes where SWMM <= 0.02m (0-2 cm) but GNN gated >= 0.05m (5-15 cm)
    dry_overpred_idx = np.where((y_true <= 0.02) & (gated >= 0.05))[0]
    
    print(f"HSR @ 50 mm/hr: Total nodes = {len(y_true)}")
    print(f"Nodes where SWMM <= 0.02m: {np.sum(y_true <= 0.02)}")
    print(f"Nodes where SWMM <= 0.02m BUT Model >= 0.05m (False Advisory/Ponding): {len(dry_overpred_idx)}")
    
    print("\nTop 30 False Advisory Nodes (SWMM 0-1 cm vs Model ~10cm):")
    print(f"{'Idx':<6s} | {'GNN Depth':<10s} | {'SWMM Depth':<10s} | {'Prob':<8s} | {'RawLin':<8s} | {'Sink D':<8s} | {'Dep D':<8s} | {'In/Out':<7s} | {'SagIdx':<8s} | {'MaxGrade':<8s} | {'RelDrop'}")
    print("-" * 115)
    
    for i in dry_overpred_idx[:30]:
        ind = x_raw[i, 3]
        outd = x_raw[i, 4]
        sinkd = x_raw[i, 23]
        depd = x_raw[i, 16]
        sag = x_raw[i, 8]
        grade = x_raw[i, 7]
        drop = x_raw[i, 0]
        print(f"{i:<6d} | {gated[i]:<10.4f} | {y_true[i]:<10.4f} | {prob[i]:<8.4f} | {raw[i]:<8.4f} | {sinkd:<8.4f} | {depd:<8.4f} | {ind:.0f}/{outd:.0f}   | {sag:<8.4f} | {grade:<8.4f} | {drop:.4f}")

if __name__ == '__main__':
    inspect_shallow_advisory()
