"""Inspect top risk nodes in HSR Layout in the web app.
Find which top highlighted nodes are false alarms.
"""
import torch, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from production_v4 import ProductionFloodPredictorV4

def inspect_top_hsr_risk_nodes():
    dl = torch.load("multi_scenario_physics_pyg_dataset.pt", weights_only=False)
    predictor = ProductionFloodPredictorV4("hydro_gine_v4_3_model.pt", device=device)
    
    g_50 = [g for g in dl if g.region == 'hsr' and abs(g.x[0, 13].item() - 50.0) < 1e-3][0]
    y_true_50 = g_50.y.cpu().numpy().ravel()
    p_50, raw_50, prob_50 = predictor.predict(g_50, 50.0, 60.0)
    x_50 = g_50.x.cpu().numpy()
    
    # Sort by predicted depth (top highlighted nodes in the UI)
    top_idx = np.argsort(p_50)[::-1]
    
    print("=" * 115)
    print("         TOP 20 MOST HIGHLIGHTED NODES IN HSR LAYOUT @ 50 mm/hr")
    print("=" * 115)
    print(f"{'Rank':<5s} | {'Idx':<6s} | {'Pred Depth':<12s} | {'SWMM Depth':<12s} | {'Diff (cm)':<10s} | {'Status':<12s} | {'Sink D':<8s} | {'In/Out':<7s} | {'ConvDef':<8s} | {'SagIdx'}")
    print("-" * 115)
    
    for rank, idx in enumerate(top_idx[:20], 1):
        pred = p_50[idx]
        swmm = y_true_50[idx]
        diff_cm = (pred - swmm) * 100.0
        sd = x_50[idx, 23]
        ind = x_50[idx, 3]
        outd = x_50[idx, 4]
        conv = x_50[idx, 30]
        sag = x_50[idx, 8]
        
        status = "CORRECT" if abs(diff_cm) <= 15.0 else ("FALSE HIGH" if diff_cm > 15.0 else "UNDER")
        print(f"{rank:<5d} | {idx:<6d} | {pred:<12.4f} | {swmm:<12.4f} | {diff_cm:<+10.1f} | {status:<12s} | {sd:<8.4f} | {ind:.0f}/{outd:.0f}   | {conv:<8.4f} | {sag:.4f}")

if __name__ == '__main__':
    inspect_top_hsr_risk_nodes()
