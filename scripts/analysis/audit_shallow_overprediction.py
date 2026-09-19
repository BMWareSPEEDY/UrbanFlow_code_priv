"""Forensic audit of 15-25cm overpredictions on shallow (0-8cm) SWMM nodes in HSR Layout.
"""
import torch, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from production_v4 import ProductionFloodPredictorV4

def audit_shallow_overprediction():
    dl = torch.load("multi_scenario_physics_pyg_dataset.pt", weights_only=False)
    predictor = ProductionFloodPredictorV4("hydro_gine_v4_model.pt", device=device)
    
    hsr_graphs = [g for g in dl if g.region == 'hsr']
    hsr_graphs.sort(key=lambda g: g.x[0, 13].item())
    
    print("=" * 115)
    print("      FORENSIC AUDIT: NODES WHERE SWMM <= 8cm BUT MODEL PREDICTS >= 15cm")
    print("=" * 115)
    
    for g in hsr_graphs:
        I = g.x[0, 13].item()
        dur = g.x[0, 14].item()
        y_true = g.y.cpu().numpy().ravel()
        gated, raw, prob = predictor.predict(g, I, dur)
        x_raw = g.x.cpu().numpy()
        
        # Nodes where SWMM is shallow (<= 8cm = 0.08m) but model predicts >= 15cm (0.15m)
        over_idx = np.where((y_true <= 0.08) & (gated >= 0.15))[0]
        
        print(f"\n--- HSR @ {I:.0f} mm/hr ({dur:.0f} min) ---")
        print(f"Total nodes: {len(y_true)} | SWMM <= 8cm: {np.sum(y_true <= 0.08)} | Model >= 15cm on <=8cm SWMM: {len(over_idx)}")
        
        if len(over_idx) > 0:
            print(f"{'Idx':<6s} | {'GNN Depth':<10s} | {'SWMM Depth':<10s} | {'Prob':<8s} | {'RawLin':<8s} | {'Sink D':<8s} | {'Dep D':<8s} | {'In/Out':<7s} | {'SagIdx':<8s} | {'ConvDef':<8s} | {'Accum'}")
            print("-" * 110)
            for i in over_idx[:15]:
                ind = x_raw[i, 3]
                outd = x_raw[i, 4]
                sinkd = x_raw[i, 23]
                depd = x_raw[i, 16]
                sag = x_raw[i, 8]
                conv = x_raw[i, 30]
                accum = x_raw[i, 5]
                print(f"{i:<6d} | {gated[i]:<10.4f} | {y_true[i]:<10.4f} | {prob[i]:<8.4f} | {raw[i]:<8.4f} | {sinkd:<8.4f} | {depd:<8.4f} | {ind:.0f}/{outd:.0f}   | {sag:<8.4f} | {conv:<8.4f} | {accum:.4f}")

if __name__ == '__main__':
    audit_shallow_overprediction()
