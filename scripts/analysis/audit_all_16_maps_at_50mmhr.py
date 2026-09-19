"""Audit all maps at 50 mm/hr to count FP and FN.
"""
import torch, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from production_v4 import ProductionFloodPredictorV4

predictor = ProductionFloodPredictorV4("hydro_gine_v5_model.pt", device=device)
dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)

# Get unique regions
regions = []
for g in dl:
    r = getattr(g, 'region', '') or getattr(g, 'city', '')
    if r and r not in regions:
        regions.append(r)

print("=" * 95)
print(f"AUDIT ACROSS ALL {len(regions)} MAPS AT 50 mm/hr:")
print("=" * 95)
print(f"{'Region':<15s} | {'Total':<7s} | {'SWMM Floods':<12s} | {'Detected(TP)':<13s} | {'FP (Alarms)':<12s} | {'FN (Missed)':<12s}")
print("-" * 95)

total_swmm = 0
total_tp = 0
total_fp = 0
total_fn = 0

for r in regions:
    r_50 = [g for g in dl if (getattr(g, 'region', '') == r or getattr(g, 'city', '') == r) and abs(g.rain_intensity - 50.0) < 1.0]
    if not r_50:
        continue
    g = r_50[0]
    y_true = g.y.cpu().numpy().ravel()
    p, _, _ = predictor.predict(g, 50.0, 60.0)
    
    sf = (y_true >= 0.15)
    gf = (p >= 0.15)
    
    tp = int(np.sum(sf & gf))
    fp = int(np.sum(~sf & gf))
    fn = int(np.sum(sf & ~gf))
    
    total_swmm += int(np.sum(sf))
    total_tp += tp
    total_fp += fp
    total_fn += fn
    
    print(f"{r:<15s} | {len(y_true):<7d} | {int(np.sum(sf)):<12d} | {tp:<13d} | {fp:<12d} | {fn:<12d}")

print("=" * 95)
print(f"{'TOTAL ACROSS ALL MAPS':<15s} | {'':<7s} | {total_swmm:<12d} | {total_tp:<13d} | {total_fp:<12d} | {total_fn:<12d}")
