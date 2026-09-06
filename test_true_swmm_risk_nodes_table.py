"""Test UI risk nodes table accuracy when using true SWMM simulation ground truth.
"""
import torch, sys, numpy as np
from app import REGION_CACHE, PRODUCTION_PREDICTOR

sys.stdout.reconfigure(line_buffering=True)

dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
swmm_truth_by_region = {}
for g in dl:
    r = getattr(g, 'region', '') or getattr(g, 'city', '')
    if r and abs(g.rain_intensity - 50.0) < 1.0 and r not in swmm_truth_by_region:
        swmm_truth_by_region[r] = g.y.cpu().numpy().ravel()

print("=" * 115)
print("AUDIT ACROSS ALL 16 CITIES WITH TRUE SWMM SIMULATION TARGETS (@ 50 mm/hr):")
print("=" * 115)
print(f"{'City':<15s} | {'Total':<8s} | {'Real Floods':<12s} | {'Detected':<10s} | {'False Alarms':<13s} | {'Missed':<8s} | {'Match %':<9s} | {'Depth MAE'}")
print("-" * 115)

tot_nodes = 0
tot_floods = 0
tot_tp = 0
tot_fp = 0
tot_fn = 0
errs = []

for r_key, r_data in REGION_CACHE.items():
    if r_key not in swmm_truth_by_region:
        continue
    preds, _, _ = PRODUCTION_PREDICTOR.predict(r_data['pyg_data'], 50.0, 60.0)
    y_swmm = swmm_truth_by_region[r_key]
    
    # In live app: Hazard threshold is 0.15m
    sf = (y_swmm >= 0.15)
    gf = (preds >= 0.15)
    
    tp = int(np.sum(sf & gf))
    fp = int(np.sum(~sf & gf))
    fn = int(np.sum(sf & ~gf))
    tn = int(np.sum(~sf & ~gf))
    
    tot_nodes += len(y_swmm)
    tot_floods += int(np.sum(sf))
    tot_tp += tp
    tot_fp += fp
    tot_fn += fn
    
    mae = np.mean(np.abs(preds - y_swmm)) * 100.0
    errs.append(np.abs(preds - y_swmm))
    acc = ((tp + tn) / len(y_swmm)) * 100.0
    
    print(f"{r_key:<15s} | {len(y_swmm):<8d} | {int(np.sum(sf)):<12d} | {tp:<10d} | {fp:<13d} | {fn:<8d} | {acc:<8.1f}% | {mae:<6.2f} cm")

print("=" * 115)
all_e = np.concatenate(errs)
tot_acc = ((tot_tp + (tot_nodes - tot_floods - tot_fp)) / tot_nodes) * 100.0
print(f"{'TOTAL':<15s} | {tot_nodes:<8d} | {tot_floods:<12d} | {tot_tp:<10d} | {tot_fp:<13d} | {tot_fn:<8d} | {tot_acc:<8.1f}% | {np.mean(all_e)*100.0:<6.2f} cm")
