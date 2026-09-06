"""Diagnose all 31 FP and 40 FN nodes in HSR Layout at 50 mm/hr.
"""
import torch, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from production_v4 import ProductionFloodPredictorV4

predictor = ProductionFloodPredictorV4("hydro_gine_v5_model.pt", device=device)
dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
hsr_50 = [g for g in dl if getattr(g, 'region', '') == 'hsr' and abs(g.rain_intensity - 50.0) < 1e-3][0]

y_true = hsr_50.y.cpu().numpy().ravel()
pred, _, _ = predictor.predict(hsr_50, 50.0, 60.0)

is_swmm_flood = (y_true >= 0.15)
is_gnn_flood = (pred >= 0.15)

fp_indices = np.where(~is_swmm_flood & is_gnn_flood)[0]
fn_indices = np.where(is_swmm_flood & ~is_gnn_flood)[0]

print(f"Total FP (False Alarms): {len(fp_indices)}")
print(f"Total FN (Missed Floods): {len(fn_indices)}")

x_raw = hsr_50.x.cpu().numpy()

print("\n--- TOP 15 FALSE ALARMS (FP: Pred >= 0.15m, SWMM < 0.15m) ---")
print(f"{'Idx':<6s} | {'Pred':<8s} | {'SWMM':<8s} | {'RelDrop':<8s} | {'DepD':<8s} | {'SinkD':<8s} | {'InD':<4s} | {'OutD':<5s} | {'AccumS':<7s} | {'ConvDef':<8s}")
print("-" * 85)
for idx in fp_indices[:15]:
    p = pred[idx]
    s = y_true[idx]
    rd = x_raw[idx, 0]
    dep = x_raw[idx, 16]
    sink = x_raw[idx, 23]
    in_d = int(x_raw[idx, 3])
    out_d = int(x_raw[idx, 4])
    acc = x_raw[idx, 5]
    cdef = x_raw[idx, 30]
    print(f"{idx:<6d} | {p:<8.4f} | {s:<8.4f} | {rd:<8.4f} | {dep:<8.4f} | {sink:<8.4f} | {in_d:<4d} | {out_d:<5d} | {acc:<7.2f} | {cdef:<8.4f}")

print("\n--- TOP 15 MISSED FLOODS (FN: SWMM >= 0.15m, Pred < 0.15m) ---")
print(f"{'Idx':<6s} | {'Pred':<8s} | {'SWMM':<8s} | {'RelDrop':<8s} | {'DepD':<8s} | {'SinkD':<8s} | {'InD':<4s} | {'OutD':<5s} | {'AccumS':<7s} | {'ConvDef':<8s}")
print("-" * 85)
for idx in fn_indices[:15]:
    p = pred[idx]
    s = y_true[idx]
    rd = x_raw[idx, 0]
    dep = x_raw[idx, 16]
    sink = x_raw[idx, 23]
    in_d = int(x_raw[idx, 3])
    out_d = int(x_raw[idx, 4])
    acc = x_raw[idx, 5]
    cdef = x_raw[idx, 30]
    print(f"{idx:<6d} | {p:<8.4f} | {s:<8.4f} | {rd:<8.4f} | {dep:<8.4f} | {sink:<8.4f} | {in_d:<4d} | {out_d:<5d} | {acc:<7.2f} | {cdef:<8.4f}")
