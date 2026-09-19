"""Inspect hydraulic pipe features of FP vs TP nodes.
"""
import torch, numpy as np

dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
hsr_50 = [g for g in dl if getattr(g, 'region', '') == 'hsr' and abs(g.rain_intensity - 50.0) < 1e-3][0]

x = hsr_50.x.cpu().numpy()
y = hsr_50.y.cpu().numpy().ravel()

fp_nodes = [21, 116, 167, 170, 338, 415, 560, 586, 675, 762]
tp_nodes = [24, 38, 41, 79, 115, 169, 392, 450, 558, 674]

print(f"{'Type':<4s} | {'Idx':<5s} | {'SWMM':<7s} | {'HydCap(6)':<10s} | {'PathCap(20)':<12s} | {'Surcharge(19)':<14s} | {'PathHops(21)':<12s} | {'DistOut(22)':<12s}")
print("-" * 85)

for idx in fp_nodes:
    print(f"FP   | {idx:<5d} | {y[idx]:<7.4f} | {x[idx,6]:<10.4f} | {x[idx,20]:<12.4f} | {x[idx,19]:<14.4f} | {x[idx,21]:<12.4f} | {x[idx,22]:<12.4f}")

print("-" * 85)
for idx in tp_nodes:
    print(f"TP   | {idx:<5d} | {y[idx]:<7.4f} | {x[idx,6]:<10.4f} | {x[idx,20]:<12.4f} | {x[idx,19]:<14.4f} | {x[idx,21]:<12.4f} | {x[idx,22]:<12.4f}")
