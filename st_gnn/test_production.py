import sys, os
sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")
os.chdir(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")

import torch, numpy as np
from production import ProductionEnsemble

print("Loading model...")
ens = ProductionEnsemble()
print(f"Model loaded on {ens.device}")
xm = ens.stats["xm"]
print(f"x_mean shape: {xm.shape}, yl_mean: {ens.stats['ylm'].item():.4f}")

print("\nLoading test graph from dataset...")
dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_full22_pyg_dataset.pt", weights_only=False)
test_graphs = [g for g in dl if g.city == 'hongkong']
print(f"Found {len(test_graphs)} HK test graphs")

g = test_graphs[0]
print(f"Graph: {g.x.shape[0]} nodes, {g.x.shape[1]} features")

gated, raw, prob = ens.predict([g], [150.0])
print(f"\nPredictions for HK I=150:")
print(f"  Gated depth: {gated.shape}, range [{raw.min():.4f}, {raw.max():.4f}]")
print(f"  Prob: {prob.shape}, range [{prob.min():.4f}, {prob.max():.4f}]")
print(f"  Flood fraction (gated >= 0.15): {(gated >= 0.15).mean()*100:.1f}%")
print(f"  Flood fraction (raw >= 0.15): {(raw >= 0.15).mean()*100:.1f}%")

# Batch prediction
print("\nBatch test (all HK graphs, I=150)...")
gated_batch, raw_batch, prob_batch = ens.predict(test_graphs, [150.0] * len(test_graphs))
THR = 0.15
y_batch = np.concatenate([g.y.numpy() for g in test_graphs])
tp = np.sum((gated_batch >= THR) & (y_batch >= THR))
fp = np.sum((gated_batch >= THR) & (y_batch < THR))
fn = np.sum((gated_batch < THR) & (y_batch >= THR))
pr = tp / max(1, tp + fp)
rc = tp / max(1, tp + fn)
f1 = 2 * pr * rc / max(1e-9, pr + rc)
print(f"  HK batch F1: {f1:.4f} (P {pr:.3f} R {rc:.3f})")

# Test different intensities
for I in [50.0, 120.0, 200.0, 300.0]:
    g_i, r_i, p_i = ens.predict(test_graphs, [I] * len(test_graphs))
    y_i = y_batch
    tp = np.sum((g_i >= THR) & (y_i >= THR))
    fp = np.sum((g_i >= THR) & (y_i < THR))
    fn = np.sum((g_i < THR) & (y_i >= THR))
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    f1_i = 2 * pr * rc / max(1e-9, pr + rc)
    print(f"  HK I={I:.0f}: F1 {f1_i:.4f}")

print("\nSMOKE TEST PASSED")
