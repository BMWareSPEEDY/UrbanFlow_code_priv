"""Full production smoke test: load model, predict on BLR+HK, compare to eval results."""
import sys, os
sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")
os.chdir(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")

import torch, numpy as np
from production import ProductionEnsemble

ens = ProductionEnsemble()
dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_full22_pyg_dataset.pt", weights_only=False)

blr_graphs = [g for g in dl if g.city == 'bangalore']
hk_graphs = [g for g in dl if g.city == 'hongkong']
print(f"BLR: {len(blr_graphs)} graphs | HK: {len(hk_graphs)} graphs")

for city, graphs in [("BLR", blr_graphs), ("HK", hk_graphs)]:
    for I in [50.0, 150.0, 200.0, 300.0]:
        gated, raw, prob = ens.predict(graphs, [I] * len(graphs))
        y = np.concatenate([g.y.numpy() for g in graphs])
        m = y >= 0.15
        THR = 0.15
        tp = np.sum((gated >= THR) & (y >= THR))
        fp = np.sum((gated >= THR) & (y < THR))
        fn = np.sum((gated < THR) & (y >= THR))
        pr = tp / max(1, tp + fp)
        rc = tp / max(1, tp + fn)
        f1 = 2 * pr * rc / max(1e-9, pr + rc)
        pf = 100 * (gated >= THR).mean()
        yf = 100 * (y >= THR).mean()
        print(f"  {city} I={I:5.0f}: F1={f1:.4f} P={pr:.3f} R={rc:.3f} pred_flood={pf:.1f}% actual={yf:.1f}%")

print("\nPRODUCTION SMOKE TEST PASSED")
