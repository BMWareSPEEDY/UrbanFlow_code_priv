"""Check new Bangalore stats + evaluate production iter13 on the real-terrain test set."""
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")
sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW")
from production import ProductionEnsemble

dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_full22_pyg_dataset.pt", weights_only=False)
blr = [g for g in dl if g.city == 'bangalore']
by_reg = {}
for g in blr:
    by_reg.setdefault(g.region, []).append(g)

print("=== Bangalore regions: real-terrain stats (I=300) ===")
for rg, gs in sorted(by_reg.items()):
    g300 = [g for g in gs if abs(g.x[0, 15].item() - 300.0) < 1e-6][0]
    x = g300.x.numpy()
    y = g300.y.numpy().ravel()
    e2 = x[:, 17]
    print(f"{rg:<12} n={g300.num_nodes:5d} elev_std2={e2.mean():6.2f} flood%={100*(y >= 0.15).mean():5.1f} "
          f"mean_y={y.mean():.3f}")

print("\n=== Production iter13 (unchanged weights) on REAL-terrain Bangalore test ===")
ens = ProductionEnsemble()
THR = 0.15
m_i = lambda i: i >= 150.0

preds = ens.predict(blr, [g.x[0, 15].item() for g in blr])
intensities = np.array([g.x[0, 15].item() for g in blr])
sel = intensities >= 150.0
for rg in sorted(by_reg):
    idx = [i for i, g in enumerate(blr) if g.region == rg]
    i_s = np.array([blr[i].x[0, 15].item() for i in idx])
    m = i_s >= 150.0
    idx = np.array(idx)[m]
    p = np.concatenate([preds[0][sum(g.num_nodes for g in blr[:i]):sum(g.num_nodes for g in blr[:i + 1])] for i in idx])
    y = np.concatenate([blr[i].y.numpy().ravel() for i in idx])
    tp = np.sum((p >= THR) & (y >= THR)); fp = np.sum((p >= THR) & (y < THR)); fn = np.sum((p < THR) & (y >= THR))
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    f = 2 * pr * rc / max(1e-9, pr + rc)
    print(f"  {rg:<12} F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")
off = 0
offsets = np.concatenate([np.full(g.num_nodes, i) for i, g in enumerate(blr)])
keep = np.isin(offsets, np.where(sel)[0])
p_all = preds[0][keep]
y_all = np.concatenate([g.y.numpy().ravel() for i, g in enumerate(blr) if sel[i]])
tp = np.sum((p_all >= THR) & (y_all >= THR)); fp = np.sum((p_all >= THR) & (y_all < THR)); fn = np.sum((p_all < THR) & (y_all >= THR))
pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
f = 2 * pr * rc / max(1e-9, pr + rc)
print(f"  {'BLR pooled':<12} F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")