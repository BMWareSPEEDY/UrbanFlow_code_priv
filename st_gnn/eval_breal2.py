"""Per-intensity + predicted-vs-actual flood% for real-terrain Bangalore."""
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")
sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW")
from production import ProductionEnsemble

dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_full22_pyg_dataset.pt", weights_only=False)
blr = [g for g in dl if g.city == 'bangalore']
ens = ProductionEnsemble()
pred, _, _ = ens.predict(blr, [g.x[0, 15].item() for g in blr])
THR = 0.15

off = 0
rows = []
for g in blr:
    I = g.x[0, 15].item()
    p = pred[off:off + g.num_nodes]
    y = g.y.numpy().ravel()
    off += g.num_nodes
    tp = np.sum((p >= THR) & (y >= THR)); fp = np.sum((p >= THR) & (y < THR)); fn = np.sum((p < THR) & (y >= THR))
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    f = 2 * pr * rc / max(1e-9, pr + rc)
    rows.append((I, f, pr, rc, 100 * (p >= THR).mean(), 100 * (y >= THR).mean(), g.region))
for I in [150.0, 200.0, 250.0, 300.0]:
    sel = [r for r in rows if r[0] == I]
    f = np.mean([r[1] for r in sel]); pr = np.mean([r[2] for r in sel]); rc = np.mean([r[3] for r in sel])
    pf = np.mean([r[4] for r in sel]); yf = np.mean([r[5] for r in sel])
    print(f"I={I:5.0f}: F1 {f:.4f} (P {pr:.3f} R {rc:.3f}) | pred flood% {pf:.1f} vs actual {yf:.1f}")
print()
for rg in sorted(set(r[6] for r in rows)):
    sel = [r for r in rows if r[6] == rg and r[0] == 300.0][0]
    print(f"{rg:<12} I=300: pred flood% {sel[4]:.1f} vs actual {sel[5]:.1f} (F1 {sel[1]:.4f})")