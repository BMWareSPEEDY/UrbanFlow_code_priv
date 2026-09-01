"""Smoke test production ensemble artifact on held-out test graphs."""
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")
sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW")
from production import ProductionEnsemble

dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_full22_pyg_dataset.pt", weights_only=False)
te = [g.clone() for g in dl if g.city in ('bangalore', 'hongkong')]
thr = 0.15
ens = ProductionEnsemble()
print("ensemble loaded")

pred, raw, prob = ens.predict([te[0]], 150.0)
y = te[0].y.numpy().ravel()
f = lambda p: 2 * (np.sum((p >= thr) & (y >= thr))) / (np.sum(p >= thr) + np.sum(y >= thr) + 1e-9)
print(f"single-graph I=150 test: ens F1 {f(pred):.4f} | raw {f(raw):.4f} | n={len(pred)}")

from torch_geometric.data import Batch
b = Batch.from_data_list(te)
pred2, raw2, prob2 = ens.predict([b], [150.0])
y2 = np.concatenate([g.y.numpy().ravel() for g in te])
f2 = lambda p: 2 * (np.sum((p >= thr) & (y2 >= thr))) / (np.sum(p >= thr) + np.sum(y2 >= thr) + 1e-9)
print(f"batched full test (n={len(pred2)}): ens F1 {f2(pred2):.4f}")