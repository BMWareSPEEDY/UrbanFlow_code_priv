"""Ensemble probes with iter12."""
import torch
import numpy as np

THR = 0.15
d12 = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter12_full.pt", weights_only=False)
d11 = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter11_full.pt", weights_only=False)
d9 = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter9_full.pt", weights_only=False)

TAUS12 = {(20.0, 0): 0.35, (20.0, 1): 0.5, (50.0, 0): 0.1, (50.0, 1): 0.15,
          (80.0, 0): 0.35, (80.0, 1): 0.35, (120.0, 0): 0.4, (120.0, 1): 0.35,
          (150.0, 0): 0.4, (150.0, 1): 0.4, (200.0, 0): 0.4, (200.0, 1): 0.4,
          (250.0, 0): 0.35, (250.0, 1): 0.4, (300.0, 0): 0.4, (300.0, 1): 0.4}
TAUS11 = {(20.0, 0): 0.5, (20.0, 1): 0.65, (50.0, 0): 0.4, (50.0, 1): 0.3,
          (80.0, 0): 0.55, (80.0, 1): 0.55, (120.0, 0): 0.6, (120.0, 1): 0.65,
          (150.0, 0): 0.65, (150.0, 1): 0.65, (200.0, 0): 0.65, (200.0, 1): 0.65,
          (250.0, 0): 0.6, (250.0, 1): 0.65, (300.0, 0): 0.65, (300.0, 1): 0.65}
TAUS9 = {(20.0, 0): 0.45, (20.0, 1): 0.25, (50.0, 0): 0.4, (50.0, 1): 0.45,
         (80.0, 0): 0.6, (80.0, 1): 0.55, (120.0, 0): 0.6, (120.0, 1): 0.6,
         (150.0, 0): 0.6, (150.0, 1): 0.65, (200.0, 0): 0.65, (200.0, 1): 0.65,
         (250.0, 0): 0.6, (250.0, 1): 0.6, (300.0, 0): 0.6, (300.0, 1): 0.65}


def gate(pred, prob, e2, intensity, taus):
    out = pred.copy()
    for I in np.unique(intensity):
        for cl in [0, 1]:
            m = (intensity == I) & ((e2 > 1.3) == (cl == 1))
            t = taus.get((float(I), cl), 0.6)
            out[m] = np.where(prob[m] < t, 0.0, out[m])
    return out


def f1(yv, p):
    tp = np.sum((p >= THR) & (yv >= THR)); fp = np.sum((p >= THR) & (yv < THR)); fn = np.sum((p < THR) & (yv >= THR))
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc), pr, rc


g12 = gate(d12['pred'], d12['prob'], d12['elev2'], d12['intensity'], TAUS12)
g11 = gate(d11['pred'], d11['prob'], d11['elev2'], d11['intensity'], TAUS11)
g9 = gate(d9['pred'], d9['prob'], d9['elev2'], d9['intensity'], TAUS9)
y, i, city = d12['y'], d12['intensity'], d12['city']
m_i = i >= 150.0

for name, p in [("iter12", g12), ("ens 9+12", np.maximum(g9, g12)),
                ("ens 11+12", np.maximum(g11, g12)), ("ens 9+11+12", np.maximum(np.maximum(g9, g11), g12))]:
    fb, prb, rcb = f1(y[(city == 'bangalore') & m_i], p[(city == 'bangalore') & m_i])
    fh, prh, rch = f1(y[(city == 'hongkong') & m_i], p[(city == 'hongkong') & m_i])
    print(f"{name:<12} BLR {fb:.4f} ({prb:.2f}/{rcb:.2f}) | HK {fh:.4f} ({prh:.2f}/{rch:.2f})")