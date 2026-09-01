"""Ensemble iter12 + iter13 (max-gate, different architectures)."""
import torch
import numpy as np

THR = 0.15
d12 = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter12_full.pt", weights_only=False)
d13 = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter13_full.pt", weights_only=False)

TAUS = {(20.0, 0): 0.4, (20.0, 1): 0.35, (50.0, 0): 0.45, (50.0, 1): 0.3,
        (80.0, 0): 0.35, (80.0, 1): 0.15, (120.0, 0): 0.35, (120.0, 1): 0.4,
        (150.0, 0): 0.4, (150.0, 1): 0.4, (200.0, 0): 0.4, (200.0, 1): 0.4,
        (250.0, 0): 0.4, (250.0, 1): 0.4, (300.0, 0): 0.4, (300.0, 1): 0.4}
TAUS12 = {(20.0, 0): 0.35, (20.0, 1): 0.5, (50.0, 0): 0.1, (50.0, 1): 0.15,
          (80.0, 0): 0.35, (80.0, 1): 0.35, (120.0, 0): 0.4, (120.0, 1): 0.35,
          (150.0, 0): 0.4, (150.0, 1): 0.4, (200.0, 0): 0.4, (200.0, 1): 0.4,
          (250.0, 0): 0.35, (250.0, 1): 0.4, (300.0, 0): 0.4, (300.0, 1): 0.4}


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
g13 = gate(d13['pred'], d13['prob'], d13['elev2'], d13['intensity'], TAUS)
y, i, city = d13['y'], d13['intensity'], d13['city']
m_i = i >= 150.0

for name, p in [("iter13", g13), ("ens 12+13", np.maximum(g12, g13)),
                ("ens 12+13 avgD", 0.5 * g12 + 0.5 * g13)]:
    fb, pb, rb = f1(y[(city == 'bangalore') & m_i], p[(city == 'bangalore') & m_i])
    fh, ph, rh = f1(y[(city == 'hongkong') & m_i], p[(city == 'hongkong') & m_i])
    f2, _, _ = f1(y[m_i], p[m_i])
    print(f"{name:<16} BLR {fb:.4f} ({pb:.2f}/{rb:.2f}) | HK {fh:.4f} ({ph:.2f}/{rh:.2f}) | POOLED {f2:.4f}")
for rg in ['hsr', 'bellandur']:
    m = (city == 'bangalore') & (d13['region'] == rg) & m_i
    f2, _, _ = f1(y[m], np.maximum(g12[m], g13[m]))
    print(f"  ens12+13 {rg}: {f2:.4f}")