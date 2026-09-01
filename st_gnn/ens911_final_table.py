"""Full per-region + per-intensity table for production ensemble iter9+iter11 (max-gate)."""
import torch
import numpy as np

THR = 0.15
d9 = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter9_full.pt", weights_only=False)
d11 = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter11_full.pt", weights_only=False)
y, i, city, reg = d9['y'], d9['intensity'], d9['city'], d9['region']

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


g9 = gate(d9['pred'], d9['prob'], d9['elev2'], d9['intensity'], TAUS9)
g11 = gate(d11['pred'], d11['prob'], d11['elev2'], d11['intensity'], TAUS11)
ens = np.maximum(g9, g11)


def f1(yv, p):
    tp = np.sum((p >= THR) & (yv >= THR)); fp = np.sum((p >= THR) & (yv < THR)); fn = np.sum((p < THR) & (yv >= THR))
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc), pr, rc


m_i = i >= 150.0
print("=== PRODUCTION ENSEMBLE iter9+iter11 (max-gate) ===")
for label, m in [("BANGALORE pooled", (city == 'bangalore') & m_i), ("HONGKONG", (city == 'hongkong') & m_i)]:
    f, pr, rc = f1(y[m], ens[m])
    print(f"{label:<18} F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")
for rg in ['bellandur', 'ecity', 'koramangala', 'whitefield', 'hsr', 'hongkong']:
    m = (reg == rg) & m_i
    f, pr, rc = f1(y[m], ens[m])
    print(f"  {rg:<12} F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")
print()
for I in [150.0, 200.0, 250.0, 300.0]:
    f, pr, rc = f1(y[(city == 'bangalore') & (i == I)], ens[(city == 'bangalore') & (i == I)])
    print(f"  BLR I={I:5.0f}: F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")
for I in [150.0, 200.0, 250.0, 300.0]:
    f, pr, rc = f1(y[(city == 'hongkong') & (i == I)], ens[(city == 'hongkong') & (i == I)])
    print(f"  HK  I={I:5.0f}: F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")