"""iter12 detailed: pooled + per-intensity + per-region."""
import torch
import numpy as np

d = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter12_full.pt", weights_only=False)
pred, y, i, reg, city = d['pred'], d['y'], d['intensity'], d['region'], d['city']
THR = 0.15


def f1(yv, p):
    tp = np.sum((p >= THR) & (yv >= THR)); fp = np.sum((p >= THR) & (yv < THR)); fn = np.sum((p < THR) & (yv >= THR))
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc), pr, rc


m_i = i >= 150.0
f, pr, rc = f1(y[m_i], pred[m_i])
print(f"POOLED BLR+HK I>=150: F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")
for c in ['bangalore', 'hongkong']:
    m = (city == c) & m_i
    f, pr, rc = f1(y[m], pred[m])
    print(f"{c:<10} I>=150 F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")
for rg in ['bellandur', 'ecity', 'koramangala', 'whitefield', 'hsr', 'hongkong']:
    m = (reg == rg) & m_i
    f, pr, rc = f1(y[m], pred[m])
    print(f"  {rg:<12} F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")
print()
for c in ['bangalore', 'hongkong']:
    for I in [150.0, 200.0, 250.0, 300.0]:
        m = (city == c) & (i == I)
        f, pr, rc = f1(y[m], pred[m])
        print(f"  {c[:3]} I={I:5.0f}: F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")