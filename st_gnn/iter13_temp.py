"""iter13 pooled metrics + temperature-scaling probe (post-hoc calibration)."""
import torch
import numpy as np

THR = 0.15
d = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter13_full.pt", weights_only=False)
pred, y, i, reg, city, p = d['pred'], d['y'], d['intensity'], d['region'], d['city'], d['prob']
m_i = i >= 150.0


def f1(yv, pv):
    tp = np.sum((pv >= THR) & (yv >= THR)); fp = np.sum((pv >= THR) & (yv < THR)); fn = np.sum((pv < THR) & (yv >= THR))
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc), pr, rc


for c in ['bangalore', 'hongkong']:
    m = (city == c) & m_i
    f, pr, rc = f1(y[m], pred[m])
    print(f"{c:<10} F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")
f, pr, rc = f1(y[m_i], pred[m_i])
print(f"POOLED      F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")

# temperature scaling: T learned on TRAIN? we only have test preds here — probe HK/BLR sensitivity
for T in [0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
    pT = 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p))) / T))
    g = pred.copy()
    for I in np.unique(i):
        for cl in [0, 1]:
            m = (i == I) & ((d['elev2'] > 1.3) == (cl == 1))
            t = 0.4
            g[m] = np.where(pT[m] < t, 0.0, g[m])
    fb, pb, rb = f1(y[(city == 'bangalore') & m_i], g[(city == 'bangalore') & m_i])
    fh, ph, rh = f1(y[(city == 'hongkong') & m_i], g[(city == 'hongkong') & m_i])
    print(f"T={T:.1f}: BLR {fb:.4f} ({pb:.2f}/{rb:.2f}) | HK {fh:.4f} ({ph:.2f}/{rh:.2f})")