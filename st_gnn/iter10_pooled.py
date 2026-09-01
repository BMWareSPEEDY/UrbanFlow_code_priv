import torch
import numpy as np

d = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter10_full.pt", weights_only=False)
pred, y, i, reg = d['pred'], d['y'], d['intensity'], d['region']
THR = 0.15

m_all = (i >= 150.0) & (reg != 'hongkong')
for label, m in [("BLR pooled", m_all)]:
    tp = np.sum((pred[m] >= THR) & (y[m] >= THR)); fp = np.sum((pred[m] >= THR) & (y[m] < THR))
    fn = np.sum((pred[m] < THR) & (y[m] >= THR))
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    f = 2 * pr * rc / max(1e-9, pr + rc)
    print(f"{label}: F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")
for I in [150.0, 200.0, 250.0, 300.0]:
    m = (i == I) & (reg != 'hongkong')
    tp = np.sum((pred[m] >= THR) & (y[m] >= THR)); fp = np.sum((pred[m] >= THR) & (y[m] < THR))
    fn = np.sum((pred[m] < THR) & (y[m] >= THR))
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    f = 2 * pr * rc / max(1e-9, pr + rc)
    print(f"  I={I:5.0f}: F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")