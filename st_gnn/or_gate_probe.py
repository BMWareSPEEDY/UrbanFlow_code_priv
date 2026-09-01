"""Analyze iter10 HK collapse: check raw depth vs gated depth, and test OR-rule gate."""
import torch
import numpy as np

d = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter10_full.pt", weights_only=False)
pred = d['pred']
prob = d['prob']
depth = d['depth']
y = d['y']
i = d['intensity']
city = d['city']

THR = 0.15
for c in ['bangalore', 'hongkong']:
    m = (city == c) & (i >= 150.0)
    print(f"=== {c} I>=150 (n={m.sum()}) ===")
    print(f"  raw depth:  F1 {2*np.sum((depth[m]>=THR)&(y[m]>=THR)) / max(1e-9, np.sum(depth[m]>=THR)+np.sum(y[m]>=THR)):.4f}")
    print(f"  gated pred: F1 {2*np.sum((pred[m]>=THR)&(y[m]>=THR)) / max(1e-9, np.sum(pred[m]>=THR)+np.sum(y[m]>=THR)):.4f}")
    print(f"  mean prob: {prob[m].mean():.3f} | p>=0.65: {(prob[m]>=0.65).mean()*100:.1f}% | depth>=0.15: {(depth[m]>=0.15).mean()*100:.1f}% | y>=0.15: {(y[m]>=0.15).mean()*100:.1f}%")

# OR-rule: keep depth if p>=0.65 OR depth>=0.3 (would need train-tuning; just probe HK sensitivity)
m = (city == 'hongkong') & (i >= 150.0)
for d_t in [0.2, 0.3, 0.4, 0.5]:
    or_pred = np.where((prob[m] >= 0.65) | (depth[m] >= d_t), depth[m], 0.0)
    tp = np.sum((or_pred >= THR) & (y[m] >= THR)); fp = np.sum((or_pred >= THR) & (y[m] < THR))
    fn = np.sum((or_pred < THR) & (y[m] >= THR))
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    f = 2 * pr * rc / max(1e-9, pr + rc)
    print(f"HK OR-rule d_t={d_t:.1f}: F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")