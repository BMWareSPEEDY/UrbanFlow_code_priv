"""Diagnostic: does hsr's inflated accumulation feature (log_imp_area) drive its
over-prediction? Replace corrupted accumulation features for hsr nodes at EVAL time
and watch hsr precision. Same for other regions = control (should barely change)."""
import sys
import torch
import numpy as np
from torch_geometric.data import Batch

sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW")
BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"
device = torch.device('cuda')
from train_iter2 import TwoHeadGINE

dl = torch.load(BASE + r"\multi_scenario_full22_pyg_dataset.pt", weights_only=False)
te_g = [g.clone() for g in dl if g.city == 'bangalore']

ck = torch.load(BASE + r"\st_gnn\iter6_model.pt", weights_only=False)
m = TwoHeadGINE(in_c=22).to(device)
m.load_state_dict(ck['model'])
m.eval()

TAUS = {(-1.0, 0): 0.6, (-1.0, 1): 0.45, (0.0, 0): 0.6, (0.0, 1): 0.55,
        (1.0, 0): 0.6, (1.0, 1): 0.55, (2.0, 0): 0.6, (2.0, 1): 0.55}
THR = 0.15


def f1(y, p):
    tp = np.sum((p >= THR) & (y >= THR)); fp = np.sum((p >= THR) & (y < THR)); fn = np.sum((p < THR) & (y >= THR))
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc), pr, rc


def run(reg_keep, cap_cols):
    D, P, meta = [], [], []
    for g in te_g:
        x = g.x.clone()
        if g.region == reg_keep:
            for c in cap_cols:
                x[:, c] = torch.clamp(x[:, c], max=cap_cols[c])
        g2 = g.clone()
        g2.x = x
        b = Batch.from_data_list([g2])
        b = b.to(device)
        with torch.no_grad():
            d, c = m((b.x - ck['x_mean'].to(device)) / ck['x_std'].to(device), b.edge_index,
                     (b.edge_attr - ck['e_mean'].to(device)) / ck['e_std'].to(device))
        D.append(np.clip(np.expm1(d.cpu().numpy() * ck['yl_std'].item() + ck['yl_mean'].item()), 0, 3.0).ravel())
        P.append(torch.sigmoid(c).cpu().numpy().ravel())
        meta.append((g2.y.numpy().ravel(), g2.x.numpy()[:, 15], g2.x.numpy()[:, 17], np.array([g2.region] * g2.y.shape[0])))
    d = np.concatenate(D); p = np.concatenate(P)
    y = np.concatenate([mm[0] for mm in meta]); i = np.concatenate([mm[1] for mm in meta])
    e2 = np.concatenate([mm[2] for mm in meta]); reg = np.concatenate([mm[3] for mm in meta])
    pred = d.copy()
    for I in np.unique(i):
        for cl in [0, 1]:
            mm = (i == I) & ((e2 > 1.3) == (cl == 1))
            t = TAUS.get((float(I), cl), 0.6)
            pred[mm] = np.where(p[mm] < t, 0.0, pred[mm])
    return y, pred, i, reg


CAP = {13: np.log1p(3.0), 12: np.log1p(4.0), 19: np.log1p(4.0), 20: 2.0}  # log1p caps

for cap_name, cols in [("none", {}), ("capped accimp/area/surcharge/pathcap", CAP)]:
    y, pred, i, reg = run('hsr', cols)
    mask = (i >= 150.0) & (reg == 'hsr')
    f, pr, rc = f1(y[mask], pred[mask])
    print(f"hsr  {cap_name:<38} F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")
y, pred, i, reg = run('bellandur', CAP)
mask = (i >= 150.0) & (reg == 'bellandur')
f, pr, rc = f1(y[mask], pred[mask])
print(f"bellandur capped (control)                     F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")