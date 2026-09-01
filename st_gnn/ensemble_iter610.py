"""Ensemble iter6 + iter10 on full held-out test (bangalore + hongkong)."""
import sys
import torch
import numpy as np
from torch_geometric.data import Batch

sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW")
BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"
device = torch.device('cuda')
from train_iter2 import TwoHeadGINE

dl = torch.load(BASE + r"\multi_scenario_full22_pyg_dataset.pt", weights_only=False)
te_g = [g.clone() for g in dl if g.city in ('bangalore', 'hongkong')]

TAUS = {(20.0, 0): 0.45, (20.0, 1): 0.25, (50.0, 0): 0.4, (50.0, 1): 0.45,
        (80.0, 0): 0.6, (80.0, 1): 0.55, (120.0, 0): 0.6, (120.0, 1): 0.6,
        (150.0, 0): 0.6, (150.0, 1): 0.65, (200.0, 0): 0.65, (200.0, 1): 0.65,
        (250.0, 0): 0.6, (250.0, 1): 0.6, (300.0, 0): 0.6, (300.0, 1): 0.65}
THR = 0.15


def f1(y, p):
    tp = np.sum((p >= THR) & (y >= THR)); fp = np.sum((p >= THR) & (y < THR)); fn = np.sum((p < THR) & (y >= THR))
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc), pr, rc


def load(name):
    ck = torch.load(BASE + rf"\st_gnn\{name}_model.pt", weights_only=False)
    m = TwoHeadGINE(in_c=22).to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    return m, ck


models = [load("iter6"), load("iter10")]

D, P = [], []
for i in range(0, len(te_g), 2):
    b = Batch.from_data_list(te_g[i:i + 2]).to(device)
    ds, ps = [], []
    for m, ck in models:
        with torch.no_grad():
            d, c = m((b.x - ck['x_mean'].to(device)) / ck['x_std'].to(device), b.edge_index,
                     (b.edge_attr - ck['e_mean'].to(device)) / ck['e_std'].to(device))
        ds.append(np.clip(np.expm1(d.cpu().numpy() * ck['yl_std'].item() + ck['yl_mean'].item()), 0, 3.0).ravel())
        ps.append(torch.sigmoid(c).cpu().numpy().ravel())
    D.append(np.mean(ds, 0))
    P.append(np.mean(ps, 0))
d_all = np.concatenate(D)
p_all = np.concatenate(P)
y_all = np.concatenate([g.y.numpy().ravel() for g in te_g])
i_all = np.concatenate([g.x.numpy()[:, 15] for g in te_g])
e2_all = np.concatenate([g.x.numpy()[:, 17] for g in te_g])
reg_all = np.concatenate([[g.region] * g.y.shape[0] for g in te_g])
city_all = np.concatenate([[g.city] * g.y.shape[0] for g in te_g])

pred = d_all.copy()
for I in np.unique(i_all):
    for cl in [0, 1]:
        m = (i_all == I) & ((e2_all > 1.3) == (cl == 1))
        t = TAUS.get((float(I), cl), 0.6)
        pred[m] = np.where(p_all[m] < t, 0.0, pred[m])

m_i = i_all >= 150.0
f, pr, rc = f1(y_all[m_i], pred[m_i])
print(f"POOLED I>=150: F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")
for city in ['bangalore', 'hongkong']:
    mc = city_all == city
    f, pr, rc = f1(y_all[mc & m_i], pred[mc & m_i])
    print(f"  {city:<10} I>=150 F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")
for rg in ['bellandur', 'ecity', 'koramangala', 'whitefield', 'hsr']:
    m = (reg_all == rg) & m_i
    f, pr, rc = f1(y_all[m], pred[m])
    print(f"  {rg:<12} F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")