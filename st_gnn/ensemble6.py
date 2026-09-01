"""Ensemble iter2 + iter6 models: avg depth + avg prob, gate with per-(I,cluster) taus
tuned on TRAIN ensemble. No Bangalore labels."""
import sys
import torch
import numpy as np
from torch_geometric.data import Batch

sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW")
sys.stdout.reconfigure(line_buffering=True)
BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"

device = torch.device('cuda')
from train_iter2 import TwoHeadGINE

dl = torch.load(BASE + r"\multi_scenario_pyg_dataset.pt", weights_only=False)
tr_g = [g.clone() for g in dl if g.city != 'bangalore']
te_g = [g.clone() for g in dl if g.city == 'bangalore']


def load(name):
    ck = torch.load(BASE + rf"\st_gnn\{name}_model.pt", weights_only=False)
    m = TwoHeadGINE(in_c=22).to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    return m, ck


m2, ck2 = load("iter2")
m6, ck6 = load("iter6")


def predict(models, graphs, batch_size=4):
    D, P = [], []
    for i in range(0, len(graphs), batch_size):
        sub = graphs[i:i + batch_size]
        b = Batch.from_data_list(sub).to(device)
        ds, ps = [], []
        for m, ck in models:
            with torch.no_grad():
                d, c = m((b.x - ck['x_mean'].to(device)) / ck['x_std'].to(device), b.edge_index,
                         (b.edge_attr - ck['e_mean'].to(device)) / ck['e_std'].to(device))
            ds.append(np.clip(np.expm1(d.cpu().numpy() * ck['yl_std'].item() + ck['yl_mean'].item()), 0, 3.0).ravel())
            ps.append(torch.sigmoid(c).cpu().numpy().ravel())
        D.append(np.mean(ds, 0))
        P.append(np.mean(ps, 0))
    return np.concatenate(D), np.concatenate(P)


print("forward train...", flush=True)
tr_d, tr_p = predict([(m2, ck2), (m6, ck6)], tr_g)
tr_y = np.concatenate([g.y.numpy().ravel() for g in tr_g])
tr_i = np.concatenate([g.x.numpy()[:, 15] for g in tr_g])
tr_e2 = np.concatenate([g.x.numpy()[:, 17] for g in tr_g])
print("forward test...", flush=True)
te_d, te_p = predict([(m2, ck2), (m6, ck6)], te_g)
te_y = np.concatenate([g.y.numpy().ravel() for g in te_g])
te_i = np.concatenate([g.x.numpy()[:, 15] for g in te_g])
te_e2 = np.concatenate([g.x.numpy()[:, 17] for g in te_g])
te_reg = np.concatenate([[g.region] * g.y.shape[0] for g in te_g])

THR = 0.15


def f1(y, p):
    tp = np.sum((p >= THR) & (y >= THR)); fp = np.sum((p >= THR) & (y < THR)); fn = np.sum((p < THR) & (y >= THR))
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc), pr, rc


tau = {}
for I in np.unique(tr_i):
    for cl in [0, 1]:
        m = (tr_i == I) & ((tr_e2 > 1.3) == (cl == 1))
        if m.sum() < 2000:
            continue
        best_t, best_f = 0.5, -1
        for t in np.arange(0.05, 0.95, 0.05):
            g = tr_d[m].copy(); g[tr_p[m] < t] = 0.0
            f, _, _ = f1(tr_y[m], g)
            if f > best_f:
                best_f, best_t = f, t
        tau[(float(I), cl)] = best_t
        print(f"train I={I:5.0f} cl={cl}: tau {best_t:.2f} F1 {best_f:.4f}", flush=True)

te_final = te_d.copy()
for I in np.unique(te_i):
    for cl in [0, 1]:
        m = (te_i == I) & ((te_e2 > 1.3) == (cl == 1))
        if m.sum() < 10:
            continue
        t = tau.get((float(I), cl), 0.6)
        te_final[m] = np.where(te_p[m] < t, 0.0, te_final[m])

torch.save({'pred': te_final}, BASE + r"\st_gnn\preds_ens6.pt")
m_all = te_i >= 150.0
f, pr, rc = f1(te_y[m_all], te_final[m_all])
print(f"\nBLR pooled I>=150 ensemble F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")
for rg in sorted(set(te_reg)):
    m = m_all & (te_reg == rg)
    f_, pr_, rc_ = f1(te_y[m], te_final[m])
    print(f"  {rg:<13} F1 {f_:.4f} (P {pr_:.3f} R {rc_:.3f})")
for I in [150.0, 200.0, 250.0, 300.0]:
    m = te_i == I
    f_, _, _ = f1(te_y[m], te_final[m])
    print(f"  I={I:5.0f}: F1 {f_:.4f}")