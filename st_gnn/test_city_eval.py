"""Evaluate iter2+iter6 ensemble on held-out real-terrain test cities (nyc, london).
Gates: train-tuned iter6 per-(intensity, cluster) taus, hardcoded. No test labels used."""
import sys
import torch
import numpy as np
from torch_geometric.data import Batch

sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW")
sys.stdout.reconfigure(line_buffering=True)
BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"

device = torch.device('cuda')
from train_iter2 import TwoHeadGINE

dl = torch.load(BASE + r"\multi_scenario_testcities_pyg_dataset.pt", weights_only=False)

TAUS = {  # iter6 train-tuned per (intensity, cluster): cluster = elev_std2 > 1.3
    (-1.0, 0): 0.6, (-1.0, 1): 0.45, (-0.0, 0): 0.6, (-0.0, 1): 0.55,
    (0.0, 0): 0.6, (0.0, 1): 0.55, (1.0, 0): 0.6, (1.0, 1): 0.55,
    (2.0, 0): 0.6, (2.0, 1): 0.55,
}
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


models = [load("iter2"), load("iter6")]


def predict(graphs, batch_size=4):
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


d_all, p_all = predict(dl)
y_all = np.concatenate([g.y.numpy().ravel() for g in dl])
i_all = np.concatenate([g.x.numpy()[:, 15] for g in dl])
e2_all = np.concatenate([g.x.numpy()[:, 17] for g in dl])
city_all = np.concatenate([[g.city] * g.y.shape[0] for g in dl])

pred = d_all.copy()
for I in np.unique(i_all):
    for cl in [0, 1]:
        m = (i_all == I) & ((e2_all > 1.3) == (cl == 1))
        t = TAUS.get((float(I), cl), 0.6)
        pred[m] = np.where(p_all[m] < t, 0.0, pred[m])

for city in ['newyork', 'london']:
    m_c = city_all == city
    y, p, i = y_all[m_c], pred[m_c], i_all[m_c]
    print(f"\n=== {city.upper()} ({m_c.sum()} nodes) ===")
    f, pr, rc = f1(y, p)
    mae = np.abs(p - y).mean()
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"OVERALL: MAE {mae:.4f} | R2 {r2:.3f} | F1@0.15 {f:.4f} (P {pr:.3f} R {rc:.3f})")
    m_i = i >= 150.0
    f, pr, rc = f1(y[m_i], p[m_i])
    print(f"I>=150:  F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")
    for I in [20.0, 50.0, 80.0, 120.0, 150.0, 200.0, 250.0, 300.0]:
        m = i == I
        f_, pr_, rc_ = f1(y[m], p[m])
        print(f"  I={I:5.0f}: F1 {f_:.4f} (P {pr_:.3f} R {rc_:.3f}) | dry% {(y[m]<0.15).mean()*100:5.1f} | mean_y {y[m].mean():.3f} | mean_pred {p[m].mean():.3f}")

print("\n=== terrain/feature check ===")
for city in ['newyork', 'london']:
    m_c = city_all == city
    x = np.concatenate([g.x.numpy() for g in dl])[m_c]
    print(f"{city:<10} elev_std2 mean {x[:,17].mean():.2f} (max {x[:,17].max():.2f}) | imp mean {x[:,3].mean():.3f} | "
          f"log_imp_area mean {x[:,13].mean():.2f} | sink% {x[:,8].mean()*100:.1f} | grade_e mean {np.abs(x[:,9]).mean():.4f}")