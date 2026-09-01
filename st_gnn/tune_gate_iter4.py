"""Iteration 4: cluster-aware gate calibration + per-region F1 breakdown.
Clusters = flat (elev_std2 < 1.0) vs hilly (>= 1.0) regions. Fit per-intensity per-cluster
gate tau on TRAIN cities only; apply to Bangalore by cluster identity. No Bangalore labels."""
import sys
import torch
import numpy as np
from torch_geometric.data import Batch
from sklearn.metrics import f1_score

sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW")
sys.stdout.reconfigure(line_buffering=True)
BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"

device = torch.device('cuda')
ck = torch.load(BASE + r"\st_gnn\iter2_model.pt", weights_only=False)
from train_iter2 import TwoHeadGINE

model = TwoHeadGINE(in_c=22).to(device)
model.load_state_dict(ck['model'])
model.eval()

dl = torch.load(BASE + r"\multi_scenario_pyg_dataset.pt", weights_only=False)
tr_g = [g.clone() for g in dl if g.city != 'bangalore']
te_g = [g.clone() for g in dl if g.city == 'bangalore']


def predict(graphs, batch_size=4):
    depths, probs, ys, ints, clusters = [], [], [], [], []
    for i in range(0, len(graphs), batch_size):
        sub = graphs[i:i + batch_size]
        b = Batch.from_data_list(sub).to(device)
        with torch.no_grad():
            d, c = model((b.x - ck['x_mean'].to(device)) / ck['x_std'].to(device), b.edge_index,
                         (b.edge_attr - ck['e_mean'].to(device)) / ck['e_std'].to(device))
        depths.append(np.clip(np.expm1(d.cpu().numpy() * ck['yl_std'].item() + ck['yl_mean'].item()), 0, 3.0).ravel())
        probs.append(torch.sigmoid(c).cpu().numpy().ravel())
        ys.append(b.y.cpu().numpy().ravel())
        ints.append(np.concatenate([np.full(g.y.shape[0], float(g.rain_intensity)) for g in sub]))
        cls = []
        for g in sub:
            e2 = g.x.numpy()[:, 17].mean()
            cls.append(np.full(g.y.shape[0], 1 if e2 >= 1.0 else 0))
        clusters.append(np.concatenate(cls))
    return (np.concatenate(a) for a in (depths, probs, ys, ints, clusters))


print("forwarding train...", flush=True)
tr_d, tr_p, tr_y, tr_i, tr_cl = predict(tr_g)
print("forwarding test...", flush=True)
te_d, te_p, te_y, te_i, te_cl = predict(te_g)

THR = 0.15
GLOBAL_TARGET = [150.0, 200.0, 250.0, 300.0]

# --- per-cluster per-intensity gate tuning on train ---
tau = {}
for cl in [0, 1]:
    m = (tr_i >= 150.0) & (tr_cl == cl)
    if m.sum() < 500:
        print(f"train cluster {cl}: NOT ENOUGH DATA (n={m.sum()}) -> fallback to global", flush=True)
        tau[cl] = None
        continue
    best_t, best_f = 0.5, -1
    for t in np.arange(0.10, 0.95, 0.05):
        g = tr_d[m].copy(); g[tr_p[m] < t] = 0.0
        f = f1_score(tr_y[m] > THR, g > THR)
        if f > best_f:
            best_f, best_t = f, t
    tau[cl] = best_t
    print(f"train cluster {cl} (I>=150): tau {best_t:.2f} (F1 {best_f:.4f}, n={m.sum()})", flush=True)
m_glob = tr_i >= 150.0
best_gt, best_gf = 0.5, -1
for t in np.arange(0.10, 0.95, 0.05):
    g = tr_d[m_glob].copy(); g[tr_p[m_glob] < t] = 0.0
    f = f1_score(tr_y[m_glob] > THR, g > THR)
    if f > best_gf:
        best_gf, best_gt = f, t
for cl in [0, 1]:
    if tau[cl] is None:
        tau[cl] = best_gt
print(f"train global tau (I>=150): {best_gt:.2f} (F1 {best_gf:.4f})", flush=True)

# --- per-region Bangalore F1 at I>=150 (diagnosis) ---
te_reg = np.concatenate([[g.region] * g.y.shape[0] for g in te_g])
print("BLR per-region F1@I>=150 (no gate / cluster gate):")
for rg in sorted(set(te_reg)):
    m = (te_i >= 150.0) & (te_reg == rg)
    if m.sum() == 0:
        continue
    cl = 1 if te_cl[m][0] == 1 else 0
    g0 = te_d[m].copy()
    g1 = te_d[m].copy(); g1[te_p[m] < tau[cl]] = 0.0
    print(f"  {rg:<13} n={m.sum():6d} F1_nogate {f1_score(te_y[m]>THR, g0>THR):.4f} "
          f"F1_cluster {f1_score(te_y[m]>THR, g1>THR):.4f}", flush=True)

# --- apply cluster-aware gate to Bangalore, pooled I>=150 ---
te_final = te_d.copy()
for I in GLOBAL_TARGET:
    mI = te_i == I
    for cl in [0, 1]:
        m = mI & (te_cl == cl)
        if m.sum() == 0:
            continue
        te_final[m] = np.where(te_p[m] < tau[cl], 0.0, te_final[m])

m = np.isin(te_i, GLOBAL_TARGET)
f = f1_score(te_y[m] > THR, te_final[m] > THR)
print(f"\nBLR pooled I>=150 cluster-gated F1: {f:.4f}")
for I in GLOBAL_TARGET:
    mI = te_i == I
    f = f1_score(te_y[mI] > THR, te_final[mI] > THR)
    print(f"  I={I:5.1f}: F1 {f:.4f}")
torch.save({'pred': te_final}, BASE + r"\st_gnn\preds_iter4.pt")