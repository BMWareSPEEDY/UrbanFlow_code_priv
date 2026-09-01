"""Iteration 3 (CPU gate tuning + 1 GPU forward): per-intensity gate thresholds tuned on
train cities only, applied to Bangalore. No Bangalore labels used."""
import sys
import time
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
    depths, probs, ys, ints = [], [], [], []
    for i in range(0, len(graphs), batch_size):
        b = Batch.from_data_list(graphs[i:i + batch_size]).to(device)
        with torch.no_grad():
            d, c = model((b.x - ck['x_mean'].to(device)) / ck['x_std'].to(device), b.edge_index,
                         (b.edge_attr - ck['e_mean'].to(device)) / ck['e_std'].to(device))
        depths.append(np.clip(np.expm1(d.cpu().numpy() * ck['yl_std'].item() + ck['yl_mean'].item()), 0, 3.0).ravel())
        probs.append(torch.sigmoid(c).cpu().numpy().ravel())
        ys.append(b.y.cpu().numpy().ravel())
        ints.append(np.concatenate([np.full(g.y.shape[0], float(g.rain_intensity)) for g in graphs[i:i + batch_size]]))
    return (np.concatenate(a) for a in (depths, probs, ys, ints))


print("forwarding train...", flush=True)
tr_d, tr_p, tr_y, tr_i = predict(tr_g)
print("forwarding test...", flush=True)
te_d, te_p, te_y, te_i = predict(te_g)

THR = 0.15
GLOBAL_TARGET = [150.0, 200.0, 250.0, 300.0]

# --- per-intensity gate tuning on train ---
per_int_tau = {}
for I in np.unique(tr_i):
    m = tr_i == I
    if m.sum() < 500:
        continue
    best_t, best_f = 0.5, -1
    for t in np.arange(0.10, 0.95, 0.05):
        g = tr_d[m].copy(); g[tr_p[m] < t] = 0.0
        f = f1_score(tr_y[m] > THR, g > THR)
        if f > best_f:
            best_f, best_t = f, t
    per_int_tau[float(I)] = best_t
    print(f"train I={I:5.1f}: best tau {best_t:.2f} (train F1 {best_f:.4f})", flush=True)

# --- global tau on train for comparison ---
best_gt, best_gf = 0.5, -1
for t in np.arange(0.10, 0.95, 0.05):
    g = tr_d.copy(); g[tr_p < t] = 0.0
    f = f1_score(tr_y > THR, g > THR)
    if f > best_gf:
        best_gf, best_gt = f, t
print(f"train global tau {best_gt:.2f} (F1 {best_gf:.4f})", flush=True)

# --- apply per-intensity taus to Bangalore ---
te_final = te_d.copy()
for I in np.unique(te_i):
    m = te_i == I
    if m.sum() < 10:
        continue
    tau = per_int_tau.get(float(I), best_gt)
    te_final[m] = np.where(te_p[m] < tau, 0.0, te_final[m])
torch.save({'pred': te_final}, BASE + r"\st_gnn\preds_iter3.pt")

# --- report I>=150 F1 ---
for I in GLOBAL_TARGET:
    m = te_i == I
    f = f1_score(te_y[m] > THR, te_final[m] > THR)
    print(f"BLR I={I:5.1f}: F1 {f:.4f} (tau {per_int_tau.get(I, best_gt):.2f})", flush=True)
m = np.isin(te_i, GLOBAL_TARGET)
f = f1_score(te_y[m] > THR, te_final[m] > THR)
print(f"BLR pooled I>=150: F1 {f:.4f}", flush=True)
print(f"done {time.time()-0:.0f}s", flush=True)