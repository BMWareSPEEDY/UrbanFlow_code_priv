"""Bias-correction experiment: fit a residual-bias model on the 23 TRAIN cities' errors
(LOOCV-validated), apply to Bangalore. No Bangalore labels used for fitting.
Baseline model = blr_gine6_final.pt (clipped at 3.0m)."""
import sys
import time
import torch
import numpy as np
from torch_geometric.data import Batch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW")
sys.stdout.reconfigure(line_buffering=True)
BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"

device = torch.device('cuda')
ck = torch.load(BASE + r"\blr_gine6_final.pt", weights_only=False)
from blr_gine6_test import GINE4

model = GINE4(in_c=22).to(device)
model.load_state_dict(ck['model'])
model.eval()

dl = torch.load(BASE + r"\multi_scenario_pyg_dataset.pt", weights_only=False)
tr_g = [g.clone() for g in dl if g.city != 'bangalore']
te_g = [g.clone() for g in dl if g.city == 'bangalore']

# --- forward pass on all graphs ---
def predict(graphs, batch_size=4):
    preds = []
    for i in range(0, len(graphs), batch_size):
        b = Batch.from_data_list(graphs[i:i + batch_size]).to(device)
        with torch.no_grad():
            out = model((b.x - ck['x_mean'].to(device)) / ck['x_std'].to(device), b.edge_index,
                        (b.edge_attr - ck['e_mean'].to(device)) / ck['e_std'].to(device))
        p = np.clip(np.expm1(out.cpu().numpy() * ck['yl_std'].item() + ck['yl_mean'].item()), 0, 3.0).ravel()
        preds.append(p)
    return np.concatenate(preds)

print("forwarding train graphs...", flush=True)
tr_pred = predict(tr_g)
tr_y = np.concatenate([g.y.numpy().ravel() for g in tr_g])
tr_int = np.concatenate([np.full(g.y.shape[0], float(g.rain_intensity)) for g in tr_g])
tr_feat = np.concatenate([g.x.numpy() for g in tr_g])
tr_city = np.concatenate([[g.city] * g.y.shape[0] for g in tr_g])

print("forwarding test graphs...", flush=True)
te_pred = predict(te_g)
te_y = np.concatenate([g.y.numpy().ravel() for g in te_g])
te_int = np.concatenate([np.full(g.y.shape[0], float(g.rain_intensity)) for g in te_g])
te_feat = np.concatenate([g.x.numpy() for g in te_g])

err_tr = tr_y - tr_pred
print(f"train MAE {np.mean(np.abs(err_tr)):.4f} | test MAE {np.mean(np.abs(te_y-te_pred)):.4f}", flush=True)

# --- LOOCV on 23 train cities: does bias correction help held-out cities? ---
cities = sorted(set(tr_city))
imp_loocv, mae_loocv = [], []
for c in cities:
    m_te = tr_city == c
    m_tr = ~m_te
    if m_tr.sum() < 20000:
        continue
    h = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=5, random_state=0)
    h.fit(tr_feat[m_tr], err_tr[m_tr])
    corr = np.clip(tr_pred[m_te] + h.predict(tr_feat[m_te]), 0, 3.0)
    imp_loocv.append(100 * (np.mean(np.abs(err_tr[m_te])) - np.mean(np.abs(tr_y[m_te] - corr))) / max(1e-9, np.mean(np.abs(err_tr[m_te]))))
    mae_loocv.append(np.mean(np.abs(tr_y[m_te] - corr)))
print(f"LOOCV mean MAE improvement: {np.mean(imp_loocv):+.2f}% (range {np.min(imp_loocv):+.2f}..{np.max(imp_loocv):+.2f})", flush=True)

if np.mean(imp_loocv) > 2.0:
    print("applying bias correction to Bangalore...", flush=True)
    h = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=5, random_state=0)
    h.fit(tr_feat, err_tr)
    te_corr = np.clip(te_pred + h.predict(te_feat), 0, 3.0)
    torch.save({'pred': te_corr}, BASE + r"\st_gnn\preds_biasfix.pt")
    print(f"Bangalore corrected MAE {np.mean(np.abs(te_y-te_corr)):.4f} vs {np.mean(np.abs(te_y-te_pred)):.4f}", flush=True)
else:
    print("LOOCV shows no gain; NOT applying.", flush=True)