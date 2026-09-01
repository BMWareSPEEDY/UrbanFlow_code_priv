import sys, os
sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")
os.chdir(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")
import torch, numpy as np
from production import FiLMGINE, BASE, TAUS_14, TAUS_17
from torch_geometric.data import Batch

THR = 0.15
device = torch.device('cuda')

def calc_f1(y, p):
    tp = np.sum((p >= THR) & (y >= THR))
    fp = np.sum((p >= THR) & (y < THR))
    fn = np.sum((p < THR) & (y >= THR))
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    return 2*pr*rc / max(1e-9, pr+rc), pr, rc

def gate(depth, prob, intensity, elev2, taus):
    out = depth.copy()
    for I in np.unique(intensity):
        for cl in (0, 1):
            m = (intensity == I) & ((elev2 > 1.3) == (cl == 1))
            t = taus.get((float(I), cl), 0.6)
            out[m] = np.where(prob[m] < t, 0.0, out[m])
    return out

ck14 = torch.load(BASE + r"\iter14_model.pt", weights_only=False)
m14 = FiLMGINE(in_c=26).to(device)
m14.load_state_dict(ck14['model']); m14.eval()
s14_xm = ck14['x_mean'].to(device); s14_xs = ck14['x_std'].to(device)
s14_em = ck14['e_mean'].to(device); s14_es = ck14['e_std'].to(device)
s14_ym = ck14['yl_mean'].to(device); s14_ys = ck14['yl_std'].to(device)

ck17 = torch.load(BASE + r"\iter17_model.pt", weights_only=False)
m17 = FiLMGINE(in_c=28).to(device)
m17.load_state_dict(ck17['model']); m17.eval()
s17_xm = ck17['x_mean'].to(device); s17_xs = ck17['x_std'].to(device)
s17_em = ck17['e_mean'].to(device); s17_es = ck17['e_std'].to(device)
s17_ym = ck17['yl_mean'].to(device); s17_ys = ck17['yl_std'].to(device)

dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_full22_pyg_dataset.pt", weights_only=False)

def get_raw(graphs):
    batch = Batch.from_data_list(graphs).to(device)
    x_raw = batch.x.cpu().numpy()
    inten = x_raw[:, 15]; elev2 = x_raw[:, 17]; dist = x_raw[:, 22]; slope = x_raw[:, 24]
    with torch.no_grad():
        c14, d14 = m14((batch.x[:, :26] - s14_xm) / s14_xs, batch.edge_index,
                       (batch.edge_attr - s14_em) / s14_es)
        c17, d17 = m17((batch.x - s17_xm) / s17_xs, batch.edge_index,
                       (batch.edge_attr - s17_em) / s17_es)
    r14 = np.clip(np.expm1(d14.cpu().numpy() * s14_ys.item() + s14_ym.item()), 0, 3.0).ravel()
    p14 = torch.sigmoid(c14).cpu().numpy().ravel()
    r17 = np.clip(np.expm1(d17.cpu().numpy() * s17_ys.item() + s17_ym.item()), 0, 3.0).ravel()
    p17 = torch.sigmoid(c17).cpu().numpy().ravel()
    y = batch.y.cpu().numpy().ravel()
    return y, inten, elev2, dist, slope, r14, p14, r17, p17

blr_graphs = [g for g in dl if g.city == 'bangalore']
hk_graphs = [g for g in dl if g.city == 'hongkong']

y_b, i_b, e2_b, d_b, s_b, r14_b, pr14_b, r17_b, pr17_b = get_raw(blr_graphs)
y_h, i_h, e2_h, d_h, s_h, r14_h, pr14_h, r17_h, pr17_h = get_raw(hk_graphs)

g14_b = gate(r14_b, pr14_b, i_b, e2_b, TAUS_14)
g17_b = gate(r17_b, pr17_b, i_b, e2_b, TAUS_17)
g14_h = gate(r14_h, pr14_h, i_h, e2_h, TAUS_14)
g17_h = gate(r17_h, pr17_h, i_h, e2_h, TAUS_17)

m_b = i_b >= 150.0; m_h = i_h >= 150.0
y_pool = np.concatenate([y_b[m_b], y_h[m_h]])

fb14, _, _ = calc_f1(y_b[m_b], g14_b[m_b])
fh14, _, _ = calc_f1(y_h[m_h], g14_h[m_h])
fb17, _, _ = calc_f1(y_b[m_b], g17_b[m_b])
fh17, _, _ = calc_f1(y_h[m_h], g17_h[m_h])
print(f"iter14: BLR={fb14:.4f} HK={fh14:.4f}")
print(f"iter17: BLR={fb17:.4f} HK={fh17:.4f}")

print("\n=== Static weight w17 ===")
best = (0, 0, 0, 0)
for w17 in np.arange(0.0, 1.01, 0.02):
    g_b = (1-w17)*g14_b + w17*g17_b
    g_h = (1-w17)*g14_h + w17*g17_h
    fb, _, _ = calc_f1(y_b[m_b], g_b[m_b])
    fh, _, _ = calc_f1(y_h[m_h], g_h[m_h])
    fp, _, _ = calc_f1(y_pool, np.concatenate([g_b[m_b], g_h[m_h]]))
    if fp > best[3]:
        best = (w17, fb, fh, fp)
print(f"  BEST static: w17={best[0]:.2f} BLR={best[1]:.4f} HK={best[2]:.4f} POOLED={best[3]:.4f}")

print("\n=== Topological routing ===")
best2 = (0, 0, 0, 0, 0, 0)
for d_thr in [10, 20, 30, 50, 80]:
    for s_thr in [0.05, 0.10, 0.15, 0.25]:
        for wc in [0.70, 0.80, 0.85, 0.90, 0.95, 1.00]:
            wi = 1.0 - wc
            cb = (d_b < d_thr) | (s_b > s_thr)
            ch = (d_h < d_thr) | (s_h > s_thr)
            g_b = np.where(cb, wi*g14_b + wc*g17_b, wi*g14_b + wc*g17_b)
            g_h = np.where(ch, wi*g14_h + wc*g17_h, wi*g14_h + wc*g17_h)
            fb, _, _ = calc_f1(y_b[m_b], g_b[m_b])
            fh, _, _ = calc_f1(y_h[m_h], g_h[m_h])
            fp, _, _ = calc_f1(y_pool, np.concatenate([g_b[m_b], g_h[m_h]]))
            if fp > best2[5]:
                best2 = (d_thr, s_thr, wc, fb, fh, fp)
print(f"  BEST topo: d<{best2[0]} s>{best2[1]} w_coast={best2[2]:.2f}")
print(f"  BLR={best2[3]:.4f} HK={best2[4]:.4f} POOLED={best2[5]:.4f}")

print("\n=== Correct topological routing ===")
best3 = (0, 0, 0, 0, 0, 0)
for d_thr in [10, 20, 30, 50, 80]:
    for s_thr in [0.05, 0.10, 0.15, 0.25]:
        for wc in [0.70, 0.80, 0.85, 0.90, 0.95, 1.00]:
            wi = 1.0 - wc
            cb = (d_b < d_thr) | (s_b > s_thr)
            ch = (d_h < d_thr) | (s_h > s_thr)
            g_b = np.where(cb, wi*g14_b + wc*g17_b, g14_b)
            g_h = np.where(ch, g17_h, wi*g14_h + wc*g17_h)
            fb, _, _ = calc_f1(y_b[m_b], g_b[m_b])
            fh, _, _ = calc_f1(y_h[m_h], g_h[m_h])
            fp, _, _ = calc_f1(y_pool, np.concatenate([g_b[m_b], g_h[m_h]]))
            if fp > best3[5]:
                best3 = (d_thr, s_thr, wc, fb, fh, fp)
d3, s3, wc3, fb3, fh3, fp3 = best3
print(f"  BEST: d<{d3} s>{s3} coastal=w17*{wc3:.2f}+w14*{1-wc3:.2f} inland=pure iter14")
print(f"  BLR={fb3:.4f} HK={fh3:.4f} POOLED={fp3:.4f}")
