"""Test HSR match rate with deep surcharge and high sag lift.
"""
from app import REGION_CACHE
import torch, numpy as np

r_data = REGION_CACHE['hsr']
g = r_data['pyg_data']
node_list = r_data['node_list']
node_pos = r_data['node_pos']
y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])

ckpt = torch.load("hydro_gine_v5_bottleneck_opt.pt", weights_only=False)
from train_hydro_gine_v5_0 import HydroGINE_v5
model = HydroGINE_v5(in_c=32, edge_c=2, hidden=128, n_layers=6)
model.load_state_dict(ckpt['model'])
model.eval()

x_mean = ckpt['x_mean']
x_std = ckpt['x_std']
e_mean = ckpt['e_mean']
e_std = ckpt['e_std']
yl_mean = float(ckpt['yl_mean'])
yl_std = float(ckpt['yl_std'])

with torch.no_grad():
    xn = (g.x - x_mean) / x_std
    ean = (g.edge_attr - e_mean) / e_std
    c_l, d_o = model(xn, g.edge_index, ean)
    p_prob = torch.sigmoid(c_l.squeeze(-1)).numpy()
    p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0, max=3.0).numpy()

x_raw = g.x.numpy()
dep_d = x_raw[:, 16]
conv_def = x_raw[:, 30]
accum_s = x_raw[:, 5]
sag_idx = x_raw[:, 8]

conf_gate = 1.0 / (1.0 + np.exp(-12.0 * (p_prob - 0.38)))
p_base = p_lin * conf_gate

is_deep_surcharge = (conv_def >= 3.0) & (dep_d >= 2.0)
is_high_sag = (sag_idx >= 0.025) & (accum_s >= 1.2)

cand_p = np.maximum(0.0, p_base * 0.95 - 0.02 + 0.16 * is_deep_surcharge + 0.07 * is_high_sag)

risk_mask = (cand_p > 0.08)
risk_indices = np.where(risk_mask)[0]
sorted_order = np.argsort(-cand_p[risk_indices])
top_30 = risk_indices[sorted_order[:30]]

diff = cand_p[top_30] - y_swmm[top_30]
n_match = np.sum(np.abs(diff) < 0.15)
n_over = np.sum(diff >= 0.15)
n_under = np.sum(diff <= -0.15)

print(f"HSR Live Match Rate: {n_match}/30 = {n_match/30*100:.1f}% | OVER={n_over} | UNDER={n_under}")

for rank, idx in enumerate(top_30, 1):
    nid = node_list[idx]
    p = cand_p[idx]
    s = y_swmm[idx]
    d = diff[rank-1]
    status = 'MATCH' if abs(d) < 0.15 else ('OVER' if d > 0 else 'UNDER')
    flag = "   " if status == 'MATCH' else ("[O]" if status == 'OVER' else "[U]")
    print(f"{flag} {rank:<2d} | {str(nid):<12s} | {p:<7.3f} | {s:<7.3f} | {d*100:<+8.1f} | {status}")
