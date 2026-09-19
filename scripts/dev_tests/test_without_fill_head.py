"""Test HSR match rate when fill_head override is removed.
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

# Test raw neural predictions with simple confidence gating
for tau_val in [0.30, 0.35, 0.40, 0.45]:
    conf_gate = 1.0 / (1.0 + np.exp(-12.0 * (p_prob - tau_val)))
    p_final = p_lin * conf_gate
    
    risk_mask = (p_final > 0.08)
    risk_indices = np.where(risk_mask)[0]
    sorted_order = np.argsort(-p_final[risk_indices])
    top_30 = risk_indices[sorted_order[:30]]
    
    diff = p_final[top_30] - y_swmm[top_30]
    n_match = np.sum(np.abs(diff) < 0.15)
    n_over = np.sum(diff >= 0.15)
    n_under = np.sum(diff <= -0.15)
    print(f"Tau={tau_val:.2f} | Matches: {n_match:2d}/30 ({n_match/30*100:.1f}%) | OVER: {n_over:2d} | UNDER: {n_under:2d}")
