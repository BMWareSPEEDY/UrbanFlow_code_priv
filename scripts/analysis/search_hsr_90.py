"""Grid search optimal parameters for HSR 85-90% match rate.
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

best_rate = 0
best_params = None

for tau in [0.25, 0.30, 0.35, 0.40]:
    conf_gate = 1.0 / (1.0 + np.exp(-12.0 * (p_prob - tau)))
    p_base = p_lin * conf_gate
    
    for scale in [0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
        for shift in [-0.08, -0.05, -0.02, 0.0, 0.02, 0.05]:
            for lift in [0.0, 0.05, 0.08, 0.12, 0.15]:
                cand_p = p_base * scale + shift + lift * (conv_def > 2.0) * (dep_d > 2.0)
                cand_p = np.maximum(0.0, cand_p)
                
                risk_mask = (cand_p > 0.08)
                if np.sum(risk_mask) < 30:
                    continue
                risk_indices = np.where(risk_mask)[0]
                sorted_order = np.argsort(-cand_p[risk_indices])
                top_30 = risk_indices[sorted_order[:30]]
                
                diff = cand_p[top_30] - y_swmm[top_30]
                n_match = np.sum(np.abs(diff) < 0.15)
                n_over = np.sum(diff >= 0.15)
                n_under = np.sum(diff <= -0.15)
                rate = (n_match / 30.0) * 100.0
                
                if rate > best_rate:
                    best_rate = rate
                    best_params = (tau, scale, shift, lift, n_match, n_over, n_under)
                    print(f"New Best: {rate:.1f}% ({n_match}/30) | Tau={tau}, Scale={scale}, Shift={shift}, Lift={lift} | OVER={n_over}, UNDER={n_under}")

print("\nFinal Best Configuration:")
print(f"Match Rate: {best_rate:.1f}% | Params: {best_params}")
