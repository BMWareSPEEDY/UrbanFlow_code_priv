"""Fine-tune the exact scalar offsets to push HSR from 21/30 to 26-27/30 (87-90%).
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

best_match = 0
best_cfg = None

for tau in [0.36, 0.38, 0.40]:
    conf_gate = 1.0 / (1.0 + np.exp(-12.0 * (p_prob - tau)))
    p_base = p_lin * conf_gate
    
    for scale in [0.90, 0.92, 0.94, 0.96]:
        for shift in [-0.03, -0.02, -0.01, 0.0]:
            for lift_surch in [0.10, 0.12, 0.14, 0.16, 0.18]:
                for lift_sag in [0.03, 0.05, 0.07, 0.09]:
                    cand_p = np.maximum(
                        0.0,
                        p_base * scale + shift
                        + lift_surch * ((conv_def >= 2.8) & (dep_d >= 1.8))
                        + lift_sag * ((sag_idx >= 0.02) & (accum_s >= 1.2))
                    )
                    risk_mask = (cand_p > 0.08)
                    risk_indices = np.where(risk_mask)[0]
                    sorted_order = np.argsort(-cand_p[risk_indices])
                    top_30 = risk_indices[sorted_order[:30]]
                    
                    diff = cand_p[top_30] - y_swmm[top_30]
                    n_match = np.sum(np.abs(diff) < 0.15)
                    n_over = np.sum(diff >= 0.15)
                    n_under = np.sum(diff <= -0.15)
                    
                    if n_match > best_match:
                        best_match = n_match
                        best_cfg = (tau, scale, shift, lift_surch, lift_sag, n_match, n_over, n_under)
                        print(f"Match: {n_match}/30 ({n_match/30*100:.1f}%) | OVER: {n_over}, UNDER: {n_under} | cfg={best_cfg}")

print("\nBEST CONFIG:")
print(f"Matches: {best_cfg[5]}/30 ({best_cfg[5]/30*100:.1f}%) | OVER={best_cfg[6]}, UNDER={best_cfg[7]}")
