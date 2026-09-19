"""Inspect remaining 8 mismatches in HSR when Tau=0.35.
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

conf_gate = 1.0 / (1.0 + np.exp(-12.0 * (p_prob - 0.35)))
p_final = p_lin * conf_gate

risk_mask = (p_final > 0.08)
risk_indices = np.where(risk_mask)[0]
sorted_order = np.argsort(-p_final[risk_indices])
top_30 = risk_indices[sorted_order[:30]]

print(f"{'Rank':<5s} | {'Node ID':<12s} | {'GNN (m)':<8s} | {'SWMM (m)':<8s} | {'Diff (cm)':<10s} | {'Status':<8s} | {'Prob':<6s} | {'Elev':<6s} | {'Dep':<6s} | {'ConvDef'}")
print("-" * 105)

for rank, idx in enumerate(top_30, 1):
    nid = node_list[idx]
    p = p_final[idx]
    s = y_swmm[idx]
    diff = p - s
    diff_cm = diff * 100.0
    status = 'MATCH' if abs(diff) < 0.15 else ('OVER' if diff > 0 else 'UNDER')
    prob = p_prob[idx]
    elev = node_pos[nid]['elevation']
    dep = g.x[idx, 16].item()
    cdef = g.x[idx, 30].item()
    if status != 'MATCH':
        flag = "[O]" if status == 'OVER' else "[U]"
        print(f"{flag} {rank:<3d} | {str(nid):<12s} | {p:<8.3f} | {s:<8.3f} | {diff_cm:<+10.1f} | {status:<8s} | {prob:<6.2f} | {elev:<6.1f} | {dep:<6.2f} | {cdef:.2f}")
