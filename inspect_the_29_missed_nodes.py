"""Inspect the missed critical nodes in HSR Layout at 50 mm/hr.
"""
import torch, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from train_hydro_gine_v5_0 import HydroGINE_v5

ck = torch.load("hydro_gine_v5_model.pt", map_location=device, weights_only=False)
model = HydroGINE_v5(in_c=ck['in_c'], edge_c=2, hidden=ck.get('hidden', 128), n_layers=ck.get('n_layers', 6)).to(device)
model.load_state_dict(ck['model'])
model.eval()

x_mean = ck['x_mean'].to(device)
x_std = ck['x_std'].to(device)
e_mean = ck['e_mean'].to(device)
e_std = ck['e_std'].to(device)
yl_mean = float(ck['yl_mean'])
yl_std = float(ck['yl_std'])

dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
hsr_50 = [g for g in dl if getattr(g, 'region', '') == 'hsr' and abs(g.rain_intensity - 50.0) < 1e-3][0].to(device)

y_true = hsr_50.y.cpu().numpy().ravel()
gx = (hsr_50.x - x_mean) / x_std
gea = (hsr_50.edge_attr - e_mean) / e_std

with torch.no_grad():
    c_l, d_o = model(gx, hsr_50.edge_index, gea)
    raw_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
    raw_depth = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0).cpu().numpy().ravel()

is_missed = (y_true >= 0.30) & (raw_depth < 0.30)
x_raw = hsr_50.x.cpu().numpy()

print(f"Total Missed Critical Nodes: {np.sum(is_missed)}")
print(f"{'Node Idx':<10s} | {'SWMM Depth':<12s} | {'Raw GNN':<10s} | {'Prob':<8s} | {'RelDrop':<8s} | {'InDeg':<6s} | {'OutDeg':<6s} | {'DepDepth':<10s} | {'SagIdx':<8s}")
print("-" * 95)

missed_indices = np.where(is_missed)[0]
for idx in missed_indices[:15]:
    swmm = y_true[idx]
    gnn = raw_depth[idx]
    p = raw_prob[idx]
    rd = x_raw[idx, 0]
    in_d = int(x_raw[idx, 3])
    out_d = int(x_raw[idx, 4])
    dep = x_raw[idx, 16]
    sag = x_raw[idx, 8]
    print(f"{idx:<10d} | {swmm:<12.4f} | {gnn:<10.4f} | {p:<8.4f} | {rd:<8.4f} | {in_d:<6d} | {out_d:<6d} | {dep:<10.4f} | {sag:<8.4f}")
