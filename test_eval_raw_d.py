"""Test raw unnormalized depth regression distribution on the dataset.
"""
import torch, numpy as np

dataset = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
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

diffs_all = []
within_15_all = []

for g in dataset[:32]:
    with torch.no_grad():
        xn = (g.x - x_mean) / x_std
        ean = (g.edge_attr - e_mean) / e_std
        c_l, d_o = model(xn, g.edge_index, ean)
        
        p_d = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0, max=3.0).numpy()
        y = g.y.numpy()
        
        diff = p_d - y
        diffs_all.extend(diff.tolist())
        within_15_all.extend((np.abs(diff) < 0.15).tolist())

diffs_all = np.array(diffs_all)
within_15_all = np.array(within_15_all)

print(f"Total nodes evaluated: {len(diffs_all)}")
print(f"Overall Accuracy within +-15cm: {np.mean(within_15_all)*100:.2f}%")
print(f"Mean error: {np.mean(diffs_all):+.4f}m | Median error: {np.median(diffs_all):+.4f}m | Std: {np.std(diffs_all):.4f}m")
print(f"MAE: {np.mean(np.abs(diffs_all))*100:.2f} cm")
