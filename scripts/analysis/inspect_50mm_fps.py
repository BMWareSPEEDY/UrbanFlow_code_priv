"""Inspect False Positives at 50 mm/hr in HSR Layout.
"""
import torch, numpy as np
from train_hydro_gine_v4 import HydroGINE_v4

dl = torch.load("multi_scenario_physics_pyg_dataset.pt", weights_only=False)
g_50 = [g for g in dl if g.region == 'hsr' and abs(g.x[0, 13].item() - 50.0) < 1e-3][0]

y_true = g_50.y.cpu().numpy().ravel()

ck = torch.load("hydro_gine_v4_model.pt", weights_only=False)
model = HydroGINE_v4(in_c=ck['in_c'], edge_c=2, hidden=ck['hidden'], n_layers=ck['n_layers'])
model.load_state_dict(ck['model'])
model.eval()

x_mean = ck['x_mean']
x_std = ck['x_std']
e_mean = ck['e_mean']
e_std = ck['e_std']
yl_mean = ck['yl_mean']
yl_std = ck['yl_std']

with torch.no_grad():
    gx = (g_50.x - x_mean) / x_std
    gea = (g_50.edge_attr - e_mean) / e_std
    c_l, d_o = model(gx, g_50.edge_index, gea)
    p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0, max=3.0).numpy().ravel()
    p_prob = torch.sigmoid(c_l.squeeze(-1)).numpy().ravel()

# Predict with test_hydrologic_fixes
x_raw = g_50.x.numpy()
in_d = x_raw[:, 3]
out_d = x_raw[:, 4]
dep_d = x_raw[:, 16]
sink_d = x_raw[:, 23]
total_r = x_raw[:, 27]
conv_def = x_raw[:, 30]

is_true_sink = (out_d <= 1) & (sink_d >= 0.20)
is_surcharged = (total_r >= 60.0) | (x_raw[:, 13] >= 100.0)
is_open_escape = (out_d >= in_d) & (sink_d < 0.03) & (dep_d < 0.20) & (~is_surcharged) & (~is_true_sink)

tau = np.where(is_true_sink, 0.10, np.where(is_surcharged, 0.20, np.where(is_open_escape, 0.55, 0.32)))
conf_gate = 1.0 / (1.0 + np.exp(-10.0 * (p_prob - tau)))
raw_gated = p_lin * conf_gate

light_mass_cap = np.where(is_true_sink, np.clip(0.30 + 0.35 * (total_r / 20.0) * sink_d, 0.30, 0.65), np.clip(0.10 + 0.12 * (total_r / 20.0) * (1.0 + 0.2 * x_raw[:, 11]), 0.02, 0.22))
mod_mass_cap = np.where(is_true_sink | (x_raw[:, 8] >= 0.05), 1.50, np.where(is_open_escape, 0.25, 0.80))
heavy_mass_cap = np.full_like(raw_gated, 3.0)
dyn_bound = np.where(total_r <= 25.0, light_mass_cap, np.where(total_r <= 60.0, mod_mass_cap, heavy_mass_cap))

pred = np.minimum(raw_gated, dyn_bound)

fp_idx = np.where((pred > 0.30) & (y_true <= 0.30))[0]
print(f"HSR @ 50 mm/hr: Total False Positives = {len(fp_idx)}")
print(f"{'Idx':<6s} | {'Pred Depth':<12s} | {'SWMM Depth':<12s} | {'Prob':<8s} | {'Sink Depth':<12s} | {'Dep Depth':<12s} | {'In/Out Deg':<10s} | {'Sag Idx':<8s} | {'Rel Drop'}")
print("-" * 105)
for i in fp_idx[:20]:
    print(f"{i:<6d} | {pred[i]:<12.4f} | {y_true[i]:<12.4f} | {p_prob[i]:<8.4f} | {sink_d[i]:<12.4f} | {dep_d[i]:<12.4f} | {in_d[i]:.0f}/{out_d[i]:.0f}       | {x_raw[i, 8]:<8.4f} | {x_raw[i, 0]:.4f}")
