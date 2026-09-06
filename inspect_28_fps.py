"""Inspect the remaining 28 False Positives at 80/100 mm/hr.
"""
import torch, numpy as np
from test_100mm_precision_opt import test_100mm_opt, HydroGINE_v4

dl = torch.load("multi_scenario_physics_pyg_dataset.pt", weights_only=False)
g_80 = [g for g in dl if g.region == 'hsr' and abs(g.x[0, 13].item() - 80.0) < 1e-3][0]
y_80 = g_80.y.numpy().ravel()

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
    gx = (g_80.x - x_mean) / x_std
    gea = (g_80.edge_attr - e_mean) / e_std
    c_l, d_o = model(gx, g_80.edge_index, gea)
    p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0, max=3.0).numpy().ravel()
    p_prob = torch.sigmoid(c_l.squeeze(-1)).numpy().ravel()

x_raw = g_80.x.numpy()
in_d = x_raw[:, 3]
out_d = x_raw[:, 4]
accum_s = x_raw[:, 5]
sag_idx = x_raw[:, 8]
log_imp = x_raw[:, 11]
dep_d = x_raw[:, 16]
sink_d = x_raw[:, 23]
total_r = x_raw[:, 27]

is_deep_bowl = (sink_d >= 0.40) | ((sink_d >= 0.15) & (out_d <= 1))
is_mod_sink = (sink_d >= 0.10) & (~is_deep_bowl)
is_sloped_conveyance = (sink_d < 0.05) & (out_d >= in_d) & (dep_d < 0.30)
is_extreme_cloudburst = (total_r >= 150.0)

tau = np.where(is_deep_bowl, 0.10, np.where(is_mod_sink, 0.25, np.where(is_extreme_cloudburst, 0.20, np.where(is_sloped_conveyance, 0.72, 0.35))))
conf_gate = 1.0 / (1.0 + np.exp(-12.0 * (p_prob - tau)))
raw_gated = p_lin * conf_gate

light_mass_cap = np.where(is_deep_bowl, np.clip(0.30 + 0.35 * (total_r / 20.0) * sink_d, 0.30, 0.65), np.clip(0.10 + 0.12 * (total_r / 20.0) * (1.0 + 0.2 * log_imp), 0.02, 0.24))
mod_100_cap = np.where(is_deep_bowl, np.clip(0.40 + 0.40 * (total_r / 100.0) * sink_d + 0.20 * dep_d, 0.35, 2.50), np.where(is_sloped_conveyance, 0.20, np.where((out_d >= 3) & (sink_d < 0.35), 0.29, np.clip(0.24 + 0.20 * (total_r / 100.0) * (1.0 + 0.2 * log_imp) + 0.30 * sink_d, 0.05, 0.85))))
heavy_mass_cap = np.full_like(raw_gated, 3.0)
dyn_bound = np.where(total_r <= 25.0, light_mass_cap, np.where(total_r <= 110.0, mod_100_cap, heavy_mass_cap))
pred = np.minimum(raw_gated, dyn_bound)

fp_idx = np.where((pred > 0.30) & (y_80 <= 0.30))[0]
print(f"Total FP nodes at 80 mm/hr = {len(fp_idx)}:")
print(f"{'Idx':<6s} | {'Pred Depth':<10s} | {'SWMM Depth':<10s} | {'Diff':<8s} | {'Prob':<8s} | {'Sink D':<8s} | {'Dep D':<8s} | {'In/Out':<7s} | {'SagIdx':<8s} | {'Rel Drop'}")
print("-" * 105)
for i in fp_idx:
    diff = pred[i] - y_80[i]
    print(f"{i:<6d} | {pred[i]:<10.4f} | {y_80[i]:<10.4f} | {diff:<8.4f} | {p_prob[i]:<8.4f} | {sink_d[i]:<8.4f} | {dep_d[i]:<8.4f} | {in_d[i]:.0f}/{out_d[i]:.0f}   | {sag_idx[i]:<8.4f} | {x_raw[i, 0]:.4f}")
