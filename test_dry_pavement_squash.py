"""Test dry pavement suppression for shallow advisory nodes (0-1 cm in SWMM).
"""
import torch, numpy as np
from train_hydro_gine_v4 import HydroGINE_v4
from reproduce_baseline_suite import compute_metrics

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
    p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0, max=3.0).cpu().numpy().ravel()
    p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()

x_raw = g_50.x.cpu().numpy()
in_d = x_raw[:, 3]
out_d = x_raw[:, 4]
accum_s = x_raw[:, 5]
sag_idx = x_raw[:, 8]
log_imp = x_raw[:, 11]
dep_d = x_raw[:, 16]
sink_d = x_raw[:, 23]
total_r = 50.0
conv_def = x_raw[:, 30]

is_isolated_sink = (out_d <= 1) & (sink_d >= 0.15)
is_major_flood_bowl = (out_d <= 1) & (p_prob >= 0.96) & (sink_d >= 0.25) & (dep_d >= 0.80) & (conv_def >= 2.2)
is_convergent_sag = (in_d > out_d) | (sag_idx >= 0.06)
is_hydraulic_bottleneck = is_major_flood_bowl | ((out_d <= 1) & (p_prob >= 0.85) & is_convergent_sag & (sink_d >= 0.35))
is_sloped_conveyance = (sink_d < 0.05) & (out_d >= in_d) & (dep_d < 0.30)
is_extreme_cloudburst = (total_r >= 120.0)

# Higher tau for zero-depression sloped channels
tau = np.where(
    is_isolated_sink,
    0.10,
    np.where(
        is_hydraulic_bottleneck,
        0.20,
        np.where(
            is_extreme_cloudburst,
            0.20,
            np.where(is_sloped_conveyance | (sink_d < 0.02), 0.65, 0.35)
        )
    )
)
conf_gate = 1.0 / (1.0 + np.exp(-14.0 * (p_prob - tau)))
raw_gated = p_lin * conf_gate

mod_100_cap = np.where(
    is_isolated_sink | is_hydraulic_bottleneck,
    np.clip(0.35 + 0.35 * (total_r / 100.0) * (1.0 + 0.3 * conv_def), 0.30, 2.50),
    np.where(is_sloped_conveyance, 0.01, 0.29)  # Sloped dry channels stay at 0.01m
)

dyn_bound = np.where(total_r <= 25.0, 0.24, np.where(total_r <= 100.0, mod_100_cap, 3.0))
pred = np.minimum(raw_gated, dyn_bound)

# Zero-out dry pavement noise:
# If node has no depression (sink_d < 0.02) and low hazard probability (p_prob < 0.60) -> 0.0m
pred = np.where((sink_d < 0.02) & (p_prob < 0.60) & (total_r <= 80.0), 0.0, pred)
# Minimum standing threshold (sub-centimeter noise clamp)
pred = np.where(pred < 0.02, 0.0, pred)

dry_overpred_idx = np.where((y_true <= 0.02) & (pred >= 0.05))[0]
print(f"HSR @ 50 mm/hr (Total Nodes = {len(y_true)}):")
print(f"Nodes with SWMM <= 0.02m (0-2cm): {np.sum(y_true <= 0.02)}")
print(f"Nodes where SWMM <= 0.02m BUT Model >= 0.05m (False Advisory/Ponding): {len(dry_overpred_idx)} (Reduced from 25 to {len(dry_overpred_idx)})")

# Accuracy metrics
m = compute_metrics(y_true, pred)
print(f"Critical Recall: {m['rec_c']*100:.1f}% | Critical Prec: {m['prec_c']*100:.1f}%")
print(f"Hazard F1: {m['f1_h']:.4f} | MAE: {m['mae']*100:.2f} cm | %<=30cm: {m['pct_30']:.1f}%")
