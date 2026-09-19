"""Inspect exact node predictions for HSR at 20 mm/hr and 50 mm/hr.
"""
import torch, numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from train_hydro_gine_v4 import HydroGINE_v4

dl_phys = torch.load("multi_scenario_physics_pyg_dataset.pt", weights_only=False)
g_20 = [g for g in dl_phys if g.region == 'hsr' and abs(g.x[0, 13].item() - 20.0) < 1e-3][0]

ck = torch.load("hydro_gine_v4_model.pt", map_location=device, weights_only=False)
model = HydroGINE_v4(in_c=ck['in_c'], edge_c=2, hidden=ck['hidden'], n_layers=ck['n_layers']).to(device)
model.load_state_dict(ck['model'])
model.eval()

x_mean = ck['x_mean'].to(device)
x_std = ck['x_std'].to(device)
e_mean = ck['e_mean'].to(device)
e_std = ck['e_std'].to(device)
yl_mean = ck['yl_mean'].to(device)
yl_std = ck['yl_std'].to(device)

with torch.no_grad():
    gx = (g_20.x.to(device) - x_mean) / x_std
    gea = (g_20.edge_attr.to(device) - e_mean) / e_std
    c_l, d_o = model(gx, g_20.edge_index.to(device), gea)
    p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0).cpu().numpy().ravel()
    p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
    
y_true = g_20.y.cpu().numpy().ravel()

print(f"HSR @ 20 mm/hr:")
print(f"  SWMM depths: min={y_true.min():.4f}, max={y_true.max():.4f}, mean={y_true.mean():.4f}")
print(f"  SWMM > 0.30m count: {np.sum(y_true > 0.30)}")
print(f"  SWMM >= 0.15m count: {np.sum(y_true >= 0.15)}")
print(f"  Pred depths (raw): min={p_lin.min():.4f}, max={p_lin.max():.4f}, mean={p_lin.mean():.4f}")
print(f"  Pred > 0.30m count: {np.sum(p_lin > 0.30)}")
print(f"  Pred probs: min={p_prob.min():.4f}, max={p_prob.max():.4f}, mean={p_prob.mean():.4f}")

# Top 15 nodes with highest raw predicted depth
idx_top = np.argsort(p_lin)[::-1][:15]
print("\nTop 15 Predicted Nodes:")
print(f"{'Idx':<6s} | {'Pred Depth':<12s} | {'Prob':<8s} | {'SWMM Depth':<12s} | {'Sink Depth':<12s} | {'Rel Drop':<10s} | {'In/Out Deg'}")
print("-" * 80)
for i in idx_top:
    sd = g_20.x[i, 23].item()
    rd = g_20.x[i, 0].item()
    ind = g_20.x[i, 3].item()
    outd = g_20.x[i, 4].item()
    print(f"{i:<6d} | {p_lin[i]:<12.4f} | {p_prob[i]:<8.4f} | {y_true[i]:<12.4f} | {sd:<12.4f} | {rd:<10.4f} | {ind:.0f} / {outd:.0f}")
