"""Test refined true sink bowl logic in production_v4.py
"""
import torch, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from production_v4 import ProductionFloodPredictorV4
predictor = ProductionFloodPredictorV4("hydro_gine_v4_3_model.pt", device=device)

dl = torch.load("multi_scenario_physics_pyg_dataset.pt", weights_only=False)
g_50 = [g for g in dl if g.region == 'hsr' and abs(g.x[0, 13].item() - 50.0) < 1e-3][0]

x_np = g_50.x.cpu().numpy()
y_swmm = g_50.y.cpu().numpy().ravel()
rel_drop = x_np[:, 0]
dep_d = x_np[:, 16]
sink_d = x_np[:, 23]
out_d = x_np[:, 4]
in_d = x_np[:, 3]

# A true sink bowl MUST be in the valley (rel_drop >= 0.50) with low outflow
is_true_valley_sink = (rel_drop >= 0.50) & (out_d <= 1) & (sink_d >= 0.15)
# Sinks on upland slopes (rel_drop < 0.50) are false elevation artifacts:
sink_d_corrected = np.where(rel_drop < 0.50, 0.0, sink_d)

g_50_mod = g_50.clone()
g_50_mod.x[:, 23] = torch.tensor(sink_d_corrected, dtype=torch.float32)

preds, raw_p, probs = predictor.predict(g_50_mod, 50.0, 60.0)

# Sort by predicted depth:
top_idx = np.argsort(preds)[::-1]

print("=" * 115)
print("TOP 15 HIGHLIGHTED NODES IN HSR LAYOUT @ 50 mm/hr (CORRECTED VALLEY SINK LOGIC):")
print("=" * 115)
print(f"{'Rank':<5s} | {'Idx':<6s} | {'GNN Depth':<12s} | {'SWMM Depth':<12s} | {'Diff (cm)':<10s} | {'Status':<12s} | {'RelDrop':<8s} | {'SinkD':<8s}")
print("-" * 115)

for r, idx in enumerate(top_idx[:15], 1):
    p = preds[idx]
    swmm = y_swmm[idx]
    diff = (p - swmm) * 100.0
    status = "CORRECT" if abs(diff) <= 15.0 else ("FALSE HIGH" if diff > 15.0 else "UNDER")
    print(f"{r:<5d} | {idx:<6d} | {p:<12.4f} | {swmm:<12.4f} | {diff:<+10.1f} | {status:<12s} | {rel_drop[idx]:<8.4f} | {sink_d_corrected[idx]:<8.4f}")
