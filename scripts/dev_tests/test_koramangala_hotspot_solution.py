"""Inspect Koramangala top 30 hotspot nodes and test head alignment.
"""
from app import REGION_CACHE, PRODUCTION_PREDICTOR
import numpy as np

r_data = REGION_CACHE['koramangala']
g = r_data['pyg_data']
node_list = r_data['node_list']
node_pos = r_data['node_pos']
y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])

preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(g, 50.0, 60.0)

x_raw = g.x.cpu().numpy()
accum_s = x_raw[:, 5]
out_d = x_raw[:, 4]
dep_d = x_raw[:, 16]
conv_def = x_raw[:, 30]

# Check top 30 nodes sorted by predicted depth
risk_mask = (preds > 0.08)
risk_indices = np.where(risk_mask)[0]
top_30 = risk_indices[np.argsort(-preds[risk_indices])[:30]]

print("Top 10 Hotspots in Koramangala:")
for i, idx in enumerate(top_30[:10]):
    print(f"#{i+1} | GNN={preds[idx]:.3f}m | SWMM={y_swmm[idx]:.3f}m | Diff={preds[idx]-y_swmm[idx]:+.3f}m | Acc={accum_s[idx]:.2f} | OutD={out_d[idx]:.0f} | Dep={dep_d[idx]:.2f} | CDef={conv_def[idx]:.2f}")

diff = preds[top_30] - y_swmm[top_30]
print(f"\nCurrent Koramangala Match Rate: {np.sum(np.abs(diff) < 0.15)}/30 ({np.sum(np.abs(diff) < 0.15)/30*100:.1f}%)")
print(f"Mean Diff on Hotspots: {np.mean(diff):+.3f}m | Mean Abs Diff: {np.mean(np.abs(diff)):.3f}m")
