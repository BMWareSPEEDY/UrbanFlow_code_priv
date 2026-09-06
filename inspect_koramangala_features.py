"""Inspect all features of the 30 worst-flooded nodes in Koramangala.
"""
from app import REGION_CACHE
import numpy as np

r_data = REGION_CACHE['koramangala']
g = r_data['pyg_data']
node_list = r_data['node_list']
node_pos = r_data['node_pos']
y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])

x_raw = g.x.cpu().numpy()
top_swmm_idx = np.argsort(-y_swmm)[:30]

print("Feature correlations with SWMM depth on Koramangala top 30:")
for col in range(x_raw.shape[1]):
    vals = x_raw[top_swmm_idx, col]
    corr = np.corrcoef(vals, y_swmm[top_swmm_idx])[0, 1]
    print(f"Col {col:2d}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, min={np.min(vals):.3f}, max={np.max(vals):.3f}, corr={corr:+.3f}")
