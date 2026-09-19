"""Inspect Tokyo predictions.
"""
from app import REGION_CACHE, PRODUCTION_PREDICTOR
import numpy as np

g = REGION_CACHE['tokyo']['pyg_data']
node_list = REGION_CACHE['tokyo']['node_list']
node_pos = REGION_CACHE['tokyo']['node_pos']
y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])

preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(g, 50.0, 60.0)

print(f"Preds min: {preds.min():.3f}, max: {preds.max():.3f}, mean: {preds.mean():.3f}")
print(f"SWMM min: {y_swmm.min():.3f}, max: {y_swmm.max():.3f}, mean: {y_swmm.mean():.3f}")

# Sort by pred descending (how risk nodes are sorted in app.py)
sorted_indices = np.argsort(-preds)
print("\nTop 10 nodes sorted by GNN prediction:")
for idx in sorted_indices[:10]:
    nid = node_list[idx]
    print(f"  Node #{nid}: GNN={preds[idx]:.3f}m | SWMM={y_swmm[idx]:.3f}m | Diff={preds[idx]-y_swmm[idx]:+.3f}m")

# Sort by SWMM descending (true highest risk nodes)
sorted_swmm = np.argsort(-y_swmm)
print("\nTop 10 nodes sorted by SWMM ground truth:")
for idx in sorted_swmm[:10]:
    nid = node_list[idx]
    print(f"  Node #{nid}: GNN={preds[idx]:.3f}m | SWMM={y_swmm[idx]:.3f}m | Diff={preds[idx]-y_swmm[idx]:+.3f}m")
