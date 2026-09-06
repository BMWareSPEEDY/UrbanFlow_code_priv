"""Test exact app prediction pipeline.
"""
import sys, numpy as np

sys.stdout.reconfigure(line_buffering=True)

from app import REGION_CACHE, PRODUCTION_PREDICTOR

hsr = REGION_CACHE['hsr']
preds, raw, probs = PRODUCTION_PREDICTOR.predict(hsr['pyg_data'], 50.0, 60.0)

node_list = hsr['node_list']
node_pos = hsr['node_pos']

top_idx = np.argsort(preds)[::-1]

print("=" * 115)
print("APP PREDICTION PIPELINE OUTPUT @ 50 mm/hr (TOP 15 NODES):")
print("=" * 115)
print(f"{'Rank':<5s} | {'Node ID':<14s} | {'GNN Depth':<12s} | {'SWMM Depth':<12s} | {'Elev':<8s} | {'DepD':<8s} | {'SinkD':<8s} | {'Diff (cm)'}")
print("-" * 115)

for r, idx in enumerate(top_idx[:15], 1):
    nid = node_list[idx]
    p = preds[idx]
    swmm = node_pos[nid]['swmm_depth']
    diff = (p - swmm) * 100.0
    elev = node_pos[nid]['elevation']
    depd = hsr['pyg_data'].x[idx, 16].item()
    sinkd = hsr['pyg_data'].x[idx, 23].item()
    print(f"{r:<5d} | {str(nid):<14s} | {p:<12.4f} | {swmm:<12.4f} | {elev:<8.2f} | {depd:<8.4f} | {sinkd:<8.4f} | {diff:<+10.1f}")
