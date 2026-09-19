"""Test strict depression definition on HSR top false nodes.
"""
import torch, numpy as np, osmnx as ox

G = ox.load_graphml("bengaluru_complete_graph.graphml")
target_nids = ['6131076159', '11259448962', '650141496', '11259448972']

from test_app_feature_computation import compute_true_graph_features
from production_v4 import ProductionFloodPredictorV4

data = compute_true_graph_features(G, 50.0, 60.0)

# Replace sink_d (col 23) with strict dep_depth (col 16)
x_np = data.x.cpu().numpy()
dep_depth = x_np[:, 16]
x_np[:, 23] = np.where(dep_depth >= 0.05, dep_depth, 0.0)
x_np[:, 6] = np.where(dep_depth >= 0.10, 1.0, 0.0) # is_sink strictly if true depression

data.x = torch.tensor(x_np, dtype=torch.float32, device='cuda')

predictor = ProductionFloodPredictorV4("hydro_gine_v4_3_model.pt")
pred, raw, prob = predictor.predict(data, 50.0, 60.0)

nodes_list = list(G.nodes())
node_to_idx = {str(nid): idx for idx, nid in enumerate(nodes_list)}

print(f"{'Node ID':<14s} | {'New Pred':<10s} | {'Prob':<8s} | {'Sink D':<8s} | {'Dep D':<8s} | {'RelDrop':<8s}")
print("-" * 75)

for nid in target_nids:
    idx = node_to_idx[nid]
    p = pred[idx]
    pr = prob[idx]
    sd = x_np[idx, 23]
    depd = x_np[idx, 16]
    reldrop = x_np[idx, 0]
    print(f"{nid:<14s} | {p:<10.4f} | {pr:<8.4f} | {sd:<8.4f} | {depd:<8.4f} | {reldrop:<8.4f}")

# Top 10 predicted nodes across HSR:
top_idx = np.argsort(pred)[::-1]
print("\nNew Top 10 Highlighted Nodes in HSR Layout @ 50mm/hr:")
for r, idx in enumerate(top_idx[:10], 1):
    nid = str(nodes_list[idx])
    print(f"Rank {r:2d}: Node #{nid:<14s} | Pred: {pred[idx]:.4f}m | Prob: {prob[idx]:.4f} | DepD: {dep_depth[idx]:.4f}m")
