"""Inspect the top 3 false critical nodes in HSR Layout:
6131076159, 11259448962, 650141496, 11259448972
"""
import torch, numpy as np, osmnx as ox

G = ox.load_graphml("bengaluru_complete_graph.graphml")
target_nids = ['6131076159', '11259448962', '650141496', '11259448972']

from test_app_feature_computation import compute_true_graph_features
from production_v4 import ProductionFloodPredictorV4

data = compute_true_graph_features(G, 50.0, 60.0)
predictor = ProductionFloodPredictorV4("hydro_gine_v4_3_model.pt")
pred, raw, prob = predictor.predict(data, 50.0, 60.0)

nodes_list = list(G.nodes())
node_to_idx = {str(nid): idx for idx, nid in enumerate(nodes_list)}
x_np = data.x.cpu().numpy()

print(f"{'Node ID':<14s} | {'Pred':<8s} | {'Raw':<8s} | {'Prob':<8s} | {'Elev':<8s} | {'In/Out':<7s} | {'Sink D':<8s} | {'Dep D':<8s} | {'RelDrop':<8s} | {'ConvDef':<8s} | {'Neighbors'}")
print("-" * 115)

for nid in target_nids:
    idx = node_to_idx[nid]
    p = pred[idx]
    r = raw[idx]
    pr = prob[idx]
    elev = G.nodes[int(nid) if int(nid) in G.nodes else nid].get('elevation', 0.0)
    ind = x_np[idx, 3]
    outd = x_np[idx, 4]
    sinkd = x_np[idx, 23]
    depd = x_np[idx, 16]
    reldrop = x_np[idx, 0]
    conv = x_np[idx, 30]
    
    # Check neighbors
    nbrs = list(G.successors(int(nid) if int(nid) in G.nodes else nid))
    nbr_elevs = [G.nodes[n].get('elevation', 0.0) for n in nbrs]
    
    print(f"{nid:<14s} | {p:<8.4f} | {r:<8.4f} | {pr:<8.4f} | {elev:<8.2f} | {ind:.0f}/{outd:.0f}   | {sinkd:<8.4f} | {depd:<8.4f} | {reldrop:<8.4f} | {conv:<8.4f} | {nbr_elevs}")
