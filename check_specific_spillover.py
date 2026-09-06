"""Check specific spillover node predictions with WSE envelope.
"""
import torch, numpy as np, sys, osmnx as ox

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from production_v4 import ProductionFloodPredictorV4
from test_wse_envelope import apply_hydrostatic_wse_envelope

predictor = ProductionFloodPredictorV4("hydro_gine_v5_model.pt", device=device)
dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
G = ox.load_graphml("bengaluru_complete_graph.graphml")

g_50 = [g for g in dl if getattr(g, 'region', '') == 'hsr' and abs(g.rain_intensity - 50.0) < 1e-3][0]
y_true = g_50.y.cpu().numpy().ravel()
preds_raw, _, probs = predictor.predict(g_50, 50.0, 60.0)
preds_wse = apply_hydrostatic_wse_envelope(G, preds_raw, g_50.x.cpu().numpy())

nodes_list = list(G.nodes())
node_to_idx = {nid: idx for idx, nid in enumerate(nodes_list)}

target_nids = [11007254966, 1488526923, 3158788119, 3149541707, 3158788140]

print(f"{'Node ID':<14s} | {'SWMM Depth':<12s} | {'Raw GNN':<10s} | {'WSE Envelope':<12s} | {'Elevation':<10s}")
print("-" * 65)

for nid in target_nids:
    idx = node_to_idx[nid]
    print(f"{nid:<14d} | {y_true[idx]:<12.4f} | {preds_raw[idx]:<10.4f} | {preds_wse[idx]:<12.4f} | {float(G.nodes[nid].get('elevation', 0)):<10.2f}")
