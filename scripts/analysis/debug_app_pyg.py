"""Debug r_data['pyg_data'].x in app.py
"""
import torch, osmnx as ox, numpy as np

from production_v4 import ProductionFloodPredictorV4
predictor = ProductionFloodPredictorV4("hydro_gine_v4_3_model.pt", device='cuda')

from app import REGION_CACHE, init_app_data
# check REGION_CACHE
hsr_data = REGION_CACHE['hsr']
pyg_data = hsr_data['pyg_data']
print(f"pyg_data.x shape: {pyg_data.x.shape}")
print(f"Node 6131076159 index in node_list:")
node_list = hsr_data['node_list']
for idx, nid in enumerate(node_list):
    if str(nid) == '6131076159':
        print(f"  Found at index {idx}:")
        x_row = pyg_data.x[idx].cpu().numpy()
        print(f"  rel_drop (0): {x_row[0]:.4f}")
        print(f"  is_sink (6): {x_row[6]:.4f}")
        print(f"  dep_depth (16): {x_row[16]:.4f}")
        print(f"  sink_depth (23): {x_row[23]:.4f}")
        
preds, raw_p, probs = predictor.predict(pyg_data, 50.0, 60.0)
print(f"  Predicted depth: {preds[idx]:.4f}m | raw: {raw_p[idx]:.4f}m | prob: {probs[idx]:.4f}")
