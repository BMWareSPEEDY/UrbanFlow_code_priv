"""Check node alignment between expanded_master_physics_dataset and REGION_CACHE in app.py.
"""
import torch, sys
from app import REGIONS, ox

dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)

for r_key, r_info in REGIONS.items():
    g_file = r_info['file']
    G = ox.load_graphml(g_file)
    n_nodes = len(G.nodes)
    
    matching = [g for g in dl if (getattr(g, 'region', '') == r_key or getattr(g, 'city', '') == r_key) and abs(g.rain_intensity - 50.0) < 1.0]
    if matching:
        g = matching[0]
        match_str = "MATCH" if len(g.y) == n_nodes else "MISMATCH"
        print(f"Region: {r_key:<15s} | GraphML Nodes: {n_nodes:6d} | Dataset Nodes: {len(g.y):6d} | {match_str}")
    else:
        print(f"Region: {r_key:<15s} | GraphML Nodes: {n_nodes:6d} | No matching dataset graph")
