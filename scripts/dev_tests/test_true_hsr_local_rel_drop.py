"""Test true local catchment rel_drop calculation strictly on local catchment nodes.
"""
import osmnx as ox, numpy as np, torch

G = ox.load_graphml("bengaluru_complete_graph.graphml")
node_list = list(G.nodes())

# In app.py for HSR, node_list is all 1379 nodes of HSR
hsr_elevs = np.array([float(G.nodes[nid].get('elevation', 880.0)) for nid in node_list])
min_el = np.min(hsr_elevs)
max_el = np.max(hsr_elevs)
el_range = max_el - min_el

print(f"HSR Layout Actual Terrain: min={min_el:.2f}m, max={max_el:.2f}m, range={el_range:.2f}m")

# Test target nodes
target_nids = [10994556963, 11663109187, 3158788119, 3149541707, 3158788140, 2399992907]
for nid in target_nids:
    el = float(G.nodes[nid].get('elevation', 880.0))
    rd = (max_el - el) / el_range
    print(f"Node {nid}: Elev = {el:.2f}m | Local RelDrop = {rd:.4f} ({'Valley Bottom' if rd >= 0.50 else 'Upland Hill'})")
