import osmnx as ox
import numpy as np

files = [
    'bengaluru_complete_graph.graphml',
    'bengaluru_hsr_graph.graphml',
    'bengaluru_bellandur_graph.graphml',
    'bengaluru_whitefield_graph.graphml',
    'bengaluru_ecity_graph.graphml',
    'bengaluru_koramangala_graph.graphml',
]
for f in files:
    G = ox.load_graphml(f)
    if not all('elevation' in d for n, d in G.nodes(data=True)):
        print(f"WARN {f}: missing elevations, skipping")
        continue
    G = ox.elevation.add_edge_grades(G, add_absolute=False)
    ox.save_graphml(G, f)
    gr = [d.get('grade', 0.0) for u, v, k, d in G.edges(keys=True, data=True)]
    gr = [g for g in gr if abs(g) < 1]
    print(f"{f}: grade mean {np.mean(gr):.5f} std {np.std(gr):.5f} (fixed)")