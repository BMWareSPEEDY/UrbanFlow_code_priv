import osmnx as ox
import numpy as np

files = [
    'bengaluru_hsr_graph.graphml',
    'bengaluru_bellandur_graph.graphml',
    'bengaluru_whitefield_graph.graphml',
    'bengaluru_ecity_graph.graphml',
    'bengaluru_koramangala_graph.graphml',
    'bengaluru_complete_graph.graphml',
    'city_hyderabad_graph.graphml',
    'city_chennai_graph.graphml',
    'city_pune_graph.graphml',
    'city_kolkata_graph.graphml',
    'city_ahmedabad_graph.graphml',
]
for f in files:
    G = ox.load_graphml(f)
    gr = [d.get('grade', 0.0) for u, v, k, d in G.edges(keys=True, data=True)]
    gr = [g for g in gr if abs(g) < 1]
    elevs = [float(d.get('elevation', 0)) for n, d in G.nodes(data=True)]
    print(f"{f:<40} nodes {len(G.nodes()):>6} | grade mean {np.mean(gr):.5f} std {np.std(gr):.5f} | elev range {np.ptp(elevs):.1f}m")