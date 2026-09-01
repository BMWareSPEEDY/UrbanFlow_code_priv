import osmnx as ox
import numpy as np

files = {
    'hsr': 'bengaluru_complete_graph.graphml',
    'bellandur': 'bengaluru_bellandur_graph.graphml',
    'whitefield': 'bengaluru_whitefield_graph.graphml',
    'ecity': 'bengaluru_ecity_graph.graphml',
    'koramangala': 'bengaluru_koramangala_graph.graphml',
    'hyderabad': 'city_hyderabad_graph.graphml',
    'chennai': 'city_chennai_graph.graphml',
    'pune': 'city_pune_graph.graphml',
}
for reg, f in files.items():
    G = ox.load_graphml(f)
    nodes = list(G.nodes())
    xs = [float(G.nodes[n].get('x', 0)) for n in nodes]
    ys = [float(G.nodes[n].get('y', 0)) for n in nodes]
    elevs = [float(G.nodes[n].get('elevation', 880)) for n in nodes]
    imps = [float(G.nodes[n].get('impervious_ratio', 0.2)) for n in nodes]
    lens = [float(d.get('length', 10)) for u, v, k, d in G.edges(keys=True, data=True)]
    print(f"{reg:<12} nodes {len(nodes):>6} edges {G.number_of_edges():>7} | span_x {np.ptp(xs):.0f}m span_y {np.ptp(ys):.0f}m | "
          f"elev_range {np.ptp(elevs):.1f}m | mean_len {np.mean(lens):.1f}m | imp {np.mean(imps):.3f}")