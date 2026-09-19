"""Regenerate the 5 hilly-terrain cities with LOW imperviousness range (0.15-0.40)
so the model learns 'hilly + low imp -> mostly dry' (hsr's regime). No Bangalore data."""
import osmnx as ox
import numpy as np

cities = {
    "varanasi": ("city_varanasi_graph.graphml", 80),
    "rajkot": ("city_rajkot_graph.graphml", 130),
    "ludhiana": ("city_ludhiana_graph.graphml", 240),
    "ranchi": ("city_ranchi_graph.graphml", 650),
    "agra": ("city_agra_graph.graphml", 170),
}

for key, (fname, base) in cities.items():
    G = ox.load_graphml(fname)
    rng = np.random.RandomState(hash(key) % 100000)
    for nid, data in G.nodes(data=True):
        data['impervious_ratio'] = float(rng.uniform(0.15, 0.40))
        data['manning_n'] = float(data.get('manning_n', rng.uniform(0.012, 0.035)))
    ox.save_graphml(G, fname)
    imps = [G.nodes[n]['impervious_ratio'] for n in G.nodes]
    print(f"{key}: imp mean {np.mean(imps):.3f} range [{np.min(imps):.2f},{np.max(imps):.2f}]", flush=True)