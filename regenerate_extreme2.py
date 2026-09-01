"""Derive 5 EXTREME-relief cities (+5m extra noise on current b5 graphmls) so training
contains hsr's flood% regime (~31% at I=300). New files, originals untouched."""
import osmnx as ox
import numpy as np

cities = ["varanasi", "rajkot", "ludhiana", "ranchi", "agra"]

for key in cities:
    src = f"city_{key}_graph.graphml"
    dst = f"city_{key}_extreme.graphml"
    G = ox.load_graphml(src)
    rng = np.random.RandomState((hash(key + "ext") * 104729) % 2**31)
    old_elev = {nid: float(G.nodes[nid].get('elevation', 880.0)) for nid in G.nodes}
    for nid in G.nodes:
        G.nodes[nid]['elevation'] = old_elev[nid] + rng.uniform(-5.0, 5.0)
    for u, v, k, data in G.edges(keys=True, data=True):
        length = max(float(data.get('length', 10.0)), 1.0)
        data['grade'] = (G.nodes[u]['elevation'] - G.nodes[v]['elevation']) / length
    ox.save_graphml(G, dst)
    print(f"{key}: +-5m extra noise -> {dst}", flush=True)