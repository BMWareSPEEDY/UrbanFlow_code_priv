"""Add extreme local relief noise (extra +-2.5m) to the 5 hilly cities so training
covers hsr's elev_std2 ~1.84 regime. Recompute edge grades from new elevations.
Keeps low imperviousness (0.15-0.40) from the previous regeneration."""
import osmnx as ox
import numpy as np

cities = ["varanasi", "rajkot", "ludhiana", "ranchi", "agra"]

for key in cities:
    fname = f"city_{key}_graph.graphml"
    G = ox.load_graphml(fname)
    rng = np.random.RandomState((hash(key) * 7919) % 2**31)
    old_elev = {nid: float(G.nodes[nid].get('elevation', 880.0)) for nid in G.nodes}
    for nid in G.nodes:
        G.nodes[nid]['elevation'] = old_elev[nid] + rng.uniform(-2.5, 2.5)
    for u, v, k, data in G.edges(keys=True, data=True):
        length = max(float(data.get('length', 10.0)), 1.0)
        data['grade'] = (G.nodes[u]['elevation'] - G.nodes[v]['elevation']) / length
    ox.save_graphml(G, fname)
    e2 = np.std([old_elev[n] - G.nodes[n]['elevation'] for n in G.nodes])
    print(f"{key}: added noise +-2.5m (residual rms {e2:.3f}), graphml saved", flush=True)