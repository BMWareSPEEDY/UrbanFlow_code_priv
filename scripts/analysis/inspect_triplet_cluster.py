"""Inspect the exact 3-node cluster: 3149541704, 3149541707, 3149541710
and 3158788119, 3158788140.
"""
import osmnx as ox, torch, numpy as np

G = ox.load_graphml("bengaluru_complete_graph.graphml")

triplet1 = [3149541704, 3149541707, 3149541710]
triplet2 = [3158788119, 3158788140]

print("=" * 90)
print("TRIPLET 1 FORENSIC ANALYSIS:")
print("=" * 90)
for nid in triplet1:
    data = G.nodes[nid]
    elev = float(data.get('elevation', 0.0))
    nbrs = list(G.neighbors(nid))
    nbr_elevs = [(n, float(G.nodes[n].get('elevation', 0.0))) for n in nbrs]
    dep = max(0.0, np.mean([e for _, e in nbr_elevs]) - elev) if nbr_elevs else 0.0
    print(f"Node #{nid}: Elev={elev:.3f}m | InDeg={G.in_degree(nid)} | OutDeg={G.out_degree(nid)} | DepDepth={dep:.4f}m | Neighbors={nbr_elevs}")

print("\n" + "=" * 90)
print("TRIPLET 2 FORENSIC ANALYSIS:")
print("=" * 90)
for nid in triplet2:
    data = G.nodes[nid]
    elev = float(data.get('elevation', 0.0))
    nbrs = list(G.neighbors(nid))
    nbr_elevs = [(n, float(G.nodes[n].get('elevation', 0.0))) for n in nbrs]
    dep = max(0.0, np.mean([e for _, e in nbr_elevs]) - elev) if nbr_elevs else 0.0
    print(f"Node #{nid}: Elev={elev:.3f}m | InDeg={G.in_degree(nid)} | OutDeg={G.out_degree(nid)} | DepDepth={dep:.4f}m | Neighbors={nbr_elevs}")
