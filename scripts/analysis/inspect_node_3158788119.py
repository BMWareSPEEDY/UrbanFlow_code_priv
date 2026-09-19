"""Inspect node 3158788119 and surrounding graph topology.
"""
import osmnx as ox, numpy as np

G = ox.load_graphml("bengaluru_complete_graph.graphml")
nid = 3158788119
data = G.nodes[nid]
print(f"Node {nid} Data: {data}")

print("\nIncoming Edges (Predecessors):")
for u in G.predecessors(nid):
    d = G[u][nid]
    el_u = float(G.nodes[u].get('elevation', 0))
    print(f"  From {u}: Elev={el_u:.2f}m, Grade={d[0].get('grade')}, Length={d[0].get('length')}")

print("\nOutgoing Edges (Successors):")
for v in G.successors(nid):
    d = G[nid][v]
    el_v = float(G.nodes[v].get('elevation', 0))
    print(f"  To {v}: Elev={el_v:.2f}m, Grade={d[0].get('grade')}, Length={d[0].get('length')}")
