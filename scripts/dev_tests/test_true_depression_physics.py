"""Test true hydraulic depression depth vs flawed mean-neighbor elevation.
"""
import osmnx as ox, numpy as np

G = ox.load_graphml("bengaluru_complete_graph.graphml")
node_list = list(G.nodes())
elev_map = {nid: float(G.nodes[nid].get('elevation', 880.0)) for nid in node_list}

# Flawed calculation (currently in app.py)
flawed_dep = {}
for nid in node_list:
    nbrs = list(G.neighbors(nid))
    if nbrs:
        flawed_dep[nid] = max(0.0, float(np.mean([elev_map[nb] for nb in nbrs]) - elev_map[nid]))
    else:
        flawed_dep[nid] = 0.0

# True physical depression calculation (Sill-based depression)
true_dep = {}
for nid in node_list:
    # Outgoing neighbors (or undirected neighbors if directed out is empty)
    succs = list(G.successors(nid)) if G.is_directed() else list(G.neighbors(nid))
    if succs:
        lowest_exit = min(elev_map[s] for s in succs)
        true_dep[nid] = max(0.0, lowest_exit - elev_map[nid])
    else:
        nbrs = list(G.neighbors(nid))
        if nbrs:
            lowest_exit = min(elev_map[nb] for nb in nbrs)
            true_dep[nid] = max(0.0, lowest_exit - elev_map[nid])
        else:
            true_dep[nid] = 0.0

flawed_nonzero = sum(1 for v in flawed_dep.values() if v > 0.05)
true_nonzero = sum(1 for v in true_dep.values() if v > 0.05)

print(f"Total HSR Nodes: {len(node_list)}")
print(f"Nodes with flawed dep_depth > 5cm: {flawed_nonzero} ({flawed_nonzero/len(node_list)*100:.1f}%)")
print(f"Nodes with TRUE physical depression > 5cm: {true_nonzero} ({true_nonzero/len(node_list)*100:.1f}%)")

# Inspect nodes on hill peaks that had fake depression
hill_nodes = [10994556963, 11663109187, 3158788119, 3158788140]
print("\nHill Nodes Comparison:")
for nid in hill_nodes:
    if nid in elev_map:
        print(f"Node {nid}: Elev = {elev_map[nid]:.2f}m | Flawed DepDepth = {flawed_dep[nid]:.4f}m | TRUE Physical DepDepth = {true_dep[nid]:.4f}m")
