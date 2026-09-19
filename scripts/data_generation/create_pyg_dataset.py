import osmnx as ox
import pandas as pd
import torch
from torch_geometric.data import Data


def build_pyg_dataset():
    print("1. Loading graph and SWMM target data...")
    G = ox.load_graphml("bengaluru_complete_graph.graphml")

    df_targets = pd.read_csv("swmm_groundtruth_targets.csv")

    target_lookup = {}
    for _, row in df_targets.iterrows():
        node_str = str(row['swmm_node_id'])
        val = float(row['max_water_depth_m'])
        if node_str.startswith('J_'):
            raw_id = node_str[2:]
            try:
                target_lookup[int(raw_id)] = val
            except ValueError:
                pass
            target_lookup[raw_id] = val
        else:
            target_lookup[node_str] = val

    nodes_list = list(G.nodes())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(nodes_list)}

    print("2. Constructing Node Feature Matrix (X)...")
    node_features = []
    node_targets = []

    for node_id in nodes_list:
        data = G.nodes[node_id]
        x_coord = float(data.get('x', 0.0))
        y_coord = float(data.get('y', 0.0))
        elevation = float(data.get('elevation', 880.0))
        impervious = float(data.get('impervious_ratio', 0.2))
        manning_n = float(data.get('manning_n', 0.013))

        node_features.append([x_coord, y_coord, elevation, impervious, manning_n])

        depth = float(target_lookup.get(node_id, 0.0))
        node_targets.append([depth])

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(node_targets, dtype=torch.float)

    print("3. Constructing Edge Index Tensor and Edge Feature Matrix (E)...")
    src_nodes = []
    dst_nodes = []
    edge_features = []

    for u, v, k, data in G.edges(keys=True, data=True):
        src_idx = node_to_idx[u]
        dst_idx = node_to_idx[v]

        src_nodes.append(src_idx)
        dst_nodes.append(dst_idx)

        length = float(data.get('length', 10.0))
        grade = float(data.get('grade', 0.0))
        edge_features.append([length, grade])

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    print("\n--- PyTorch Geometric Dataset Summary ---")
    print(pyg_data)
    print(f"Node Features Matrix (x): shape {pyg_data.x.shape}")
    print(f"Edge Index Tensor (edge_index): shape {pyg_data.edge_index.shape}")
    print(f"Edge Attributes Tensor (edge_attr): shape {pyg_data.edge_attr.shape}")
    print(f"Target Labels Tensor (y): shape {pyg_data.y.shape}")

    output_path = "bengaluru_pyg_dataset.pt"
    torch.save(pyg_data, output_path)
    print(f"\nPyG Dataset successfully saved to '{output_path}'!")


if __name__ == "__main__":
    build_pyg_dataset()