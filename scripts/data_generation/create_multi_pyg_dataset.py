import osmnx as ox
import pandas as pd
import torch
from torch_geometric.data import Data, Batch

region_files = {
    'hsr': 'bengaluru_complete_graph.graphml',
    'bellandur': 'bengaluru_bellandur_graph.graphml',
    'whitefield': 'bengaluru_whitefield_graph.graphml',
    'ecity': 'bengaluru_ecity_graph.graphml',
    'koramangala': 'bengaluru_koramangala_graph.graphml'
}

def build_multi_region_dataset():
    print("1. Loading ground truth targets from 'swmm_groundtruth_targets.csv'...")
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

    data_list = []
    total_nodes = 0
    total_edges = 0

    for reg, file_path in region_files.items():
        print(f"Processing region '{reg}' ({file_path})...")
        G = ox.load_graphml(file_path)
        nodes_list = list(G.nodes())
        node_to_idx = {node_id: idx for idx, node_id in enumerate(nodes_list)}

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
            depth = float(target_lookup.get(node_id, target_lookup.get(str(node_id), 0.0)))
            node_targets.append([depth])

        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_targets, dtype=torch.float)

        src_nodes, dst_nodes, edge_features = [], [], []

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
        data_list.append(pyg_data)
        total_nodes += pyg_data.x.shape[0]
        total_edges += pyg_data.edge_index.shape[1]

    # Combine all regions into a batched multi-graph PyG object
    batch_dataset = Batch.from_data_list(data_list)

    print("\n--- Multi-Region PyTorch Geometric Dataset Summary ---")
    print(batch_dataset)
    print(f"Total Combined Nodes: {total_nodes}")
    print(f"Total Combined Edges: {total_edges}")

    output_path = "bengaluru_pyg_dataset.pt"
    torch.save(batch_dataset, output_path)
    print(f"\nMulti-region PyG Dataset successfully saved to '{output_path}'!")

if __name__ == "__main__":
    build_multi_region_dataset()
