import osmnx as ox
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv

region_files = {
    'hsr': 'bengaluru_complete_graph.graphml',
    'bellandur': 'bengaluru_bellandur_graph.graphml',
    'whitefield': 'bengaluru_whitefield_graph.graphml',
    'ecity': 'bengaluru_ecity_graph.graphml',
    'koramangala': 'bengaluru_koramangala_graph.graphml'
}

def create_enhanced_dataset():
    print("1. Loading target depths from 'swmm_groundtruth_targets.csv'...")
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
    
    for reg, file_path in region_files.items():
        print(f"Processing region '{reg}' ({file_path}) with 8 hydraulic features...")
        G = ox.load_graphml(file_path)
        nodes_list = list(G.nodes())
        node_to_idx = {node_id: idx for idx, node_id in enumerate(nodes_list)}
        
        # Calculate elevation bounds and upstream flow accumulation per node
        elevs = [float(G.nodes[nid].get('elevation', 880.0)) for nid in nodes_list]
        min_elev = min(elevs)
        max_elev = max(elevs)
        elev_range = max(1.0, max_elev - min_elev)
        
        in_deg_map = dict(G.in_degree())
        out_deg_map = dict(G.out_degree())
        
        node_features = []
        node_targets = []
        
        for node_id in nodes_list:
            data = G.nodes[node_id]
            x_coord = float(data.get('x', 0.0))
            y_coord = float(data.get('y', 0.0))
            elev = float(data.get('elevation', 880.0))
            imp = float(data.get('impervious_ratio', 0.2))
            manning_n = float(data.get('manning_n', 0.013))
            
            # Hydrodynamic accumulation features
            rel_drop = (max_elev - elev) / elev_range # Higher near low elevation sinks
            in_deg = in_deg_map.get(node_id, 0)
            out_deg = out_deg_map.get(node_id, 0)
            accum_score = np.log1p(in_deg * 2.5 + (1.0 if out_deg == 0 else 0.0))
            is_sink = 1.0 if (rel_drop > 0.85 and in_deg >= 2) else 0.0
            
            node_features.append([
                x_coord, y_coord, elev, imp, manning_n, rel_drop, accum_score, is_sink
            ])
            
            depth = float(target_lookup.get(node_id, target_lookup.get(str(node_id), 0.0)))
            node_targets.append([depth])
            
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_targets, dtype=torch.float)
        
        src_nodes, dst_nodes, edge_features = [], [], []
        for u, v, k, data in G.edges(keys=True, data=True):
            src_nodes.append(node_to_idx[u])
            dst_nodes.append(node_to_idx[v])
            length = float(data.get('length', 10.0))
            grade = float(data.get('grade', 0.0))
            edge_features.append([length, grade])
            
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data_list.append(pyg_data)
        
    batch_dataset = Batch.from_data_list(data_list)
    torch.save(batch_dataset, "bengaluru_pyg_dataset.pt")
    print(f"Successfully saved enhanced 8-feature PyG dataset to 'bengaluru_pyg_dataset.pt'!")

if __name__ == "__main__":
    create_enhanced_dataset()
