"""Inspect all PyG datasets in the repository to see available training data and scenarios.
"""
import os, torch

pt_files = [
    "multi_scenario_physics_pyg_dataset.pt",
    "multi_scenario_full31_pyg_dataset.pt",
    "multi_scenario_full22_pyg_dataset.pt",
    "multi_scenario_pyg_dataset.pt",
    "multi_scenario_pyg_dataset_v2.pt",
    "multi_scenario_testcities_pyg_dataset.pt",
    "bengaluru_pyg_dataset.pt"
]

for fn in pt_files:
    if os.path.exists(fn):
        try:
            data = torch.load(fn, weights_only=False)
            if isinstance(data, list):
                cities = set(getattr(g, 'city', 'unknown') for g in data)
                total_nodes = sum(g.x.shape[0] for g in data)
                feat_dim = data[0].x.shape[1] if hasattr(data[0], 'x') else None
                edge_dim = data[0].edge_attr.shape[1] if hasattr(data[0], 'edge_attr') and data[0].edge_attr is not None else None
                print(f"{fn:<42s} | Graphs: {len(data):4d} | Nodes: {total_nodes:8d} | Feat: {feat_dim} | Edge: {edge_dim} | Cities: {len(cities)} {list(cities)[:5]}")
            else:
                print(f"{fn:<42s} | Type: {type(data)}")
        except Exception as e:
            print(f"{fn:<42s} | Error: {e}")
