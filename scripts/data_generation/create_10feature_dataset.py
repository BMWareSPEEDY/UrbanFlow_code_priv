import os
import osmnx as ox
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Batch

region_files = {
    'hsr': 'bengaluru_complete_graph.graphml',
    'bellandur': 'bengaluru_bellandur_graph.graphml',
    'whitefield': 'bengaluru_whitefield_graph.graphml',
    'ecity': 'bengaluru_ecity_graph.graphml',
    'koramangala': 'bengaluru_koramangala_graph.graphml'
}

STORM_DURATIONS = {
    20.0: 60.0,
    50.0: 60.0,
    80.0: 45.0,
    120.0: 30.0,
    150.0: 30.0,
    200.0: 45.0,
    250.0: 30.0,
    300.0: 30.0
}

def create_12feature_dataset():
    csv_file = "swmm_multi_scenario_targets.csv"
    if not os.path.exists(csv_file):
        csv_file = "swmm_groundtruth_targets.csv"
        print(f"Multi-scenario target CSV not found yet, falling back to '{csv_file}'...")
        df_targets = pd.read_csv(csv_file)
        if 'intensity_mmhr' not in df_targets.columns:
            df_targets['intensity_mmhr'] = 50.0
            df_targets['region'] = 'all'
            df_targets['scenario_idx'] = 0
    else:
        print(f"1. Loading ground truth targets from '{csv_file}'...")
        df_targets = pd.read_csv(csv_file)

    # Group target lookup by (region/all, scenario_idx/intensity, node_id)
    target_groups = df_targets.groupby(['intensity_mmhr', 'region'])

    # Pre-load graphs
    graphs = {}
    graph_node_maps = {}
    graph_stats = {}

    for reg, file_path in region_files.items():
        print(f"Pre-loading graph for region '{reg}' ({file_path})...")
        G = ox.load_graphml(file_path)
        nodes_list = list(G.nodes())
        node_to_idx = {node_id: idx for idx, node_id in enumerate(nodes_list)}
        
        xs = [float(G.nodes[nid].get('x', 0.0)) for nid in nodes_list]
        ys = [float(G.nodes[nid].get('y', 0.0)) for nid in nodes_list]
        elevs = [float(G.nodes[nid].get('elevation', 880.0)) for nid in nodes_list]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_elev, max_elev = min(elevs), max(elevs)
        
        x_range = max(1.0, max_x - min_x)
        y_range = max(1.0, max_y - min_y)
        elev_range = max(1.0, max_elev - min_elev)
        
        in_deg_map = dict(G.in_degree())
        out_deg_map = dict(G.out_degree())
        
        node_in_grades = {nid: [] for nid in nodes_list}
        node_out_grades = {nid: [] for nid in nodes_list}
        
        for u, v, k, data in G.edges(keys=True, data=True):
            grade = float(data.get('grade', 0.0))
            if u in node_out_grades:
                node_out_grades[u].append(grade)
            if v in node_in_grades:
                node_in_grades[v].append(grade)

        graphs[reg] = G
        graph_node_maps[reg] = node_to_idx
        graph_stats[reg] = {
            'nodes_list': nodes_list,
            'min_x': min_x, 'x_range': x_range,
            'min_y': min_y, 'y_range': y_range,
            'min_elev': min_elev, 'max_elev': max_elev,
            'elev_range': elev_range,
            'in_deg_map': in_deg_map,
            'out_deg_map': out_deg_map,
            'node_in_grades': node_in_grades,
            'node_out_grades': node_out_grades
        }

    scenarios = df_targets[['intensity_mmhr', 'scenario_idx']].drop_duplicates().to_dict('records')
    scenarios = sorted(scenarios, key=lambda s: s['intensity_mmhr'])
    print(f"\nBuilding 12-feature PyG graph objects for {len(scenarios)} storm scenarios across {len(region_files)} regions...")

    all_graph_objects = []

    for scen in scenarios:
        intensity = float(scen['intensity_mmhr'])
        duration = float(STORM_DURATIONS.get(intensity, 60.0))
        
        # Sub-dataframe for this scenario
        df_scen = df_targets[df_targets['intensity_mmhr'] == intensity]
        
        # Lookup map (region, node_id) -> max_water_depth_m
        target_lookup = {}
        for _, row in df_scen.iterrows():
            r_key = str(row['region']).lower().strip()
            node_str = str(row['swmm_node_id'])
            val = float(row['max_water_depth_m'])
            target_lookup[(r_key, node_str)] = val
            if node_str.startswith('J_'):
                target_lookup[(r_key, node_str[2:])] = val
            else:
                target_lookup[(r_key, f"J_{node_str}")] = val

        region_data_list = []

        for reg in region_files.keys():
            G = graphs[reg]
            stats = graph_stats[reg]
            nodes_list = stats['nodes_list']
            node_to_idx = graph_node_maps[reg]
            
            node_features = []
            node_targets = []
            
            for node_id in nodes_list:
                data = G.nodes[node_id]
                x_coord = float(data.get('x', 0.0))
                y_coord = float(data.get('y', 0.0))
                elev = float(data.get('elevation', 880.0))
                imp = float(data.get('impervious_ratio', 0.2))
                manning_n = float(data.get('manning_n', 0.013))
                
                rel_x = (x_coord - stats['min_x']) / stats['x_range']
                rel_y = (y_coord - stats['min_y']) / stats['y_range']
                rel_drop = (stats['max_elev'] - elev) / stats['elev_range']
                
                in_deg = stats['in_deg_map'].get(node_id, 0)
                out_deg = stats['out_deg_map'].get(node_id, 0)
                accum_score = np.log1p(in_deg * 2.5 + (1.0 if out_deg == 0 else 0.0))
                is_sink = 1.0 if (rel_drop > 0.85 and in_deg >= 2) else 0.0
                
                in_grades = stats['node_in_grades'].get(node_id, [0.0])
                out_grades = stats['node_out_grades'].get(node_id, [0.0])
                
                max_in_grade = max(in_grades) if len(in_grades) > 0 else 0.0
                min_out_grade = min(out_grades) if len(out_grades) > 0 else 0.0
                sag_index = max(0.0, max_in_grade - min_out_grade) * max(1, in_deg)
                hydraulic_capacity = float(in_deg) / max(1.0, float(out_deg))
                
                # 14 Node Features: Spatially invariant relative topology + Saint-Venant hydraulics + dynamic meteorological forcing
                node_features.append([
                    rel_x, rel_y, rel_drop, imp, manning_n, float(in_deg), float(out_deg), accum_score, is_sink, max_in_grade, sag_index, hydraulic_capacity,
                    intensity, duration
                ])



                nid_str = str(node_id)
                depth = target_lookup.get((reg, nid_str),
                        target_lookup.get((reg, f"J_{nid_str}"),
                        target_lookup.get((reg, nid_str.replace('J_', '')), 0.0)))
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
            pyg_data.rain_intensity = intensity
            pyg_data.rain_duration = duration
            pyg_data.region = reg
            region_data_list.append(pyg_data)
            all_graph_objects.append(pyg_data)

    # Combine all individual region-scenario graphs into a single master PyG Batch object for backwards compatibility
    master_batch = Batch.from_data_list(all_graph_objects)
    
    # Save both single combined batch and unbatched scenario dataset list
    torch.save(master_batch, "bengaluru_pyg_dataset.pt")
    torch.save(all_graph_objects, "multi_scenario_pyg_dataset.pt")
    
    print(f"\nSuccessfully built & saved 12-feature dataset!")
    print(f"  - 'multi_scenario_pyg_dataset.pt': List of {len(all_graph_objects)} individual region-scenario graphs.")
    print(f"  - 'bengaluru_pyg_dataset.pt': Combined batch containing {master_batch.x.shape[0]} total nodes across all scenarios.")

if __name__ == "__main__":
    create_12feature_dataset()


