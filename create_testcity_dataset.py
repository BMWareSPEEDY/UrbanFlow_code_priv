"""Build 22-feature PyG dataset for the held-out real-terrain test cities (nyc, london).
Mirrors the ORIGINAL 22-feature schema used to train iter2/iter6 models."""
import sys
import osmnx as ox
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW")
from create_combined_dataset import compute_flow_accumulation

region_files = {
    'nyc': 'city_nyc_graph.graphml',
    'london': 'city_london_graph.graphml',
}

CITY_OF_REGION = {'nyc': 'newyork', 'london': 'london'}

STORM_DURATIONS = {
    20.0: 60.0, 50.0: 60.0, 80.0: 45.0, 120.0: 30.0,
    150.0: 30.0, 200.0: 45.0, 250.0: 30.0, 300.0: 30.0,
}

SUBCATCHMENT_AREA_HA = 0.5


def edge_capacity(G, u, v):
    g = abs(float(G.edges[(u, v, list(G[u][v].keys())[0])].get('grade', 0.0)))
    n = float(G.edges[(u, v, list(G[u][v].keys())[0])].get('manning_n', 0.013))
    return (1.0 / max(n, 1e-4)) * 1.5 * 0.520 * np.sqrt(max(g, 1e-6))


def build():
    df = pd.read_csv("swmm_city_testcities_targets.csv")
    print(f"Loaded {len(df)} test-city target records.")
    graphs = {}
    graph_node_maps = {}
    graph_stats = {}

    for reg, file_path in region_files.items():
        print(f"Pre-loading graph for region '{reg}'...")
        G = ox.load_graphml(file_path)
        nodes_list = list(G.nodes())
        node_to_idx = {nid: idx for idx, nid in enumerate(nodes_list)}

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

        acc_area, acc_imperv_area, dist_frac = compute_flow_accumulation(
            G, nodes_list, node_to_idx, in_deg_map, out_deg_map)

        elev_map = {nid: float(G.nodes[nid].get('elevation', 880.0)) for nid in nodes_list}
        und_adj = {nid: set() for nid in nodes_list}
        for u, v, k, data in G.edges(keys=True, data=True):
            if u in und_adj and v in und_adj:
                und_adj[u].add(v)
                und_adj[v].add(u)
        elev_std2, dep_depth = {}, {}
        for nid in nodes_list:
            hood = {nid}
            for nb in und_adj[nid]:
                hood.add(nb)
                for nb2 in und_adj[nb]:
                    hood.add(nb2)
            he = [elev_map[n] for n in hood]
            elev_std2[nid] = float(np.std(he))
            nbr_elevs = [elev_map[nb] for nb in und_adj[nid]]
            dep_depth[nid] = float(max(0.0, np.mean(nbr_elevs) - elev_map[nid])) if nbr_elevs else 0.0
        surcharge = {}
        for nid in nodes_list:
            gs = node_out_grades[nid]
            max_out_grade = max(gs) if len(gs) > 0 else 1e-4
            surcharge[nid] = float(np.log1p(acc_area[nid] / max(1e-4, abs(max_out_grade) + 1e-4)))

        path_cap = {nid: float('inf') for nid in nodes_list}
        path_hops = {nid: 0 for nid in nodes_list}
        ordered = sorted(nodes_list, key=lambda n: float(G.nodes[n].get('elevation', 880.0)), reverse=True)
        for u in ordered:
            outs = out_deg_map[u]
            if outs == 0:
                path_cap[u] = 0.0
                path_hops[u] = 0
                continue
            best_v = None
            best_g = -1.0
            for v in G.successors(u):
                if u == v:
                    continue
                for k, d in G[u][v].items():
                    gg = abs(float(d.get('grade', 0.0)))
                    if gg > best_g:
                        best_g, best_v = gg, v
            if best_v is None:
                path_cap[u] = 0.0
                continue
            cap_e = edge_capacity(G, u, best_v)
            path_cap[u] = min(cap_e, path_cap[best_v])
            path_hops[u] = path_hops[best_v] + 1
        path_cap = {nid: float(np.log1p(v)) if v != float('inf') else 0.0 for nid, v in path_cap.items()}

        graphs[reg] = G
        graph_node_maps[reg] = node_to_idx
        graph_stats[reg] = {
            'nodes_list': nodes_list, 'min_x': min_x, 'x_range': x_range,
            'min_y': min_y, 'y_range': y_range,
            'min_elev': min_elev, 'max_elev': max_elev, 'elev_range': elev_range,
            'in_deg_map': in_deg_map, 'out_deg_map': out_deg_map,
            'node_in_grades': node_in_grades, 'node_out_grades': node_out_grades,
            'acc_area': acc_area, 'acc_imperv_area': acc_imperv_area, 'dist_frac': dist_frac,
            'elev_std2': elev_std2, 'dep_depth': dep_depth, 'surcharge': surcharge,
            'path_cap': path_cap, 'path_hops': path_hops,
        }

    scenarios = sorted(set(df['intensity_mmhr']))
    all_graph_objects = []

    for intensity in scenarios:
        intensity = float(intensity)
        duration = float(STORM_DURATIONS.get(intensity, 60.0))
        df_scen = df[df['intensity_mmhr'] == intensity]

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

        for reg in region_files.keys():
            G = graphs[reg]
            st = graph_stats[reg]
            nodes_list = st['nodes_list']
            node_to_idx = graph_node_maps[reg]

            node_features = []
            node_targets = []

            for node_id in nodes_list:
                d = G.nodes[node_id]
                x_coord = float(d.get('x', 0.0))
                y_coord = float(d.get('y', 0.0))
                elev = float(d.get('elevation', 880.0))
                imp = float(d.get('impervious_ratio', 0.2))
                manning_n = float(d.get('manning_n', 0.013))

                rel_x = (x_coord - st['min_x']) / st['x_range']
                rel_y = (y_coord - st['min_y']) / st['y_range']
                rel_drop = (st['max_elev'] - elev) / st['elev_range']

                in_deg = st['in_deg_map'].get(node_id, 0)
                out_deg = st['out_deg_map'].get(node_id, 0)
                accum_score = np.log1p(in_deg * 2.5 + (1.0 if out_deg == 0 else 0.0))
                is_sink = 1.0 if (rel_drop > 0.85 and in_deg >= 2) else 0.0

                in_grades = st['node_in_grades'].get(node_id, [0.0])
                out_grades = st['node_out_grades'].get(node_id, [0.0])
                max_in_grade = max(in_grades) if len(in_grades) > 0 else 0.0
                min_out_grade = min(out_grades) if len(out_grades) > 0 else 0.0
                sag_index = max(0.0, max_in_grade - min_out_grade) * max(1, in_deg)
                hydraulic_capacity = float(in_deg) / max(1.0, float(out_deg))

                log_area = float(np.log1p(st['acc_area'].get(node_id, 0.5)))
                log_imp_area = float(np.log1p(st['acc_imperv_area'].get(node_id, 0.5)))
                dist_frac = float(st['dist_frac'].get(node_id, 0.0))

                node_features.append([
                    rel_x, rel_y, rel_drop, imp, manning_n, float(in_deg), float(out_deg),
                    accum_score, is_sink, max_in_grade, sag_index, hydraulic_capacity,
                    log_area, log_imp_area, dist_frac,
                    intensity, duration,
                    st['elev_std2'].get(node_id, 0.0), st['dep_depth'].get(node_id, 0.0),
                    st['surcharge'].get(node_id, 0.0),
                    st['path_cap'].get(node_id, 0.0), st['path_hops'].get(node_id, 0.0),
                ])

                nid_str = str(node_id)
                depth = target_lookup.get(
                    (reg, nid_str),
                    target_lookup.get((reg, f"J_{nid_str}"),
                                      target_lookup.get((reg, nid_str.replace('J_', '')), 0.0)))
                node_targets.append([depth])

            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.tensor(node_targets, dtype=torch.float)

            src_nodes, dst_nodes, edge_features = [], [], []
            for u, v, k, data in G.edges(keys=True, data=True):
                src_nodes.append(node_to_idx[u])
                dst_nodes.append(node_to_idx[v])
                edge_features.append([
                    float(data.get('length', 10.0)),
                    float(data.get('grade', 0.0)),
                ])

            edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

            pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            pyg_data.rain_intensity = intensity
            pyg_data.rain_duration = duration
            pyg_data.region = reg
            pyg_data.city = CITY_OF_REGION[reg]
            all_graph_objects.append(pyg_data)

    torch.save(all_graph_objects, "multi_scenario_testcities_pyg_dataset.pt")
    print(f"\nSaved {len(all_graph_objects)} test-city graphs with 22 node features.")
    for g in all_graph_objects:
        y = g.y.numpy().ravel()
        if abs(g.rain_intensity - 300.0) < 1e-6:
            print(f"{g.city:<10} nodes {len(y):6d} mean_y {y.mean():.3f} flood% {(y>=0.15).mean()*100:5.1f}")


if __name__ == "__main__":
    build()