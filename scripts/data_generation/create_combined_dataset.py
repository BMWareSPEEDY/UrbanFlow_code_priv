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
    'koramangala': 'bengaluru_koramangala_graph.graphml',
    'hyderabad': 'city_hyderabad_graph.graphml',
    'chennai': 'city_chennai_graph.graphml',
    'pune': 'city_pune_graph.graphml',
    'mumbai': 'city_mumbai_graph.graphml',
    'delhi': 'city_delhi_graph.graphml',
    'kolkata': 'city_kolkata_graph.graphml',
    'ahmedabad': 'city_ahmedabad_graph.graphml',
    'jaipur': 'city_jaipur_graph.graphml',
    'lucknow': 'city_lucknow_graph.graphml',
    'kochi': 'city_kochi_graph.graphml',
    'surat': 'city_surat_graph.graphml',
    'indore': 'city_indore_graph.graphml',
    'bhopal': 'city_bhopal_graph.graphml',
    'nagpur': 'city_nagpur_graph.graphml',
    'visakhapatnam': 'city_visakhapatnam_graph.graphml',
    'coimbatore': 'city_coimbatore_graph.graphml',
    'patna': 'city_patna_graph.graphml',
    'kanpur': 'city_kanpur_graph.graphml',
    'chandigarh': 'city_chandigarh_graph.graphml',
    'vadodara': 'city_vadodara_graph.graphml',
    'madurai': 'city_madurai_graph.graphml',
    'guwahati': 'city_guwahati_graph.graphml',
    'vijayawada': 'city_vijayawada_graph.graphml',
    'varanasi': 'city_varanasi_graph.graphml',
    'rajkot': 'city_rajkot_graph.graphml',
    'ludhiana': 'city_ludhiana_graph.graphml',
    'ranchi': 'city_ranchi_graph.graphml',
    'agra': 'city_agra_graph.graphml',
    'e_varanasi': 'city_varanasi_extreme.graphml',
    'e_rajkot': 'city_rajkot_extreme.graphml',
    'e_ludhiana': 'city_ludhiana_extreme.graphml',
    'e_ranchi': 'city_ranchi_extreme.graphml',
    'e_agra': 'city_agra_extreme.graphml',
}

CITY_OF_REGION = {
    'hsr': 'bangalore', 'bellandur': 'bangalore', 'whitefield': 'bangalore',
    'ecity': 'bangalore', 'koramangala': 'bangalore',
    'hyderabad': 'hyderabad', 'chennai': 'chennai', 'pune': 'pune',
    'mumbai': 'mumbai', 'delhi': 'delhi',
    'kolkata': 'kolkata', 'ahmedabad': 'ahmedabad', 'jaipur': 'jaipur',
    'lucknow': 'lucknow', 'kochi': 'kochi',
    'surat': 'surat', 'indore': 'indore', 'bhopal': 'bhopal',
    'nagpur': 'nagpur', 'visakhapatnam': 'visakhapatnam',
    'coimbatore': 'coimbatore', 'patna': 'patna',
    'kanpur': 'kanpur', 'chandigarh': 'chandigarh', 'vadodara': 'vadodara',
    'madurai': 'madurai', 'guwahati': 'guwahati', 'vijayawada': 'vijayawada',
    'varanasi': 'varanasi', 'rajkot': 'rajkot', 'ludhiana': 'ludhiana',
    'ranchi': 'ranchi', 'agra': 'agra',
    'e_varanasi': 'e_varanasi', 'e_rajkot': 'e_rajkot', 'e_ludhiana': 'e_ludhiana',
    'e_ranchi': 'e_ranchi', 'e_agra': 'e_agra',
}

STORM_DURATIONS = {
    20.0: 60.0, 50.0: 60.0, 80.0: 45.0, 120.0: 30.0,
    150.0: 30.0, 200.0: 45.0, 250.0: 30.0, 300.0: 30.0,
}

SUBCATCHMENT_AREA_HA = 0.5


def compute_flow_accumulation(G, nodes_list, node_to_idx, in_deg_map, out_deg_map):
    """Approximate upstream drainage area via elevation-ordered propagation."""
    elevs = {nid: float(G.nodes[nid].get('elevation', 880.0)) for nid in nodes_list}
    imps = {nid: float(G.nodes[nid].get('impervious_ratio', 0.2)) for nid in nodes_list}

    out_neighbors = {nid: [] for nid in nodes_list}
    for u, v, k, data in G.edges(keys=True, data=True):
        if u in out_neighbors and u != v:
            out_neighbors[u].append(v)

    acc_area = {nid: SUBCATCHMENT_AREA_HA for nid in nodes_list}
    acc_imperv_area = {nid: SUBCATCHMENT_AREA_HA * imps[nid] for nid in nodes_list}

    ordered = sorted(nodes_list, key=lambda n: elevs[n], reverse=True)
    for u in ordered:
        contrib_area = acc_area[u]
        contrib_imp = acc_imperv_area[u]
        for v in out_neighbors[u]:
            acc_area[v] += contrib_area
            acc_imperv_area[v] += contrib_imp

    # Distance (number of hops + length weighted) to lowest-elevation outlet
    lowest = min(nodes_list, key=lambda n: elevs[n])
    from collections import deque
    dist = {nid: float('inf') for nid in nodes_list}
    dist[lowest] = 0.0
    dq = deque([lowest])
    while dq:
        u = dq.popleft()
        for v in out_neighbors[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                dq.append(v)
    finite = [d for d in dist.values() if d != float('inf')]
    max_dist = max(finite) if finite else 0.0
    dist_frac = {}
    for nid in nodes_list:
        d = dist[nid]
        dist_frac[nid] = (0.0 if d == float('inf') or max_dist == 0 else d / max_dist)

    return acc_area, acc_imperv_area, dist_frac


def compute_capacity_limited_flow(G, nodes_list, intensity):
    """Per-node routed flow and ponded excess (m3/s) with capacity-limited propagation.
    Flow is split across downstream edges proportional to conduit full-flow capacity;
    excess that cannot be passed ponds at the node (SWMM-like surcharge prior)."""
    imps = {nid: float(G.nodes[nid].get('impervious_ratio', 0.2)) for nid in nodes_list}
    elevs = {nid: float(G.nodes[nid].get('elevation', 880.0)) for nid in nodes_list}
    runoff_coeff = {nid: imps[nid] * 0.9 + (1.0 - imps[nid]) * 0.3 for nid in nodes_list}
    q_own = {nid: intensity * SUBCATCHMENT_AREA_HA * 10000.0 * runoff_coeff[nid] / 3600000.0
             for nid in nodes_list}

    out_neigh = {nid: [] for nid in nodes_list}
    for u, v, k, data in G.edges(keys=True, data=True):
        if u in out_neigh and u != v:
            g = abs(float(data.get('grade', 0.0)))
            n = float(data.get('manning_n', 0.013))
            cap = (1.0 / max(n, 1e-4)) * 1.5 * 0.520 * np.sqrt(max(g, 1e-6))
            out_neigh[u].append((v, cap))

    inflow = {nid: 0.0 for nid in nodes_list}
    flow = {nid: 0.0 for nid in nodes_list}
    pond = {nid: 0.0 for nid in nodes_list}
    ordered = sorted(nodes_list, key=lambda n: elevs[n], reverse=True)
    for u in ordered:
        total_in = inflow[u] + q_own[u]
        flow[u] = total_in
        outs = out_neigh[u]
        total_cap = sum(c for _, c in outs)
        if len(outs) == 0 or total_cap <= 1e-9:
            pond[u] = total_in
            continue
        passed = min(total_in, total_cap)
        pond[u] = total_in - passed
        for v, c in outs:
            inflow[v] += passed * (c / total_cap)

    return flow, pond


def create_dataset():
    df_targets = pd.read_csv("swmm_multi_scenario_targets.csv")
    df_city = pd.read_csv("swmm_city_targets.csv")
    df_b2 = pd.read_csv("swmm_city_b2_targets.csv")
    df_b3 = pd.read_csv("swmm_city_b3_targets.csv")
    df_b4 = pd.read_csv("swmm_city_b4_targets.csv")
    df_b5 = pd.read_csv("swmm_city_b5_targets.csv")
    df_ext = pd.read_csv("swmm_city_ext_targets.csv")
    df_all = pd.concat([df_targets, df_city, df_b2, df_b3, df_b4, df_b5, df_ext], ignore_index=True)
    print(f"Loaded {len(df_all)} target records.")

    graphs = {}
    graph_node_maps = {}
    graph_stats = {}

    for reg, file_path in region_files.items():
        print(f"Pre-loading graph for region '{reg}'...")
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

        acc_area, acc_imperv_area, dist_frac = compute_flow_accumulation(
            G, nodes_list, node_to_idx, in_deg_map, out_deg_map)

        # Sub-grid micro-topography: elevation std within 2-hop catchment,
        # depression depth (local pit below neighbors), surcharge ratio
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

        # Downstream path capacity: min full-flow capacity along downstream route
        # to the graph outlet, and hop-distance to outlet. Bottlenecked nodes flood.
        def edge_capacity(u, v):
            g = abs(float(G.edges[(u, v, list(G[u][v].keys())[0])].get('grade', 0.0)))
            n = float(G.edges[(u, v, list(G[u][v].keys())[0])].get('manning_n', 0.013))
            return (1.0 / max(n, 1e-4)) * 1.5 * 0.520 * np.sqrt(max(g, 1e-6))

        path_cap = {nid: float('inf') for nid in nodes_list}
        path_hops = {nid: 0 for nid in nodes_list}
        ordered = sorted(nodes_list, key=lambda n: float(G.nodes[n].get('elevation', 880.0)), reverse=True)
        for u in ordered:
            outs = out_deg_map[u]
            if outs == 0:
                path_cap[u] = 0.0  # outlet node
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
            cap_e = edge_capacity(u, best_v)
            path_cap[u] = min(cap_e, path_cap[best_v])
            path_hops[u] = path_hops[best_v] + 1
        path_cap = {nid: float(np.log1p(v)) if v != float('inf') else 0.0 for nid, v in path_cap.items()}

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
            'node_out_grades': node_out_grades,
            'acc_area': acc_area,
            'acc_imperv_area': acc_imperv_area,
            'dist_frac': dist_frac,
            'elev_std2': elev_std2,
            'dep_depth': dep_depth,
            'surcharge': surcharge,
            'path_cap': path_cap,
            'path_hops': path_hops,
        }

    scenarios = sorted(set(df_all['intensity_mmhr']))
    all_graph_objects = []

    for intensity in scenarios:
        intensity = float(intensity)
        duration = float(STORM_DURATIONS.get(intensity, 60.0))
        df_scen = df_all[df_all['intensity_mmhr'] == intensity]

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

            flow, pond = compute_capacity_limited_flow(G, nodes_list, intensity)

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
                    float(np.log1p(flow.get(node_id, 0.0))),
                    float(np.log1p(pond.get(node_id, 0.0))),
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

    torch.save(all_graph_objects, "multi_scenario_pyg_dataset.pt")
    print(f"\nSaved {len(all_graph_objects)} graphs with 24 node features.")
    print(f"Cities: {sorted({g.city for g in all_graph_objects})}")


if __name__ == "__main__":
    create_dataset()