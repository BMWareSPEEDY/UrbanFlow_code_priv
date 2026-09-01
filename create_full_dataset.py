"""Full 22-feature dataset: 29 original cities + 5 extreme-relief synthetic +
6 real-terrain international cities + bangalore. Held-out: bangalore, hongkong."""
import sys
import osmnx as ox
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW")
from create_combined_dataset import compute_flow_accumulation

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
    'nyc': 'city_nyc_graph.graphml',
    'london': 'city_london_graph.graphml',
    'singapore': 'city_singapore_graph.graphml',
    'tokyo': 'city_tokyo_graph.graphml',
    'paris': 'city_paris_graph.graphml',
    'hongkong': 'city_hongkong_graph.graphml',
    'mumbai_r': 'city_mumbai_r_graph.graphml',
    'jakarta': 'city_jakarta_graph.graphml',
    'bangkok': 'city_bangkok_graph.graphml',
    'istanbul': 'city_istanbul_graph.graphml',
    'berlin': 'city_berlin_graph.graphml',
    'chicago': 'city_chicago_graph.graphml',
    'sanfrancisco': 'city_sanfrancisco_graph.graphml',
    'seoul': 'city_seoul_graph.graphml',
    'sydney': 'city_sydney_graph.graphml',
    'taipei': 'city_taipei_graph.graphml',
    'lisbon': 'city_lisbon_graph.graphml',
    'vancouver': 'city_vancouver_graph.graphml',
    'rio': 'city_rio_graph.graphml',
    'naples': 'city_naples_graph.graphml',
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
    'nyc': 'newyork', 'london': 'london', 'singapore': 'singapore',
    'tokyo': 'tokyo', 'paris': 'paris', 'hongkong': 'hongkong',
    'mumbai_r': 'mumbai_r', 'jakarta': 'jakarta', 'bangkok': 'bangkok',
    'istanbul': 'istanbul', 'berlin': 'berlin', 'chicago': 'chicago',
    'sanfrancisco': 'sanfrancisco', 'seoul': 'seoul', 'sydney': 'sydney',
    'taipei': 'taipei', 'lisbon': 'lisbon', 'vancouver': 'vancouver',
    'rio': 'rio', 'naples': 'naples',
}

STORM_DURATIONS = {
    20.0: 60.0, 50.0: 60.0, 80.0: 45.0, 120.0: 30.0,
    150.0: 30.0, 200.0: 45.0, 250.0: 30.0, 300.0: 30.0,
}

SUBCATCHMENT_AREA_HA = 0.5

TARGET_FILES = ["swmm_multi_scenario_targets.csv", "swmm_city_targets.csv",
                "swmm_city_b2_targets.csv", "swmm_city_b3_targets.csv",
                "swmm_city_b4_targets.csv", "swmm_city_b5_targets.csv",
                "swmm_city_ext_targets.csv", "swmm_city_testcities_targets.csv",
                "swmm_city_real4_targets.csv", "swmm_city_real10_targets.csv",
                "swmm_city_coastal_targets.csv", "swmm_city_breal_targets.csv",
                "swmm_city_indreal_targets.csv"]


def edge_capacity(G, u, v):
    g = abs(float(G.edges[(u, v, list(G[u][v].keys())[0])].get('grade', 0.0)))
    n = float(G.edges[(u, v, list(G[u][v].keys())[0])].get('manning_n', 0.013))
    return (1.0 / max(n, 1e-4)) * 1.5 * 0.520 * np.sqrt(max(g, 1e-6))


def build():
    dfs = [pd.read_csv(f) for f in TARGET_FILES]
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df_all)} target records.")
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

        inlet_cap = {}
        highway_to_cap = {
            'motorway': 0.50, 'trunk': 0.50, 'primary': 0.35,
            'secondary': 0.25, 'tertiary': 0.18, 'unclassified': 0.15,
            'residential': 0.12, 'service': 0.06, 'living_street': 0.04,
            'pedestrian': 0.03, 'track': 0.02, 'path': 0.01
        }
        default_cap = 0.12
        for nid in nodes_list:
            max_cap = 0.0
            for u, v, k, data in G.out_edges(nid, keys=True, data=True):
                hw = data.get('highway', 'residential')
                if isinstance(hw, list):
                    hw = hw[0]
                cap = highway_to_cap.get(hw, default_cap)
                max_cap = max(max_cap, cap)
            inlet_cap[nid] = max_cap if max_cap > 0 else default_cap

        surcharge_ratio_base = {}
        for nid in nodes_list:
            cap = inlet_cap[nid] if inlet_cap[nid] > 0 else default_cap
            surcharge_ratio_base[nid] = float(acc_area[nid] / max(cap, 1e-4))

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

        outlet_id = min(nodes_list, key=lambda n: elev_map[n])
        outlet_elev = elev_map[outlet_id]
        dist_outlet = {nid: float('inf') for nid in nodes_list}
        dist_outlet[outlet_id] = 0.0
        dq = [outlet_id]
        for u in dq:
            for v in und_adj[u]:
                if dist_outlet[v] > dist_outlet[u] + 1:
                    dist_outlet[v] = dist_outlet[u] + 1
                    dq.append(v)
        node_grade_max = {nid: 0.0 for nid in nodes_list}
        for u, v, k, data in G.edges(keys=True, data=True):
            g = abs(float(data.get('grade', 0.0)))
            if u in node_grade_max:
                node_grade_max[u] = max(node_grade_max[u], g)
            if v in node_grade_max:
                node_grade_max[v] = max(node_grade_max[v], g)
        slope_outlet_ratio, sink_depth = {}, {}
        for nid in nodes_list:
            hood = {nid}
            for nb in und_adj[nid]:
                hood.add(nb)
                for nb2 in und_adj[nb]:
                    hood.add(nb2)
            gmax = max(node_grade_max[n] for n in hood)
            above = max(0.0, elev_map[nid] - outlet_elev)
            slope_outlet_ratio[nid] = float(gmax / (0.3 + above))
            nbr_elevs = [elev_map[nb] for nb in und_adj[nid]]
            sink_depth[nid] = float(max(0.0, min(nbr_elevs) - elev_map[nid])) if nbr_elevs else 0.0

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
            'dist_outlet': dist_outlet, 'outlet_elev': outlet_elev,
            'slope_outlet_ratio': slope_outlet_ratio, 'sink_depth': sink_depth,
            'inlet_cap': inlet_cap, 'surcharge_ratio_base': surcharge_ratio_base,
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
                inlet_cap = float(st['inlet_cap'].get(node_id, 0.12))
                surcharge_ratio = float(st['surcharge_ratio_base'].get(node_id, 0.0)) * intensity

                node_features.append([
                    rel_x, rel_y, rel_drop, imp, manning_n, float(in_deg), float(out_deg),
                    accum_score, is_sink, max_in_grade, sag_index, hydraulic_capacity,
                    log_area, log_imp_area, dist_frac,
                    intensity, duration,
                    st['elev_std2'].get(node_id, 0.0), st['dep_depth'].get(node_id, 0.0),
                    st['surcharge'].get(node_id, 0.0),
                    st['path_cap'].get(node_id, 0.0), st['path_hops'].get(node_id, 0.0),
                    st['dist_outlet'].get(node_id, 999.0),
                    max(0.0, elev - st['outlet_elev']),
                    st['slope_outlet_ratio'].get(node_id, 0.0),
                    st['sink_depth'].get(node_id, 0.0),
                    inlet_cap, surcharge_ratio,
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
            down_src, down_dst, down_feats = [], [], []
            up_src, up_dst, up_feats = [], [], []
            elev_map = {nid: float(G.nodes[nid].get('elevation', 0.0)) for nid in nodes_list}
            for u, v, k, data in G.edges(keys=True, data=True):
                src_nodes.append(node_to_idx[u])
                dst_nodes.append(node_to_idx[v])
                edge_features.append([
                    float(data.get('length', 10.0)),
                    float(data.get('grade', 0.0)),
                ])
                z_u = elev_map.get(u, 0.0)
                z_v = elev_map.get(v, 0.0)
                if z_u > z_v + 1e-4:
                    down_src.append(node_to_idx[u])
                    down_dst.append(node_to_idx[v])
                    down_feats.append([float(data.get('length', 10.0)), float(data.get('grade', 0.0))])
                elif z_v > z_u + 1e-4:
                    up_src.append(node_to_idx[v])
                    up_dst.append(node_to_idx[u])
                    up_feats.append([float(data.get('length', 10.0)), float(data.get('grade', 0.0))])

            edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            edge_index_down = torch.tensor([down_src, down_dst], dtype=torch.long)
            edge_attr_down = torch.tensor(down_feats, dtype=torch.float)
            edge_index_up = torch.tensor([up_src, up_dst], dtype=torch.long)
            edge_attr_up = torch.tensor(up_feats, dtype=torch.float)

            pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            pyg_data.edge_index_down = edge_index_down
            pyg_data.edge_attr_down = edge_attr_down
            pyg_data.edge_index_up = edge_index_up
            pyg_data.edge_attr_up = edge_attr_up
            pyg_data.rain_intensity = intensity
            pyg_data.rain_duration = duration
            pyg_data.region = reg
            pyg_data.city = CITY_OF_REGION[reg]
            all_graph_objects.append(pyg_data)

    torch.save(all_graph_objects, "multi_scenario_full22_pyg_dataset.pt")
    print(f"\nSaved {len(all_graph_objects)} graphs with 22 node features.")
    for g in all_graph_objects:
        if abs(g.rain_intensity - 300.0) < 1e-6:
            y = g.y.numpy().ravel()
            x = g.x.numpy()
            print(f"{g.city:<12} n {len(y):6d} mean_y {y.mean():.3f} flood% {(y>=0.15).mean()*100:5.1f} "
                  f"elev2 {x[:,17].mean():6.2f}")


if __name__ == "__main__":
    build()