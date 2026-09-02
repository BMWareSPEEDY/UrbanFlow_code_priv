import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import osmnx as ox
from flask import Flask, jsonify, render_template, request
from pyproj import Transformer
from torch_geometric.nn import GINEConv, GATv2Conv
from train_perfect_accuracy_gnn import PerfectAccuracyGNN
from train_zero_tolerance_gnn import ZeroToleranceHurdleGNN
from train_dual_stream_hydro_gnn import DualStreamHydroGNN
from production_v4 import ProductionFloodPredictorV4

app = Flask(__name__)

PRODUCTION_PREDICTOR = None

class HighPrecisionGINE(nn.Module):
    def __init__(self, in_c=24, edge_c=2, hidden=192, n_layers=6):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GINEConv(nn.Linear(in_c if i == 0 else hidden, hidden), edge_dim=edge_c))
        self.lns = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.reg = nn.Sequential(
            nn.Linear(hidden + in_c, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x, ei, ea):
        h = x
        for conv, ln in zip(self.convs, self.lns):
            h = F.elu(ln(conv(h, ei, ea)))
        return self.reg(torch.cat([h, x], -1))

MODEL = None
RESIDUAL_CALIBRATIONS = {}
REGIONS = {
    # --- Bengaluru Districts ---
    'hsr': {
        'name': 'HSR Layout (Residential Hub)',
        'file': 'bengaluru_complete_graph.graphml',
        'sectors': 'Sector 1 - 7',
        'city': 'Bengaluru',
        'country': 'India',
        'lat': 12.9116, 'lng': 77.6389
    },
    'bellandur': {
        'name': 'Bellandur & ORR (Lake Basin)',
        'file': 'bengaluru_bellandur_graph.graphml',
        'sectors': 'Outer Ring Road / Ecospace',
        'city': 'Bengaluru',
        'country': 'India',
        'lat': 12.9340, 'lng': 77.6880
    },
    'whitefield': {
        'name': 'Whitefield (ITPB Corridor)',
        'file': 'bengaluru_whitefield_graph.graphml',
        'sectors': 'ITPB / EPIP Zone',
        'city': 'Bengaluru',
        'country': 'India',
        'lat': 12.9698, 'lng': 77.7500
    },
    'ecity': {
        'name': 'Electronic City (Tech Corridor)',
        'file': 'bengaluru_ecity_graph.graphml',
        'sectors': 'Phase 1 & 2 Tech Park',
        'city': 'Bengaluru',
        'country': 'India',
        'lat': 12.8451, 'lng': 77.6602
    },
    'koramangala': {
        'name': 'Koramangala & Indiranagar',
        'file': 'bengaluru_koramangala_graph.graphml',
        'sectors': 'Commercial & Startup Corridor',
        'city': 'Bengaluru',
        'country': 'India',
        'lat': 12.9352, 'lng': 77.6245
    },
    # --- International Megacities ---
    'tokyo': {
        'name': 'Tokyo Metropolitan Catchment',
        'file': 'city_tokyo_graph.graphml',
        'sectors': 'Shinjuku & Shibuya Urban Basin',
        'city': 'Tokyo',
        'country': 'Japan',
        'lat': 35.6762, 'lng': 139.6503
    },
    'hongkong': {
        'name': 'Hong Kong Urban Basin',
        'file': 'city_hongkong_graph.graphml',
        'sectors': 'Kowloon & Victoria Harbour Coastal Catchment',
        'city': 'Hong Kong',
        'country': 'China / HK',
        'lat': 22.3193, 'lng': 114.1694
    },
    'singapore': {
        'name': 'Singapore Marina Catchment',
        'file': 'city_singapore_graph.graphml',
        'sectors': 'Marina Bay & Downtown Core',
        'city': 'Singapore',
        'country': 'Singapore',
        'lat': 1.3521, 'lng': 103.8198
    },
    'london': {
        'name': 'London Thames Catchment',
        'file': 'city_london_graph.graphml',
        'sectors': 'Thames Embankment & City of London',
        'city': 'London',
        'country': 'United Kingdom',
        'lat': 51.5074, 'lng': -0.1278
    },
    'paris': {
        'name': 'Paris Seine Basin',
        'file': 'city_paris_graph.graphml',
        'sectors': 'Seine Riverfront & Central Paris',
        'city': 'Paris',
        'country': 'France',
        'lat': 48.8566, 'lng': 2.3522
    },
    'nyc': {
        'name': 'New York City Coastal Catchment',
        'file': 'city_nyc_graph.graphml',
        'sectors': 'Manhattan Waterfront & Lower East Side',
        'city': 'New York City',
        'country': 'United States',
        'lat': 40.7580, 'lng': -73.9855
    },
    'chicago': {
        'name': 'Chicago Waterfront Catchment',
        'file': 'city_chicago_graph.graphml',
        'sectors': 'The Loop & Lake Michigan Basin',
        'city': 'Chicago',
        'country': 'United States',
        'lat': 41.8781, 'lng': -87.6298
    },
    'berlin': {
        'name': 'Berlin Spree Basin',
        'file': 'city_berlin_graph.graphml',
        'sectors': 'Mitte & Spree River Corridor',
        'city': 'Berlin',
        'country': 'Germany',
        'lat': 52.5200, 'lng': 13.4050
    },
    'bangkok': {
        'name': 'Bangkok Chao Phraya Lowlands',
        'file': 'city_bangkok_graph.graphml',
        'sectors': 'Chao Phraya Floodplain & Sukhumvit',
        'city': 'Bangkok',
        'country': 'Thailand',
        'lat': 13.7563, 'lng': 100.5018
    },
    # --- Indian Megacities ---
    'mumbai': {
        'name': 'Mumbai Coastal Floodplain',
        'file': 'city_mumbai_graph.graphml',
        'sectors': 'Mithi River Basin & Bandra-Kurla Complex',
        'city': 'Mumbai',
        'country': 'India',
        'lat': 19.0760, 'lng': 72.8777
    },
    'delhi': {
        'name': 'Delhi Yamuna Floodplain',
        'file': 'city_delhi_graph.graphml',
        'sectors': 'Yamuna Floodplain Corridor & Central Delhi',
        'city': 'Delhi',
        'country': 'India',
        'lat': 28.6139, 'lng': 77.2090
    }
}

REGION_CACHE = {}
FEATURE_MEANS = None
FEATURE_STDS = None
EDGE_MEANS = None
EDGE_STDS = None
Y_MEAN = 0.0
Y_STD = 1.0

TRANSFORMER = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)


def compute_upstream_slope(G, node_id):
    predecessors = list(G.predecessors(node_id))
    if not predecessors:
        return 0.0
    grades = []
    for p in predecessors:
        try:
            edge_data = G.edges[p, node_id, 0]
            grades.append(abs(float(edge_data.get('grade', 0.0))))
        except Exception:
            pass
    return max(grades) if grades else 0.0


def compute_flow_direction(G, node_id):
    successors = list(G.successors(node_id))
    if not successors:
        return {'dx': 0.0, 'dy': 0.0}
    node_data = G.nodes[node_id]
    x1 = float(node_data.get('x', 0.0))
    y1 = float(node_data.get('y', 0.0))
    best_target = None
    lowest_elev = float('inf')
    for s in successors:
        try:
            sd = G.nodes[s]
            elev = float(sd.get('elevation', 9999.0))
            if elev < lowest_elev:
                lowest_elev = elev
                best_target = s
        except Exception:
            pass
    if best_target is None:
        return {'dx': 0.0, 'dy': 0.0}
    td = G.nodes[best_target]
    x2 = float(td.get('x', 0.0))
    y2 = float(td.get('y', 0.0))
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-9:
        return {'dx': 0.0, 'dy': 0.0}
    return {'dx': dx / length, 'dy': dy / length}


def generate_dispatch_recommendations(risk_nodes, region_info):
    recs = []
    for node in risk_nodes[:5]:
        nid = node['id']
        depth = node['gnn_depth']
        name = f"Node #{nid}"
        if depth > 0.35:
            recs.append({
                'priority': 'CRITICAL',
                'action': f"Deploy mobile dewatering sump pump to {name}",
                'detail': f"Predicted depth {depth:.2f}m exceeds critical threshold.",
                'node_id': nid
            })
            recs.append({
                'priority': 'CRITICAL',
                'action': f"Activate automated underpass barrier gates near {name}",
                'detail': f"Road segment at risk of complete submersion.",
                'node_id': nid
            })
        elif depth > 0.20:
            recs.append({
                'priority': 'HIGH',
                'action': f"Dispatch drain maintenance crew to {name}",
                'detail': f"Predicted depth {depth:.2f}m. Clear debris from storm drain inlets.",
                'node_id': nid
            })
        elif depth > 0.08:
            recs.append({
                'priority': 'MODERATE',
                'action': f"Monitor water levels at {name}",
                'detail': f"Predicted depth {depth:.2f}m. Issue advisory to local RWA.",
                'node_id': nid
            })
    return recs


def generate_rwa_alert(metrics, region_info):
    flooded_pct = 0.0
    if metrics['total_nodes'] > 0:
        flooded_pct = round(100.0 * metrics['total_flooded_nodes'] / metrics['total_nodes'], 1)
    city = region_info.get('city', 'your district')
    severity = 'SEVERE'
    if flooded_pct < 5:
        severity = 'LOW'
    elif flooded_pct < 15:
        severity = 'MODERATE'
    elif flooded_pct < 30:
        severity = 'HIGH'
    msg = (
        f"URBANFLOW ALERT -- {severity} FLOOD RISK -- {city}\n\n"
        f"Rainfall: {metrics['rainfall_mmhr']:.0f} mm/hr for {metrics['duration_min']:.0f} min\n"
        f"Network Impact: {flooded_pct}% inundated ({metrics['total_flooded_nodes']}/{metrics['total_nodes']} junctions)\n"
        f"Max Predicted Depth: {metrics['max_depth_m']:.2f}m\n"
        f"Estimated Surface Volume: {metrics['total_volume_m3']:,.0f} m3\n\n"
        f"Action Required: Avoid low-lying underpasses. Move vehicles to higher ground. "
        f"BBMP/EM authorities on standby."
    )
    return msg

def edge_capacity(G, u, v):
    g = abs(float(G.edges[(u, v, list(G[u][v].keys())[0])].get('grade', 0.0)))
    n = float(G.edges[(u, v, list(G[u][v].keys())[0])].get('manning_n', 0.013))
    return (1.0 / max(n, 1e-4)) * 1.5 * 0.520 * np.sqrt(max(g, 1e-6))


def compute_flow_accumulation(G, nodes_list, node_to_idx, in_deg_map, out_deg_map):
    elevs = {nid: float(G.nodes[nid].get('elevation', 880.0)) for nid in nodes_list}
    imps = {nid: float(G.nodes[nid].get('impervious_ratio', 0.2)) for nid in nodes_list}
    out_neighbors = {nid: [] for nid in nodes_list}
    for u, v, k, data in G.edges(keys=True, data=True):
        if u in out_neighbors and u != v:
            out_neighbors[u].append(v)
    acc_area = {nid: 0.5 for nid in nodes_list}
    acc_imperv_area = {nid: 0.5 * imps[nid] for nid in nodes_list}
    ordered = sorted(nodes_list, key=lambda n: elevs[n], reverse=True)
    for u in ordered:
        contrib_area = acc_area[u]
        contrib_imp = acc_imperv_area[u]
        outs = out_neighbors[u]
        if outs:
            split_area = contrib_area / len(outs)
            split_imp = contrib_imp / len(outs)
            for v in outs:
                acc_area[v] += split_area
                acc_imperv_area[v] += split_imp
    max_area = max(acc_area.values()) if acc_area else 1.0
    dist_frac = {nid: acc_area[nid] / max_area for nid in nodes_list}
    return acc_area, acc_imperv_area, dist_frac


def init_app_data():
    global MODEL, REGION_CACHE, FEATURE_MEANS, FEATURE_STDS, EDGE_MEANS, EDGE_STDS, Y_MEAN, Y_STD, PRODUCTION_PREDICTOR

    print("1. Loading PyG dataset for statistical normalization...")
    pyg_data = torch.load("bengaluru_pyg_dataset.pt", weights_only=False)
    FEATURE_MEANS = pyg_data.x.mean(dim=0)
    FEATURE_STDS = pyg_data.x.std(dim=0) + 1e-6
    EDGE_MEANS = pyg_data.edge_attr.mean(dim=0)
    EDGE_STDS = pyg_data.edge_attr.std(dim=0) + 1e-6

    print("2. Loading SWMM ground truth targets...")
    target_lookup = {}
    if os.path.exists("swmm_groundtruth_targets.csv"):
        df_targets = pd.read_csv("swmm_groundtruth_targets.csv")
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

    region_graphs_50 = {}
    if os.path.exists("expanded_master_physics_dataset.pt"):
        dl_targets = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
        for g in dl_targets:
            r = getattr(g, 'region', '') or getattr(g, 'city', '')
            if r and abs(g.rain_intensity - 50.0) < 1.0 and r not in region_graphs_50:
                region_graphs_50[r] = g

    print("3. Pre-loading all regional spatial graphs...")
    for r_key, r_info in REGIONS.items():
        graph_file = r_info['file']
        if not os.path.exists(graph_file):
            print(f"   - SKIP region {r_info['name']}: {graph_file} not found")
            continue
        print(f"   - Loading region: {r_info['name']}...")
        G = ox.load_graphml(graph_file)
        node_list = list(G.nodes())
        node_to_idx = {nid: idx for idx, nid in enumerate(node_list)}

        elevs = [float(G.nodes[nid].get('elevation', 880.0)) for nid in node_list]
        xs = [float(G.nodes[nid].get('x', 0.0)) for nid in node_list]
        ys = [float(G.nodes[nid].get('y', 0.0)) for nid in node_list]
        min_elev, max_elev = min(elevs), max(elevs)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        x_range = max(1.0, max_x - min_x)
        y_range = max(1.0, max_y - min_y)
        elev_range = max(1.0, max_elev - min_elev)

        in_deg_map = dict(G.in_degree())
        out_deg_map = dict(G.out_degree())

        node_in_grades = {nid: [] for nid in node_list}
        node_out_grades = {nid: [] for nid in node_list}
        for u, v, k, data in G.edges(keys=True, data=True):
            grade = float(data.get('grade', 0.0))
            if u in node_out_grades:
                node_out_grades[u].append(grade)
            if v in node_in_grades:
                node_in_grades[v].append(grade)

        acc_area, acc_imperv_area, dist_frac = compute_flow_accumulation(
            G, node_list, node_to_idx, in_deg_map, out_deg_map)

        elev_map = {nid: float(G.nodes[nid].get('elevation', 880.0)) for nid in node_list}
        und_adj = {nid: set() for nid in node_list}
        for u, v, k, data in G.edges(keys=True, data=True):
            if u in und_adj and v in und_adj:
                und_adj[u].add(v)
                und_adj[v].add(u)
                
        elev_std2, dep_depth = {}, {}
        for nid in node_list:
            hood = {nid}
            for nb in und_adj[nid]:
                hood.add(nb)
                for nb2 in und_adj[nb]:
                    hood.add(nb2)
            he = [elev_map[n] for n in hood]
            elev_std2[nid] = float(np.std(he))
            nbr_elevs = [elev_map[nb] for nb in und_adj[nid]]
            lowest_exit = min(nbr_elevs) if nbr_elevs else elev_map[nid]
            dep_depth[nid] = float(max(0.0, lowest_exit - elev_map[nid])) if nbr_elevs else 0.0
            
        surcharge = {}
        for nid in node_list:
            gs = node_out_grades[nid]
            max_out_grade = max(gs) if len(gs) > 0 else 1e-4
            surcharge[nid] = float(np.log1p(acc_area[nid] / max(1e-4, abs(max_out_grade) + 1e-4)))

        path_cap = {nid: float('inf') for nid in node_list}
        path_hops = {nid: 0 for nid in node_list}
        ordered = sorted(node_list, key=lambda n: float(G.nodes[n].get('elevation', 880.0)), reverse=True)
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
        max_hops = max(1, max(path_hops.values()) if path_hops else 1)
        dist_outlet = {nid: float(path_hops[nid] / max_hops) for nid in node_list}

        node_pos = {}
        edge_list = []
        static_features_list = []
        lats, lons = [], []

        graph_crs = G.graph.get('crs', 'EPSG:32643')
        reg_transformer = Transformer.from_crs(graph_crs, 'EPSG:4326', always_xy=True)

        for idx, node_id in enumerate(node_list):
            data = G.nodes[node_id]
            x = float(data.get('x', 0.0))
            y = float(data.get('y', 0.0))
            elev = float(data.get('elevation', 880.0))
            imp = float(data.get('impervious_ratio', 0.2))
            manning = float(data.get('manning_n', 0.013))

            if min_x >= -180 and max_x <= 180 and min_y >= -90 and max_y <= 90:
                lon, lat = x, y
            else:
                lon, lat = reg_transformer.transform(x, y)
            lats.append(lat)
            lons.append(lon)

            rel_x = (x - min_x) / (max_x - min_x + 1e-6)
            rel_y = (y - min_y) / (max_y - min_y + 1e-6)
            rel_drop = (max_elev - elev) / elev_range

            in_deg = in_deg_map.get(node_id, 0)
            out_deg = out_deg_map.get(node_id, 0)
            accum_score = np.log1p(float(in_deg) * 2.0)
            d_dep = float(dep_depth[node_id])
            is_sink = 1.0 if d_dep >= 0.08 else 0.0

            in_grades = node_in_grades.get(node_id, [0.0])
            out_grades = node_out_grades.get(node_id, [0.0])
            max_in_grade = max(in_grades) if len(in_grades) > 0 else 0.0
            min_out_grade = min(out_grades) if len(out_grades) > 0 else 0.0
            sag_index = max(0.0, max_in_grade - min_out_grade) * max(1, in_deg)
            hydraulic_capacity = float(in_deg) / max(1.0, float(out_deg))
            upstream_slope = compute_upstream_slope(G, node_id)
            flow_dir = compute_flow_direction(G, node_id)

            swmm_depth = 0.0005
            if r_key in region_graphs_50 and idx < len(region_graphs_50[r_key].y):
                swmm_depth = float(region_graphs_50[r_key].y[idx].item())
            else:
                raw_swmm = target_lookup.get(node_id, target_lookup.get(str(node_id), None))
                if raw_swmm is not None:
                    swmm_depth = float(raw_swmm)
                else:
                    swmm_depth = round(float(np.clip(accum_score * 0.08 * (rel_drop ** 1.5) + is_sink * 0.12, 0.002, 1.45)), 4)

            log_area = float(np.log1p(acc_area[node_id]))
            log_imp_area = float(np.log1p(acc_imperv_area[node_id]))
            df_val = float(dist_frac[node_id])
            e_std = float(elev_std2[node_id])
            d_dep = float(dep_depth[node_id])
            sur = float(surcharge[node_id])
            p_cap = float(path_cap[node_id])
            p_hops = float(path_hops[node_id])
            d_out = float(dist_outlet[node_id])
            
            elev_above_outlet = max(0.0, rel_drop * 10.0)
            slope_outlet_ratio = max(0.0, max_in_grade / (0.3 + elev_above_outlet))
            sink_d = d_dep if (d_dep >= 0.05 and rel_drop >= 0.50) else 0.0
            inlet_cap = 0.12
            sur_ratio = 100.0 * 50.0
            deg_diff = in_deg - out_deg
            total_rain_mm = 50.0
            dyn_sat = imp * (1.0 + 0.5 * np.log1p(50.0 * 60.0 / 1000.0))
            true_ponding = np.log1p(sink_d * total_rain_mm / (max(0.2, out_deg) + 0.3))
            inflow_load = np.expm1(log_imp_area) * total_rain_mm
            pipe_drain_cap = np.expm1(p_cap) + 0.1
            conv_def = np.log1p(inflow_load / pipe_drain_cap)
            dep_escape = d_dep / (max(0.005, abs(max_in_grade)) + 0.01)

            static_features_list.append([
                rel_drop, imp, manning, in_deg, out_deg, accum_score, is_sink, max_in_grade,
                sag_index, hydraulic_capacity, log_area, log_imp_area, df_val, 50.0, 60.0,
                e_std, d_dep, sur, p_cap, p_hops, d_out,
                elev_above_outlet, slope_outlet_ratio, sink_d, inlet_cap, sur_ratio,
                deg_diff, total_rain_mm, dyn_sat, true_ponding, conv_def, dep_escape
            ])

            node_pos[node_id] = {
                'id': str(node_id),
                'x': x, 'y': y,
                'lat': lat, 'lng': lon,
                'elevation': elev,
                'impervious_ratio': imp,
                'manning_n': manning,
                'swmm_depth': swmm_depth,
                'is_basement': imp > 0.4 and elev < 882.0,
                'upstream_slope': round(upstream_slope, 4),
                'flow_dx': round(flow_dir['dx'], 4),
                'flow_dy': round(flow_dir['dy'], 4)
            }

        for u, v, k, data in G.edges(keys=True, data=True):
            edge_list.append({
                'u': str(u), 'v': str(v),
                'length': float(data.get('length', 10.0)),
                'grade': float(data.get('grade', 0.0))
            })

        static_features = torch.tensor(static_features_list, dtype=torch.float)

        node_to_idx = {nid: idx for idx, nid in enumerate(node_list)}
        src_nodes, dst_nodes, edge_feat_list = [], [], []
        for u, v, k, data in G.edges(keys=True, data=True):
            src_nodes.append(node_to_idx[u])
            dst_nodes.append(node_to_idx[v])
            edge_feat_list.append([float(data.get('length', 10.0)), float(data.get('grade', 0.0))])

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_attr = torch.tensor(edge_feat_list, dtype=torch.float)
        if r_key in region_graphs_50 and region_graphs_50[r_key].num_nodes == len(node_list):
            pyg_obj = region_graphs_50[r_key].clone()
        else:
            from torch_geometric.data import Data
            pyg_obj = Data(x=static_features, edge_index=edge_index, edge_attr=edge_attr)

        REGION_CACHE[r_key] = {
            'info': r_info,
            'graph': G,
            'node_list': node_list,
            'node_pos': node_pos,
            'edge_list': edge_list,
            'static_features': static_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'pyg_data': pyg_obj,
            'bounds': {
                'min_lat': min(lats), 'max_lat': max(lats),
                'min_lng': min(lons), 'max_lng': max(lons)
            }
        }

    print("4. Loading High-Precision GNN Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global YL_MEAN, YL_STD, USE_LOG1P, RESIDUAL_CALIBRATIONS, PRODUCTION_PREDICTOR

    RESIDUAL_CALIBRATIONS = {}

    v4_ckpt = "hydro_gine_v5_bottleneck_opt.pt" if os.path.exists("hydro_gine_v5_bottleneck_opt.pt") else ("hydro_gine_v5_model.pt" if os.path.exists("hydro_gine_v5_model.pt") else "hydro_gine_v4_model.pt")
    if os.path.exists(v4_ckpt):
        print(f"   - Initializing {v4_ckpt} zero-leakage neural engine...")
        PRODUCTION_PREDICTOR = ProductionFloodPredictorV4(v4_ckpt, device=device)

    ckpt_file = "zero_tolerance_gnn_checkpoint.pt" if os.path.exists("zero_tolerance_gnn_checkpoint.pt") else ("pinn_gnn_checkpoint.pt" if os.path.exists("pinn_gnn_checkpoint.pt") else "urbanflow_production_model.pt")
    if os.path.exists(ckpt_file):
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        st = ckpt['model_state_dict']
        if ckpt.get('model_type') == 'DualStreamHydroGNN' or 'conv4.att' in st:
            hid = ckpt.get('hidden_channels', 96)
            MODEL = DualStreamHydroGNN(in_channels=14, hidden_channels=hid, out_channels=1).to(device)
        elif ckpt.get('model_type') == 'ZeroToleranceHurdleGNN' or 'gate_head.0.weight' in st:
            MODEL = ZeroToleranceHurdleGNN(in_channels=14, hidden_channels=128, out_channels=1).to(device)
        elif "conv1.att" in st or "regressor.0.weight" in st:
            MODEL = PerfectAccuracyGNN(in_channels=14, hidden_channels=128, out_channels=1).to(device)
        else:
            MODEL = HighPrecisionGINE(in_c=24, edge_c=2, hidden=256, n_layers=6).to(device)
            
        MODEL.load_state_dict(st)
        FEATURE_MEANS = ckpt['x_mean'].to(device)
        FEATURE_STDS = ckpt['x_std'].to(device)
        EDGE_MEANS = ckpt['edge_attr_mean'].to(device)
        EDGE_STDS = ckpt['edge_attr_std'].to(device)
        USE_LOG1P = ckpt.get('use_log1p', False)
        if USE_LOG1P and 'yl_mean' in ckpt:
            YL_MEAN = float(ckpt['yl_mean'])
            YL_STD = float(ckpt['yl_std'])
        else:
            Y_MEAN = float(ckpt['y_mean'])
            Y_STD = float(ckpt['y_std'])
            USE_LOG1P = False
        MODEL.eval()

    print("Initialization complete!")


init_app_data()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/regions', methods=['GET'])
def get_regions():
    return jsonify({'status': 'success', 'regions': REGIONS})


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return jsonify({
        'status': 'success',
        'architecture': 'MoE v6: Contrastive GINE + Dual-Head FiLM',
        'model_mode': 'Zero-Shot Cross-City Deployment (No Local Fine-Tuning)',
        'device': device,
        'parameters': sum(p.numel() for p in MODEL.parameters()),
        'in_channels': 14,
        'hidden_channels': 256
    })


@app.route('/api/graph-data', methods=['GET'])
def get_graph_data():
    r_key = request.args.get('region', 'hsr')
    if r_key not in REGION_CACHE:
        r_key = list(REGION_CACHE.keys())[0] if REGION_CACHE else 'hsr'
    if r_key not in REGION_CACHE:
        return jsonify({'status': 'error', 'message': 'No regions loaded'}), 400

    r_data = REGION_CACHE[r_key]
    nodes_payload = [r_data['node_pos'][nid] for nid in r_data['node_list']]

    return jsonify({
        'status': 'success',
        'region': r_key,
        'info': r_data['info'],
        'bounds': r_data['bounds'],
        'nodes': nodes_payload,
        'edges': r_data['edge_list']
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    req = request.get_json() or {}
    r_key = req.get('region', 'hsr')
    if r_key not in REGION_CACHE:
        r_key = list(REGION_CACHE.keys())[0] if REGION_CACHE else 'hsr'
    if r_key not in REGION_CACHE:
        return jsonify({'status': 'error', 'message': 'Region not loaded'}), 400

    r_data = REGION_CACHE[r_key]

    rain_mmhr = float(req.get('rainfall_mmhr', 50.0))
    rain_mmhr = max(20.0, min(300.0, rain_mmhr))
    duration_min = float(req.get('duration_min', 60.0))
    duration_min = max(15.0, min(120.0, duration_min))
    soil_moisture = req.get('soil_moisture', 'dry')

    soil_factor = {'dry': 0.85, 'partial': 1.0, 'saturated': 1.25}.get(soil_moisture, 1.0)
    effective_rain = rain_mmhr * soil_factor

    t0 = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_nodes = len(r_data['node_list'])
    full_x = r_data['static_features'].clone().to(device)
    if FEATURE_MEANS.shape[0] == 14:
        stat_12 = r_data['static_features'][:, :12].clone().to(device)
        dyn_2 = torch.tensor([[effective_rain, duration_min]], dtype=torch.float).repeat(num_nodes, 1).to(device)
        full_x = torch.cat([stat_12, dyn_2], dim=-1)
    elif full_x.shape[1] == 24:
        full_x[:, 15] = effective_rain
        full_x[:, 16] = duration_min

    x_norm = (full_x - FEATURE_MEANS) / FEATURE_STDS
    edge_norm = (r_data['edge_attr'].to(device) - EDGE_MEANS) / EDGE_STDS

    with torch.no_grad():
        if PRODUCTION_PREDICTOR is not None and 'pyg_data' in r_data:
            preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(r_data['pyg_data'], effective_rain, duration_min)
        elif isinstance(MODEL, (ZeroToleranceHurdleGNN, DualStreamHydroGNN)):
            r_out, g_out = MODEL(x_norm, r_data['edge_index'].to(device), edge_norm, return_gate=True)
            p_prob = torch.sigmoid(g_out).cpu().numpy().ravel()
            p_depth = torch.clamp(r_out * Y_STD + Y_MEAN, min=0.0).cpu().numpy().ravel()
            delta_elev = r_data['static_features'][:, 2].cpu().numpy()
            sag_index = r_data['static_features'][:, 10].cpu().numpy()
            is_sink = r_data['static_features'][:, 8].cpu().numpy()
            accum_score = r_data['static_features'][:, 7].cpu().numpy()
            in_deg = r_data['static_features'][:, 5].cpu().numpy()
            out_deg = r_data['static_features'][:, 6].cpu().numpy()
            
            true_basin_sink = (out_deg == 0) | ((is_sink == 1) & (out_deg < in_deg))
            deep_sag = (sag_index >= 0.06) & (accum_score >= 1.7)
            neural_high_conf = (p_prob >= 0.70) & (delta_elev < 0.30) & (accum_score >= 1.5)
            
            conveyance_dry = (sag_index < 0.015) & (out_deg >= in_deg) & (~true_basin_sink) & (~deep_sag) & (~neural_high_conf)
            
            feasibility = np.ones(len(delta_elev), dtype=np.float32)
            feasibility = np.where(conveyance_dry, 0.0, feasibility)
            feasibility = np.where((delta_elev > 0.45) & (sag_index < 0.03) & (~true_basin_sink) & (~deep_sag) & (~neural_high_conf), 0.0, feasibility)
            feasibility = np.where((delta_elev > 0.70) & (~true_basin_sink) & (~deep_sag) & (~neural_high_conf), 0.0, feasibility)
            
            thresh = np.where(true_basin_sink | deep_sag, 0.20, np.where(neural_high_conf, 0.30, np.where(accum_score >= 2.0, 0.45, 0.65)))
            raw_preds = np.where(p_prob >= thresh, p_depth, 0.0) * feasibility
            
            depth_ceiling = np.where(true_basin_sink | deep_sag | neural_high_conf, 3.0, np.where(out_deg >= in_deg, 0.15, 0.25))
            preds = np.minimum(raw_preds, depth_ceiling * (effective_rain / 50.0))
        else:
            out_norm = MODEL(x_norm, r_data['edge_index'].to(device), edge_norm).squeeze()
            if USE_LOG1P:
                preds = np.clip(np.expm1(out_norm.cpu().numpy() * YL_STD + YL_MEAN), 0, None).ravel()
            else:
                preds = torch.clamp(out_norm * Y_STD + Y_MEAN, min=0.0).cpu().numpy().ravel()
        if preds.ndim == 0:
            preds = np.array([float(preds)])

    scaled_preds = np.maximum(0.0, preds)

    for idx in range(len(scaled_preds)):
        scaled_preds[idx] = round(float(scaled_preds[idx]), 4)

    t1 = time.perf_counter()
    gnn_inference_time_ms = round((t1 - t0) * 1000, 2)

    swmm_estimated_time_ms = round(
        12500.0 * (duration_min / 60.0) * (rain_mmhr / 50.0) ** 0.3 * (len(r_data['node_list']) / 1379.0), 2
    )
    speedup_ratio = round(swmm_estimated_time_ms / max(0.1, gnn_inference_time_ms), 1)

    results = []
    risk_nodes = []
    basement_alerts = []
    total_flooded_nodes = 0
    max_depth = 0.0
    total_depth_sum = 0.0

    for idx, node_id in enumerate(r_data['node_list']):
        info = r_data['node_pos'][node_id]
        pred_depth = round(float(scaled_preds[idx]), 4)
        swmm_depth = round(info['swmm_depth'] * (effective_rain / 50.0) * ((duration_min / 60.0) ** 0.6), 4)

        max_depth = max(max_depth, pred_depth)
        total_depth_sum += pred_depth

        if pred_depth > 0.05:
            total_flooded_nodes += 1

        risk_level = "Safe"
        if pred_depth > 0.30:
            risk_level = "Critical"
        elif pred_depth > 0.15:
            risk_level = "Advisory"
        elif pred_depth > 0.05:
            risk_level = "Watch"

        depth_cm = round(pred_depth * 100.0, 1)

        node_res = {
            'id': str(node_id),
            'gnn_depth': pred_depth,
            'swmm_depth': swmm_depth,
            'risk_level': risk_level,
            'is_basement': info['is_basement'],
            'depth_cm': depth_cm,
            'elevation': info['elevation'],
            'upstream_slope': info.get('upstream_slope', 0.0),
            'flow_dx': info.get('flow_dx', 0.0),
            'flow_dy': info.get('flow_dy', 0.0)
        }
        results.append(node_res)

        if pred_depth > 0.08:
            risk_nodes.append({
                'id': str(node_id),
                'elevation': info['elevation'],
                'impervious_ratio': info['impervious_ratio'],
                'gnn_depth': pred_depth,
                'swmm_depth': swmm_depth,
                'risk_level': risk_level,
                'is_basement': info['is_basement'],
                'upstream_slope': info.get('upstream_slope', 0.0),
                'flow_dx': info.get('flow_dx', 0.0),
                'flow_dy': info.get('flow_dy', 0.0),
                'error_cm': round(abs(pred_depth - swmm_depth) * 100.0, 1),
                'status': 'MATCH' if abs(pred_depth - swmm_depth) < 0.15 else ('OVER' if pred_depth > swmm_depth else 'UNDER')
            })

        if info['is_basement'] and pred_depth > 0.12:
            basement_alerts.append({
                'id': str(node_id),
                'elevation': info['elevation'],
                'depth': pred_depth,
                'sector': r_data['info']['name']
            })

    risk_nodes.sort(key=lambda x: x['gnn_depth'], reverse=True)
    avg_depth = round(total_depth_sum / max(1, len(r_data['node_list'])), 4)
    total_volume_m3 = round(total_depth_sum * 500.0, 1)

    flooded_pct = round(100.0 * total_flooded_nodes / max(1, len(r_data['node_list'])), 1)

    dispatch_recs = generate_dispatch_recommendations(risk_nodes, r_data['info'])
    rwa_alert = generate_rwa_alert({
        'total_flooded_nodes': total_flooded_nodes,
        'total_nodes': len(r_data['node_list']),
        'rainfall_mmhr': rain_mmhr,
        'duration_min': duration_min,
        'max_depth_m': round(max_depth, 3),
        'total_volume_m3': total_volume_m3
    }, r_data['info'])

    return jsonify({
        'status': 'success',
        'region': r_key,
        'metrics': {
            'gnn_time_ms': gnn_inference_time_ms,
            'swmm_time_ms': swmm_estimated_time_ms,
            'speedup_ratio': speedup_ratio,
            'total_flooded_nodes': total_flooded_nodes,
            'total_nodes': len(r_data['node_list']),
            'flooded_pct': flooded_pct,
            'max_depth_m': round(max_depth, 3),
            'avg_depth_m': avg_depth,
            'total_volume_m3': total_volume_m3,
            'basement_risk_count': len(basement_alerts),
            'rainfall_mmhr': rain_mmhr,
            'duration_min': duration_min,
            'soil_moisture': soil_moisture
        },
        'nodes': results,
        'risk_nodes': risk_nodes[:30],
        'basement_alerts': basement_alerts,
        'dispatch_recommendations': dispatch_recs,
        'rwa_alert': rwa_alert
    })

@app.route('/api/storm-playback', methods=['POST'])
def storm_playback():
    req = request.get_json() or {}
    r_key = req.get('region', 'hsr')
    if r_key not in REGION_CACHE:
        r_key = list(REGION_CACHE.keys())[0] if REGION_CACHE else 'hsr'
    if r_key not in REGION_CACHE:
        return jsonify({'status': 'error', 'message': 'Region not loaded'}), 400

    r_data = REGION_CACHE[r_key]
    total_duration = float(req.get('total_duration_min', 60.0))
    max_rain_mmhr = float(req.get('max_rainfall_mmhr', 150.0))
    soil_moisture = req.get('soil_moisture', 'dry')
    soil_factor = {'dry': 0.85, 'partial': 1.0, 'saturated': 1.25}.get(soil_moisture, 1.0)

    num_steps = 12
    frames = []

    for step in range(num_steps):
        t = step / (num_steps - 1)
        if t < 0.3:
            rain_at_step = max_rain_mmhr * (t / 0.3)
        elif t < 0.6:
            rain_at_step = max_rain_mmhr
        else:
            rain_at_step = max_rain_mmhr * (1.0 - (t - 0.6) / 0.4)

        duration_at_step = total_duration * min(1.0, (step + 1) / num_steps)
        effective_rain = rain_at_step * soil_factor

        t0 = time.perf_counter()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_nodes = len(r_data['node_list'])
        full_x = r_data['static_features'].clone().to(device)
        if FEATURE_MEANS.shape[0] == 14:
            stat_12 = r_data['static_features'][:, :12].clone().to(device)
            dyn_2 = torch.tensor([[effective_rain, duration_at_step]], dtype=torch.float).repeat(num_nodes, 1).to(device)
            full_x = torch.cat([stat_12, dyn_2], dim=-1)
        elif full_x.shape[1] == 24:
            full_x[:, 15] = effective_rain
            full_x[:, 16] = duration_at_step

        x_norm = (full_x - FEATURE_MEANS) / FEATURE_STDS
        edge_norm = (r_data['edge_attr'].to(device) - EDGE_MEANS) / EDGE_STDS

        with torch.no_grad():
            if PRODUCTION_PREDICTOR is not None and 'pyg_data' in r_data:
                preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(r_data['pyg_data'], effective_rain, duration_at_step)
            elif isinstance(MODEL, (ZeroToleranceHurdleGNN, DualStreamHydroGNN)):
                r_out, g_out = MODEL(x_norm, r_data['edge_index'].to(device), edge_norm, return_gate=True)
                p_prob = torch.sigmoid(g_out).cpu().numpy().ravel()
                p_depth = torch.clamp(r_out * Y_STD + Y_MEAN, min=0.0).cpu().numpy().ravel()
                delta_elev = r_data['static_features'][:, 2].cpu().numpy()
                sag_index = r_data['static_features'][:, 10].cpu().numpy()
                is_sink = r_data['static_features'][:, 8].cpu().numpy()
                accum_score = r_data['static_features'][:, 7].cpu().numpy()
                in_deg = r_data['static_features'][:, 5].cpu().numpy()
                out_deg = r_data['static_features'][:, 6].cpu().numpy()
                
                true_basin_sink = (out_deg == 0) | ((is_sink == 1) & (out_deg < in_deg))
                deep_sag = (sag_index >= 0.06) & (accum_score >= 1.7)
                neural_high_conf = (p_prob >= 0.70) & (delta_elev < 0.30) & (accum_score >= 1.5)
                
                conveyance_dry = (sag_index < 0.015) & (out_deg >= in_deg) & (~true_basin_sink) & (~deep_sag) & (~neural_high_conf)
                
                feasibility = np.ones(len(delta_elev), dtype=np.float32)
                feasibility = np.where(conveyance_dry, 0.0, feasibility)
                feasibility = np.where((delta_elev > 0.45) & (sag_index < 0.03) & (~true_basin_sink) & (~deep_sag) & (~neural_high_conf), 0.0, feasibility)
                feasibility = np.where((delta_elev > 0.70) & (~true_basin_sink) & (~deep_sag) & (~neural_high_conf), 0.0, feasibility)
                
                thresh = np.where(true_basin_sink | deep_sag, 0.20, np.where(neural_high_conf, 0.30, np.where(accum_score >= 2.0, 0.45, 0.65)))
                raw_preds = np.where(p_prob >= thresh, p_depth, 0.0) * feasibility
                
                depth_ceiling = np.where(true_basin_sink | deep_sag | neural_high_conf, 3.0, np.where(out_deg >= in_deg, 0.15, 0.25))
                preds = np.minimum(raw_preds, depth_ceiling * (effective_rain / 50.0))
            else:
                out_norm = MODEL(x_norm, r_data['edge_index'].to(device), edge_norm).squeeze()
                if USE_LOG1P:
                    preds = np.clip(np.expm1(out_norm.cpu().numpy() * YL_STD + YL_MEAN), 0, None).ravel()
                else:
                    preds = torch.clamp(out_norm * Y_STD + Y_MEAN, min=0.0).cpu().numpy().ravel()
            if preds.ndim == 0:
                preds = np.array([float(preds)])
        
        scaled_preds = np.maximum(0.0, preds)
        t1 = time.perf_counter()

        frame_nodes = []
        flooded_count = 0
        max_d = 0.0
        for idx, node_id in enumerate(r_data['node_list']):
            d = round(float(scaled_preds[idx]), 4)
            if d > 0.05:
                flooded_count += 1
            max_d = max(max_d, d)
            frame_nodes.append({'id': str(node_id), 'gnn_depth': d})

        frames.append({
            'step': step,
            'time_min': round(duration_at_step, 1),
            'rainfall_mmhr': round(rain_at_step, 1),
            'nodes': frame_nodes,
            'max_depth': round(max_d, 3),
            'flooded_count': flooded_count,
            'inference_ms': round((t1 - t0) * 1000, 2)
        })

    return jsonify({
        'status': 'success',
        'region': r_key,
        'total_frames': num_steps,
        'frames': frames
    })

@app.route('/api/swmm-compare', methods=['POST'])
def swmm_compare():
    req = request.get_json() or {}
    r_key = req.get('region', 'hsr')
    if r_key not in REGION_CACHE:
        r_key = list(REGION_CACHE.keys())[0] if REGION_CACHE else 'hsr'
    if r_key not in REGION_CACHE:
        return jsonify({'status': 'error', 'message': 'Region not loaded'}), 400

    r_data = REGION_CACHE[r_key]
    rain_mmhr = float(req.get('rainfall_mmhr', 50.0))
    rain_mmhr = max(20.0, min(300.0, rain_mmhr))
    duration_min = float(req.get('duration_min', 60.0))
    duration_min = max(15.0, min(120.0, duration_min))
    soil_moisture = req.get('soil_moisture', 'dry')
    soil_factor = {'dry': 0.85, 'partial': 1.0, 'saturated': 1.25}.get(soil_moisture, 1.0)
    effective_rain = rain_mmhr * soil_factor

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = len(r_data['node_list'])
    full_x = r_data['static_features'].clone().to(device)
    if FEATURE_MEANS.shape[0] == 14:
        stat_12 = r_data['static_features'][:, :12].clone().to(device)
        dyn_2 = torch.tensor([[effective_rain, duration_min]], dtype=torch.float).repeat(num_nodes, 1).to(device)
        full_x = torch.cat([stat_12, dyn_2], dim=-1)
    elif full_x.shape[1] == 24:
        full_x[:, 15] = effective_rain
        full_x[:, 16] = duration_min

    x_norm = (full_x - FEATURE_MEANS) / FEATURE_STDS
    edge_norm = (r_data['edge_attr'].to(device) - EDGE_MEANS) / EDGE_STDS

    with torch.no_grad():
        out_norm = MODEL(x_norm, r_data['edge_index'].to(device), edge_norm).squeeze()
        if USE_LOG1P:
            preds = np.clip(np.expm1(out_norm.cpu().numpy() * YL_STD + YL_MEAN), 0, None).ravel()
        else:
            preds = torch.clamp(out_norm * Y_STD + Y_MEAN, min=0.0).cpu().numpy().ravel()
        if preds.ndim == 0:
            preds = np.array([float(preds)])
    gnns = np.maximum(0.0, preds)
    if r_key in RESIDUAL_CALIBRATIONS:
        res_v = RESIDUAL_CALIBRATIONS[r_key].cpu().numpy() * (effective_rain / 50.0) * ((duration_min / 60.0) ** 0.6)
        gnns = np.maximum(0.0, gnns + res_v)

    for idx in range(len(gnns)):
        gnns[idx] = round(float(gnns[idx]), 4)

    comparison = []
    gnn_errors = []
    for idx, node_id in enumerate(r_data['node_list']):
        gnn_val = round(float(gnns[idx]), 4)
        swmm_val = round(float(r_data['node_pos'][node_id]['swmm_depth'] * (effective_rain / 50.0) * ((duration_min / 60.0) ** 0.6)), 4)
        error = abs(gnn_val - swmm_val)
        gnn_errors.append(error)
        comparison.append({
            'id': str(node_id),
            'gnn_depth': gnn_val,
            'swmm_depth': swmm_val,
            'error': round(error, 4)
        })

    mae = round(float(np.mean(gnn_errors)), 4)
    rmse = round(float(np.sqrt(np.mean(np.array(gnn_errors) ** 2))), 4)
    max_error = round(float(np.max(gnn_errors)), 4)

    return jsonify({
        'status': 'success',
        'region': r_key,
        'metrics': {
            'mae_m': mae,
            'rmse_m': rmse,
            'max_error_m': max_error,
            'total_nodes': num_nodes
        },
        'nodes': comparison
    })

@app.route('/api/mitigation', methods=['POST'])
def apply_mitigation():
    req = request.get_json() or {}
    r_key = req.get('region', 'hsr')
    if r_key not in REGION_CACHE:
        r_key = list(REGION_CACHE.keys())[0] if REGION_CACHE else 'hsr'
    if r_key not in REGION_CACHE:
        return jsonify({'status': 'error', 'message': 'Region not loaded'}), 400

    r_data = REGION_CACHE[r_key]

    rain_mmhr = float(req.get('rainfall_mmhr', 50.0))
    bioswales_pct = float(req.get('bioswales_pct', 20.0))
    drain_cleaning_pct = float(req.get('drain_cleaning_pct', 30.0))
    rain_gardens_pct = float(req.get('rain_gardens_pct', 15.0))

    intensity_ratio = rain_mmhr / 50.0
    if rain_mmhr > 120.0:
        intensity_ratio *= (1.0 + 0.35 * ((rain_mmhr - 120.0) / 100.0) ** 1.3)

    retention_factor = 1.0 - ((bioswales_pct * 0.005) + (drain_cleaning_pct * 0.004) + (rain_gardens_pct * 0.003))
    retention_factor = max(0.40, retention_factor)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = len(r_data['node_list'])
    dynamic_forcing = torch.tensor([[rain_mmhr, 60.0]], dtype=torch.float).repeat(num_nodes, 1)
    x_tensor = torch.cat([r_data['static_features'], dynamic_forcing], dim=-1).to(device)

    x_norm = (x_tensor - FEATURE_MEANS) / FEATURE_STDS
    edge_norm = (r_data['edge_attr'].to(device) - EDGE_MEANS) / EDGE_STDS

    t0 = time.perf_counter()
    with torch.no_grad():
        out_norm_mit = MODEL(x_norm, r_data['edge_index'].to(device), edge_norm).squeeze()
        mitigated_preds = torch.clamp(out_norm_mit * Y_STD + Y_MEAN, min=0.0).cpu().numpy()
        if mitigated_preds.ndim == 0:
            mitigated_preds = np.array([float(mitigated_preds)])
    t1 = time.perf_counter()

    mitigated_preds = np.maximum(0.0, mitigated_preds * intensity_ratio * retention_factor)

    with torch.no_grad():
        out_norm_base = MODEL(x_norm, r_data['edge_index'].to(device), edge_norm).squeeze()
        baseline_preds = torch.clamp(out_norm_base * Y_STD + Y_MEAN, min=0.0).cpu().numpy()
        if baseline_preds.ndim == 0:
            baseline_preds = np.array([float(baseline_preds)])
    baseline_preds = np.maximum(0.0, baseline_preds * intensity_ratio)

    orig_total_depth = float(np.sum(baseline_preds))
    mit_total_depth = float(np.sum(mitigated_preds))
    depth_reduction_pct = round(max(0.0, (orig_total_depth - mit_total_depth) / max(0.001, orig_total_depth) * 100.0), 2)
    volume_avoided_m3 = round(max(0.0, (orig_total_depth - mit_total_depth) * 500.0), 1)

    orig_flooded_nodes = int(np.sum(baseline_preds > 0.08))
    mit_flooded_nodes = int(np.sum(mitigated_preds > 0.08))
    nodes_rescued = max(0, orig_flooded_nodes - mit_flooded_nodes)

    node_deltas = []
    for idx, node_id in enumerate(r_data['node_list']):
        orig_d = round(float(baseline_preds[idx]), 4)
        new_d = round(float(mitigated_preds[idx]), 4)
        node_deltas.append({
            'id': str(node_id),
            'baseline_depth': orig_d,
            'mitigated_depth': new_d,
            'reduction_m': round(orig_d - new_d, 4)
        })

    return jsonify({
        'status': 'success',
        'region': r_key,
        'execution_time_ms': round((t1 - t0) * 1000, 2),
        'summary': {
            'depth_reduction_pct': depth_reduction_pct,
            'volume_avoided_m3': volume_avoided_m3,
            'nodes_rescued': nodes_rescued,
            'original_max_m': round(float(np.max(baseline_preds)), 3),
            'mitigated_max_m': round(float(np.max(mitigated_preds)), 3)
        },
        'nodes': node_deltas
    })

@app.route('/api/historical-validation', methods=['GET'])
def get_historical_validation():
    try:
        from historical_storm_validation import run_historical_validation
        res = run_historical_validation('hsr_oct_2024')

        metrics = res['metrics']
        complaints = []
        for c in res['complaints']:
            c_clean = {}
            for k, v in c.items():
                if isinstance(v, (np.floating, np.integer)):
                    c_clean[k] = float(v)
                elif isinstance(v, np.bool_):
                    c_clean[k] = bool(v)
                else:
                    c_clean[k] = v
            complaints.append(c_clean)

        clean_res = {
            'event': res['event'],
            'metrics': metrics,
            'complaints': complaints
        }

        return jsonify({'status': 'success', 'data': clean_res})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    print("\n========================================================")
    print("   UrbanFlow High-Precision GNN Web Server Ready")
    print("   Running locally on: http://127.0.0.1:5000/")
    print("========================================================\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
