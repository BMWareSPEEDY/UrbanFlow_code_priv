"""Test exact node-level feature computation in app.py.
Compares hardcoded vs true topological features for HSR and Hong Kong.
"""
import os, sys, torch, numpy as np, osmnx as ox

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from production_v4 import ProductionFloodPredictorV4
from create_combined_dataset import compute_flow_accumulation

def edge_capacity(G, u, v):
    g = abs(float(G.edges[(u, v, list(G[u][v].keys())[0])].get('grade', 0.0)))
    n = float(G.edges[(u, v, list(G[u][v].keys())[0])].get('manning_n', 0.013))
    return (1.0 / max(n, 1e-4)) * 1.5 * 0.520 * np.sqrt(max(g, 1e-6))

def compute_true_graph_features(G, intensity_mmhr=50.0, duration_min=60.0):
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
    dist_outlet = {nid: float(path_hops[nid] / max(1, max(path_hops.values()))) for nid in nodes_list}

    feature_matrix = []
    for nid in nodes_list:
        data = G.nodes[nid]
        x = float(data.get('x', 0.0))
        y = float(data.get('y', 0.0))
        elev = float(data.get('elevation', 880.0))
        imp = float(data.get('impervious_ratio', 0.2))
        manning = float(data.get('manning_n', 0.013))

        rel_drop = (max_elev - elev) / elev_range
        in_deg = in_deg_map.get(nid, 0)
        out_deg = out_deg_map.get(nid, 0)
        accum_score = np.log1p(in_deg * 2.5 + (1.0 if out_deg == 0 else 0.0))
        is_sink = 1.0 if (rel_drop > 0.85 and in_deg >= 2) else 0.0

        in_grades = node_in_grades.get(nid, [0.0])
        out_grades = node_out_grades.get(nid, [0.0])
        max_in_grade = max(in_grades) if len(in_grades) > 0 else 0.0
        min_out_grade = min(out_grades) if len(out_grades) > 0 else 0.0
        sag_index = max(0.0, max_in_grade - min_out_grade) * max(1, in_deg)
        hyd_cap = float(in_deg) / max(1.0, float(out_deg))

        log_area = float(np.log1p(acc_area[nid]))
        log_imp_area = float(np.log1p(acc_imperv_area[nid]))
        df_val = float(dist_frac[nid])
        e_std = float(elev_std2[nid])
        d_dep = float(dep_depth[nid])
        sur = float(surcharge[nid])
        p_cap = float(path_cap[nid])
        p_hops = float(path_hops[nid])
        d_out = float(dist_outlet[nid])
        
        elev_above_outlet = max(0.0, rel_drop * 10.0)
        slope_outlet_ratio = max(0.0, max_in_grade / (0.3 + elev_above_outlet))
        sink_d = max(0.20, d_dep) if is_sink == 1 else (d_dep if d_dep > 0.10 else 0.0)
        inlet_cap = 0.12
        sur_ratio = 100.0 * intensity_mmhr
        
        deg_diff = in_deg - out_deg
        total_rain_mm = intensity_mmhr * (duration_min / 60.0)
        dyn_sat = imp * (1.0 + 0.5 * np.log1p(intensity_mmhr * duration_min / 1000.0))
        true_ponding = np.log1p(sink_d * total_rain_mm / (max(0.2, out_deg) + 0.3))
        inflow_load = np.expm1(log_imp_area) * total_rain_mm
        pipe_drain_cap = np.expm1(p_cap) + 0.1
        conv_def = np.log1p(inflow_load / pipe_drain_cap)
        dep_escape = d_dep / (max(0.005, abs(max_in_grade)) + 0.01)

        row = [
            rel_drop, imp, manning, in_deg, out_deg, accum_score, is_sink, max_in_grade,
            sag_index, hyd_cap, log_area, log_imp_area, df_val, intensity_mmhr, duration_min,
            e_std, d_dep, sur, p_cap, p_hops, d_out,
            elev_above_outlet, slope_outlet_ratio, sink_d, inlet_cap, sur_ratio,
            deg_diff, total_rain_mm, dyn_sat, true_ponding, conv_def, dep_escape
        ]
        feature_matrix.append(row)
        
    src_nodes, dst_nodes, edge_feats = [], [], []
    for u, v, k, data in G.edges(keys=True, data=True):
        src_nodes.append(node_to_idx[u])
        dst_nodes.append(node_to_idx[v])
        edge_feats.append([float(data.get('length', 10.0)), float(data.get('grade', 0.0))])
        
    x_t = torch.tensor(feature_matrix, dtype=torch.float32, device=device)
    ei_t = torch.tensor([src_nodes, dst_nodes], dtype=torch.long, device=device)
    ea_t = torch.tensor(edge_feats, dtype=torch.float32, device=device)
    
    from torch_geometric.data import Data
    return Data(x=x_t, edge_index=ei_t, edge_attr=ea_t)

def test_feature_computation():
    print("Testing dynamic topological feature computation...")
    predictor = ProductionFloodPredictorV4("hydro_gine_v4_3_model.pt", device=device)
    
    # 1. Test Hong Kong
    G_hk = ox.load_graphml("city_hongkong_graph.graphml")
    data_hk = compute_true_graph_features(G_hk, 50.0, 60.0)
    pred_hk, raw_hk, prob_hk = predictor.predict(data_hk, 50.0, 60.0)
    
    print(f"\nHong Kong (3,848 nodes @ 50mm/hr):")
    print(f"  Mean Predicted Depth: {np.mean(pred_hk)*100:.2f} cm")
    print(f"  Max Predicted Depth:  {np.max(pred_hk)*100:.2f} cm")
    print(f"  Flooded Nodes (>15cm): {np.sum(pred_hk > 0.15)} ({np.sum(pred_hk > 0.15)/len(pred_hk)*100:.1f}%)")
    print(f"  Watch Nodes (5-15cm):  {np.sum((pred_hk > 0.05) & (pred_hk <= 0.15))}")
    print(f"  Safe Nodes (<=5cm):    {np.sum(pred_hk <= 0.05)} ({np.sum(pred_hk <= 0.05)/len(pred_hk)*100:.1f}%)")
    
    # 2. Test HSR Layout
    G_hsr = ox.load_graphml("bengaluru_complete_graph.graphml")
    data_hsr = compute_true_graph_features(G_hsr, 50.0, 60.0)
    pred_hsr, raw_hsr, prob_hsr = predictor.predict(data_hsr, 50.0, 60.0)
    
    print(f"\nHSR Layout (1,379 nodes @ 50mm/hr):")
    print(f"  Mean Predicted Depth: {np.mean(pred_hsr)*100:.2f} cm")
    print(f"  Max Predicted Depth:  {np.max(pred_hsr)*100:.2f} cm")
    print(f"  Flooded Nodes (>15cm): {np.sum(pred_hsr > 0.15)}")
    print(f"  Safe Nodes (<=5cm):    {np.sum(pred_hsr <= 0.05)}")

if __name__ == '__main__':
    test_feature_computation()
