import osmnx as ox
import numpy as np
import torch
from torch_geometric.data import Data, Batch

SUBCATCHMENT_AREA_HA = 0.5


def manning_full_capacity(n, s):
    # RECT_OPEN 1.0 x 1.5: A=1.5, R=0.375, R^(2/3)=0.520
    return (1.0 / np.maximum(n, 1e-4)) * 1.5 * 0.520 * np.sqrt(np.maximum(s, 1e-6))


def manning_depth(q, s, n, width=1.0, hmax=3.0):
    # solve Q(h) = (1/n) * (w*h) * (h*w/(w+2h))^(2/3) * sqrt(s) = q, bisection
    lo = np.zeros_like(q)
    hi = np.full_like(q, hmax)
    for _ in range(40):
        h = 0.5 * (lo + hi)
        A = width * h
        P = width + 2 * h
        R = np.maximum(A / np.maximum(P, 1e-9), 1e-9)
        Qh = (1.0 / np.maximum(n, 1e-4)) * A * R ** (2.0 / 3.0) * np.sqrt(np.maximum(s, 1e-6))
        lo = np.where(Qh < q, h, lo)
        hi = np.where(Qh < q, hi, h)
    return 0.5 * (lo + hi)


def compute_physics(G, nodes_list, node_to_idx):
    """Steady-state hydraulic prior features per node."""
    elevs = {nid: float(G.nodes[nid].get('elevation', 880.0)) for nid in nodes_list}
    imps = {nid: float(G.nodes[nid].get('impervious_ratio', 0.2)) for nid in nodes_list}

    out_neighbors = {nid: [] for nid in nodes_list}
    out_grades = {nid: [] for nid in nodes_list}
    out_n = {nid: [] for nid in nodes_list}
    for u, v, k, data in G.edges(keys=True, data=True):
        if u in out_neighbors and u != v:
            out_neighbors[u].append(v)
            out_grades[u].append(abs(float(data.get('grade', 0.0))))
            out_n[u].append(float(data.get('manning_n', 0.013)))

    acc_imp_area = {nid: SUBCATCHMENT_AREA_HA * imps[nid] for nid in nodes_list}
    ordered = sorted(nodes_list, key=lambda n: elevs[n], reverse=True)
    for u in ordered:
        for v in out_neighbors[u]:
            acc_imp_area[v] += acc_imp_area[u]

    for nid in nodes_list:
        gs = out_grades[nid]
        ns = out_n[nid]
        if len(gs) == 0:
            out_grades[nid] = [1e-4]
            out_n[nid] = [0.013]
            gs = out_grades[nid]
            ns = out_n[nid]
        # steepest downstream edge carries the flow
        i = int(np.argmax(np.array(gs)))
        out_grades[nid] = gs[i]
        out_n[nid] = ns[i]

    arr_nodes = np.array([node_to_idx[n] for n in nodes_list])
    arr_imp_area = np.array([acc_imp_area[n] for n in nodes_list])
    arr_grade = np.array([out_grades[n] for n in nodes_list])
    arr_n = np.array([out_n[n] for n in nodes_list])
    return arr_nodes, arr_imp_area, arr_grade, arr_n


def main():
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
    }
    dl = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
    dl_by_reg = {g.region: g for g in dl}

    print(f"{'region':<13} {'corr(log-prior,log-y)':>24} {'n':>7}  per-intensity corr")
    for reg, f in region_files.items():
        G = ox.load_graphml(f)
        nodes_list = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes_list)}
        arr_nodes, imp_area, grade, n_ = compute_physics(G, nodes_list, node_to_idx)

        g0 = dl_by_reg[reg]
        x = g0.x.numpy()
        y = g0.y.numpy().ravel()
        intensity = x[:, 15]
        # q_in = intensity(mm/hr) * acc_imp_area(ha) * 10000 m2/ha / 3600000 -> m3/s, runoff coeff 0.9
        q = intensity * imp_area * 10000.0 * 0.9 / 3600000.0
        depth_prior = manning_depth(q, grade, n_)

        safe = y > 0.02
        c1 = np.corrcoef(np.log1p(depth_prior[safe]), np.log1p(y[safe]))[0, 1]
        per_int = {}
        for I in np.unique(intensity):
            m = safe & (intensity == I)
            if m.sum() > 50:
                per_int[I] = np.corrcoef(np.log1p(depth_prior[m]), np.log1p(y[m]))[0, 1]
        pis = " ".join(f"{k}:{v:.2f}" for k, v in sorted(per_int.items()))
        print(f"{reg:<13} {c1:>24.4f} {safe.sum():>7}  {pis}")


if __name__ == "__main__":
    main()