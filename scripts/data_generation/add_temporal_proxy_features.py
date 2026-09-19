"""Add 3 static-temporal proxy features (29-31) to existing dataset.

Feature 28: T_c = L^0.6 / S^0.3  (time-to-peak, Kirpich approx)
Feature 29: Rainfall Volume Index = I * duration  (total water mass)
Feature 30: Dynamic Saturation Proxy = imp * (1 + 0.5 * log1p(I*duration/1000))
"""
import torch
import numpy as np
from collections import defaultdict

DATASET_PATH = r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_full22_pyg_dataset.pt"
OUT_PATH = r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_full31_pyg_dataset.pt"

print("Loading dataset...")
dl = torch.load(DATASET_PATH, weights_only=False)
print(f"Loaded {len(dl)} graphs, x.shape={dl[0].x.shape}")


def compute_tc(edge_index, edge_attr, elev, n_nodes):
    """Compute T_c = L^0.6 / S^0.3 for each node using topological upstream path length."""
    n = n_nodes
    adj = defaultdict(list)
    in_degree = np.zeros(n, dtype=int)
    e_length = {}
    e_grade = {}
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    lengths = edge_attr[:, 0].numpy()
    grades = np.abs(edge_attr[:, 1].numpy())

    for i in range(len(src)):
        adj[src[i]].append(dst[i])
        in_degree[dst[i]] += 1
        e_length[(src[i], dst[i])] = lengths[i]
        e_grade[(src[i], dst[i])] = grades[i]

    topo_order = []
    queue = [i for i in range(n) if in_degree[i] == 0]
    in_deg = in_degree.copy()
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for nb in adj[node]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)

    upstream_len = np.zeros(n, dtype=np.float32)
    upstream_grade_sum = np.zeros(n, dtype=np.float32)
    upstream_grade_cnt = np.zeros(n, dtype=np.float32)

    for node in topo_order:
        for nb in adj[node]:
            l = e_length.get((node, nb), 10.0)
            g = max(e_grade.get((node, nb), 0.01), 1e-4)
            cand = upstream_len[node] + l
            if cand > upstream_len[nb]:
                upstream_len[nb] = cand
            upstream_grade_sum[nb] += g
            upstream_grade_cnt[nb] += 1

    avg_grade = np.where(upstream_grade_cnt > 0,
                         upstream_grade_sum / upstream_grade_cnt, 0.01)
    avg_grade = np.maximum(avg_grade, 1e-4)

    tc = np.power(np.maximum(upstream_len, 1.0) / 1000.0, 0.6) / np.power(avg_grade, 0.3)
    tc = np.log1p(tc)
    tc_max = tc.max()
    if tc_max > 0:
        tc = tc / tc_max
    return tc


for i, g in enumerate(dl):
    intensity = g.x[0, 15].item()
    duration = g.x[0, 16].item()
    n_nodes = g.x.shape[0]
    imp = g.x[:, 3].numpy()

    tc = compute_tc(g.edge_index, g.edge_attr, g.x[:, 2].numpy(), n_nodes)

    rainfall_vol_idx = np.full(n_nodes, intensity * duration, dtype=np.float32)

    dyn_sat = imp * (1.0 + 0.5 * np.log1p(intensity * duration / 1000.0))
    dyn_sat = np.clip(dyn_sat, 0.0, 1.0).astype(np.float32)

    new_feats = np.stack([tc, rainfall_vol_idx, dyn_sat], axis=1)
    g.x = torch.cat([g.x, torch.tensor(new_feats, dtype=torch.float)], dim=1)

    if i % 50 == 0:
        print(f"  [{i}/{len(dl)}] {g.city}/{g.region} I={intensity:.0f}: "
              f"T_c=[{tc.min():.3f},{tc.max():.3f}] "
              f"vol_idx={rainfall_vol_idx[0]:.0f} "
              f"dyn_sat=[{dyn_sat.min():.3f},{dyn_sat.max():.3f}]")

print(f"\nFinal x.shape: {dl[0].x.shape}")
torch.save(dl, OUT_PATH)
print(f"Saved to {OUT_PATH}")
