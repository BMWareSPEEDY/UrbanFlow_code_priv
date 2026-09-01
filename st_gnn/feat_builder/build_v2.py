"""Build multi_scenario_pyg_dataset_v2.pt: original 22 features (cols 0..21, unchanged) +
7 new static graph features (cols 22..28). Order and y identical to the original dataset."""
import sys
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"
SRC = BASE + r"\multi_scenario_pyg_dataset.pt"
DST = BASE + r"\multi_scenario_pyg_dataset_v2.pt"

NEW_FEATURES = [
    "imp_pct",      # 22 percentile rank of log_imp_area within region graph (0..1)
    "area_pct",     # 23 percentile rank of log_area within region graph (0..1)
    "pond_vol",     # 24 log1p(upstream accumulation of dep_depth*imp*0.5ha) ponding volume proxy
    "down_sum_cap", # 25 log1p(sum of edge capacities along steepest-downstream path)
    "cap_deficit",  # 26 surcharge - path_cap
    "conv_ratio",   # 27 down_sum_cap - log_area (conveyance ratio in log space)
    "red_idx",      # 28 log1p(#distinct downstream nodes reachable in <=3 hops)
    "imp_grade",    # 29 log1p(expm1(log_imp_area)*max(0,max_in_grade)+1e-9) impervious x grade
]


def build_region_features(g):
    """Compute the 8 new static features for one region graph (node-aligned). Returns [N,8]."""
    x = g.x.numpy()
    ei = g.edge_index.numpy()
    ea = g.edge_attr.numpy()
    grades = ea[:, 1]
    n = x.shape[0]
    rel_drop = x[:, 2]
    imp = x[:, 3]
    max_in_grade = x[:, 9]
    log_area = x[:, 12]
    log_imp_area = x[:, 13]
    dep_depth = x[:, 18]
    surcharge = x[:, 19]
    path_cap = x[:, 20]

    out_list = [[] for _ in range(n)]
    cap_per_edge = np.zeros(ei.shape[1])
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        cap_per_edge[i] = 60.0 * np.sqrt(max(abs(float(grades[i])), 1e-6))
        if u != v:
            out_list[u].append(v)

    def propagate(base):
        acc = base.copy()
        order = np.argsort(rel_drop)  # ascending rel_drop -> highest elevation first
        for u in order:
            a = acc[u]
            for v in out_list[u]:
                acc[v] += a
        return acc

    # 24 pond_vol: upstream depression-storage volume proxy
    pond = propagate(0.5 * dep_depth * imp)
    pond_vol = np.log1p(pond)

    # 22/23 percentile ranks
    def pct_of(col):
        s = np.argsort(np.argsort(col))
        return s / max(1.0, n - 1)
    imp_pct = pct_of(log_imp_area)
    area_pct = pct_of(log_area)

    # 25 down_sum_cap: steepest-downstream path sum of edge capacities
    succ = np.full(n, -1, dtype=np.int64)
    succ_grade = np.zeros(n)
    succ_cap = np.zeros(n)
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        if u != v:
            gg = abs(float(grades[i]))
            if gg > succ_grade[u]:
                succ_grade[u] = gg
                succ[u] = v
                succ_cap[u] = cap_per_edge[i]

    order = np.argsort(rel_drop)[::-1]  # descending rel_drop = lowest elevation first (outlet end)
    sum_cap = np.zeros(n)
    for u in order:
        if succ[u] == -1:
            sum_cap[u] = 0.0
        else:
            sum_cap[u] = succ_cap[u] + sum_cap[succ[u]]
    down_sum_cap = np.log1p(sum_cap)

    # 26 cap_deficit, 27 conv_ratio (col arithmetic)
    cap_deficit = surcharge - path_cap
    conv_ratio = down_sum_cap - log_area

    # 28 red_idx: distinct downstream nodes within 3 hops (boolean sparse matvec frontiers)
    row = torch.from_numpy(ei[0])
    col = torch.from_numpy(ei[1])
    val = torch.ones(ei.shape[1])
    A = torch.sparse_coo_tensor(torch.stack([row, col]), val, (n, n))
    A = A.coalesce()
    ones = torch.ones(n)
    frontier = (A @ ones > 0)
    count_vec = frontier.clone()
    reached = frontier.clone()
    for _ in range(2):
        f = (A @ frontier.float() > 0)
        frontier = f & ~reached
        reached |= frontier
        count_vec += frontier
    red_idx = np.log1p(count_vec.numpy())

    # 29 imp_grade interaction
    imp_grade = np.log1p(np.expm1(np.clip(log_imp_area, 0, None)) * np.maximum(0.0, max_in_grade) + 1e-9)

    feats = np.stack([imp_pct, area_pct, pond_vol, down_sum_cap,
                      cap_deficit, conv_ratio, red_idx, imp_grade], axis=1).astype(np.float32)
    return feats


dl = torch.load(SRC, weights_only=False)
print(f"loaded {len(dl)} graphs", flush=True)

region_first = {}
for idx, g in enumerate(dl):
    reg = g.region
    if reg not in region_first:
        region_first[reg] = idx
        new_feats = build_region_features(g)
        print(f"[{idx:4d}] computed {len(NEW_FEATURES)} new feats for region {reg} ({g.x.shape[0]} nodes)", flush=True)
    else:
        ref = dl[region_first[reg]].x.numpy()
        x = g.x.numpy()
        assert np.allclose(ref[:, [i for i in range(22) if i not in (15, 16)]],
                           x[:, [i for i in range(22) if i not in (15, 16)]], atol=1e-5), \
            f"static cols mismatch region {reg} idx {idx}"
        new_feats = build_region_features(g)
    dl[idx].x = torch.cat([g.x, torch.from_numpy(new_feats)], dim=1)

orig = torch.load(SRC, weights_only=False)
assert len(orig) == len(dl)
for a, b in zip(orig, dl):
    assert torch.equal(a.y, b.y), "y mismatch"
    assert torch.equal(a.edge_index, b.edge_index), "edge_index mismatch"
    assert torch.equal(a.x[:, :22], b.x[:, :22]), "original 22 feats must be untouched"
    assert a.region == b.region and a.city == b.city and a.rain_intensity == b.rain_intensity
n_te = sum(1 for g in dl if g.city == 'bangalore')
n_te_nodes = sum(g.x.shape[0] for g in dl if g.city == 'bangalore')
print(f"validation OK: {len(dl)} graphs, bangalore graphs={n_te} nodes={n_te_nodes}, x now {dl[0].x.shape[1]} cols", flush=True)
torch.save(dl, DST)
print(f"saved {DST}", flush=True)

te = [g for g in dl if g.city == 'bangalore']
X = torch.cat([g.x for g in te], 0).numpy()
for j, name in enumerate(NEW_FEATURES):
    c = X[:, 22 + j]
    print(f"  {name:14s} mean={c.mean():8.4f} std={c.std():8.4f} min={c.min():9.4f} max={c.max():10.4f}", flush=True)

# verify propagation reproduction of log_area (excluding self loops)
reg0 = dl[region_first['hsr']]
x0 = reg0.x.numpy()
ei0 = reg0.edge_index.numpy()
n0 = x0.shape[0]
outl = [[] for _ in range(n0)]
for i in range(ei0.shape[1]):
    u, v = int(ei0[0, i]), int(ei0[1, i])
    if u != v:
        outl[u].append(v)
acc = np.full(n0, 0.5)
for u in np.argsort(x0[:, 2]):
    a = acc[u]
    for v in outl[u]:
        acc[v] += a
err = np.abs(np.log1p(acc) - x0[:, 12]).max()
corr = np.corrcoef(np.log1p(acc), x0[:, 12])[0, 1]
print(f"log_area reproduction on hsr: max_abs_err={err:.4f} corr={corr:.5f}", flush=True)
