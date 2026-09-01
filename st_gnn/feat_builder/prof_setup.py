import sys
import time
import numpy as np
import torch
from torch_geometric.data import Batch

sys.stdout.reconfigure(line_buffering=True)

BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"
DATASET = BASE + r"\multi_scenario_pyg_dataset.pt"

dl = torch.load(DATASET, weights_only=False)
print(f"n_graphs={len(dl)}", flush=True)

from collections import Counter, OrderedDict
city_nodes = OrderedDict()
city_graphs = Counter()
for g in dl:
    city = g.city
    city_graphs[city] += 1
    city_nodes[city] = city_nodes.get(city, 0) + g.x.shape[0]

for c, n in city_nodes.items():
    print(f"city={c:15s} graphs={city_graphs[c]:3d} nodes={n:8d} feats={dl[0].x.shape[1]}", flush=True)

g = dl[0]
print("x cols sample:", g.x[0].tolist(), flush=True)

# --- validate that we can reproduce log_area (col12) via elevation-ordered propagation ---
def propagate_check(g):
    x = g.x.numpy()
    ei = g.edge_index.numpy()
    n = x.shape[0]
    rel_drop = x[:, 2]
    log_area = x[:, 12]
    order = np.argsort(rel_drop)  # ascending rel_drop = highest elevation first
    base = np.full(n, 0.5)
    acc = base.copy()
    out_list = [[] for _ in range(n)]
    for i in range(ei.shape[1]):
        out_list[ei[0, i]].append(ei[1, i])
    for u in order:
        a = acc[u]
        for v in out_list[u]:
            acc[v] += a
    err = np.abs(np.log1p(acc) - log_area).max()
    corr = np.corrcoef(np.log1p(acc), log_area)[0, 1]
    return err, corr

errs = []
corrs = []
for g in dl[:8]:
    e, c = propagate_check(g)
    errs.append(e)
    corrs.append(c)
print(f"log_area reproduction: max_abs_err={max(errs):.4f} corr range=[{min(corrs):.4f},{max(corrs):.4f}]", flush=True)

# --- time a few epochs on a chosen subset (CUDA) ---
def time_subset(cities, hidden, layers, epochs):
    tr_g = [g.clone() for g in dl if g.city in cities]
    tr = Batch.from_data_list(tr_g).to("cuda")
    x_mean, x_std = tr.x.mean(0), tr.x.std(0) + 1e-6
    yl_mean, yl_std = tr.y.log1p().mean(), tr.y.log1p().std() + 1e-6
    ty = (tr.y.log1p() - yl_mean) / yl_std
    tr.x = (tr.x - x_mean) / x_std
    import torch.nn as nn
    from torch_geometric.nn import GINEConv
    class M(nn.Module):
        def __init__(self, in_c, hidden=96, n_layers=4):
            super().__init__()
            self.convs = nn.ModuleList()
            for i in range(n_layers):
                self.convs.append(GINEConv(nn.Linear(in_c if i == 0 else hidden, hidden), edge_dim=2))
            self.lns = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
            self.reg = nn.Sequential(nn.Linear(hidden + in_c, 192), nn.LayerNorm(192), nn.LeakyReLU(0.1), nn.Linear(192, 1))
        def forward(self, x, ei, ea):
            h = x
            for conv, ln in zip(self.convs, self.lns):
                h = torch.nn.functional.elu(ln(conv(h, ei, ea)))
            return self.reg(torch.cat([h, x], -1))
    model = M(tr.x.shape[1], hidden, layers).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    import torch.nn.functional as F
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(tr.x, tr.edge_index, tr.edge_attr)
        loss_log = F.mse_loss(out, ty)
        pred_lin = torch.expm1(out * yl_std + yl_mean)
        diff = pred_lin - tr.y
        w = torch.where(diff < 0, torch.full_like(diff, 2.5), torch.ones_like(diff))
        loss = loss_log + 0.5 * torch.mean(w * diff ** 2)
        loss.backward()
        opt.step()
    dt = time.time() - t0
    print(f"time: cities={cities} hidden={hidden} layers={layers} epochs={epochs} -> {dt:.1f}s ({dt/epochs:.2f}s/ep)", flush=True)
    return dt / epochs

print(f"GPU: {torch.cuda.get_device_name(0)}  mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)
time_subset({'hyderabad', 'chennai', 'pune', 'mumbai', 'delhi'}, 96, 4, 5)
time_subset({'hyderabad', 'chennai', 'pune', 'mumbai', 'delhi', 'kolkata', 'ahmedabad', 'jaipur'}, 96, 4, 5)
