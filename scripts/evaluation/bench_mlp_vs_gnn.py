import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv

sys.stdout.reconfigure(line_buffering=True)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / max(1e-6, ss_tot))


class GNN(nn.Module):
    def __init__(self, in_c=17, hidden=64):
        super().__init__()
        self.c1 = GATv2Conv(in_c, hidden, heads=4, concat=True, edge_dim=2)
        self.c2 = GATv2Conv(hidden * 4, hidden, heads=2, concat=True, edge_dim=2)
        self.l1 = nn.LayerNorm(hidden * 4)
        self.l2 = nn.LayerNorm(hidden * 2)
        self.reg = nn.Sequential(nn.Linear(hidden * 2 + in_c, 64), nn.LayerNorm(64),
                                 nn.LeakyReLU(0.1), nn.Linear(64, 1))

    def forward(self, x, ei, ea):
        h1 = F.elu(self.l1(self.c1(x, ei, ea)))
        h2 = F.elu(self.l2(self.c2(h1, ei, ea)))
        return self.reg(torch.cat([h2, x], -1))


class MLP(nn.Module):
    def __init__(self, in_c=17, hidden=128):
        super().__init__()
        self.reg = nn.Sequential(nn.Linear(in_c, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.1),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.1),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.reg(x)


def main():
    device = torch.device('cuda')
    dl = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
    holdout = 'hyderabad'
    train_graphs = [g.clone() for g in dl if g.city != holdout]
    test_graphs = [g.clone() for g in dl if g.city == holdout]
    tr = Batch.from_data_list(train_graphs).to(device)
    te = Batch.from_data_list(test_graphs).to(device)

    x_mean, x_std = tr.x.mean(0), tr.x.std(0) + 1e-6
    e_mean, e_std = tr.edge_attr.mean(0), tr.edge_attr.std(0) + 1e-6
    y_mean, y_std = tr.y.mean(), tr.y.std() + 1e-6
    tr.x = (tr.x - x_mean) / x_std
    tr.edge_attr = (tr.edge_attr - e_mean) / e_std
    ty = (tr.y - y_mean) / y_std

    def run(name, model, train_fn, predict_fn):
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1e-5)
        t0 = time.time()
        for ep in range(1, 301):
            model.train()
            opt.zero_grad()
            out = train_fn()
            loss = F.mse_loss(out, ty)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        dt = (time.time() - t0) / 300

        model.eval()
        with torch.no_grad():
            tr_pred = torch.clamp(predict_fn(tr) * y_std + y_mean, min=0.0).cpu().numpy()
            te_pred = torch.clamp(predict_fn(te) * y_std + y_mean, min=0.0).cpu().numpy()
        tr_t = tr.y.cpu().numpy()
        te_t = te.y.cpu().numpy()
        print(f"\n[{name}] {dt:.2f}s/epoch", flush=True)
        print(f"  TRAIN R2 {r2_score(tr_t, tr_pred):.4f} MAE {np.mean(np.abs(tr_pred-tr_t)):.4f}", flush=True)
        print(f"  TEST  R2 {r2_score(te_t, te_pred):.4f} MAE {np.mean(np.abs(te_pred-te_t)):.4f}", flush=True)
        return r2_score(te_t, te_pred)

    gnn = GNN(in_c=17, hidden=64).to(device)
    run("GNN hidden64", gnn,
        lambda: gnn(tr.x, tr.edge_index, tr.edge_attr),
        lambda b: gnn(b.x, b.edge_index, b.edge_attr))

    mlp = MLP(in_c=17, hidden=128).to(device)
    run("MLP hidden128", mlp,
        lambda: mlp(tr.x),
        lambda b: mlp(b.x))


if __name__ == "__main__":
    main()