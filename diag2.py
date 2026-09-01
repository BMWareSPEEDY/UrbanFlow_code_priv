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

    model = GNN(in_c=17, hidden=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=250, eta_min=1e-5)
    t0 = time.time()
    for ep in range(1, 251):
        model.train()
        opt.zero_grad()
        out = model(tr.x, tr.edge_index, tr.edge_attr)
        loss = F.mse_loss(out, ty)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    print(f"Trained {250} epochs in {time.time()-t0:.0f}s ({(time.time()-t0)/250:.2f}s/epoch)")

    model.eval()
    with torch.no_grad():
        tr_pred = torch.clamp(model(tr.x, tr.edge_index, tr.edge_attr) * y_std + y_mean, min=0.0).cpu().numpy()
        te_pred = torch.clamp(model((te.x - x_mean.to(device)) / x_std.to(device), te.edge_index,
                                    (te.edge_attr - e_mean.to(device)) / e_std.to(device)) * y_std + y_mean, min=0.0).cpu().numpy()
    tr_t = tr.y.cpu().numpy().ravel()
    te_t = te.y.cpu().numpy().ravel()
    tr_pred = tr_pred.ravel()
    te_pred = te_pred.ravel()

    print(f"\nTRAIN R2 {r2_score(tr_t, tr_pred):.4f} MAE {np.mean(np.abs(tr_pred-tr_t)):.4f}")
    print(f"TEST  R2 {r2_score(te_t, te_pred):.4f} MAE {np.mean(np.abs(te_pred-te_t)):.4f}")

    print("\n--- target vs pred distribution (test) ---")
    print("tgt  q:", np.round(np.percentile(te_t, [5, 25, 50, 75, 90, 99]), 3))
    print("pred q:", np.round(np.percentile(te_pred, [5, 25, 50, 75, 90, 99]), 3))

    print("\n--- per-intensity test R2 ---")
    te_x = te.x.cpu().numpy()
    int_col = list(range(17))[15]  # intensity
    for i in np.sort(np.unique(te_x[:, 15])):
        m = te_x[:, 15] == i
        if m.sum() < 10:
            continue
        print(f"  I={i:6.1f}: n={m.sum():6d} tgt_mean={te_t[m].mean():.3f} pred_mean={te_pred[m].mean():.3f} "
              f"R2={r2_score(te_t[m], te_pred[m]):+.3f} MAE={np.mean(np.abs(te_pred[m]-te_t[m])):.3f}")

    print("\n--- corr of pred with key test features ---")
    fnames = ['rel_drop', 'in_deg', 'out_deg', 'accum_score', 'is_sink', 'max_in_grade',
              'sag_index', 'log_area', 'log_imp_area', 'dist_frac', 'intensity', 'duration']
    idxs = [2, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16]
    for f, i in zip(fnames, idxs):
        c = te_x[:, i]
        if c.std() < 1e-9:
            continue
        print(f"  {f:<12} tgt {np.corrcoef(c, te_t)[0,1]:+.3f}  pred {np.corrcoef(c, te_pred)[0,1]:+.3f}")


if __name__ == "__main__":
    main()