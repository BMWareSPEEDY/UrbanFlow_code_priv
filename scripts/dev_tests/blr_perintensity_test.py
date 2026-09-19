import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv

sys.stdout.reconfigure(line_buffering=True)

N_EPOCHS = 250
TARGET_I = 200.0


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / max(1e-6, ss_tot))


class GNN4(nn.Module):
    def __init__(self, in_c, hidden=64):
        super().__init__()
        self.c1 = GATv2Conv(in_c, hidden, heads=4, concat=True, edge_dim=2)
        self.c2 = GATv2Conv(hidden * 4, hidden, heads=2, concat=True, edge_dim=2)
        self.c3 = GATv2Conv(hidden * 2, hidden, heads=2, concat=True, edge_dim=2)
        self.c4 = GATv2Conv(hidden * 2, hidden, heads=2, concat=False, edge_dim=2)
        self.l1 = nn.LayerNorm(hidden * 4)
        self.l2 = nn.LayerNorm(hidden * 2)
        self.l3 = nn.LayerNorm(hidden * 2)
        self.l4 = nn.LayerNorm(hidden)
        self.reg = nn.Sequential(nn.Linear(hidden + hidden * 2 + in_c, 96), nn.LayerNorm(96),
                                 nn.LeakyReLU(0.1), nn.Linear(96, 1))

    def forward(self, x, ei, ea):
        h1 = F.elu(self.l1(self.c1(x, ei, ea)))
        h2 = F.elu(self.l2(self.c2(h1, ei, ea)))
        h3 = F.elu(self.l3(self.c3(h2, ei, ea)))
        h4 = F.elu(self.l4(self.c4(h3, ei, ea)))
        return self.reg(torch.cat([h4, h3, x], -1))


def main():
    device = torch.device('cuda')
    dl = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)

    # Per-intensity model: only graphs at TARGET_I
    tr_g = [g.clone() for g in dl if g.city != 'bangalore' and abs(g.rain_intensity - TARGET_I) < 1e-6]
    te_g = [g.clone() for g in dl if g.city == 'bangalore' and abs(g.rain_intensity - TARGET_I) < 1e-6]
    tr = Batch.from_data_list(tr_g).to(device)
    te = Batch.from_data_list(te_g).to(device)
    print(f"Train graphs at I={TARGET_I}: {len(tr_g)} (nodes {tr.x.shape[0]}), test: {len(te_g)} (nodes {te.x.shape[0]})")

    x_mean, x_std = tr.x.mean(0), tr.x.std(0) + 1e-6
    e_mean, e_std = tr.edge_attr.mean(0), tr.edge_attr.std(0) + 1e-6
    y_mean, y_std = tr.y.mean(), tr.y.std() + 1e-6
    tr.x = (tr.x - x_mean) / x_std
    tr.edge_attr = (tr.edge_attr - e_mean) / e_std
    ty = (tr.y - y_mean) / y_std

    model = GNN4(in_c=17).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)
    t0 = time.time()
    for ep in range(1, N_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        out = model(tr.x, tr.edge_index, tr.edge_attr)
        loss = F.mse_loss(out, ty)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        tr_out = model(tr.x, tr.edge_index, tr.edge_attr)
        te_out = model((te.x - x_mean.to(device)) / x_std.to(device), te.edge_index,
                       (te.edge_attr - e_mean.to(device)) / e_std.to(device))
    tr_pred = np.clip(tr_out.cpu().numpy() * y_std.item() + y_mean.item(), 0, None).ravel()
    te_pred = np.clip(te_out.cpu().numpy() * y_std.item() + y_mean.item(), 0, None).ravel()
    tr_t = tr.y.cpu().numpy().ravel()
    te_t = te.y.cpu().numpy().ravel()

    a = np.sum((tr_pred - tr_pred.mean()) * (tr_t - tr_t.mean())) / max(1e-9, np.sum((tr_pred - tr_pred.mean()) ** 2))
    b = tr_t.mean() - a * tr_pred.mean()
    te_pred_cal = np.clip(a * te_pred + b, 0, None)

    print(f"\n=== Per-intensity model I={TARGET_I} ({time.time()-t0:.0f}s) ===")
    print(f"  TRAIN R2 {r2_score(tr_t, tr_pred):.4f} MAE {np.mean(np.abs(tr_pred-tr_t)):.4f}")
    for tag, pred in [("raw", te_pred), ("calib", te_pred_cal)]:
        print(f"  BLR [{tag}] R2 {r2_score(te_t, pred):.4f} MAE {np.mean(np.abs(pred-te_t)):.4f} | "
              f"+-5cm {np.mean(np.abs(pred-te_t)<=0.05)*100:.1f}% +-10cm {np.mean(np.abs(pred-te_t)<=0.10)*100:.1f}% "
              f"+-20cm {np.mean(np.abs(pred-te_t)<=0.20)*100:.1f}%")
    print(f"  tgt q: {np.round(np.percentile(te_t, [5,25,50,75,90,99]),3)}")
    print(f"  predq: {np.round(np.percentile(te_pred_cal, [5,25,50,75,90,99]),3)}")

    # Per-region
    print("\n  per-region (calib):")
    for reg in sorted({g.region for g in te_g}):
        g0 = next(g for g in te_g if g.region == reg)
        n = g0.x.shape[0]
        # node ids in te batch are contiguous per graph
        cum = 0
        idx = None
        for g in te_g:
            if g.region == reg:
                idx = slice(cum, cum + g.x.shape[0])
                break
            cum += g.x.shape[0]
        p = te_pred_cal[idx]
        t = te_t[idx]
        print(f"    {reg:<12} n={n:>6} tgt {t.mean():.3f} pred {p.mean():.3f} | +-10cm {np.mean(np.abs(p-t)<=0.10)*100:.1f}% +-20cm {np.mean(np.abs(p-t)<=0.20)*100:.1f}%")


if __name__ == "__main__":
    main()