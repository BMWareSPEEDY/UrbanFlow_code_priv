import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv

sys.stdout.reconfigure(line_buffering=True)

N_EPOCHS = 200
HOLDOUTS = ['hyderabad', 'chennai', 'kochi', 'patna']


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


def evaluate(model, batch, x_mean, x_std, e_mean, e_std, y_mean, y_std, device):
    model.eval()
    with torch.no_grad():
        out = model((batch.x - x_mean.to(device)) / x_std.to(device), batch.edge_index,
                    (batch.edge_attr - e_mean.to(device)) / e_std.to(device))
        pred = torch.clamp(out * y_std + y_mean, min=0.0).cpu().numpy().ravel()
    t = batch.y.cpu().numpy().ravel()
    return pred, t


def main():
    device = torch.device('cuda')
    dl = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
    cities = sorted({g.city for g in dl})

    for holdout in HOLDOUTS:
        t_fold = time.time()
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
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)
        scaler = torch.amp.GradScaler('cuda')

        for ep in range(1, N_EPOCHS + 1):
            model.train()
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(tr.x, tr.edge_index, tr.edge_attr)
                loss = F.mse_loss(out, ty)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

        tr_pred, tr_t = evaluate(model, tr, x_mean, x_std, e_mean, e_std, y_mean, y_std, device)
        te_pred, te_t = evaluate(model, te, x_mean, x_std, e_mean, e_std, y_mean, y_std, device)
        te_x = te.x.cpu().numpy()

        print(f"\n=== HOLDOUT {holdout} ({time.time()-t_fold:.0f}s) ===", flush=True)
        print(f"  TRAIN R2 {r2_score(tr_t, tr_pred):.4f} MAE {np.mean(np.abs(tr_pred-tr_t)):.4f}")
        print(f"  TEST  R2 {r2_score(te_t, te_pred):.4f} MAE {np.mean(np.abs(te_pred-te_t)):.4f} "
              f"RMSE {np.sqrt(np.mean((te_pred-te_t)**2)):.4f}")
        for tol in [0.05, 0.10, 0.20]:
            print(f"  +-{tol*100:.0f}cm: {np.mean(np.abs(te_pred-te_t)<=tol)*100:.1f}%", end="")
        print()

        print(f"  tgt q: {np.round(np.percentile(te_t, [5,25,50,75,90,99]), 3)}")
        print(f"  predq: {np.round(np.percentile(te_pred, [5,25,50,75,90,99]), 3)}")

        # Dry-node behavior
        dry = te_t < 0.10
        wet = te_t >= 0.10
        print(f"  dry nodes (<10cm): {dry.sum()} ({dry.mean()*100:.1f}%), mean pred {te_pred[dry].mean():.3f}")
        print(f"  wet nodes (>=10cm): {wet.sum()} ({wet.mean()*100:.1f}%), mean pred {te_pred[wet].mean():.3f}, mean tgt {te_t[wet].mean():.3f}")

        # Per-intensity tolerance
        print("  per-intensity (+-10cm / +-20cm / R2):")
        for i in np.sort(np.unique(te_x[:, 15])):
            m = te_x[:, 15] == i
            if m.sum() < 10:
                continue
            p10 = np.mean(np.abs(te_pred[m] - te_t[m]) <= 0.10) * 100
            p20 = np.mean(np.abs(te_pred[m] - te_t[m]) <= 0.20) * 100
            r2 = r2_score(te_t[m], te_pred[m])
            print(f"    I={i:6.1f}: {p10:5.1f}% / {p20:5.1f}% / R2 {r2:+.3f}   (tgt_mean {te_t[m].mean():.3f})")


if __name__ == "__main__":
    main()