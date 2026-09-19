import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv

sys.stdout.reconfigure(line_buffering=True)

RESULTS_CSV = "loo_cv_18city_results.csv"
N_EPOCHS = 200


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
    cities = sorted({g.city for g in dl})
    print(f"Device: {device} | Cities: {cities}\n", flush=True)

    done = set()
    if os.path.exists(RESULTS_CSV):
        done = set(pd.read_csv(RESULTS_CSV)['holdout_city'].astype(str))

    for holdout in cities:
        if str(holdout) in done:
            print(f"=== SKIP {holdout} (already done) ===", flush=True)
            continue

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

        model.eval()
        with torch.no_grad():
            tr_pred = torch.clamp(model(tr.x, tr.edge_index, tr.edge_attr) * y_std + y_mean, min=0.0).cpu().numpy()
            te_out = model((te.x - x_mean.to(device)) / x_std.to(device), te.edge_index,
                           (te.edge_attr - e_mean.to(device)) / e_std.to(device))
            te_pred = torch.clamp(te_out * y_std + y_mean, min=0.0).cpu().numpy()
        tr_t = tr.y.cpu().numpy().ravel()
        te_t = te.y.cpu().numpy().ravel()
        tr_pred = tr_pred.ravel()
        te_pred = te_pred.ravel()

        train_r2 = r2_score(tr_t, tr_pred)
        train_mae = np.mean(np.abs(tr_pred - tr_t))
        test_r2 = r2_score(te_t, te_pred)
        test_mae = np.mean(np.abs(te_pred - te_t))
        test_rmse = np.sqrt(np.mean((te_pred - te_t) ** 2))
        p10 = np.mean(np.abs(te_pred - te_t) <= 0.10) * 100
        p20 = np.mean(np.abs(te_pred - te_t) <= 0.20) * 100

        row = {
            'holdout_city': holdout, 'train_r2': train_r2, 'train_mae': train_mae,
            'test_r2': test_r2, 'test_mae': test_mae, 'test_rmse': test_rmse,
            'p10cm_pct': p10, 'p20cm_pct': p20, 'fold_seconds': time.time() - t_fold,
        }
        df = pd.DataFrame([row])
        if os.path.exists(RESULTS_CSV):
            df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
        else:
            df.to_csv(RESULTS_CSV, index=False)

        print(f"=== HOLDOUT {holdout}: TRAIN R2 {train_r2:.4f} | TEST R2 {test_r2:.4f} | "
              f"MAE {test_mae:.3f}m | RMSE {test_rmse:.3f} | +-10cm {p10:.1f}% | +-20cm {p20:.1f}% | "
              f"{time.time()-t_fold:.0f}s ===", flush=True)

    df = pd.read_csv(RESULTS_CSV)
    print("\n" + "=" * 80)
    print("SUMMARY (18-city leave-one-out CV):")
    print(df[['holdout_city', 'train_r2', 'test_r2', 'test_mae', 'p10cm_pct', 'p20cm_pct']].to_string(index=False))
    print(f"\nMEAN TEST R2: {df['test_r2'].mean():.4f} | MEAN MAE: {df['test_mae'].mean():.3f}m | "
          f"MEAN +-10cm: {df['p10cm_pct'].mean():.1f}% | MEAN +-20cm: {df['p20cm_pct'].mean():.1f}%")


if __name__ == "__main__":
    main()