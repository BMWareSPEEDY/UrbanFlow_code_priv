import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv

sys.stdout.reconfigure(line_buffering=True)


class GNN(nn.Module):
    def __init__(self, in_channels=14, hidden=128, out=1):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden, heads=4, concat=True, edge_dim=2)
        self.conv2 = GATv2Conv(hidden * 4, hidden, heads=2, concat=True, edge_dim=2)
        self.conv3 = GATv2Conv(hidden * 2, hidden, heads=2, concat=False, edge_dim=2)
        self.ln1 = nn.LayerNorm(hidden * 4)
        self.ln2 = nn.LayerNorm(hidden * 2)
        self.ln3 = nn.LayerNorm(hidden)
        self.reg = nn.Sequential(
            nn.Linear(hidden + in_channels, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden, out),
        )

    def forward(self, x, edge_index, edge_attr):
        h1 = F.elu(self.ln1(self.conv1(x, edge_index, edge_attr)))
        h2 = F.elu(self.ln2(self.conv2(h1, edge_index, edge_attr)))
        h3 = F.elu(self.ln3(self.conv3(h2, edge_index, edge_attr)))
        fusion = torch.cat([h3, x], dim=-1)
        return self.reg(fusion)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / max(1e-6, ss_tot))


REGIONS = ['hsr', 'bellandur', 'whitefield', 'ecity', 'koramangala']


def run_loro():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_list = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
    all_regions = sorted({g.region for g in data_list})

    results = []
    for holdout in all_regions:
        train_graphs = [g.clone() for g in data_list if g.region != holdout]
        val_graphs = [g.clone() for g in data_list if g.region == holdout]

        train_batch = Batch.from_data_list(train_graphs)
        val_batch = Batch.from_data_list(val_graphs)

        x_mean, x_std = train_batch.x.mean(0), train_batch.x.std(0) + 1e-6
        e_mean, e_std = train_batch.edge_attr.mean(0), train_batch.edge_attr.std(0) + 1e-6
        y_mean, y_std = train_batch.y.mean(), train_batch.y.std() + 1e-6

        train_batch.x = (train_batch.x - x_mean) / x_std
        train_batch.edge_attr = (train_batch.edge_attr - e_mean) / e_std
        ty = (train_batch.y - y_mean) / y_std

        val_batch.x = (val_batch.x - x_mean) / x_std
        val_batch.edge_attr = (val_batch.edge_attr - e_mean) / e_std

        model = GNN(in_channels=14, hidden=128).to(device)
        tb = train_batch.to(device)
        tbn = ty.to(device)
        vb = val_batch.to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1e-5)

        best = float('inf')
        best_state = None
        for ep in range(300):
            model.train()
            opt.zero_grad()
            out = model(tb.x, tb.edge_index, tb.edge_attr)
            loss = F.mse_loss(out, tbn) + 0.5 * F.l1_loss(out, tbn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                vout = model(vb.x, vb.edge_index, vb.edge_attr)
                vpred = torch.clamp(vout * y_std + y_mean, min=0.0)
                vmae = float(F.l1_loss(vpred, vb.y).cpu())
            if vmae < best:
                best = vmae
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            vout = model(vb.x, vb.edge_index, vb.edge_attr)
            vpred = torch.clamp(vout * y_std + y_mean, min=0.0).cpu().numpy().squeeze()
            vt = vb.y.cpu().numpy().squeeze()
            tout = model(tb.x, tb.edge_index, tb.edge_attr)
            tpred = torch.clamp(tout * y_std + y_mean, min=0.0).cpu().numpy().squeeze()
            tt = tb.y.cpu().numpy().squeeze()

        vd = np.abs(vpred - vt)
        td = np.abs(tpred - tt)
        res = {
            'holdout_region': holdout,
            'train_mae': float(np.mean(td)),
            'train_r2': float(r2_score(tt, tpred)),
            'val_mae': float(np.mean(vd)),
            'val_rmse': float(np.sqrt(np.mean(vd ** 2))),
            'val_r2': float(r2_score(vt, vpred)),
        }
        results.append(res)
        print(f"{holdout:12s} | Train MAE {res['train_mae']:.4f} R2 {res['train_r2']:.3f} | "
              f"Val MAE {res['val_mae']:.4f} RMSE {res['val_rmse']:.4f} R2 {res['val_r2']:.3f}", flush=True)

    df = pd.DataFrame(results)
    print("\n=== LORO SUMMARY ===")
    print(df.to_string(index=False))
    print(f"\nMean held-out R2: {df['val_r2'].mean():.4f} | Mean held-out MAE: {df['val_mae'].mean():.4f}")


if __name__ == "__main__":
    run_loro()