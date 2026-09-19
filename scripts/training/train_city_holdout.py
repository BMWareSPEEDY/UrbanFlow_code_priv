import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv

sys.stdout.reconfigure(line_buffering=True)


class FloodGNN(nn.Module):
    """
    Graph Attention Network predicting node flood depth (m) from 17 static+dynamic
    features and edge length/grade. Multi-head GATv2 with residual fusion.
    """
    def __init__(self, in_channels=17, hidden_channels=128, out_channels=1):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, concat=True, edge_dim=2)
        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=2, concat=True, edge_dim=2)
        self.conv3 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=2, concat=False, edge_dim=2)
        self.ln1 = nn.LayerNorm(hidden_channels * 4)
        self.ln2 = nn.LayerNorm(hidden_channels * 2)
        self.ln3 = nn.LayerNorm(hidden_channels)
        self.reg = nn.Sequential(
            nn.Linear(hidden_channels + in_channels, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(64, out_channels),
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


def evaluate(model, batch, x_mean, x_std, e_mean, e_std, y_mean, y_std, device):
    model.eval()
    batch = batch.to(device)
    with torch.no_grad():
        out = model((batch.x - x_mean.to(device)) / x_std.to(device),
                    batch.edge_index,
                    (batch.edge_attr - e_mean.to(device)) / e_std.to(device)).squeeze()
        preds = torch.clamp(out * y_std + y_mean, min=0.0).cpu().numpy()
    targets = batch.y.squeeze().cpu().numpy()
    return targets, preds


def run_fold(train_graphs, val_graphs, test_graphs, epochs=400, lr=2e-3,
             hidden=128, seed=0, device=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_batch = Batch.from_data_list(train_graphs)
    val_batch = Batch.from_data_list(val_graphs)

    # Standardize using TRAIN ONLY statistics (critical to avoid leakage)
    x_mean, x_std = train_batch.x.mean(0), train_batch.x.std(0) + 1e-6
    e_mean, e_std = train_batch.edge_attr.mean(0), train_batch.edge_attr.std(0) + 1e-6
    y_mean, y_std = train_batch.y.mean(), train_batch.y.std() + 1e-6

    model = FloodGNN(in_channels=train_batch.x.shape[1], hidden_channels=hidden).to(device)
    tb = train_batch.to(device)
    ty = (train_batch.y - y_mean) / y_std
    ty = ty.to(device)
    vb = val_batch.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    best_val_mae = float('inf')
    best_state = None
    patience = 60
    wait = 0

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(tb.x, tb.edge_index, tb.edge_attr)
        mse = F.mse_loss(out, ty)
        huber = F.huber_loss(out, ty, delta=0.5)
        loss = mse + huber
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if ep % 25 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                tout = model(tb.x, tb.edge_index, tb.edge_attr)
                tpred = torch.clamp(tout * y_std + y_mean, min=0.0)
                tmae = float(F.l1_loss(tpred, tb.y).cpu())
                vout = model(vb.x, vb.edge_index, vb.edge_attr)
                vpred = torch.clamp(vout * y_std + y_mean, min=0.0)
                vmae = float(F.l1_loss(vpred, vb.y).cpu())
            print(f"  Epoch {ep:04d} | loss {loss.item():.4f} | train MAE {tmae:.4f} | val MAE {vmae:.4f}", flush=True)
            if vmae < best_val_mae:
                best_val_mae = vmae
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 25
                if wait >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_graphs_list = test_graphs if isinstance(test_graphs, list) else [test_graphs]
    test_batch = Batch.from_data_list(test_graphs_list)
    targets, preds = evaluate(model, test_batch, x_mean, x_std, e_mean, e_std,
                              torch.tensor(y_mean), torch.tensor(y_std), device)

    diffs = np.abs(preds - targets)
    mae = float(np.mean(diffs))
    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    r2 = float(r2_score(targets, preds))

    # Also evaluate on the training batch for overfitting diagnostics
    tr_targets, tr_preds = evaluate(model, tb, x_mean, x_std, e_mean, e_std,
                                    torch.tensor(y_mean), torch.tensor(y_std), device)
    tr_r2 = float(r2_score(tr_targets, tr_preds))
    tr_mae = float(np.mean(np.abs(tr_preds - tr_targets)))

    return {
        'targets': targets,
        'preds': preds,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'train_r2': tr_r2,
        'train_mae': tr_mae,
        'y_mean': float(y_mean),
        'y_std': float(y_std),
    }


CITIES = ['bangalore', 'hyderabad', 'chennai', 'pune', 'mumbai', 'delhi']
INTENSITIES = [20.0, 50.0, 80.0, 120.0, 150.0, 200.0, 250.0, 300.0]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    data_list = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
    if not isinstance(data_list, list):
        data_list = data_list.to_data_list()
    print(f"Loaded {len(data_list)} graphs from cities: "
          f"{sorted({g.city for g in data_list})}\n", flush=True)

    holdout_city = sys.argv[1] if len(sys.argv) > 1 else 'hyderabad'
    print(f"================================================", flush=True)
    print(f"  HOLDOUT CITY: {holdout_city.upper()}", flush=True)
    print(f"  (Model NEVER sees {holdout_city} during training)", flush=True)
    print(f"================================================", flush=True)

    train_graphs = [g.clone() for g in data_list if g.city != holdout_city]
    test_graphs = [g.clone() for g in data_list if g.city == holdout_city]

    # Within training cities: hold out one intensity (120mm/hr) for early stopping
    val_graphs = [g for g in train_graphs if g.rain_intensity == 120.0]
    train_final = [g for g in train_graphs if g.rain_intensity != 120.0]

    print(f"Train: {len(train_final)} graphs "
          f"(cities {sorted({g.city for g in train_final})})", flush=True)
    print(f"Val:   {len(val_graphs)} graphs (intensity 120mm/hr)", flush=True)
    print(f"Test:  {len(test_graphs)} graphs (all {holdout_city} intensities)\n", flush=True)

    res = run_fold(train_final, val_graphs, test_graphs,
                   epochs=400, lr=2e-3, hidden=128, seed=0, device=device)

    t, p = res['targets'], res['preds']
    print(f"\n=== HOLDOUT CITY: {holdout_city.upper()} ===", flush=True)
    print(f"Test nodes: {len(t)}", flush=True)
    print(f"Train R2 : {res['train_r2']:.4f} | Train MAE: {res['train_mae']:.4f} m", flush=True)
    print(f"MAE : {res['mae']:.4f} m ({res['mae']*100:.2f} cm)", flush=True)
    print(f"RMSE: {res['rmse']:.4f} m ({res['rmse']*100:.2f} cm)", flush=True)
    print(f"R2  : {res['r2']:.4f} ({res['r2']*100:.2f}%)", flush=True)
    d = np.abs(p - t)
    for tol in [0.02, 0.05, 0.10, 0.20]:
        print(f"  within ±{tol*100:.0f}cm: {(d<=tol).mean()*100:.2f}%", flush=True)
    print(f"\nAccuracy (R2 >= 90%): {'YES' if res['r2'] >= 0.90 else 'NO'}", flush=True)


if __name__ == "__main__":
    main()