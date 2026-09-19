import sys
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv

sys.stdout.reconfigure(line_buffering=True)


class FloodGNN(nn.Module):
    def __init__(self, in_channels=17, hidden_channels=64, out_channels=1):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, concat=True, edge_dim=2)
        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=2, concat=True, edge_dim=2)
        self.conv3 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=2, concat=False, edge_dim=2)
        self.ln1 = nn.LayerNorm(hidden_channels * 4)
        self.ln2 = nn.LayerNorm(hidden_channels * 2)
        self.ln3 = nn.LayerNorm(hidden_channels)
        self.reg = nn.Sequential(
            nn.Linear(hidden_channels + in_channels, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            nn.Linear(64, out_channels),
        )

    def forward(self, x, edge_index, edge_attr):
        h1 = F.elu(self.ln1(self.conv1(x, edge_index, edge_attr)))
        h2 = F.elu(self.ln2(self.conv2(h1, edge_index, edge_attr)))
        h3 = F.elu(self.ln3(self.conv3(h2, edge_index, edge_attr)))
        fusion = torch.cat([h3, x], dim=-1)
        return self.reg(fusion)


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / max(1e-6, ss_tot))


def train_fold(train_graphs, val_graphs, test_graphs, epochs=350, lr=3e-3,
               hidden=64, seed=0, device=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_batch = Batch.from_data_list(train_graphs)
    val_batch = Batch.from_data_list(val_graphs)

    x_mean, x_std = train_batch.x.mean(0), train_batch.x.std(0) + 1e-6
    e_mean, e_std = train_batch.edge_attr.mean(0), train_batch.edge_attr.std(0) + 1e-6
    y_mean, y_std = train_batch.y.mean(), train_batch.y.std() + 1e-6

    model = FloodGNN(in_channels=train_batch.x.shape[1], hidden_channels=hidden).to(device)
    tb = train_batch.to(device)
    ty = ((train_batch.y - y_mean) / y_std).to(device)
    vb = val_batch.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    best_val_mae = float('inf')
    best_state = None
    patience = 80
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

        if ep % 20 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                vout = model(vb.x, vb.edge_index, vb.edge_attr)
                vpred = torch.clamp(vout * y_std.to(device) + y_mean.to(device), min=0.0)
                vmae = float(F.l1_loss(vpred, vb.y).cpu())
            if vmae < best_val_mae:
                best_val_mae = vmae
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 20
                if wait >= patience:
                    break

    model.load_state_dict(best_state)

    # Evaluate test
    test_batch = Batch.from_data_list(test_graphs).to(device)
    model.eval()
    with torch.no_grad():
        out = model((test_batch.x - x_mean.to(device)) / x_std.to(device),
                    test_batch.edge_index,
                    (test_batch.edge_attr - e_mean.to(device)) / e_std.to(device)).squeeze()
        preds = torch.clamp(out * y_std.to(device) + y_mean.to(device), min=0.0).cpu().numpy()
    targets = test_batch.y.squeeze().cpu().numpy()

    # Train eval for overfitting check
    model.eval()
    with torch.no_grad():
        tr_out = model(tb.x, tb.edge_index, tb.edge_attr)
        tr_preds = torch.clamp(tr_out * y_std.to(device) + y_mean.to(device), min=0.0).cpu().numpy()
    tr_targets = tb.y.squeeze().cpu().numpy()

    diffs = np.abs(preds.ravel() - targets.ravel())
    return {
        'targets': targets,
        'preds': preds,
        'mae': float(np.mean(diffs)),
        'rmse': float(np.sqrt(np.mean(diffs ** 2))),
        'r2': float(r2_score(targets, preds)),
        'train_r2': float(r2_score(tr_targets, tr_preds)),
        'train_mae': float(np.mean(np.abs(tr_preds.ravel() - tr_targets.ravel()))),
        'y_mean': float(y_mean),
        'y_std': float(y_std),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    data_list = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
    if not isinstance(data_list, list):
        data_list = data_list.to_data_list()

    cities = sorted({g.city for g in data_list})
    print(f"Cities: {cities}\n", flush=True)

    all_results = []
    for holdout in cities:
        t0 = time.time()
        train_graphs = [g.clone() for g in data_list if g.city != holdout]
        test_graphs = [g.clone() for g in data_list if g.city == holdout]

        # Validation: hold out a random (region, intensity) subset within training cities
        rng = np.random.RandomState(42)
        val_graphs = []
        train_final = []
        for g in train_graphs:
            if g.rain_intensity == 120.0 and rng.rand() < 0.3:
                val_graphs.append(g.clone())
            else:
                train_final.append(g.clone())
        if len(val_graphs) == 0:
            val_graphs = [g.clone() for g in train_graphs if g.rain_intensity == 120.0]
        if len(val_graphs) == 0:
            val_graphs = [g.clone() for g in train_graphs[:2]]

        print(f"=== HOLDOUT CITY: {holdout.upper()} ===", flush=True)
        print(f"  train {len(train_final)} graphs ({len(Batch.from_data_list(train_final).y)} nodes) | "
              f"val {len(val_graphs)} | test {len(test_graphs)}", flush=True)

        res = train_fold(train_final, val_graphs, test_graphs,
                         epochs=350, lr=3e-3, hidden=64, seed=0, device=device)
        dt = time.time() - t0

        t, p = res['targets'], res['preds']
        d = np.abs(p.ravel() - t.ravel())
        acc10 = (d <= 0.10).mean() * 100
        acc20 = (d <= 0.20).mean() * 100
        print(f"  Train R2 {res['train_r2']:.4f} (MAE {res['train_mae']:.4f}m) | "
              f"TEST R2 {res['r2']:.4f} | MAE {res['mae']:.4f}m | RMSE {res['rmse']:.4f}m | "
              f"±10cm {acc10:.1f}% | ±20cm {acc20:.1f}% | {dt:.0f}s", flush=True)

        all_results.append({
            'city': holdout,
            'nodes': len(t),
            'train_r2': res['train_r2'],
            'test_r2': res['r2'],
            'mae': res['mae'],
            'rmse': res['rmse'],
            'acc_10cm': acc10,
            'acc_20cm': acc20,
        })

    df = pd.DataFrame(all_results)
    print("\n=== LEAVE-ONE-CITY-OUT SUMMARY ===")
    print(df.to_string(index=False))
    print(f"\nMEAN TEST R2 : {df['test_r2'].mean():.4f}")
    print(f"MEAN MAE     : {df['mae'].mean():.4f} m")
    print(f"MEAN ±10cm   : {df['acc_10cm'].mean():.2f}%")
    print(f"MEAN ±20cm   : {df['acc_20cm'].mean():.2f}%")
    print(f"MEAN TRAIN R2: {df['train_r2'].mean():.4f}")


if __name__ == "__main__":
    main()