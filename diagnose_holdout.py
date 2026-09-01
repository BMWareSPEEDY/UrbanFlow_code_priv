import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

    model = FloodGNN(in_channels=17, hidden_channels=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=1e-5)

    for ep in range(1, 501):
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
        tr_pred = torch.clamp(tr_out * y_std + y_mean, min=0.0).cpu().numpy()
        te_out = model((te.x - x_mean.to(device)) / x_std.to(device), te.edge_index,
                       (te.edge_attr - e_mean.to(device)) / e_std.to(device))
        te_pred = torch.clamp(te_out * y_std + y_mean, min=0.0).cpu().numpy()
    tr_t = tr.y.cpu().numpy()
    te_t = te.y.cpu().numpy()

    def r2(t, p):
        return 1 - np.sum((t - p) ** 2) / max(1e-9, np.sum((t - t.mean()) ** 2))

    print(f"TRAIN R2 {r2(tr_t, tr_pred):.4f} MAE {np.mean(np.abs(tr_pred-tr_t)):.4f}")
    print(f"TEST  R2 {r2(te_t, te_pred):.4f} MAE {np.mean(np.abs(te_pred-te_t)):.4f}")

    print("\n--- per-intensity test diagnostics (hyderabad) ---")
    for g in test_graphs:
        idx0 = g.x.shape[0]
        # find slice in te
        pass
    # compute per intensity using te batch with batch pointer
    print("\n--- test pred vs target distribution ---")
    print("target quantiles:", np.percentile(te_t, [5, 25, 50, 75, 90, 99]))
    print("pred   quantiles:", np.percentile(te_pred, [5, 25, 50, 75, 90, 99]))
    print("\n--- correlation of pred with key features on test ---")
    te_x = te.x.cpu().numpy()
    for i, nm in enumerate(['rel_drop', 'in_deg', 'log_area', 'log_imp_area', 'intensity']):
        print(f"  {nm}: target corr {np.corrcoef(te_x[:,i], te_t)[0,1]:+.3f} | pred corr {np.corrcoef(te_x[:,i], te_pred)[0,1]:+.3f}")


if __name__ == "__main__":
    main()