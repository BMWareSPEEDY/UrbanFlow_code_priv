import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv

sys.stdout.reconfigure(line_buffering=True)


class FloodGNN(nn.Module):
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


def main():
    device = torch.device('cuda')
    data_list = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)

    # Fit test: train on ALL cities except the holdout, measure fit on train itself
    holdout = 'hyderabad'
    train_graphs = [g.clone() for g in data_list if g.city != holdout]
    train_batch = Batch.from_data_list(train_graphs).to(device)

    x_mean, x_std = train_batch.x.mean(0), train_batch.x.std(0) + 1e-6
    e_mean, e_std = train_batch.edge_attr.mean(0), train_batch.edge_attr.std(0) + 1e-6
    y_mean, y_std = train_batch.y.mean(), train_batch.y.std() + 1e-6

    train_batch.x = (train_batch.x - x_mean) / x_std
    train_batch.edge_attr = (train_batch.edge_attr - e_mean) / e_std
    ty = ((train_batch.y - y_mean) / y_std)

    model = FloodGNN(in_channels=17, hidden_channels=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=800, eta_min=1e-5)

    for ep in range(1, 801):
        model.train()
        opt.zero_grad()
        out = model(train_batch.x, train_batch.edge_index, train_batch.edge_attr)
        loss = F.mse_loss(out, ty)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ep % 100 == 0:
            model.eval()
            with torch.no_grad():
                tout = model(train_batch.x, train_batch.edge_index, train_batch.edge_attr)
                tpred = torch.clamp(tout * y_std + y_mean, min=0.0).cpu().numpy()
            tt = train_batch.y.cpu().numpy()
            r2 = r2_score(tt, tpred)
            mae = float(np.mean(np.abs(tpred - tt)))
            print(f"Epoch {ep:04d} | loss {loss.item():.4f} | TRAIN R2 {r2:.4f} | MAE {mae:.4f}", flush=True)


if __name__ == "__main__":
    main()