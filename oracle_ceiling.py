import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv

sys.stdout.reconfigure(line_buffering=True)

N_EPOCHS = 400


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / max(1e-6, ss_tot))


class GNN(nn.Module):
    def __init__(self, in_c=17, hidden=96):
        super().__init__()
        self.c1 = GATv2Conv(in_c, hidden, heads=4, concat=True, edge_dim=2)
        self.c2 = GATv2Conv(hidden * 4, hidden, heads=2, concat=True, edge_dim=2)
        self.c3 = GATv2Conv(hidden * 2, hidden, heads=2, concat=False, edge_dim=2)
        self.l1 = nn.LayerNorm(hidden * 4)
        self.l2 = nn.LayerNorm(hidden * 2)
        self.l3 = nn.LayerNorm(hidden)
        self.reg = nn.Sequential(nn.Linear(hidden + in_c, 96), nn.LayerNorm(96),
                                 nn.LeakyReLU(0.1), nn.Linear(96, 1))

    def forward(self, x, ei, ea):
        h1 = F.elu(self.l1(self.c1(x, ei, ea)))
        h2 = F.elu(self.l2(self.c2(h1, ei, ea)))
        h3 = F.elu(self.l3(self.c3(h2, ei, ea)))
        return self.reg(torch.cat([h3, x], -1))


def main():
    device = torch.device('cuda')
    dl = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)

    # Ceiling: train ONLY on Bangalore (its own regions), eval on Bangalore
    tr_g = [g.clone() for g in dl if g.city == 'bangalore']
    tr = Batch.from_data_list(tr_g).to(device)
    print(f"Train on Bangalore only: {len(tr_g)} graphs, {tr.x.shape[0]} nodes")

    x_mean, x_std = tr.x.mean(0), tr.x.std(0) + 1e-6
    e_mean, e_std = tr.edge_attr.mean(0), tr.edge_attr.std(0) + 1e-6
    y_mean, y_std = tr.y.mean(), tr.y.std() + 1e-6
    tr.x = (tr.x - x_mean) / x_std
    tr.edge_attr = (tr.edge_attr - e_mean) / e_std
    ty = (tr.y - y_mean) / y_std

    model = GNN(in_c=17, hidden=96).to(device)
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
        if ep % 100 == 0:
            print(f"  ep {ep} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

    model.eval()
    with torch.no_grad():
        out = model(tr.x, tr.edge_index, tr.edge_attr)
    pred = np.clip(out.cpu().numpy() * y_std.item() + y_mean.item(), 0, None).ravel()
    t = tr.y.cpu().numpy().ravel()
    int_ = tr.x.cpu().numpy()[:, 15] * x_std[15].item() + x_mean[15].item()

    print(f"\n=== ORACLE CEILING (Bangalore-only train) ({time.time()-t0:.0f}s) ===")
    print(f"  TRAIN R2 {r2_score(t, pred):.4f} MAE {np.mean(np.abs(pred-t)):.4f} | "
          f"+-5cm {np.mean(np.abs(pred-t)<=0.05)*100:.1f}% +-10cm {np.mean(np.abs(pred-t)<=0.10)*100:.1f}% "
          f"+-20cm {np.mean(np.abs(pred-t)<=0.20)*100:.1f}%")
    print("  per-intensity:")
    for i in np.sort(np.unique(int_)):
        m = int_ == i
        p10 = np.mean(np.abs(pred[m] - t[m]) <= 0.10) * 100
        p20 = np.mean(np.abs(pred[m] - t[m]) <= 0.20) * 100
        r2i = r2_score(t[m], pred[m])
        print(f"    I={i:6.1f}: R2 {r2i:+.3f} MAE {np.mean(np.abs(pred[m]-t[m])):.3f} | +-10cm {p10:.1f}% +-20cm {p20:.1f}%")


if __name__ == "__main__":
    main()