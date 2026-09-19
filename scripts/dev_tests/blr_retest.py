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


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / max(1e-6, ss_tot))


class GNN2(nn.Module):
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


class GNN4(nn.Module):
    def __init__(self, in_c=17, hidden=64):
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
    tr_g = [g.clone() for g in dl if g.city != 'bangalore']
    te_g = [g.clone() for g in dl if g.city == 'bangalore']
    tr = Batch.from_data_list(tr_g).to(device)
    te = Batch.from_data_list(te_g).to(device)

    x_mean, x_std = tr.x.mean(0), tr.x.std(0) + 1e-6
    e_mean, e_std = tr.edge_attr.mean(0), tr.edge_attr.std(0) + 1e-6
    y_mean, y_std = tr.y.mean(), tr.y.std() + 1e-6
    tr.x = (tr.x - x_mean) / x_std
    tr.edge_attr = (tr.edge_attr - e_mean) / e_std
    ty = (tr.y - y_mean) / y_std

    for name, model in [("GNN2", GNN2(in_c=22)), ("GNN4", GNN4(in_c=22))]:
        model = model.to(device)
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
        te_int = te.x.cpu().numpy()[:, 15]

        calib = {}
        for i in np.unique(tr.x.cpu().numpy()[:, 15] * x_std[15].item() + x_mean[15].item()):
            m = (tr.x.cpu().numpy()[:, 15] * x_std[15].item() + x_mean[15].item()) == i
            if m.sum() < 50:
                continue
            p, t = tr_pred[m], tr_t[m]
            a = np.sum((p - p.mean()) * (t - t.mean())) / max(1e-9, np.sum((p - p.mean()) ** 2))
            b = t.mean() - a * p.mean()
            calib[float(i)] = (a, b)
        te_pred_cal = te_pred.copy()
        for i in np.unique(te_int):
            m = te_int == i
            if float(i) in calib:
                a, b = calib[float(i)]
                te_pred_cal[m] = a * te_pred[m] + b
        te_pred_cal = np.clip(te_pred_cal, 0, None)

        print(f"\n=== {name} ({time.time()-t0:.0f}s) ===")
        print(f"  TRAIN R2 {r2_score(tr_t, tr_pred):.4f} MAE {np.mean(np.abs(tr_pred-tr_t)):.4f}")
        for tag, pred in [("raw", te_pred), ("calib", te_pred_cal)]:
            print(f"  BLR [{tag}] R2 {r2_score(te_t, pred):.4f} MAE {np.mean(np.abs(pred-te_t)):.4f} | "
                  f"+-5cm {np.mean(np.abs(pred-te_t)<=0.05)*100:.1f}% +-10cm {np.mean(np.abs(pred-te_t)<=0.10)*100:.1f}% "
                  f"+-20cm {np.mean(np.abs(pred-te_t)<=0.20)*100:.1f}%")
        print("  per-intensity (+-10cm / +-20cm):")
        for i in np.sort(np.unique(te_int)):
            m = te_int == i
            if m.sum() < 10:
                continue
            p10 = np.mean(np.abs(te_pred_cal[m] - te_t[m]) <= 0.10) * 100
            p20 = np.mean(np.abs(te_pred_cal[m] - te_t[m]) <= 0.20) * 100
            print(f"    I={i:6.1f}: tgt {te_t[m].mean():.3f} pred {te_pred_cal[m].mean():.3f} | +-10cm {p10:.1f}% +-20cm {p20:.1f}%")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()