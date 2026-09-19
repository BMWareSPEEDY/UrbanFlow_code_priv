import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv

sys.stdout.reconfigure(line_buffering=True)

N_EPOCHS = 800


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / max(1e-6, ss_tot))


class EdgeMLP(nn.Module):
    def __init__(self, edge_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, e):
        return self.net(e)


class GINE4(nn.Module):
    """6-layer edge-conditioned GNN (GINEConv), with raw-feature fusion head."""
    def __init__(self, in_c, edge_c=2, hidden=160, n_layers=6):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GINEConv(nn.Linear(in_c if i == 0 else hidden, hidden), edge_dim=edge_c))
        self.lns = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.reg = nn.Sequential(nn.Linear(hidden + in_c, 192), nn.LayerNorm(192),
                                 nn.LeakyReLU(0.1), nn.Linear(192, 1))

    def forward(self, x, ei, ea):
        h = x
        for conv, ln in zip(self.convs, self.lns):
            h = F.elu(ln(conv(h, ei, ea)))
        return self.reg(torch.cat([h, x], -1))


def main():
    device = torch.device('cuda')
    dl = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
    tr_g = [g.clone() for g in dl if g.city != 'bangalore']
    te_g = [g.clone() for g in dl if g.city == 'bangalore']
    tr = Batch.from_data_list(tr_g).to(device)
    te = Batch.from_data_list(te_g).to(device)
    print(f"Train: {len(tr_g)} graphs ({tr.x.shape[0]} nodes) | Test: {len(te_g)} graphs ({te.x.shape[0]} nodes)", flush=True)

    x_mean, x_std = tr.x.mean(0), tr.x.std(0) + 1e-6
    e_mean, e_std = tr.edge_attr.mean(0), tr.edge_attr.std(0) + 1e-6

    # log-compressed target: y_log = ln(y+1)
    yl_mean, yl_std = tr.y.log1p().mean(), tr.y.log1p().std() + 1e-6
    ty = (tr.y.log1p() - yl_mean) / yl_std
    y_lin_mean, y_lin_std = tr.y.mean(), tr.y.std() + 1e-6

    tr.x = (tr.x - x_mean) / x_std
    tr.edge_attr = (tr.edge_attr - e_mean) / e_std

    model = GINE4(in_c=tr.x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)
    t0 = time.time()

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        out = model(tr.x, tr.edge_index, tr.edge_attr)

        loss_log = F.mse_loss(out, ty)

        # asymmetric penalty on linear depth: under-prediction penalized x2.5
        pred_lin = torch.expm1(out * yl_std + yl_mean)
        diff = pred_lin - tr.y
        w = torch.where(diff < 0, torch.full_like(diff, 2.5), torch.ones_like(diff))
        loss_asym = torch.mean(w * diff ** 2)

        loss = loss_log + 0.5 * loss_asym
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ep % 100 == 0:
            print(f"  ep {ep} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    with torch.no_grad():
        te_out = model((te.x - x_mean.to(device)) / x_std.to(device), te.edge_index,
                       (te.edge_attr - e_mean.to(device)) / e_std.to(device))
    te_pred = np.clip(np.expm1(te_out.cpu().numpy() * yl_std.item() + yl_mean.item()), 0, None).ravel()
    te_t = te.y.cpu().numpy().ravel()
    te_int = te.x.cpu().numpy()[:, 15]

    # train eval for calibration
    with torch.no_grad():
        tr_out = model(tr.x, tr.edge_index, tr.edge_attr)
    tr_pred = np.clip(np.expm1(tr_out.cpu().numpy() * yl_std.item() + yl_mean.item()), 0, None).ravel()
    tr_t = tr.y.cpu().numpy().ravel()
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

    print(f"\n=== GINE4 + micro-topo + asym log-loss ({time.time()-t0:.0f}s) ===")
    print(f"  TRAIN R2 {r2_score(tr_t, tr_pred):.4f} MAE {np.mean(np.abs(tr_pred-tr_t)):.4f}")
    for tag, pred in [("raw", te_pred), ("calib", te_pred_cal)]:
        print(f"  BLR [{tag}] R2 {r2_score(te_t, pred):.4f} MAE {np.mean(np.abs(pred-te_t)):.4f} | "
              f"+-5cm {np.mean(np.abs(pred-te_t)<=0.05)*100:.1f}% +-10cm {np.mean(np.abs(pred-te_t)<=0.10)*100:.1f}% "
              f"+-20cm {np.mean(np.abs(pred-te_t)<=0.20)*100:.1f}%")
    print("  per-intensity (+-10cm / +-20cm / tgt / pred):")
    for i in np.sort(np.unique(te_int)):
        m = te_int == i
        if m.sum() < 10:
            continue
        p10 = np.mean(np.abs(te_pred_cal[m] - te_t[m]) <= 0.10) * 100
        p20 = np.mean(np.abs(te_pred_cal[m] - te_t[m]) <= 0.20) * 100
        print(f"    I={i:6.1f}: tgt {te_t[m].mean():.3f} pred {te_pred_cal[m].mean():.3f} | +-10cm {p10:.1f}% +-20cm {p20:.1f}%")


if __name__ == "__main__":
    main()