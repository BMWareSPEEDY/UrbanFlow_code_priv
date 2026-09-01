import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv

sys.stdout.reconfigure(line_buffering=True)

N_EPOCHS = 800
CKPT = "blr_gine6_ckpt.pt"
SAVE_EVERY = 50


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / max(1e-6, ss_tot))


class GINE4(nn.Module):
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


def evaluate(model, tr, te, x_mean, x_std, e_mean, e_std, yl_mean, yl_std):
    model.eval()
    with torch.no_grad():
        te_out = model((te.x - x_mean) / x_std, te.edge_index, (te.edge_attr - e_mean) / e_std)
        tr_out = model(tr.x, tr.edge_index, tr.edge_attr)
    te_pred = np.clip(np.expm1(te_out.cpu().numpy() * yl_std.item() + yl_mean.item()), 0, None).ravel()
    te_t = te.y.cpu().numpy().ravel()
    tr_pred = np.clip(np.expm1(tr_out.cpu().numpy() * yl_std.item() + yl_mean.item()), 0, None).ravel()
    tr_t = tr.y.cpu().numpy().ravel()

    calib = {}
    iraw = tr.x.cpu().numpy()[:, 15]
    for i in np.unique(iraw):
        m = iraw == i
        if m.sum() < 50:
            continue
        p, t = tr_pred[m], tr_t[m]
        a = np.sum((p - p.mean()) * (t - t.mean())) / max(1e-9, np.sum((p - p.mean()) ** 2))
        b = t.mean() - a * p.mean()
        calib[float(i)] = (a, b)
    te_int = te.x.cpu().numpy()[:, 15]
    te_pred_cal = te_pred.copy()
    for i in np.unique(te_int):
        m = te_int == i
        if float(i) in calib:
            a, b = calib[float(i)]
            te_pred_cal[m] = a * te_pred[m] + b
    te_pred_cal = np.clip(te_pred_cal, 0, None)

    tr_r2 = r2_score(tr_t, tr_pred)
    out = [f"TRAIN R2 {tr_r2:.4f} MAE {np.mean(np.abs(tr_pred-tr_t)):.4f}"]
    for tag, pred in [("raw", te_pred), ("calib", te_pred_cal)]:
        out.append(f"BLR [{tag}] R2 {r2_score(te_t,pred):.4f} MAE {np.mean(np.abs(pred-te_t)):.4f} | "
                   f"+-5cm {np.mean(np.abs(pred-te_t)<=0.05)*100:.1f}% +-10cm {np.mean(np.abs(pred-te_t)<=0.10)*100:.1f}% "
                   f"+-20cm {np.mean(np.abs(pred-te_t)<=0.20)*100:.1f}%")
    out.append("per-intensity (+-10cm / +-20cm / tgt / pred):")
    for i in np.sort(np.unique(te_int)):
        m = te_int == i
        if m.sum() < 10:
            continue
        p10 = np.mean(np.abs(te_pred_cal[m] - te_t[m]) <= 0.10) * 100
        p20 = np.mean(np.abs(te_pred_cal[m] - te_t[m]) <= 0.20) * 100
        out.append(f"  I={i:6.1f}: tgt {te_t[m].mean():.3f} pred {te_pred_cal[m].mean():.3f} | +-10cm {p10:.1f}% +-20cm {p20:.1f}%")
    return "\n".join(out)


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
    yl_mean, yl_std = tr.y.log1p().mean(), tr.y.log1p().std() + 1e-6
    ty = (tr.y.log1p() - yl_mean) / yl_std

    tr.x = (tr.x - x_mean) / x_std
    tr.edge_attr = (tr.edge_attr - e_mean) / e_std

    model = GINE4(in_c=tr.x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)
    start_ep = 1
    if os.path.exists(CKPT):
        ck = torch.load(CKPT)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start_ep = ck['ep'] + 1
        print(f"Resuming from epoch {start_ep}", flush=True)

    t0 = time.time()
    for ep in range(start_ep, N_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        out = model(tr.x, tr.edge_index, tr.edge_attr)
        loss_log = F.mse_loss(out, ty)
        pred_lin = torch.expm1(out * yl_std + yl_mean)
        diff = pred_lin - tr.y
        w = torch.where(diff < 0, torch.full_like(diff, 2.5), torch.ones_like(diff))
        loss = loss_log + 0.5 * torch.mean(w * diff ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ep % SAVE_EVERY == 0 or ep == N_EPOCHS:
            torch.save({'ep': ep, 'model': model.state_dict(), 'opt': opt.state_dict(),
                        'sched': sched.state_dict()}, CKPT)
            print(f"  ep {ep} loss {loss.item():.4f} ({time.time()-t0:.0f}s) [ckpt]", flush=True)

    print(f"\n=== GINE6 hidden160 + micro-topo + asym ({time.time()-t0:.0f}s) ===", flush=True)
    print(evaluate(model, tr, te, x_mean, x_std, e_mean, e_std, yl_mean, yl_std), flush=True)
    torch.save({'model': model.state_dict(), 'x_mean': x_mean, 'x_std': x_std,
                'e_mean': e_mean, 'e_std': e_std, 'yl_mean': yl_mean, 'yl_std': yl_std},
               "blr_gine6_final.pt")


if __name__ == "__main__":
    main()