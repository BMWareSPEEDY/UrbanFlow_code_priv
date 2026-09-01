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

N_EPOCHS = 400
HIDDEN, N_LAYERS = 128, 5
CKPT = r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\iter2_ckpt.pt"
OUT_PREDS = r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter2.pt"


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / max(1e-6, ss_tot))


def f1(y, p, thr=0.15):
    tp = np.sum((p >= thr) & (y >= thr))
    fp = np.sum((p >= thr) & (y < thr))
    fn = np.sum((p < thr) & (y >= thr))
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc), pr, rc


class TwoHeadGINE(nn.Module):
    def __init__(self, in_c, edge_c=2, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GINEConv(nn.Linear(in_c if i == 0 else hidden, hidden), edge_dim=edge_c))
        self.lns = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.reg = nn.Sequential(nn.Linear(hidden + in_c, 192), nn.LayerNorm(192),
                                 nn.LeakyReLU(0.1), nn.Linear(192, 1))
        self.cls = nn.Sequential(nn.Linear(hidden + in_c, 96), nn.LayerNorm(96),
                                 nn.LeakyReLU(0.1), nn.Linear(96, 1))

    def forward(self, x, ei, ea):
        h = x
        for conv, ln in zip(self.convs, self.lns):
            h = F.elu(ln(conv(h, ei, ea)))
        cat = torch.cat([h, x], -1)
        return self.reg(cat), self.cls(cat)


def main():
    device = torch.device('cuda')
    dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_pyg_dataset.pt", weights_only=False)
    tr_g = [g.clone() for g in dl if g.city != 'bangalore']
    te_g = [g.clone() for g in dl if g.city == 'bangalore']
    tr = Batch.from_data_list(tr_g).to(device)
    te = Batch.from_data_list(te_g).to(device)
    print(f"Train: {len(tr_g)} graphs ({tr.x.shape[0]} nodes) | Test: {len(te_g)} graphs ({te.x.shape[0]} nodes)", flush=True)

    x_mean, x_std = tr.x.mean(0), tr.x.std(0) + 1e-6
    e_mean, e_std = tr.edge_attr.mean(0), tr.edge_attr.std(0) + 1e-6
    yl_mean, yl_std = tr.y.log1p().mean(), tr.y.log1p().std() + 1e-6
    ty = (tr.y.log1p() - yl_mean) / yl_std

    flood = (tr.y > 0.15).float()
    frac = flood.mean().item()
    pos_w = (1 - frac) / max(frac, 1e-4)

    tr.x = (tr.x - x_mean) / x_std
    tr.edge_attr = (tr.edge_attr - e_mean) / e_std

    model = TwoHeadGINE(in_c=tr.x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)
    start_ep = 1
    if os.path.exists(CKPT):
        ck = torch.load(CKPT)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        start_ep = ck['ep'] + 1
        print(f"resume from ep {start_ep}", flush=True)

    t0 = time.time()
    for ep in range(start_ep, N_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        depth_out, cls_out = model(tr.x, tr.edge_index, tr.edge_attr)
        loss_reg = F.mse_loss(depth_out, ty)
        pred_lin = torch.expm1(depth_out * yl_std + yl_mean)
        diff = pred_lin - tr.y
        w_asym = torch.where(diff < 0, torch.full_like(diff, 2.5), torch.ones_like(diff))
        loss_asym = torch.mean(w_asym * diff ** 2)
        loss_cls = F.binary_cross_entropy_with_logits(cls_out, flood, pos_weight=torch.tensor(pos_w, device=device))
        loss = loss_reg + 0.5 * loss_asym + 1.0 * loss_cls
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ep % 100 == 0:
            print(f"  ep {ep} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
        if ep % SAVE_EVERY if False else ep % 200 == 0:
            torch.save({'ep': ep, 'model': model.state_dict(), 'opt': opt.state_dict(),
                        'sched': sched.state_dict()}, CKPT)

    model.eval()
    with torch.no_grad():
        tr_depth, tr_cls = model(tr.x, tr.edge_index, tr.edge_attr)
        te_depth, te_cls = model((te.x - x_mean) / x_std, te.edge_index, (te.edge_attr - e_mean) / e_std)

    tr_pred = np.clip(np.expm1(tr_depth.cpu().numpy() * yl_std.item() + yl_mean.item()), 0, 3.0).ravel()
    tr_t = tr.y.cpu().numpy().ravel()
    tr_p = torch.sigmoid(tr_cls).cpu().numpy().ravel()
    te_pred = np.clip(np.expm1(te_depth.cpu().numpy() * yl_std.item() + yl_mean.item()), 0, 3.0).ravel()
    te_t = te.y.cpu().numpy().ravel()
    te_p = torch.sigmoid(te_cls).cpu().numpy().ravel()
    te_int = te.x.cpu().numpy()[:, 15]

    # --- gate threshold tuned on TRAIN cities only ---
    best_tau, best_f1 = 0.5, -1
    for tau in np.arange(0.30, 0.85, 0.05):
        g = tr_pred.copy(); g[tr_p < tau] = 0.0
        f, _, _ = f1(tr_t, g)
        if f > best_f1:
            best_f1, best_tau = f, tau
    print(f"train-optimal gate tau = {best_tau:.2f} (train F1 {best_f1:.4f})", flush=True)

    for tag, tau in [("no-gate", -1.0), ("gate-train-tau", best_tau)]:
        p = te_pred.copy()
        if tau >= 0:
            p[te_p < tau] = 0.0
        torch.save({'pred': p}, OUT_PREDS.replace(".pt", f"_{tag}.pt"))
        print(f"  saved {tag}")

    torch.save({'model': model.state_dict(), 'x_mean': x_mean.cpu(), 'x_std': x_std.cpu(),
                'e_mean': e_mean.cpu(), 'e_std': e_std.cpu(),
                'yl_mean': yl_mean.cpu(), 'yl_std': yl_std.cpu()},
               r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\iter2_model.pt")
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()