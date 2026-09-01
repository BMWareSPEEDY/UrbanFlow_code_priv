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
CKPT = r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\iter1_ckpt.pt"
OUT_PREDS = r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter1.pt"


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / max(1e-6, ss_tot))


class GINE6(nn.Module):
    def __init__(self, in_c, edge_c=2, hidden=HIDDEN, n_layers=N_LAYERS):
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

    # --- deep-tail node weights: upweight deep floods ---
    raw_int = tr.x[:, 15].unsqueeze(1)
    w_node = torch.ones_like(tr.y)
    w_node[tr.y > 0.3] = 2.5
    w_node[tr.y > 0.8] = 4.0
    w_node[raw_int >= 150.0] *= 2.0
    w_node = w_node.squeeze(1)

    tr.x = (tr.x - x_mean) / x_std
    tr.edge_attr = (tr.edge_attr - e_mean) / e_std

    model = GINE6(in_c=tr.x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)
    t0 = time.time()

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        out = model(tr.x, tr.edge_index, tr.edge_attr)
        loss_log = torch.mean(w_node * (out.squeeze(1) - ty.squeeze(1)) ** 2)
        pred_lin = torch.expm1(out * yl_std + yl_mean)
        diff = pred_lin - tr.y
        w_asym = torch.where(diff < 0, torch.full_like(diff, 2.5), torch.ones_like(diff))
        loss_asym = torch.mean(w_node.unsqueeze(1) * w_asym * diff ** 2)
        loss = loss_log + 0.5 * loss_asym
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ep % 100 == 0:
            print(f"  ep {ep} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

    model.eval()
    with torch.no_grad():
        te_out = model((te.x - x_mean) / x_std, te.edge_index, (te.edge_attr - e_mean) / e_std)
        tr_out = model(tr.x, tr.edge_index, tr.edge_attr)
    te_pred = np.clip(np.expm1(te_out.cpu().numpy() * yl_std.item() + yl_mean.item()), 0, 3.0).ravel()
    te_t = te.y.cpu().numpy().ravel()
    tr_pred = np.clip(np.expm1(tr_out.cpu().numpy() * yl_std.item() + yl_mean.item()), 0, 3.0).ravel()
    tr_t = tr.y.cpu().numpy().ravel()

    tr_r2 = r2_score(tr_t, tr_pred)
    print(f"TRAIN R2 {tr_r2:.4f} MAE {np.mean(np.abs(tr_pred-tr_t)):.4f}", flush=True)

    torch.save({'pred': te_pred}, OUT_PREDS)
    np.savez(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\train_preds_iter1.npz",
             pred=tr_pred, t=tr_t, intensity=tr.x.cpu().numpy()[:, 15],
             x_mean=x_mean.cpu().numpy(), x_std=x_std.cpu().numpy())
    torch.save({'model': model.state_dict(), 'x_mean': x_mean.cpu(), 'x_std': x_std.cpu(),
                'e_mean': e_mean.cpu(), 'e_std': e_std.cpu(),
                'yl_mean': yl_mean.cpu(), 'yl_std': yl_std.cpu()},
               r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\iter1_model.pt")
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()