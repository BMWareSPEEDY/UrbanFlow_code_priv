import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv

sys.stdout.reconfigure(line_buffering=True)


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


def r2(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)


def main():
    device = torch.device('cpu')
    dl = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
    tr_g = [g.clone() for g in dl if g.city != 'bangalore']
    te_g = [g.clone() for g in dl if g.city == 'bangalore']
    tr = Batch.from_data_list(tr_g).to(device)
    te = Batch.from_data_list(te_g).to(device)

    ck = torch.load("blr_gine6_final.pt", weights_only=False)
    ck = {k: (v.to('cpu') if torch.is_tensor(v) else v) for k, v in ck.items()}
    model = GINE4(in_c=tr.x.shape[1]).to(device)
    model.load_state_dict(ck['model'])
    model.eval()

    t0 = time.time()
    with torch.no_grad():
        te_out = model((te.x - ck['x_mean']) / ck['x_std'], te.edge_index,
                       (te.edge_attr - ck['e_mean']) / ck['e_std'])
        tr_out = model((tr.x - ck['x_mean']) / ck['x_std'], tr.edge_index,
                       (tr.edge_attr - ck['e_mean']) / ck['e_std'])
    print(f"forward took {time.time()-t0:.0f}s")

    yl_mean, yl_std = ck['yl_mean'], ck['yl_std']
    def unnorm(o):
        return np.clip(np.expm1(o.cpu().numpy() * yl_std.item() + yl_mean.item()), 0, None).ravel()

    tr_pred, tr_t = unnorm(tr_out), tr.y.numpy().ravel()
    te_pred, te_t = unnorm(te_out), te.y.numpy().ravel()
    tr_I = tr.x.cpu().numpy()[:, 15].astype(float)
    te_I = te.x.cpu().numpy()[:, 15].astype(float)

    print(f"TRAIN overall: R2 {r2(tr_t, tr_pred):.4f} MAE {np.abs(tr_pred-tr_t).mean():.4f} "
          f"+-10cm {np.mean(np.abs(tr_pred-tr_t)<=0.10)*100:.1f}%")
    print(f"TEST  overall: R2 {r2(te_t, te_pred):.4f} MAE {np.abs(te_pred-te_t).mean():.4f} "
          f"+-10cm {np.mean(np.abs(te_pred-te_t)<=0.10)*100:.1f}%")

    print(f"\n{'I':>6} {'trR2':>7} {'teR2':>7} {'trMAE':>7} {'teMAE':>7} {'trP10':>6} {'teP10':>6} {'trP20':>6} {'teP20':>6}")
    for I in [20.0, 50.0, 80.0, 120.0, 150.0, 200.0, 250.0, 300.0]:
        mt, mt_ = tr_I == I, te_I == I
        et, ee = np.abs(tr_pred[mt]-tr_t[mt]), np.abs(te_pred[mt_]-te_t[mt_])
        print(f"{I:6.0f} {r2(tr_t[mt], tr_pred[mt]):7.3f} {r2(te_t[mt_], te_pred[mt_]):7.3f} "
              f"{et.mean():7.3f} {ee.mean():7.3f} "
              f"{np.mean(et<=0.10)*100:6.1f} {np.mean(ee<=0.10)*100:6.1f} "
              f"{np.mean(et<=0.20)*100:6.1f} {np.mean(ee<=0.20)*100:6.1f}")

    # deep-regime train underfit check: train MAE by depth bin
    print("\nTRAIN error by true-depth bin:")
    for lo, hi in [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 0.80), (0.80, 3.01)]:
        m = (tr_t >= lo) & (tr_t < hi)
        e = np.abs(tr_pred[m]-tr_t[m])
        print(f"  y in [{lo:.2f},{hi:.2f}): n={m.sum():8d} MAE={e.mean():.3f} bias={(tr_pred[m]-tr_t[m]).mean():+.3f}")


if __name__ == '__main__':
    main()