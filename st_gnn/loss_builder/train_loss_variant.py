"""UrbanFLOW loss-builder: train GINE4 with a configurable loss variant, validate on Bangalore.

Usage:
  python train_loss_variant.py --variant 0 --hidden 96 --layers 4 --epochs 150 --cities 12 --tag v0_base

Variants:
  0 = baseline asymmetric MSE (replication)
  1 = depth-adaptive asymmetric weights + pseudo-Huber (dry-false-flood hammer, deep-under hammer)
  2 = variant 1 + flooded/dry gate head (BCE aux, gated inference)
  3 = variant 1 + soft-Dice@0.15 + tilted (q=0.30) flood push-up
"""
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv

sys.stdout.reconfigure(line_buffering=True)

LOCK = r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\gpu_lock_a"
DATASET = r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_pyg_dataset.pt"
OUTDIR = r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\loss_builder"


def acquire_lock():
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL)
            os.close(fd)
            print("[lock] acquired", flush=True)
            return
        except FileExistsError:
            print("[lock] waiting...", flush=True)
            time.sleep(20)


def release_lock():
    try:
        os.remove(LOCK)
        print("[lock] released", flush=True)
    except OSError:
        pass


class GINE4(nn.Module):
    def __init__(self, in_c, edge_c=2, hidden=96, n_layers=4, gate=False):
        super().__init__()
        self.gate = gate
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GINEConv(nn.Linear(in_c if i == 0 else hidden, hidden), edge_dim=edge_c))
        self.lns = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.reg = nn.Sequential(nn.Linear(hidden + in_c, 192), nn.LayerNorm(192),
                                 nn.LeakyReLU(0.1), nn.Linear(192, 1))
        if gate:
            self.gate_head = nn.Sequential(nn.Linear(hidden + in_c, 64), nn.LeakyReLU(0.1), nn.Linear(64, 1))

    def forward(self, x, ei, ea):
        h = x
        for conv, ln in zip(self.convs, self.lns):
            h = F.elu(ln(conv(h, ei, ea)))
        cat = torch.cat([h, x], -1)
        out = self.reg(cat)
        g = torch.sigmoid(self.gate_head(cat)) if self.gate else None
        return out, g


def huber(e, delta):
    d = torch.abs(e)
    return torch.where(d < delta, 0.5 * d ** 2 / delta, d - 0.5 * delta)


def loss_v0(out, ty, pred_lin, y, **kw):
    diff = pred_lin - y
    w = torch.where(diff < 0, torch.full_like(diff, 2.5), torch.ones_like(diff))
    return F.mse_loss(out, ty) + 0.5 * torch.mean(w * diff ** 2)


def depth_weights(y, diff, pred_lin):
    w = torch.ones_like(diff)
    dry = y < 0.05
    mid = (y >= 0.05) & (y < 0.5)
    deep = y >= 0.5
    under = diff < 0
    w = torch.where(dry & under, torch.full_like(w, 0.6), w)
    w = torch.where(dry & ~under, torch.full_like(w, 3.0), w)
    w = torch.where(mid & under, torch.full_like(w, 2.0), w)
    w = torch.where(mid & ~under, torch.full_like(w, 1.0), w)
    w = torch.where(deep & under, torch.full_like(w, 4.5), w)
    w = torch.where(deep & ~under, torch.full_like(w, 1.2), w)
    return w


def loss_v1(out, ty, pred_lin, y, **kw):
    diff = pred_lin - y
    w = depth_weights(y, diff, pred_lin)
    return F.mse_loss(out, ty) + 0.5 * torch.mean(w * huber(diff, 0.25))


def loss_v2(out, ty, pred_lin, y, gate=None, **kw):
    L = loss_v1(out, ty, pred_lin, y)
    Lb = F.binary_cross_entropy(gate, (y >= 0.05).float())
    return L + 0.8 * Lb


def loss_v3(out, ty, pred_lin, y, **kw):
    L = loss_v1(out, ty, pred_lin, y)
    p = torch.sigmoid(20.0 * (pred_lin - 0.15))
    t = (y >= 0.15).float()
    dice = (2 * (p * t).sum() + 1.0) / (p.sum() + t.sum() + 1.0)
    Ld = 1.0 - dice
    e = pred_lin - y
    m = y >= 0.05
    Lq = torch.mean(torch.maximum(0.30 * e[m], -0.70 * e[m])) if m.any() else torch.zeros_like(L)
    return L + 0.25 * Ld + 0.3 * Lq


LOSSES = {0: loss_v0, 1: loss_v1, 2: loss_v2, 3: loss_v3}


def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / max(1e-6, ss_tot)


def f1h(y, p, thr=0.15):
    tp = np.sum((p >= thr) & (y >= thr))
    fp = np.sum((p >= thr) & (y < thr))
    fn = np.sum((p < thr) & (y >= thr))
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc), pr, rc


def eval_model(model, te, x_mean, x_std, e_mean, e_std, yl_mean, yl_std, gate_used):
    model.eval()
    with torch.no_grad():
        out, g = model((te.x - x_mean) / x_std, te.edge_index, (te.edge_attr - e_mean) / e_std)
    p = np.clip(np.expm1(out.cpu().numpy() * yl_std.item() + yl_mean.item()), 0, None).ravel()
    if gate_used:
        p = p * np.clip(g.cpu().numpy().ravel(), 0, 1)
    return p, te.y.cpu().numpy().ravel()


def summarize(p, y, inten):
    lines = []
    mae = np.mean(np.abs(p - y))
    f, pr, rc = f1h(y, p)
    lines.append(f"OVERALL: MAE {mae:.4f} | RMSE {np.sqrt(np.mean((p-y)**2)):.4f} | R2 {r2(y,p):.4f} | "
                 f"+-10cm {np.mean(np.abs(p-y)<=0.10)*100:.1f}% | +-20cm {np.mean(np.abs(p-y)<=0.20)*100:.1f}% | "
                 f"F1@0.15 {f:.4f} (P {pr:.3f} R {rc:.3f})")
    for i in np.sort(np.unique(inten)):
        m = inten == i
        f_, _, _ = f1h(y[m], p[m])
        lines.append(f"  I={i:6.1f}: MAE {np.mean(np.abs(p[m]-y[m])):.4f} | R2 {r2(y[m],p[m]):.4f} | "
                     f"+-10cm {np.mean(np.abs(p[m]-y[m])<=0.10)*100:.1f}% | F1@0.15 {f_:.4f}")
    lines.append("ERROR-BY-DEPTH-BIN:")
    for lo, hi in [(0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 3.0)]:
        m = (y >= lo) & (y < hi)
        if m.sum() == 0:
            continue
        lines.append(f"  y in [{lo:.2f},{hi:.2f}): n={m.sum():6d} | MAE {np.mean(np.abs(p[m]-y[m])):.4f} | "
                     f"bias {np.mean(p[m]-y[m]):+.4f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--cities", type=int, default=12)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda')

    dl = torch.load(DATASET, weights_only=False)
    names = sorted({g.city for g in dl if g.city != 'bangalore'})
    rng = np.random.RandomState(args.seed)
    pick = set(rng.choice(names, size=args.cities, replace=False).tolist())
    tr_g = [g.clone() for g in dl if g.city in pick]
    te_g = [g.clone() for g in dl if g.city == 'bangalore']
    tr = Batch.from_data_list(tr_g).to(device)
    te = Batch.from_data_list(te_g).to(device)
    print(f"Train cities: {sorted(pick)}", flush=True)
    print(f"Train: {len(tr_g)} graphs ({tr.x.shape[0]} nodes) | Test: {len(te_g)} graphs ({te.x.shape[0]} nodes)", flush=True)

    dry_frac = float((tr.y < 0.05).float().mean())
    print(f"Train dry-frac (y<0.05): {dry_frac:.3f}", flush=True)

    x_mean, x_std = tr.x.mean(0), tr.x.std(0) + 1e-6
    e_mean, e_std = tr.edge_attr.mean(0), tr.edge_attr.std(0) + 1e-6
    yl_mean, yl_std = tr.y.log1p().mean(), tr.y.log1p().std() + 1e-6
    ty = (tr.y.log1p() - yl_mean) / yl_std
    tr.x = (tr.x - x_mean) / x_std
    tr.edge_attr = (tr.edge_attr - e_mean) / e_std

    gate_used = args.variant == 2
    model = GINE4(in_c=tr.x.shape[1], hidden=args.hidden, n_layers=args.layers, gate=gate_used).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"Model params: {nparam/1e6:.2f}M (variant {args.variant})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    loss_fn = LOSSES[args.variant]

    t0 = time.time()
    best = {}
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        out, g = model(tr.x, tr.edge_index, tr.edge_attr)
        pred_lin = torch.expm1(out * yl_std + yl_mean)
        if gate_used:
            pred_lin = pred_lin * g
        loss = loss_fn(out, ty, pred_lin, tr.y, gate=g)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ep % 25 == 0 or ep == args.epochs:
            p, y = eval_model(model, te, x_mean, x_std, e_mean, e_std, yl_mean, yl_std, gate_used)
            mae = float(np.mean(np.abs(p - y)))
            key = (mae, ep)
            if not best or mae < best[0]:
                best = (mae, ep)
                torch.save({'model': model.state_dict(), 'x_mean': x_mean, 'x_std': x_std,
                            'e_mean': e_mean, 'e_std': e_std, 'yl_mean': yl_mean, 'yl_std': yl_std},
                           os.path.join(OUTDIR, f"best_{args.tag}.pt"))
            print(f"  ep {ep} loss {loss.item():.4f} te_MAE {mae:.4f} ({time.time()-t0:.0f}s) [best {best[0]:.4f}@{best[1]}]", flush=True)

    p, y = eval_model(model, te, x_mean, x_std, e_mean, e_std, yl_mean, yl_std, gate_used)
    inten = te.x.cpu().numpy()[:, 15]
    txt = summarize(p, y, inten)
    print(txt, flush=True)
    with open(os.path.join(OUTDIR, f"summary_{args.tag}.txt"), "w") as f:
        f.write(txt + "\n")
    torch.save({'pred': p}, os.path.join(OUTDIR, f"preds_{args.tag}.pt"))
    print(f"DONE {args.tag} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()