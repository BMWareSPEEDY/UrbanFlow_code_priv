import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

sys.stdout.reconfigure(line_buffering=True)

BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"
OUTDIR = os.path.join(BASE, "st_gnn", "stgnn_builder")
LOCK = os.path.join(BASE, "st_gnn", "gpu_lock_b")
DS = os.path.join(BASE, "multi_scenario_pyg_dataset.pt")
DEVICE = torch.device('cuda')
TORCH_SEQ = [20.0, 50.0, 80.0, 120.0, 150.0, 200.0, 250.0, 300.0]


def acquire_lock():
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return
        except FileExistsError:
            time.sleep(5)


def release_lock():
    try:
        os.remove(LOCK)
    except OSError:
        pass


def r2_score(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / max(1e-6, ss_tot)


def f1_hazard(y, p, thr=0.15):
    tp = np.sum((p >= thr) & (y >= thr))
    fp = np.sum((p >= thr) & (y < thr))
    fn = np.sum((p < thr) & (y >= thr))
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc), pr, rc


# ---------------------------------------------------------------- models
class SpatialEnc(nn.Module):
    def __init__(self, in_c, edge_c, hidden, n_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GINEConv(nn.Linear(in_c if i == 0 else hidden, hidden), edge_dim=edge_c))
        self.lns = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

    def forward(self, x, ei, ea):
        h = x
        for conv, ln in zip(self.convs, self.lns):
            h = F.elu(ln(conv(h, ei, ea)))
        return h


class Head(nn.Module):
    def __init__(self, in_c, hidden):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden + in_c, 192), nn.LayerNorm(192),
                                 nn.LeakyReLU(0.1), nn.Linear(192, 1))

    def forward(self, h, x):
        return self.mlp(torch.cat([h, x], -1))


class ConvGRUModel(nn.Module):
    def __init__(self, in_c, edge_c, hidden, n_layers, int_cond=True):
        super().__init__()
        self.hidden = hidden
        self.int_cond = int_cond
        self.enc = SpatialEnc(in_c, edge_c, hidden, n_layers)
        self.gru = nn.GRUCell(hidden + (1 if int_cond else 0), hidden)
        self.head = Head(in_c, hidden)

    def forward(self, seq):
        preds = []
        h = None
        for X, EI, EA, _Y in seq:
            f = self.enc(X, EI, EA)
            if self.int_cond:
                f = torch.cat([f, X[:, 15:16]], -1)
            if h is None:
                h = f.new_zeros(f.shape[0], self.hidden)
            h = self.gru(f, h)
            preds.append(self.head(h, X))
        return preds


class StaticModel(nn.Module):
    def __init__(self, in_c, edge_c, hidden, n_layers):
        super().__init__()
        self.enc = SpatialEnc(in_c, edge_c, hidden, n_layers)
        self.head = Head(in_c, hidden)

    def forward(self, seq):
        return [self.head(self.enc(X, EI, EA), X) for X, EI, EA, _Y in seq]


class STGCNModel(nn.Module):
    def __init__(self, in_c, edge_c, hidden, n_layers, n_tconv=2, kernel=3):
        super().__init__()
        self.enc = SpatialEnc(in_c, edge_c, hidden, n_layers)
        self.tconvs = nn.ModuleList([nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
                                     for _ in range(n_tconv)])
        self.head = Head(in_c, hidden)

    def forward(self, seq):
        fs = [self.enc(X, EI, EA) for X, EI, EA, _Y in seq]      # T x [N, H]
        F = torch.stack(fs, 1).transpose(1, 2)                   # [N, H, T]
        for tc in self.tconvs:
            F = F + F.elu(tc(F))
        F = F.transpose(1, 2)                                    # [N, T, H]
        return [self.head(F[:, t], seq[t][0]) for t in range(len(seq))]


def make_model(arch, in_c, edge_c, hidden, layers, int_cond):
    if arch == 'convgru':
        return ConvGRUModel(in_c, edge_c, hidden, layers, int_cond)
    if arch == 'static':
        return StaticModel(in_c, edge_c, hidden, layers)
    if arch == 'stgcn':
        return STGCNModel(in_c, edge_c, hidden, layers)
    raise ValueError(arch)


# ---------------------------------------------------------------- data
def load_grouped(n_cities):
    dl = torch.load(DS, weights_only=False)
    train = [g for g in dl if g.city != 'bangalore']
    te = [g for g in dl if g.city == 'bangalore']

    regs = {}
    for g in train:
        regs.setdefault((g.city, g.region), []).append(g)
    tr_seq = []
    for k in sorted(regs):
        gs = sorted(regs[k], key=lambda g: g.rain_intensity)
        assert len(gs) == 8 and [g.rain_intensity for g in gs] == TORCH_SEQ
        tr_seq.append((k, gs))
    if n_cities > 0:
        tr_seq = tr_seq[:n_cities]

    te_regs = {}
    for i, g in enumerate(te):
        te_regs.setdefault((g.city, g.region), []).append((i, g))
    te_seq = []
    for k in sorted(te_regs):
        gs = sorted(te_regs[k], key=lambda x: x[1].rain_intensity)
        assert [g.rain_intensity for _, g in gs] == TORCH_SEQ
        te_seq.append((k, gs))
    return tr_seq, te_seq


def compute_stats(tr_seq):
    xs, es, ys = [], [], []
    for _k, gs in tr_seq:
        xs.append(gs[0].x)
        es.append(gs[0].edge_attr)
        for g in gs:
            ys.append(g.y)
    x_all = torch.cat(xs, 0)
    e_all = torch.cat(es, 0)
    y_all = torch.cat(ys, 0)
    return (x_all.mean(0), x_all.std(0) + 1e-6,
            e_all.mean(0), e_all.std(0) + 1e-6,
            y_all.log1p().mean(), y_all.log1p().std() + 1e-6)


def build_batches(seq_list, batch_size, stats):
    xm, xs, em, es, ym, ys = stats
    batches = []
    for i in range(0, len(seq_list), batch_size):
        grp = seq_list[i:i + batch_size]
        steps = [[] for _ in range(8)]
        n_off = 0
        for _k, gs in grp:
            n = gs[0].x.shape[0]
            for t in range(8):
                g = gs[t]
                steps[t].append(((g.x - xm) / xs, g.edge_index + n_off,
                                 (g.edge_attr - em) / es, g.y))
            n_off += n
        batch = []
        for t in range(8):
            X = torch.cat([s[0] for s in steps[t]], 0).to(DEVICE)
            EI = torch.cat([s[1] for s in steps[t]], 1).to(DEVICE)
            EA = torch.cat([s[2] for s in steps[t]], 0).to(DEVICE)
            Y = torch.cat([s[3] for s in steps[t]], 0).to(DEVICE)
            batch.append((X, EI, EA, Y))
        batches.append(batch)
    return batches


# ---------------------------------------------------------------- losses
def step_loss(pred, Y, ym, ys):
    l_log = F.mse_loss(pred, (Y.log1p() - ym) / ys)
    plin = torch.expm1(pred * ys + ym)
    d = plin - Y
    w = torch.where(d < 0, torch.full_like(d, 2.5), torch.ones_like(d))
    return l_log + 0.5 * torch.mean(w * d ** 2)


def mono_penalty(preds, ym, ys, wgt):
    if wgt <= 0:
        return 0.0
    pl = [torch.expm1(p * ys + ym) for p in preds]
    pen = sum(F.relu(a - b + 0.01).mean() for a, b in zip(pl[:-1], pl[1:]))
    return wgt * pen


# ---------------------------------------------------------------- eval
def predict_linear(model, batch, ym, ys):
    model.eval()
    with torch.no_grad():
        preds = model(batch)
        return [torch.expm1(p * ys + ym) for p in preds]


def evaluate(model, tr_batches, te_batch, te_order_idx, stats):
    ym, ys = stats[4], stats[5]
    tr_p, tr_t = [], []
    for b in tr_batches:
        pl = predict_linear(model, b, ym, ys)
        for t, (X, EI, EA, Y) in enumerate(b):
            tr_p.append(pl[t].cpu().numpy().ravel())
            tr_t.append(Y.cpu().numpy().ravel())
    tr_p = np.concatenate(tr_p)
    tr_t = np.concatenate(tr_t)
    tr_r2 = r2_score(tr_t, tr_p)
    tr_mae = np.mean(np.abs(tr_p - tr_t))

    pl = predict_linear(model, te_batch, ym, ys)
    n_te = sum(g.y.shape[0] for _k, gs in te_order_idx for _i, g in gs)
    pred_raw = np.zeros(n_te)
    y_aligned = np.zeros(n_te)
    int_aligned = np.zeros(n_te)
    ptr = 0
    for _k, gs in te_order_idx:
        n = gs[0][1].x.shape[0]
        for t, (idx, g) in enumerate(gs):
            pred_raw[idx:idx + n] = pl[t][ptr:ptr + n].cpu().numpy().ravel()
            y_aligned[idx:idx + n] = g.y.numpy().ravel()
            int_aligned[idx:idx + n] = float(g.rain_intensity)
        ptr += n
    return tr_r2, tr_mae, tr_p, tr_t, np.clip(pred_raw, 0, None), y_aligned, int_aligned


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='convgru', choices=['convgru', 'stgcn', 'static'])
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--n-cities', type=int, default=12)
    ap.add_argument('--hidden', type=int, default=96)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--mono', type=float, default=0.3)
    ap.add_argument('--int-cond', type=int, default=1)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-calib', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    tr_seq, te_seq = load_grouped(args.n_cities)
    stats = compute_stats(tr_seq)
    tr_batches = build_batches(tr_seq, args.batch, stats)
    te_batch = build_batches([(k, [g for _i, g in gs]) for k, gs in te_seq], len(te_seq), stats)[0]
    in_c = te_batch[0][0].shape[1]
    print(f"[{args.prefix}] {args.arch} | cities {args.n_cities} | hidden {args.hidden} "
          f"layers {args.layers} | epochs {args.epochs} | lr {args.lr} | mono {args.mono} | "
          f"int-cond {args.int_cond} | batches/epoch {len(tr_batches)} | in_c {in_c}", flush=True)
    n_train_nodes = sum(b[0][0].shape[0] for b in tr_batches)
    print(f"[{args.prefix}] train nodes {n_train_nodes} | te nodes {te_batch[0][0].shape[0]} | "
          f"load {time.time() - t0:.0f}s", flush=True)

    model = make_model(args.arch, in_c, 2, args.hidden, args.layers, args.int_cond).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.prefix}] params {n_params/1e6:.2f}M", flush=True)

    ckpt_path = os.path.join(OUTDIR, args.prefix + '_ckpt.pt')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    start_ep = 1
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start_ep = ck['ep'] + 1
        print(f"[{args.prefix}] resume ep {start_ep}", flush=True)

    ym, ys = stats[4], stats[5]
    best_loss = float('inf')
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        order = list(range(len(tr_batches)))
        if ep > 1:
            np.random.shuffle(order)
        tot = 0.0
        ep_t0 = time.time()
        for bi in order:
            b = tr_batches[bi]
            opt.zero_grad()
            preds = model(b)
            loss = sum(step_loss(p, Y, ym, ys) for p, (_X, _E, _A, Y) in zip(preds, b))
            loss = loss + mono_penalty(preds, ym, ys, args.mono)
            if not torch.isfinite(loss):
                print(f"[{args.prefix}] NON-FINITE loss ep {ep}; skipping step", flush=True)
                opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        sched.step()
        if ep % 10 == 0 or ep == args.epochs:
            print(f"[{args.prefix}] ep {ep}/{args.epochs} loss {tot/len(tr_batches):.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} ({time.time()-ep_t0:.1f}s/ep, "
                  f"total {time.time()-t0:.0f}s)", flush=True)
        if ep % 25 == 0 or ep == args.epochs:
            torch.save({'ep': ep, 'model': model.state_dict(), 'opt': opt.state_dict(),
                        'sched': sched.state_dict()}, ckpt_path)
            print(f"[{args.prefix}] ckpt saved ep {ep}", flush=True)

    tr_r2, tr_mae, tr_p, tr_t, pred_raw, te_y, te_int = evaluate(
        model, tr_batches, te_batch, te_seq, stats)
    print(f"[{args.prefix}] TRAIN R2 {tr_r2:.4f} MAE {tr_mae:.4f}", flush=True)

    pred_calib = pred_raw.copy()
    if not args.no_calib:
        tr_int = np.concatenate([np.full(g.y.shape[0], float(g.rain_intensity))
                                 for _k, gs in tr_seq for g in gs])
        calib = {}
        for i in np.unique(tr_int):
            m = tr_int == i
            if m.sum() < 50:
                continue
            a = np.sum((tr_p[m] - tr_p[m].mean()) * (tr_t[m] - tr_t[m].mean())) / max(
                1e-9, np.sum((tr_p[m] - tr_p[m].mean()) ** 2))
            b = tr_t[m].mean() - a * tr_p[m].mean()
            calib[float(i)] = (a, b)
        for i in np.unique(te_int):
            m = te_int == i
            if float(i) in calib:
                a, b = calib[float(i)]
                pred_calib[m] = a * pred_calib[m] + b
        pred_calib = np.clip(pred_calib, 0, None)

    def score_tag(tag, p):
        mae = np.mean(np.abs(p - te_y))
        r2 = r2_score(te_y, p)
        f, pr, rc = f1_hazard(te_y, p)
        print(f"[{args.prefix}] BANGALORE [{tag}] MAE {mae:.4f} | R2 {r2:.4f} | "
              f"+-10cm {np.mean(np.abs(p-te_y)<=0.10)*100:.1f}% | F1@0.15 {f:.4f} (P {pr:.3f} R {rc:.3f})",
              flush=True)
        return mae, r2
        y = np.concatenate([g.y.numpy().ravel() for _k, gs in te_seq for _i, g in gs])
        mae = np.mean(np.abs(p - y))
        r2 = r2_score(y, p)
        f, pr, rc = f1_hazard(y, p)
        print(f"[{args.prefix}] BANGALORE [{tag}] MAE {mae:.4f} | R2 {r2:.4f} | "
              f"+-10cm {np.mean(np.abs(p-y)<=0.10)*100:.1f}% | F1@0.15 {f:.4f} (P {pr:.3f} R {rc:.3f})",
              flush=True)
        return mae, r2

    score_tag('raw', pred_raw)
    score_tag('calib', pred_calib)

    torch.save({'pred': pred_raw}, os.path.join(OUTDIR, args.prefix + '_preds.pt'))
    if not args.no_calib:
        torch.save({'pred': pred_calib}, os.path.join(OUTDIR, args.prefix + '_preds_calib.pt'))
    with open(os.path.join(OUTDIR, args.prefix + '_train.txt'), 'w') as f:
        f.write(f"TRAIN R2 {tr_r2:.4f} MAE {tr_mae:.4f}\n")
    print(f"[{args.prefix}] DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    acquire_lock()
    try:
        main()
    finally:
        release_lock()
