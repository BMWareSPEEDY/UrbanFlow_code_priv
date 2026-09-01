import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv

BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"
LOCK = os.path.join(BASE, "st_gnn", "gpu_lock_b")


def acquire_lock(path=LOCK, timeout=3600, wait=5):
    t0 = time.time()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - t0 > timeout:
                raise RuntimeError(f"gpu lock timeout: {path}")
            time.sleep(wait)


def release_lock(path=LOCK):
    try:
        os.remove(path)
    except OSError:
        pass


def r2_score(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / max(1e-6, ss_tot)


def load_data():
    dl = torch.load(os.path.join(BASE, "multi_scenario_pyg_dataset.pt"), weights_only=False)
    tr_g = [g.clone() for g in dl if g.city != "bangalore"]
    te_g = [g.clone() for g in dl if g.city == "bangalore"]
    return tr_g, te_g


def compute_stats(tr_g, device):
    X = torch.cat([g.x for g in tr_g], 0).to(device)
    E = torch.cat([g.edge_attr for g in tr_g], 0).to(device)
    Y = torch.cat([g.y for g in tr_g], 0).to(device)
    x_mean, x_std = X.mean(0), X.std(0) + 1e-6
    e_mean, e_std = E.mean(0), E.std(0) + 1e-6
    yl_mean, yl_std = Y.log1p().mean(), Y.log1p().std() + 1e-6
    return x_mean, x_std, e_mean, e_std, yl_mean, yl_std


def make_city_batches(tr_g, x_mean, x_std, e_mean, e_std, n_cities=None, device="cuda"):
    by = {}
    for g in tr_g:
        by.setdefault(g.city, []).append(g)
    cities = sorted(by.keys())
    if n_cities is not None:
        cities = sorted(cities, key=lambda c: sum(g.x.shape[0] for g in by[c]), reverse=True)[:n_cities]
    batches = []
    for c in cities:
        b = Batch.from_data_list(by[c])
        b.x = (b.x - x_mean.cpu()) / x_std.cpu()
        b.edge_attr = (b.edge_attr - e_mean.cpu()) / e_std.cpu()
        batches.append(b.to(device))
    return batches, cities


def topk_per_graph(score, ratio, batch):
    n_g = int(batch.max()) + 1
    cnt = torch.bincount(batch, minlength=n_g)
    k = (cnt * ratio).clamp(min=1).long()
    ptr = torch.cat([cnt.new_zeros(1), cnt.cumsum(0)])
    perms = []
    for i in range(n_g):
        s = score[ptr[i]:ptr[i + 1]]
        kk = min(int(k[i]), s.numel())
        perms.append(torch.topk(s, kk).indices + ptr[i])
    return torch.cat(perms)


def sag_pool(x, edge_index, edge_attr, batch, score, ratio):
    perm = topk_per_graph(score.view(-1), ratio, batch)
    xp = x[perm] * score[perm]
    bp = batch[perm]
    mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    mask[perm] = True
    keep = mask[edge_index[0]] & mask[edge_index[1]]
    eip = edge_index[:, keep]
    idx_map = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
    idx_map[perm] = torch.arange(perm.size(0), device=x.device)
    eip = idx_map[eip]
    return xp, eip, edge_attr[keep], bp, perm


def unpool_fill(xp, perm, num_nodes):
    xu = xp.new_zeros(num_nodes, xp.size(1))
    xu[perm] = xp
    return xu


class GINE(nn.Module):
    def __init__(self, in_c, h, edge_c=2):
        super().__init__()
        self.conv = GINEConv(nn.Linear(in_c, h), edge_dim=edge_c)
        self.ln = nn.LayerNorm(h)

    def forward(self, x, ei, ea):
        return F.elu(self.ln(self.conv(x, ei, ea)))


class ScoreNet(nn.Module):
    def __init__(self, h, edge_c=2):
        super().__init__()
        self.conv = GINEConv(nn.Linear(h, 1), edge_dim=edge_c)

    def forward(self, x, ei, ea):
        return torch.tanh(self.conv(x, ei, ea))


class UrbanPoolNet(nn.Module):
    """GINE backbone + SAG-style top-k pooling stages with unpooling + skip (add).
    Each stage: score nodes with a GINE (edge-aware), pool to ratio, run one coarse
    GINE on the pooled catchment graph, unpool back to full resolution (zero-fill for
    dropped nodes) and add to the local node features -> basin-scale signal reaches
    every node before the final depth head."""

    def __init__(self, in_c=22, edge_c=2, hidden=64, n_local=2, ratios=(0.6,), head=192):
        super().__init__()
        self.n_local = n_local
        self.ratios = list(ratios)
        self.convs = nn.ModuleList()
        for i in range(n_local):
            self.convs.append(GINE(in_c if i == 0 else hidden, hidden, edge_c))
        self.scores = nn.ModuleList([ScoreNet(hidden, edge_c) for _ in self.ratios])
        self.coarse = nn.ModuleList([GINE(hidden, hidden, edge_c) for _ in self.ratios])
        self.reg = nn.Sequential(nn.Linear(hidden + in_c, head), nn.LayerNorm(head),
                                 nn.LeakyReLU(0.1), nn.Linear(head, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
        for r, sc, co in zip(self.ratios, self.scores, self.coarse):
            s = sc(h, edge_index, edge_attr)
            xp, eip, eap, bp, perm = sag_pool(h, edge_index, edge_attr, batch, s, r)
            hc = co(xp, eip, eap)
            h = h + unpool_fill(hc, perm, x.size(0))
        return self.reg(torch.cat([h, x], -1))


def step_loss(model, b, yl_mean, yl_std):
    out = model(b.x, b.edge_index, b.edge_attr, b.batch)
    ty = (b.y.log1p() - yl_mean) / yl_std
    loss_log = F.mse_loss(out, ty)
    pred_lin = torch.expm1(out * yl_std + yl_mean)
    diff = pred_lin - b.y
    w = torch.where(diff < 0, torch.full_like(diff, 2.5), torch.ones_like(diff))
    return loss_log + 0.5 * torch.mean(w * diff ** 2)


def predict(model, b, yl_mean, yl_std):
    model.eval()
    with torch.no_grad():
        out = model(b.x, b.edge_index, b.edge_attr, b.batch)
    return np.clip(np.expm1(out.cpu().numpy() * yl_std.item() + yl_mean.item()), 0, None).ravel()


def f1_hazard(y, p, thr=0.15):
    tp = np.sum((p >= thr) & (y >= thr))
    fp = np.sum((p >= thr) & (y < thr))
    fn = np.sum((p < thr) & (y >= thr))
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc), pr, rc


def evaluate(model, te, te_raw_x, te_y, yl_mean, yl_std, tr_batches=None):
    pred = predict(model, te, yl_mean, yl_std)
    t = te_y.ravel()
    res = {
        "mae": float(np.mean(np.abs(pred - t))),
        "rmse": float(np.sqrt(np.mean((pred - t) ** 2))),
        "r2": float(r2_score(t, pred)),
        "p10": float(np.mean(np.abs(pred - t) <= 0.10) * 100),
        "p20": float(np.mean(np.abs(pred - t) <= 0.20) * 100),
    }
    res["f1"], res["pr"], res["rc"] = f1_hazard(t, pred)
    inten = te_raw_x[:, 15]
    res["per_int"] = {}
    for i in np.sort(np.unique(inten)):
        m = inten == i
        if m.sum() < 10:
            continue
        f_, _, _ = f1_hazard(t[m], pred[m])
        res["per_int"][float(i)] = {
            "mae": float(np.mean(np.abs(pred[m] - t[m]))),
            "r2": float(r2_score(t[m], pred[m])),
            "p10": float(np.mean(np.abs(pred[m] - t[m]) <= 0.10) * 100),
            "f1": float(f_),
            "bias": float(np.mean(pred[m] - t[m])),
        }
    res["bins"] = {}
    for lo, hi in [(0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 3.0)]:
        m = (t >= lo) & (t < hi)
        if m.sum() == 0:
            continue
        res["bins"][f"[{lo:.2f},{hi:.2f})"] = {
            "n": int(m.sum()),
            "mae": float(np.mean(np.abs(pred[m] - t[m]))),
            "bias": float(np.mean(pred[m] - t[m])),
            "contrib": float(np.sum(np.abs(pred[m] - t[m]))),
        }
    if tr_batches is not None:
        ps, ts = [], []
        for b in tr_batches:
            ps.append(predict(model, b, yl_mean, yl_std))
            ts.append(b.y.cpu().numpy().ravel())
        p = np.concatenate(ps)
        y = np.concatenate(ts)
        res["tr_r2"] = float(r2_score(y, p))
        res["tr_mae"] = float(np.mean(np.abs(p - y)))
    return res, pred


def fmt_eval(res):
    out = [f"TE MAE {res['mae']:.4f} R2 {res['r2']:.4f} +-10cm {res['p10']:.1f}% +-20cm {res['p20']:.1f}% F1 {res['f1']:.4f}"]
    if "tr_r2" in res:
        out.append(f"TR R2 {res['tr_r2']:.4f} MAE {res['tr_mae']:.4f}")
    return " | ".join(out)