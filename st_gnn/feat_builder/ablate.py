"""Fast feature ablation harness for UrbanFLOW (feature-only changes; architecture/loss fixed).
Runs a sequence of experiments (train subset cities, small model), evaluates Bangalore held-out,
logs one line per experiment to ablation_results.csv."""
import sys
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv

sys.stdout.reconfigure(line_buffering=True)

BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"
V1 = BASE + r"\multi_scenario_pyg_dataset.pt"
V2 = BASE + r"\multi_scenario_pyg_dataset_v2.pt"
LOCK = BASE + r"\st_gnn\gpu_lock_a"


def acquire_lock(path, timeout=3600):
    t0 = time.time()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - t0 > timeout:
                raise RuntimeError("GPU lock timeout")
            time.sleep(5)


def release_lock(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / max(1e-6, ss_tot))


def f1_hazard(y, p, thr=0.15):
    tp = np.sum((p >= thr) & (y >= thr))
    fp = np.sum((p >= thr) & (y < thr))
    fn = np.sum((p < thr) & (y >= thr))
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc)


class GINE4(nn.Module):
    def __init__(self, in_c, hidden, n_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GINEConv(nn.Linear(in_c if i == 0 else hidden, hidden), edge_dim=2))
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

    err = np.abs(te_pred_cal - te_t)
    p10 = np.mean(err <= 0.10) * 100
    p20 = np.mean(err <= 0.20) * 100
    return {
        "mae": float(np.mean(err)), "r2": float(r2_score(te_t, te_pred_cal)),
        "p10": float(p10), "p20": float(p20), "f1": float(f1_hazard(te_t, te_pred_cal)),
    }


def run_experiment(tag, col_sel, cities, epochs, hidden, layers, seed, out_csv):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dl = torch.load(V2, weights_only=False)
    tr_g = [g.clone() for g in dl if g.city in cities]
    te_g = [g.clone() for g in dl if g.city == 'bangalore']
    for g in tr_g:
        g.x = g.x[:, col_sel]
    for g in te_g:
        g.x = g.x[:, col_sel]
    tr = Batch.from_data_list(tr_g).to("cuda")
    te = Batch.from_data_list(te_g).to("cuda")
    print(f"[{tag}] train {len(tr_g)} graphs ({tr.x.shape[0]} nodes), feats={len(col_sel)}", flush=True)

    x_mean, x_std = tr.x.mean(0), tr.x.std(0) + 1e-6
    e_mean, e_std = tr.edge_attr.mean(0), tr.edge_attr.std(0) + 1e-6
    yl_mean, yl_std = tr.y.log1p().mean(), tr.y.log1p().std() + 1e-6
    ty = (tr.y.log1p() - yl_mean) / yl_std
    tr.x = (tr.x - x_mean) / x_std
    tr.edge_attr = (tr.edge_attr - e_mean) / e_std

    model = GINE4(tr.x.shape[1], hidden, layers).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    t0 = time.time()
    for ep in range(1, epochs + 1):
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
        if ep % 50 == 0:
            print(f"  ep {ep} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    res = evaluate(model, tr, te, x_mean, x_std, e_mean, e_std, yl_mean, yl_std)
    line = f"{tag},{epochs},{hidden},{layers},{res['mae']:.4f},{res['r2']:.4f},{res['p10']:.1f},{res['p20']:.1f},{res['f1']:.4f},{time.time()-t0:.0f}"
    print(f"RESULT {line}", flush=True)
    with open(out_csv, "a") as f:
        f.write(line + "\n")


ALL_22 = list(range(22))
FEAT_NAMES = ["rel_x", "rel_y", "rel_drop", "imp", "manning_n", "in_deg", "out_deg", "accum_score",
              "is_sink", "max_in_grade", "sag_index", "hydraulic_capacity", "log_area", "log_imp_area",
              "dist_frac", "intensity", "duration", "elev_std2", "dep_depth", "surcharge", "path_cap",
              "path_hops"]
DROP = {"elev_std2": 17, "dep_depth": 18, "surcharge": 19, "path_cap": 20,
        "path_hops": 21, "log_imp_area": 13, "accum_score": 7}
NEW_COLS = list(range(22, 30))
ALL_30 = list(range(30))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", required=True, help="comma list: baseline,drop_<name>,add")
    ap.add_argument("--cities", default="hyderabad,chennai,pune,mumbai,delhi,kolkata,ahmedabad,jaipur,surat,indore,lucknow,kochi")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=BASE + r"\st_gnn\feat_builder\ablation_results.csv")
    args = ap.parse_args()

    cities = set(args.cities.split(","))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not os.path.exists(args.out):
        with open(args.out, "w") as f:
            f.write("tag,epochs,hidden,layers,mae,r2,p10,p20,f1,secs\n")

    acquire_lock(LOCK)
    try:
        for tag in args.experiments.split(","):
            tag = tag.strip()
            if tag == "baseline":
                sel = ALL_22
            elif tag == "add":
                sel = ALL_30
            elif tag.startswith("drop_"):
                name = tag[5:]
                assert name in DROP, f"unknown drop {name}"
                sel = [c for c in ALL_22 if c != DROP[name]]
            elif tag.startswith("dropnew_"):
                name = tag[8:]
                sel = [c for c in ALL_30 if c != 22 + NEW_COLS.index(name)]
            else:
                raise ValueError(tag)
            run_experiment(tag, sel, cities, args.epochs, args.hidden, args.layers, args.seed, args.out)
            torch.cuda.empty_cache()
    finally:
        release_lock(LOCK)
    print("done", flush=True)


if __name__ == "__main__":
    main()
