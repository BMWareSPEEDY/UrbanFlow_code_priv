import sys
import os
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv

sys.stdout.reconfigure(line_buffering=True)

RESULTS_CSV = "loo_cv_18city_cal_results.csv"
PROD_CKPT = "urbanflow_production_model.pt"
N_EPOCHS = 200


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / max(1e-6, ss_tot))


class GNN(nn.Module):
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


def predict(model, batch, x_mean, x_std, e_mean, e_std, y_mean, y_std, device, normalized=False):
    model.eval()
    with torch.no_grad():
        x = batch.x if normalized else (batch.x - x_mean.to(device)) / x_std.to(device)
        ea = batch.edge_attr if normalized else (batch.edge_attr - e_mean.to(device)) / e_std.to(device)
        out = model(x, batch.edge_index, ea)
        return torch.clamp(out * y_std + y_mean, min=0.0).cpu().numpy().ravel()


def fit_calibration(train_pred, train_t, train_x_int):
    """Per-intensity affine calibration fitted ONLY on training predictions."""
    calib = {}
    ints = np.unique(train_x_int)
    for i in ints:
        m = train_x_int == i
        if m.sum() < 50:
            continue
        p, t = train_pred[m], train_t[m]
        a = np.sum((p - p.mean()) * (t - t.mean())) / max(1e-9, np.sum((p - p.mean()) ** 2))
        b = t.mean() - a * p.mean()
        calib[float(i)] = (a, b)
    return calib


def apply_calibration(pred, x_int, calib):
    out = np.zeros_like(pred)
    for i in np.unique(x_int):
        m = x_int == i
        if float(i) in calib:
            a, b = calib[float(i)]
            out[m] = a * pred[m] + b
        else:
            out[m] = pred[m]
    return np.clip(out, 0.0, None)


def train_model(tr, ty, device):
    model = GNN(in_c=17, hidden=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    for ep in range(1, N_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            out = model(tr.x, tr.edge_index, tr.edge_attr)
            loss = F.mse_loss(out, ty)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
    return model


def metrics(t, p):
    return {
        'r2': r2_score(t, p),
        'mae': float(np.mean(np.abs(p - t))),
        'rmse': float(np.sqrt(np.mean((p - t) ** 2))),
        'p5': float(np.mean(np.abs(p - t) <= 0.05) * 100),
        'p10': float(np.mean(np.abs(p - t) <= 0.10) * 100),
        'p20': float(np.mean(np.abs(p - t) <= 0.20) * 100),
        'p30': float(np.mean(np.abs(p - t) <= 0.30) * 100),
    }


def main():
    device = torch.device('cuda')
    dl = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
    cities = sorted({g.city for g in dl})
    print(f"Device: {device} | {len(cities)} cities", flush=True)

    done = set()
    if os.path.exists(RESULTS_CSV):
        done = set(pd.read_csv(RESULTS_CSV)['holdout_city'].astype(str))

    for holdout in cities:
        if str(holdout) in done:
            print(f"SKIP {holdout} (done)", flush=True)
            continue

        t_fold = time.time()
        train_graphs = [g.clone() for g in dl if g.city != holdout]
        test_graphs = [g.clone() for g in dl if g.city == holdout]
        tr = Batch.from_data_list(train_graphs).to(device)
        te = Batch.from_data_list(test_graphs).to(device)

        x_mean, x_std = tr.x.mean(0), tr.x.std(0) + 1e-6
        e_mean, e_std = tr.edge_attr.mean(0), tr.edge_attr.std(0) + 1e-6
        y_mean, y_std = tr.y.mean(), tr.y.std() + 1e-6
        tr.x = (tr.x - x_mean) / x_std
        tr.edge_attr = (tr.edge_attr - e_mean) / e_std
        ty = (tr.y - y_mean) / y_std

        model = train_model(tr, ty, device)

        tr_pred = predict(model, tr, x_mean, x_std, e_mean, e_std, y_mean, y_std, device, normalized=True)
        te_pred = predict(model, te, x_mean, x_std, e_mean, e_std, y_mean, y_std, device)
        tr_t = tr.y.cpu().numpy().ravel()
        te_t = te.y.cpu().numpy().ravel()

        calib = fit_calibration(tr_pred, tr_t, tr.x.cpu().numpy()[:, 15] * x_std[15].item() + x_mean[15].item())
        te_int = te.x.cpu().numpy()[:, 15]
        te_pred_cal = apply_calibration(te_pred, te_int, calib)

        m_raw = metrics(te_t, te_pred)
        m_cal = metrics(te_t, te_pred_cal)

        row = {
            'holdout_city': holdout,
            'train_r2': r2_score(tr_t, tr_pred),
            'test_r2': m_cal['r2'], 'test_mae': m_cal['mae'], 'test_rmse': m_cal['rmse'],
            'p5cm_pct': m_cal['p5'], 'p10cm_pct': m_cal['p10'], 'p20cm_pct': m_cal['p20'], 'p30cm_pct': m_cal['p30'],
            'raw_r2': m_raw['r2'], 'raw_mae': m_raw['mae'],
            'raw_p10': m_raw['p10'], 'raw_p20': m_raw['p20'],
            'fold_seconds': time.time() - t_fold,
        }
        df = pd.DataFrame([row])
        if os.path.exists(RESULTS_CSV):
            df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
        else:
            df.to_csv(RESULTS_CSV, index=False)

        print(f"=== {holdout}: TEST R2 {m_cal['r2']:.4f} MAE {m_cal['mae']:.3f} | "
              f"+-5cm {m_cal['p5']:.1f}% +-10cm {m_cal['p10']:.1f}% +-20cm {m_cal['p20']:.1f}% +-30cm {m_cal['p30']:.1f}% | "
              f"raw +-10cm {m_raw['p10']:.1f}% +-20cm {m_raw['p20']:.1f}% | {time.time()-t_fold:.0f}s ===", flush=True)

        del model, tr, te
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.read_csv(RESULTS_CSV)
    print("\n" + "=" * 90)
    print(df[['holdout_city', 'train_r2', 'test_r2', 'test_mae', 'p10cm_pct', 'p20cm_pct', 'p30cm_pct']].to_string(index=False))
    print(f"\nMEAN: TEST R2 {df['test_r2'].mean():.4f} | MAE {df['test_mae'].mean():.3f}m | "
          f"+-10cm {df['p10cm_pct'].mean():.1f}% | +-20cm {df['p20cm_pct'].mean():.1f}% | +-30cm {df['p30cm_pct'].mean():.1f}%")

    # Production model: train on ALL cities, save for deployment
    print("\nTraining production model on ALL cities...", flush=True)
    all_g = [g.clone() for g in dl]
    ab = Batch.from_data_list(all_g).to(device)
    xm, xs = ab.x.mean(0), ab.x.std(0) + 1e-6
    em, es = ab.edge_attr.mean(0), ab.edge_attr.std(0) + 1e-6
    ym, ys = ab.y.mean(), ab.y.std() + 1e-6
    ab.x = (ab.x - xm) / xs
    ab.edge_attr = (ab.edge_attr - em) / es
    ty = (ab.y - ym) / ys
    prod = train_model(ab, ty, device)
    prod_pred = predict(prod, ab, xm, xs, em, es, ym, ys, device, normalized=True)
    m_prod = metrics(ab.y.cpu().numpy().ravel(), prod_pred)
    print(f"Production model (all cities) TRAIN R2 {m_prod['r2']:.4f} MAE {m_prod['mae']:.3f} | "
          f"+-10cm {m_prod['p10']:.1f}% +-20cm {m_prod['p20']:.1f}%", flush=True)
    torch.save({
        'model_state_dict': prod.state_dict(),
        'x_mean': xm.cpu(), 'x_std': xs.cpu(),
        'edge_attr_mean': em.cpu(), 'edge_attr_std': es.cpu(),
        'y_mean': ym.cpu(), 'y_std': ys.cpu(),
        'hidden': 64, 'n_cities': len(cities),
    }, PROD_CKPT)
    print(f"Production model saved to {PROD_CKPT}", flush=True)


if __name__ == "__main__":
    main()