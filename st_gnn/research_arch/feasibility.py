import torch
import numpy as np
import math
import sys

def norm_ppf(q):
    return torch.distributions.Normal(torch.tensor(0.0, dtype=torch.float64),
                                      torch.tensor(1.0, dtype=torch.float64)).icdf(
                                      torch.tensor(q, dtype=torch.float64)).item()

sys.stdout.reconfigure(line_buffering=True)

dl = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
pred = torch.load("st_gnn/baseline_preds.pt", weights_only=False)['pred'].astype(np.float64)

te_g = [g.clone() for g in dl if g.city == 'bangalore']
from torch_geometric.data import Batch
te = Batch.from_data_list(te_g)
y = te.y.numpy().ravel().astype(np.float64)
ints = np.asarray([float(g.rain_intensity) for g in te_g])
ints_all = np.repeat(ints, [g.x.shape[0] for g in te_g])

print(f"pred {pred.shape} vs y {y.shape} vs ints {ints_all.shape}")

INT = [150.0, 200.0, 250.0, 300.0]
ORACLE_P10 = {150.0: 0.64, 200.0: 0.545, 250.0: 0.535, 300.0: 0.515}
ORACLE_R2 = {150.0: 0.85, 200.0: 0.87, 250.0: 0.88, 300.0: 0.89}

def mae_req_for_p90():
    # P(|err|<=0.10)=0.90 under N(0,sigma): sigma = 0.10 / Phi^-1(0.95)
    sigma = 0.10 / norm_ppf(0.95)
    return sigma * math.sqrt(2 / math.pi), sigma

MAE_REQ, SIG_REQ = mae_req_for_p90()
print(f"\n=== GAUSSIAN REQUIREMENT FOR 90% WITHIN +-10cm ===")
print(f"required residual sigma = 0.10/Phi^-1(0.95) = {SIG_REQ:.4f} m")
print(f"required MAE = sigma*sqrt(2/pi) = {MAE_REQ:.4f} m")

print(f"\n{'I':>6} {'n':>7} {'std_y':>7} {'curMAE':>7} {'MAE/req':>7} {'curR2':>6} {'R2_req':>7} {'oracR2':>7} {'oracMAE':>8} {'oracP10':>7} {'gap':>6}")
for I in INT:
    m = ints_all == I
    yI, pI = y[m], pred[m]
    n = m.sum()
    std_y = yI.std()
    mae = np.abs(pI - yI).mean()
    r2 = 1 - np.sum((yI - pI)**2) / np.sum((yI - yI.mean())**2)
    r2_req = 1 - (SIG_REQ / std_y) ** 2
    # oracle-implied MAE from its +-10cm (Gaussian inversion)
    p_or = ORACLE_P10[I]
    sig_or = 0.10 / norm_ppf(0.5 * (1 + p_or))
    mae_or = sig_or * math.sqrt(2 / math.pi)
    gap = mae / mae_or
    print(f"{I:6.0f} {n:7d} {std_y:7.3f} {mae:7.3f} {mae/MAE_REQ:7.1f}x {r2:6.3f} {r2_req:7.3f} {ORACLE_R2[I]:7.2f} {mae_or:8.4f} {p_or*100:7.1f} {gap:6.1f}x")

# error-by-depth-bin on Bangalore (matches baseline report's true-depth bins)
print("\n=== BLR ERROR BY TRUE-DEPTH BIN ===")
bins = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 0.80), (0.80, 3.01)]
tot_abs = 0.0
for lo, hi in bins:
    m = (y >= lo) & (y < hi)
    n = m.sum()
    if n == 0:
        continue
    e = np.abs(pred[m] - y[m])
    bias = (pred[m] - y[m]).mean()
    tot_abs += e.sum()
    print(f"  y in [{lo:.2f},{hi:.2f}): n={n:7d} MAE={e.mean():.3f} bias={bias:+.3f} +-10cm={np.mean(e<=0.10)*100:5.1f}%")

print(f"  total abs error: {tot_abs:.0f} m")

# saturation at 3.0m (SWMM MaxDepth cap)
print("\n=== TARGET SATURATION (y clipped at 3.0 m by SWMM MaxDepth) ===")
for I in INT:
    m = ints_all == I
    yI = y[m]
    print(f"  I={I:6.0f}: n={m.sum():7d} frac(y>=2.95)={np.mean(yI>=2.95)*100:6.2f}% frac(y>=2.5)={np.mean(yI>=2.5)*100:6.2f}%")

# what +-10cm would perfect-intensity-mean baseline give (feature-lite baseline)
print("\n=== BASELINE CEILING CHECKS ===")
print("R2 of per-intensity-mean predictor on Bangalore:")
y_pred_mean = np.zeros_like(y)
for I in INT:
    m = ints_all == I
    y_pred_mean[m] = y[m].mean()
r2m = 1 - np.sum((y - y_pred_mean)**2) / np.sum((y - y.mean())**2)
print(f"  per-intensity mean: R2={r2m:.3f}  +-10cm={np.mean(np.abs(y-y_pred_mean)<=0.10)*100:.1f}%")
for I in INT:
    m = ints_all == I
    r2i = 1 - np.sum((y[m]-y_pred_mean[m])**2)/np.sum((y[m]-y[m].mean())**2)
    print(f"  I={I:6.0f}: R2={r2i:.3f} +-10cm={np.mean(np.abs(y[m]-y[m].mean())<=0.10)*100:.1f}%")

# implied R2 needed overall for 90% at each intensity and overall MAE budget
print(f"\n=== BUDGET: overall MAE needed for 90% +-10cm overall (Gaussian) ===")
print(f"overall required MAE = {MAE_REQ:.4f} m vs current {np.abs(pred-y).mean():.4f} m "
      f"({np.abs(pred-y).mean()/MAE_REQ:.1f}x reduction)")