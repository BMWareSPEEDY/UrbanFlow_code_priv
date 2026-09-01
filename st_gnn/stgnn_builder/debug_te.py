import sys
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"
OUTDIR = os.path.join(BASE, "st_gnn", "stgnn_builder") if False else BASE + r"\st_gnn\stgnn_builder"
import os

dl = torch.load(BASE + r"\multi_scenario_pyg_dataset.pt", weights_only=False)
te = [g for g in dl if g.city == 'bangalore']
tr = [g for g in dl if g.city != 'bangalore']
y_te = np.concatenate([g.y.numpy().ravel() for g in te])
y_tr = np.concatenate([g.y.numpy().ravel() for g in tr])
print("te y: mean", y_te.mean(), "std", y_te.std(), "median", np.median(y_te))
print("tr y: mean", y_tr.mean(), "std", y_tr.std())
for i in [20, 50, 80, 120, 150, 200, 250, 300]:
    m = np.concatenate([g.y.numpy().ravel() for g in te if g.rain_intensity == i])
    print(f"  te I={i}: n={len(m)} mean={m.mean():.4f} std={m.std():.4f} p90={np.percentile(m,90):.3f} max={m.max():.3f}")

ck = torch.load(OUTDIR + r"\smoke_convgru_preds.pt", weights_only=False)
p = np.clip(np.asarray(ck['pred'], float).ravel(), 0, None)
print("\npred: mean", p.mean(), "std", p.std(), "median", np.median(p), "shape", p.shape)
d = p - y_te
print("resid: mean", d.mean(), "std", d.std())
for i in [20, 50, 80, 120, 150, 200, 250, 300]:
    mm = np.concatenate([np.full(g.y.shape[0], i) for g in te if g.rain_intensity == i]) if False else np.concatenate([np.full(g.y.shape[0], i) for g in te if g.rain_intensity == i])
    m = np.concatenate([g.y.numpy().ravel() for g in te if g.rain_intensity == i])
    n = len(m)
    idx = np.where(np.concatenate([np.full(g.y.shape[0], i) for g in te if g.rain_intensity == i]) == i)[0]
    seg = p[idx]
    print(f"  te I={i}: pred mean={seg.mean():.4f} pred std={seg.std():.4f} | tgt mean={m.mean():.4f} | MAE={np.mean(np.abs(seg-m)):.4f} R2={1-np.sum((seg-m)**2)/max(1e-6,np.sum((m-m.mean())**2)):.4f}")
print("\ncorr(pred,y):", np.corrcoef(p, y_te)[0, 1])
oi = np.argsort(-np.abs(d))[:10]
for k in oi:
    print(f"  idx {k}: true {y_te[k]:.3f} pred {p[k]:.3f}")

# correlation check: would pred track y even if misaligned? compare block shuffles
