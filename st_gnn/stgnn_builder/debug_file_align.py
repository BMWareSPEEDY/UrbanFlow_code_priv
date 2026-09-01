import sys
import os
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"
OUTDIR = BASE + r"\st_gnn\stgnn_builder"


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / max(1e-6, np.sum((y - np.mean(y)) ** 2))


dl = torch.load(BASE + r"\multi_scenario_pyg_dataset.pt", weights_only=False)
te = [g for g in dl if g.city == 'bangalore']
y_critic = np.concatenate([g.y.numpy().ravel() for g in te])
inten = np.concatenate([np.full(g.y.shape[0], float(g.rain_intensity)) for g in te])

p_file = np.clip(np.asarray(torch.load(OUTDIR + r"\smoke_convgru_preds.pt", weights_only=False)['pred'], float).ravel(), 0, None)
print("file stats: mean", p_file.mean(), "std", p_file.std(), "median", np.median(p_file), "zeros%", (p_file == 0).mean() * 100)

# critic-aligned
mae = np.mean(np.abs(p_file - y_critic))
print(f"MAE(p_file, y_critic) = {mae:.4f}  R2 = {r2(y_critic, p_file):.4f}")

# per-intensity, correct masks
for i in [20.0, 50.0, 80.0, 120.0, 150.0, 200.0, 250.0, 300.0]:
    m = inten == i
    print(f"  I={i:6.1f}: tgt_mean {y_critic[m].mean():.4f} pred_mean {p_file[m].mean():.4f} MAE {np.mean(np.abs(p_file[m]-y_critic[m])):.4f}")

# where are the nonzeros in the file?
nz = np.where(p_file > 0.001)[0]
print("\nnonzero count:", len(nz))
if len(nz):
    print("nonzero position range:", nz.min(), nz.max())
    print("intensity at nonzero positions:", np.unique(inten[nz])[:20])

# mapping check: te order positions per region/intensity
print("\nte graph order (region, I, node offset):")
off = 0
for gi, g in enumerate(te):
    print(f"  te[{gi}] region={g.region:12s} I={g.rain_intensity:6.1f} nodes={g.y.shape[0]:5d} offset={off}")
    off += g.y.shape[0]
