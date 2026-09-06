"""Compare evaluate_v5_0.py vs app.py prediction logic on the exact same graph.
"""
import torch, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from app import REGION_CACHE, PRODUCTION_PREDICTOR
from train_hydro_gine_v5_0 import HydroGINE_v5

# 1. Prediction via app.py
r_data = REGION_CACHE['hongkong']
preds_app, raw_app, probs_app = PRODUCTION_PREDICTOR.predict(r_data['pyg_data'], 50.0, 60.0)

# 2. Prediction via evaluate_v5_0.py
dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
hk_graph = [g for g in dl if getattr(g, 'region', '') == 'hongkong' and abs(g.rain_intensity - 50.0) < 1.0][0]
preds_eval, raw_eval, probs_eval = PRODUCTION_PREDICTOR.predict(hk_graph, 50.0, 60.0)

y_true = hk_graph.y.cpu().numpy().ravel()

# Check difference between preds_app and preds_eval
max_diff = np.max(np.abs(preds_app - preds_eval))
mean_diff = np.mean(np.abs(preds_app - preds_eval))

print(f"Max diff between app.py and evaluate_v5_0.py: {max_diff:.6f} m")
print(f"Mean diff between app.py and evaluate_v5_0.py: {mean_diff:.6f} m")

# Check accuracy under evaluate_v5_0 definition:
# Hazard is depth >= 0.15m
crit_swmm = (y_true >= 0.30)
crit_app = (preds_app >= 0.30)
crit_eval = (preds_eval >= 0.30)

print(f"\nCRITICAL (>= 0.30m):")
print(f"  SWMM True Critical: {np.sum(crit_swmm)}")
print(f"  App.py   Critical : {np.sum(crit_app)} | TP={np.sum(crit_swmm & crit_app)} | FP={np.sum(~crit_swmm & crit_app)}")
print(f"  Eval.py  Critical : {np.sum(crit_eval)} | TP={np.sum(crit_swmm & crit_eval)} | FP={np.sum(~crit_swmm & crit_eval)}")

# Check <= 15cm diff (tolerance = 0.15m)
risk_mask = (preds_app > 0.08) | (y_true > 0.08)
diff_app = np.abs(preds_app - y_true)
diff_eval = np.abs(preds_eval - y_true)

print(f"\nRISK NODES (p > 0.08 or s > 0.08): {np.sum(risk_mask)}")
print(f"  App.py   <= 15cm diff: {np.sum(diff_app[risk_mask] <= 0.15)} / {np.sum(risk_mask)} ({np.mean(diff_app[risk_mask] <= 0.15)*100:.1f}%)")
print(f"  Eval.py  <= 15cm diff: {np.sum(diff_eval[risk_mask] <= 0.15)} / {np.sum(risk_mask)} ({np.mean(diff_eval[risk_mask] <= 0.15)*100:.1f}%)")
print(f"  App.py   <= 30cm diff: {np.sum(diff_app[risk_mask] <= 0.30)} / {np.sum(risk_mask)} ({np.mean(diff_app[risk_mask] <= 0.30)*100:.1f}%)")
