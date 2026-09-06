"""Compare node features in app.py pyg_data vs dataset graph.
"""
import torch, numpy as np
from app import REGION_CACHE

dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
hk_graph = [g for g in dl if getattr(g, 'region', '') == 'hongkong' and abs(g.rain_intensity - 50.0) < 1.0][0]

r_data = REGION_CACHE['hongkong']
pyg_app = r_data['pyg_data']

x_app = pyg_app.x.cpu().numpy()
x_ds = hk_graph.x.cpu().numpy()

print(f"Shape app: {x_app.shape} | Shape dataset: {x_ds.shape}")
feat_diff = np.max(np.abs(x_app - x_ds), axis=0)

for f_idx, d in enumerate(feat_diff):
    mean_app = np.mean(x_app[:, f_idx])
    mean_ds = np.mean(x_ds[:, f_idx])
    print(f"Feat {f_idx:2d}: Max Diff = {d:10.4f} | Mean App = {mean_app:10.4f} | Mean DS = {mean_ds:10.4f}")
