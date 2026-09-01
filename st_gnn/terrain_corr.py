import torch
import numpy as np

dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_pyg_dataset.pt", weights_only=False)
cities = sorted({g.city for g in dl if g.city != 'bangalore'})
rows = []
for c in cities:
    for g in dl:
        if g.city == c and abs(g.rain_intensity - 300) < 1e-6:
            x, y = g.x.numpy(), g.y.numpy().ravel()
            rows.append((c, x[:, 17].mean(), y.mean(), (y >= 0.15).mean()))
rows.sort(key=lambda r: -r[1])
print(f"{'city':<13} {'elev2':>6} {'mean_y':>7} {'flood%':>6}")
for c, e2, my, fl in rows:
    print(f"{c:<13} {e2:6.2f} {my:7.3f} {fl*100:6.1f}")