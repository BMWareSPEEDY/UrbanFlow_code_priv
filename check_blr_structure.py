import sys
import numpy as np
import torch
from torch_geometric.data import Batch

sys.stdout.reconfigure(line_buffering=True)

# Analyze Bangalore region-level depth structure: how much of test error is
# region-mean bias vs node-level scatter?

dl = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
blr = [g for g in dl if g.city == 'bangalore']
regions = sorted({g.region for g in blr})

# Per-region per-intensity target means
print("Per-region x intensity mean depth (Bangalore):")
print(f"{'region':<14}" + "".join(f"{i:>8.0f}" for i in [20, 50, 80, 120, 150, 200, 250, 300]))
for reg in regions:
    gs = [g for g in blr if g.region == reg]
    row = []
    for i in [20, 50, 80, 120, 150, 200, 250, 300]:
        g0 = next(g for g in gs if abs(g.rain_intensity - i) < 1e-6)
        row.append(g0.y.mean().item())
    print(f"{reg:<14}" + "".join(f"{v:>8.3f}" for v in row))

# Variance decomposition within Bangalore at fixed intensity 200:
print("\nWithin-region node depth std at I=200 (Bangalore):")
for reg in regions:
    g0 = next(g for g in blr if g.region == reg and abs(g.rain_intensity - 200) < 1e-6)
    print(f"  {reg:<12} std {g0.y.std().item():.3f} mean {g0.y.mean().item():.3f}")

# Same for a training city for comparison
print("\nWithin-region node depth std at I=200 (training cities):")
for c in ['hyderabad', 'chennai', 'pune', 'kolkata', 'ahmedabad']:
    g0 = next(g for g in dl if g.city == c and abs(g.rain_intensity - 200) < 1e-6)
    print(f"  {c:<12} std {g0.y.std().item():.3f} mean {g0.y.mean().item():.3f}")