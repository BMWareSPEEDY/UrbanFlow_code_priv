"""Check true SWMM ground truth in expanded_master_physics_dataset.pt.
"""
import torch

dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)

print(f"Total graphs in dataset: {len(dl)}")
# Check a few graphs from different cities
for r in ['hsr', 'tokyo', 'hongkong', 'singapore', 'london', 'paris', 'nyc']:
    matching = [g for g in dl if (getattr(g, 'region', '') == r or getattr(g, 'city', '') == r) and abs(g.rain_intensity - 50.0) < 1.0]
    if matching:
        g = matching[0]
        y = g.y.cpu().numpy().ravel()
        print(f"City: {r:<12s} | Rain: {g.rain_intensity} mm/hr | Nodes: {len(y):6d} | Max SWMM: {y.max():.4f}m | Mean SWMM: {y.mean():.4f}m | Flooded (>=0.15m): {sum(y >= 0.15)}")
