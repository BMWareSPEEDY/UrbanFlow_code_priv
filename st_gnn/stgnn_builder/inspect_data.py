import sys
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_pyg_dataset.pt", weights_only=False)
print(f"total graphs: {len(dl)}")
print(f"feature dim: {dl[0].x.shape[1]}, edge_attr dim: {dl[0].edge_attr.shape[1]}")
print(f"y shape: {dl[0].y.shape}")

cities = {}
for i, g in enumerate(dl):
    cities.setdefault(g.city, []).append((i, g))

print(f"cities: {len(cities)}")
for c, gl in cities.items():
    ints = sorted({g.rain_intensity for _, g in gl})
    regions = {g.region for _, g in gl}
    n_nodes = [g.x.shape[0] for _, g in gl]
    print(f"  {c}: n_graphs={len(gl)} regions={sorted(regions)} ints={ints} nodes_per_graph={sorted(set(n_nodes))}")

g0 = dl[0]
print("\nfeature column names (guessing by stats):")
x = g0.x.numpy()
for j in range(x.shape[1]):
    print(f"  col {j}: min {x[:, j].min():.3f} max {x[:, j].max():.3f} mean {x[:, j].mean():.3f}")
print("\nedge_attr col0 (length_m):", g0.edge_attr[:, 0].numpy()[:5], "col1 (grade):", g0.edge_attr[:, 1].numpy()[:5])
print("y:", g0.y.numpy()[:10].ravel())

# check contiguity of a region's 8 graphs and their intensity order
bl = [i for i, g in enumerate(dl) if g.city == 'bangalore']
print("\nbangalore graph indices in dataset:", bl)
for i in bl:
    print(f"  idx {i}: region={dl[i].region} int={dl[i].rain_intensity} nodes={dl[i].x.shape[0]} edges={dl[i].edge_index.shape[1]}")
