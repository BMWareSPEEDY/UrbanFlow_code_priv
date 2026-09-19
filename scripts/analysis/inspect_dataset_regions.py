"""Check how many graphs per region are in the training dataset.
"""
import torch

dataset = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
print(f"Total graphs in dataset: {len(dataset)}")

region_counts = {}
for g in dataset:
    reg = getattr(g, 'region', 'unknown')
    region_counts[reg] = region_counts.get(reg, 0) + 1

for reg, cnt in sorted(region_counts.items(), key=lambda x: -x[1]):
    print(f"  {reg:<20s}: {cnt} graphs")
