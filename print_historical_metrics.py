"""Print historical performance from checkpoint.
"""
import torch

ckpt = torch.load("zero_tolerance_gnn_checkpoint.pt", map_location='cpu', weights_only=False)
for k in ['best_tc_50', 'best_fc_50', 'best_tc_120', 'best_fc_120', 'best_mae_50', 'best_mae_120']:
    print(f"{k}: {ckpt.get(k)}")
