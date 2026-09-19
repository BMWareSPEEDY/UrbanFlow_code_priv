"""Evaluate historical zero_tolerance_gnn_checkpoint.pt on HSR and other catchments.
"""
import torch, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_historical():
    ckpt = torch.load("zero_tolerance_gnn_checkpoint.pt", map_location=device, weights_only=False)
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    print(f"Model type: {ckpt.get('model_type')}")
    print(f"Hidden channels: {ckpt.get('hidden_channels')}")
    print(f"Features: {ckpt['x_mean'].shape}")
    
if __name__ == '__main__':
    evaluate_historical()
