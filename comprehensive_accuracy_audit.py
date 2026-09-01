import os
import sys
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Batch
from train_perfect_accuracy_gnn import PerfectAccuracyGNN

sys.stdout.reconfigure(line_buffering=True)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / max(1e-6, ss_tot))

def audit():
    print("=================================================================")
    print("   URBANFLOW COMPREHENSIVE GNN vs SWMM ACCURACY AUDIT (12-FEAT)")
    print("=================================================================\n")
    
    dataset_path = "bengaluru_pyg_dataset.pt"
        
    data_or_list = torch.load(dataset_path, weights_only=False)
    if isinstance(data_or_list, list):
        data = Batch.from_data_list(data_or_list)
    else:
        data = data_or_list

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PerfectAccuracyGNN(in_channels=14, hidden_channels=256, out_channels=1)
    
    if os.path.exists("pinn_gnn_checkpoint.pt"):
        ckpt = torch.load("pinn_gnn_checkpoint.pt", map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        x_mean = ckpt['x_mean'].to(device)
        x_std = ckpt['x_std'].to(device)
        edge_attr_mean = ckpt['edge_attr_mean'].to(device)
        edge_attr_std = ckpt['edge_attr_std'].to(device)
        y_mean = float(ckpt['y_mean'])
        y_std = float(ckpt['y_std'])
    else:
        model.load_state_dict(torch.load("pinn_gnn_model.pth", map_location=device))
        x_mean = data.x.mean(dim=0).to(device)
        x_std = (data.x.std(dim=0) + 1e-6).to(device)
        edge_attr_mean = data.edge_attr.mean(dim=0).to(device)
        edge_attr_std = (data.edge_attr.std(dim=0) + 1e-6).to(device)
        y_mean = float(data.y.mean())
        y_std = float(data.y.std() + 1e-6)

    model.eval()
    model.to(device)
    data = data.to(device)

    x_norm = (data.x - x_mean) / x_std
    edge_norm = (data.edge_attr - edge_attr_mean) / edge_attr_std

    with torch.no_grad():
        out_norm = model(x_norm, data.edge_index, edge_norm).squeeze()
        preds = torch.clamp(out_norm * y_std + y_mean, min=0.0).cpu().numpy()
        
    targets = data.y.squeeze().cpu().numpy()

    diffs = np.abs(preds - targets)
    overall_mae = np.mean(diffs)
    overall_rmse = np.sqrt(np.mean(diffs ** 2))
    overall_r2 = r2_score(targets, preds)

    print(f"Overall Dataset Nodes: {len(targets)}")
    print(f"Target Depth Mean:  {np.mean(targets):.4f} m | Max: {np.max(targets):.4f} m")
    print(f"Pred Depth Mean:    {np.mean(preds):.4f} m | Max: {np.max(preds):.4f} m")
    print(f"OVERALL MAE:        {overall_mae:.4f} m ({overall_mae*100:.2f} cm)")
    print(f"OVERALL RMSE:       {overall_rmse:.4f} m ({overall_rmse*100:.2f} cm)")
    print(f"OVERALL R² SCORE:   {overall_r2:.4f} ({overall_r2*100:.2f}% variance explained)\n")

    # Depth Range Breakdown
    print("--- ACCURACY BY WATER DEPTH RANGE ---")
    ranges = [
        ("Shallow (< 0.05m)", targets < 0.05),
        ("Moderate (0.05m - 0.25m)", (targets >= 0.05) & (targets < 0.25)),
        ("Deep (0.25m - 0.75m)", (targets >= 0.25) & (targets < 0.75)),
        ("Severe Outfalls (>= 0.75m)", targets >= 0.75)
    ]
    for label, mask in ranges:
        if np.sum(mask) > 0:
            sub_mae = np.mean(diffs[mask])
            sub_r2 = r2_score(targets[mask], preds[mask])
            print(f"  * {label:26s}: Count={np.sum(mask):5d} | MAE={sub_mae:.4f}m ({sub_mae*100:.2f}cm) | R²={sub_r2:.4f}")

    # Top 10 Worst Node Discrepancies
    print("\n--- TOP 10 WORST NODE DISCREPANCIES ---")
    worst_idx = np.argsort(diffs)[-10:]
    for i in reversed(worst_idx):
        print(f"  Node Index {i:5d} | Target (SWMM): {targets[i]:.4f}m | Pred (GNN): {preds[i]:.4f}m | Error: {diffs[i]:.4f}m")

if __name__ == "__main__":
    audit()

