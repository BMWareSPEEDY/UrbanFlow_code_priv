import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch
from train_perfect_accuracy_gnn import PerfectAccuracyGNN

sys.stdout.reconfigure(line_buffering=True)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / max(1e-6, ss_tot))


REGIONS = ['hsr', 'bellandur', 'whitefield', 'ecity', 'koramangala']

def run_leave_one_region_out():
    print("=================================================================")
    print("  LEAVE-ONE-REGION-OUT (LORO) ZERO-SHOT SPATIAL GENERALIZATION AUDIT")
    print("=================================================================\n", flush=True)

    dataset_list = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
    all_region_graphs = dataset_list if isinstance(dataset_list, list) else dataset_list.to_data_list()

    print(f"Total scenario-region graph objects: {len(all_region_graphs)}", flush=True)


    results = []

    for holdout_region in REGIONS:
        print(f"\n-----------------------------------------------------------------")
        print(f"  Holdout Region: '{holdout_region.upper()}' (Model never sees this catchment during training)")
        print(f"-----------------------------------------------------------------", flush=True)

        # Filter representative storm scenarios for fast LORO spatial cross-validation
        train_graphs = [g.clone() for g in all_region_graphs if getattr(g, 'region', '') != holdout_region and getattr(g, 'rain_intensity', 0.0) in [50.0, 150.0, 250.0]]
        val_graphs = [g.clone() for g in all_region_graphs if getattr(g, 'region', '') == holdout_region and getattr(g, 'rain_intensity', 0.0) == 150.0]

        train_batch = Batch.from_data_list(train_graphs)
        val_batch = Batch.from_data_list(val_graphs)

        # Standardize features based on Training Set statistics
        x_mean, x_std = train_batch.x.mean(dim=0), train_batch.x.std(dim=0) + 1e-6
        edge_attr_mean, edge_attr_std = train_batch.edge_attr.mean(dim=0), train_batch.edge_attr.std(dim=0) + 1e-6
        y_mean, y_std = train_batch.y.mean(), train_batch.y.std() + 1e-6

        train_batch.x = (train_batch.x - x_mean) / x_std
        train_batch.edge_attr = (train_batch.edge_attr - edge_attr_mean) / edge_attr_std
        train_y_norm = (train_batch.y - y_mean) / y_std

        val_batch.x = (val_batch.x - x_mean) / x_std
        val_batch.edge_attr = (val_batch.edge_attr - edge_attr_mean) / edge_attr_std

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PerfectAccuracyGNN(in_channels=12, hidden_channels=128, out_channels=1).to(device)
        
        train_batch = train_batch.to(device)
        train_y_norm = train_y_norm.to(device)
        val_batch = val_batch.to(device)

        print(f"  Training 40 epochs on {len(train_graphs)} graphs ({len(train_batch.y)} nodes)...", flush=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-4)

        for epoch in range(1, 41):
            model.train()
            optimizer.zero_grad()
            out_norm = model(train_batch.x, train_batch.edge_index, train_batch.edge_attr)
            
            mse_loss = F.mse_loss(out_norm, train_y_norm)
            huber_loss = F.huber_loss(out_norm, train_y_norm, delta=0.1)
            pred_meters = torch.clamp(out_norm * y_std + y_mean, min=0.0)
            depth_weights = 1.0 + torch.clamp(train_batch.y * 5.0, max=10.0)
            focal_l1 = torch.mean(depth_weights * torch.abs(pred_meters - train_batch.y))
            
            total_loss = mse_loss + huber_loss + (0.5 * focal_l1)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if epoch % 10 == 0 or epoch == 40:
                print(f"  [Epoch {epoch:02d}/40] Loss: {float(total_loss.detach().cpu()):.4f}", flush=True)




        # Evaluate Zero-Shot Transfer on holdout_region
        model.eval()
        with torch.no_grad():
            val_out_norm = model(val_batch.x, val_batch.edge_index, val_batch.edge_attr)
            val_preds = torch.clamp(val_out_norm * y_std + y_mean, min=0.0).cpu().numpy().squeeze()
            val_targets = val_batch.y.cpu().numpy().squeeze()

        diffs = np.abs(val_preds - val_targets)
        mae = float(np.mean(diffs))
        rmse = float(np.sqrt(np.mean(diffs ** 2)))
        r2 = float(r2_score(val_targets, val_preds))

        print(f"  [OK] Zero-Shot Result on '{holdout_region.upper()}':", flush=True)
        print(f"      - Nodes Evaluated : {len(val_targets)}", flush=True)
        print(f"      - MAE             : {mae:.4f} m ({mae*100:.2f} cm)", flush=True)
        print(f"      - RMSE            : {rmse:.4f} m ({rmse*100:.2f} cm)", flush=True)
        print(f"      - R² Accuracy     : {r2*100:.2f}% ({r2:.4f})\n", flush=True)



        results.append({
            'holdout_region': holdout_region,
            'nodes': len(val_targets),
            'mae_m': mae,
            'rmse_m': rmse,
            'r2_score': r2,
            'accuracy_pct': r2 * 100.0
        })

    df_res = pd.DataFrame(results)
    print("\n=================================================================")
    print("  SUMMARY: LEAVE-ONE-REGION-OUT (LORO) ZERO-SHOT GENERALIZATION")
    print("=================================================================")
    print(df_res.to_string(index=False))
    mean_r2 = df_res['r2_score'].mean()
    mean_mae = df_res['mae_m'].mean()
    print(f"\nAverage Zero-Shot R² Accuracy across all regional catchments: {mean_r2*100:.2f}%")
    print(f"Average Zero-Shot MAE across all regional catchments: {mean_mae*100:.2f} cm\n")

if __name__ == "__main__":
    run_leave_one_region_out()
