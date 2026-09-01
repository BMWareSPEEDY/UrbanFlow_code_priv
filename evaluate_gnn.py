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


def eval_one_batch(model, batch, x_mean, x_std, e_mean, e_std, y_mean, y_std, device):
    batch = batch.to(device)
    x = (batch.x - x_mean) / x_std
    e = (batch.edge_attr - e_mean) / e_std
    with torch.no_grad():
        out = model(x, batch.edge_index, e).squeeze()
        preds = torch.clamp(out * y_std + y_mean, min=0.0).cpu().numpy()
    targets = batch.y.squeeze().cpu().numpy()
    return targets, preds


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    ckpt = torch.load("pinn_gnn_checkpoint.pt", map_location=device, weights_only=False)
    model = PerfectAccuracyGNN(in_channels=14, hidden_channels=256, out_channels=1).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    x_mean = ckpt['x_mean'].to(device)
    x_std = ckpt['x_std'].to(device)
    e_mean = ckpt['edge_attr_mean'].to(device)
    e_std = ckpt['edge_attr_std'].to(device)
    y_mean = float(ckpt['y_mean'])
    y_std = float(ckpt['y_std'])

    data_list = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
    if not isinstance(data_list, list):
        data_list = data_list.to_data_list()

    print(f"Evaluating {len(data_list)} scenario graphs...\n")

    rows = []
    for g in data_list:
        t, p = eval_one_batch(model, g, x_mean, x_std, e_mean, e_std, y_mean, y_std, device)
        d = np.abs(p - t)
        rows.append({
            'region': g.region,
            'intensity': g.rain_intensity,
            'nodes': len(t),
            'mae': float(np.mean(d)),
            'rmse': float(np.sqrt(np.mean(d ** 2))),
            'r2': float(r2_score(t, p)),
            'mean_target': float(np.mean(t)),
        })

    df = pd.DataFrame(rows)
    df['accuracy_pct'] = df['r2'] * 100

    print("=== BY REGION (all storms pooled) ===")
    by_region = df.groupby('region').agg(
        nodes=('nodes', 'sum'),
        mae=('mae', 'mean'),
        rmse=('rmse', 'mean'),
        r2=('r2', 'mean'),
    ).reset_index()
    by_region['accuracy_pct'] = by_region['r2'] * 100
    print(by_region.to_string(index=False))

    print("\n=== BY INTENSITY (all regions pooled) ===")
    by_int = df.groupby('intensity').agg(
        nodes=('nodes', 'sum'),
        mae=('mae', 'mean'),
        rmse=('rmse', 'mean'),
        r2=('r2', 'mean'),
    ).reset_index()
    by_int['accuracy_pct'] = by_int['r2'] * 100
    print(by_int.to_string(index=False))

    # Global pooled metrics
    all_t = []
    all_p = []
    for g in data_list:
        t, p = eval_one_batch(model, g, x_mean, x_std, e_mean, e_std, y_mean, y_std, device)
        all_t.append(t)
        all_p.append(p)
    all_t = np.concatenate(all_t)
    all_p = np.concatenate(all_p)
    d = np.abs(all_p - all_t)
    print("\n=== GLOBAL POOLED ===")
    print(f"Nodes: {len(all_t)}")
    print(f"MAE:  {np.mean(d):.4f} m")
    print(f"RMSE: {np.sqrt(np.mean(d**2)):.4f} m")
    print(f"R2:   {r2_score(all_t, all_p):.4f}")
    for tol in [0.02, 0.05, 0.10, 0.20]:
        print(f"  within ±{tol*100:.0f}cm: {(np.abs(d)<=tol).mean()*100:.2f}%")


if __name__ == "__main__":
    main()