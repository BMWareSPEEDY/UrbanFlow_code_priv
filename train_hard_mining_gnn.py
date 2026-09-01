import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATv2Conv

sys.stdout.reconfigure(line_buffering=True)


class ZeroToleranceHurdleGNN(nn.Module):
    """
    Decoupled Zero-Tolerance Graph Neural Network with Multi-Scale GATv2 Attention.
    """
    def __init__(self, in_channels=14, hidden_channels=128, out_channels=1):
        super(ZeroToleranceHurdleGNN, self).__init__()
        
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, concat=True, edge_dim=2)
        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=2, concat=True, edge_dim=2)
        self.conv3 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=2, concat=False, edge_dim=2)
        self.conv4 = GATv2Conv(hidden_channels, hidden_channels, heads=1, concat=False, edge_dim=2)

        self.ln1 = nn.LayerNorm(hidden_channels * 4)
        self.ln2 = nn.LayerNorm(hidden_channels * 2)
        self.ln3 = nn.LayerNorm(hidden_channels)
        self.ln4 = nn.LayerNorm(hidden_channels)

        fusion_dim = (hidden_channels * 2) + in_channels

        # Head 1: Flood Initiation Classification Gate (Wet vs Dry)
        self.gate_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )

        # Head 2: Pure Physical Depth Magnitude Regressor (Conditioned on Flooding)
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, return_gate=False):
        x_in = x
        h1 = F.elu(self.ln1(self.conv1(x, edge_index, edge_attr)))
        h2 = F.elu(self.ln2(self.conv2(h1, edge_index, edge_attr)))
        h3 = F.elu(self.ln3(self.conv3(h2, edge_index, edge_attr)))
        h4 = F.elu(self.ln4(self.conv4(h3, edge_index, edge_attr)))

        fusion = torch.cat([h4, h3, x_in], dim=-1)
        
        gate_logit = self.gate_head(fusion)
        reg_out = self.reg_head(fusion)
        
        if return_gate:
            return reg_out, gate_logit
        
        gate_prob = torch.sigmoid(gate_logit)
        gated_out = torch.where(gate_prob >= 0.35, reg_out, torch.zeros_like(reg_out))
        return gated_out


def train_hard_mining():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}", flush=True)

    data_file = "bengaluru_pyg_dataset.pt"
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!", flush=True)
        return

    print("Loading PyG dataset...", flush=True)
    g = torch.load(data_file, weights_only=False).to(device)

    x_mean = g.x[:, :14].mean(dim=0)
    x_std = g.x[:, :14].std(dim=0).clamp(min=1e-5)
    e_mean = g.edge_attr.mean(dim=0)
    e_std = g.edge_attr.std(dim=0).clamp(min=1e-5)
    y_mean = g.y.mean()
    y_std = g.y.std().clamp(min=1e-5)

    x_norm = (g.x[:, :14] - x_mean) / x_std
    e_norm = (g.edge_attr - e_mean) / e_std
    targets = g.y.view(-1, 1)

    wet_mask = (targets >= 0.02).float()
    dry_mask = (targets < 0.02).float()

    model = ZeroToleranceHurdleGNN(in_channels=14, hidden_channels=128, out_channels=1).to(device)

    # Warm-start
    if os.path.exists("zero_tolerance_gnn_checkpoint.pt"):
        try:
            ckpt = torch.load("zero_tolerance_gnn_checkpoint.pt", weights_only=False)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"Warm-started {len(pretrained_dict)} layers from existing checkpoint.", flush=True)
        except Exception as e:
            print(f"Warm-start skipped: {e}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-6)

    print("\nStarting Multi-Iteration Hard-Sample Residual Boosting (400 Epochs)...", flush=True)
    t_start = time.time()

    best_gt_30 = float('inf')
    best_mae = float('inf')

    for ep in range(1, 401):
        model.train()
        optimizer.zero_grad()

        reg_out, gate_logit = model(x_norm, g.edge_index, e_norm, return_gate=True)
        preds_depth = torch.clamp(reg_out * y_std + y_mean, min=0.0)

        err_m = torch.abs(preds_depth - targets)

        # Quadratic focal hard-mining weighting (supercharges gradient on error > 30cm)
        hard_focal_weight = 1.0 + 25.0 * torch.clamp(err_m / 0.30, min=0.0, max=3.0) ** 2
        # Asymmetric under-prediction penalty
        under_pred_penalty = torch.where(preds_depth < targets, 2.5, 1.0)
        # Wet depth loss
        wet_depth_loss = (wet_mask * hard_focal_weight * under_pred_penalty * F.smooth_l1_loss(preds_depth, targets, reduction='none', beta=0.03)).sum() / (wet_mask.sum() + 1e-6)
        
        # Heavy dry penalty to suppress false positive surcharges
        dry_loss = (dry_mask * 20.0 * torch.square(torch.clamp(preds_depth - 0.02, min=0.0))).sum() / (dry_mask.sum() + 1e-6)
        # Gate BCE loss
        gate_loss = F.binary_cross_entropy_with_logits(gate_logit, wet_mask)

        total_loss = wet_depth_loss + 2.5 * gate_loss + dry_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if ep % 50 == 0 or ep == 400:
            model.eval()
            with torch.no_grad():
                r_eval, g_eval = model(x_norm, g.edge_index, e_norm, return_gate=True)
                p_prob = torch.sigmoid(g_eval).cpu().numpy().ravel()
                p_depth = torch.clamp(r_eval * y_std + y_mean, min=0.0).cpu().numpy().ravel()
                p_final = np.where(p_prob >= 0.35, p_depth, 0.0)
                t_eval = targets.cpu().numpy().ravel()

                diffs_cm = np.abs(p_final - t_eval) * 100.0
                mae = np.mean(diffs_cm)
                median_err = np.median(diffs_cm)
                le_5 = np.sum(diffs_cm <= 5.0) / len(diffs_cm) * 100.0
                le_20 = np.sum(diffs_cm <= 20.0) / len(diffs_cm) * 100.0
                le_30 = np.sum(diffs_cm <= 30.0) / len(diffs_cm) * 100.0
                gt_30 = np.sum(diffs_cm > 30.0) / len(diffs_cm) * 100.0
                gt_30_count = int(np.sum(diffs_cm > 30.0))

                print(f"Epoch {ep:3d} | Loss: {total_loss.item():.4f} | MAE: {mae:.2f}cm | <=30cm: {le_30:.1f}% | >30cm: {gt_30:.1f}% ({gt_30_count:,}) | <=5cm: {le_5:.1f}% | <=20cm: {le_20:.1f}%", flush=True)

                if gt_30 < best_gt_30 or mae < best_mae:
                    best_gt_30 = gt_30
                    best_mae = mae
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_type': 'ZeroToleranceHurdleGNN',
                        'x_mean': x_mean.cpu(),
                        'x_std': x_std.cpu(),
                        'edge_attr_mean': e_mean.cpu(),
                        'edge_attr_std': e_std.cpu(),
                        'y_mean': y_mean.cpu(),
                        'y_std': y_std.cpu(),
                        'use_log1p': False,
                        'best_mae': mae,
                        'best_gt_30': best_gt_30
                    }, "zero_tolerance_gnn_checkpoint.pt")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_type': 'ZeroToleranceHurdleGNN',
                        'x_mean': x_mean.cpu(),
                        'x_std': x_std.cpu(),
                        'edge_attr_mean': e_mean.cpu(),
                        'edge_attr_std': e_std.cpu(),
                        'y_mean': y_mean.cpu(),
                        'y_std': y_std.cpu(),
                        'use_log1p': False,
                        'best_mae': mae,
                        'best_gt_30': best_gt_30
                    }, "pinn_gnn_checkpoint.pt")

    elapsed = time.time() - t_start
    print(f"\nTraining completed in {elapsed / 60.0:.2f} minutes!", flush=True)
    print(f"Best >30cm Error Rate: {best_gt_30:.2f}% | Best MAE: {best_mae:.2f} cm", flush=True)
    print("Checkpoint saved to 'zero_tolerance_gnn_checkpoint.pt' and 'pinn_gnn_checkpoint.pt'!", flush=True)


if __name__ == "__main__":
    train_hard_mining()
