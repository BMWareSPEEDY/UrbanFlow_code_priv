import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GATv2Conv

class PhysicsInformedGNN(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=1, dropout_rate=0.08):
        super(PhysicsInformedGNN, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, concat=True, edge_dim=2)
        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=2, concat=False, edge_dim=2)

        self.dropout_rate = dropout_rate
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, force_dropout=False):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        p = self.dropout_rate if (self.training or force_dropout) else 0.0
        x = F.dropout(x, p=p, training=True)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)

        out = self.regressor(x)
        return out

def compute_saint_venant_pinn_loss(pred, target, data, lambda_mass=0.05, lambda_momentum=0.02, lambda_phys=0.2):
    """
    Computes numerically stable Saint-Venant 1D Shallow Water Physics Penalties:
    1. MSE Data Loss vs SWMM targets
    2. Continuity Equation (Conservation of Mass)
    3. Energy / Momentum Conservation (Manning Friction Slope S_f)
    4. Physical Non-Negativity Penalty
    5. Junction Surcharge Saturation Constraint
    """
    mse_loss = F.mse_loss(pred, target)
    non_neg_penalty = torch.mean(F.relu(-pred) ** 2)
    surcharge_penalty = torch.mean(F.relu(pred - 3.0) ** 2)

    edge_src, edge_dst = data.edge_index[0], data.edge_index[1]
    pos_depth_src = torch.clamp(pred[edge_src], min=1e-3)
    pos_depth_dst = torch.clamp(pred[edge_dst], min=1e-3)
    
    grade = data.edge_attr[:, 1:2]
    length = torch.clamp(data.edge_attr[:, 0:1], min=1.0)
    
    friction_n = torch.clamp(data.x[edge_src, 4:5], min=0.01)
    slope = torch.clamp(torch.abs(grade), min=1e-4)
    velocity = (1.0 / friction_n) * torch.sqrt(slope)
    flow_q = pos_depth_src * velocity
    
    node_inflow = torch.zeros_like(pred)
    node_outflow = torch.zeros_like(pred)
    node_inflow.index_add_(0, edge_dst, flow_q)
    node_outflow.index_add_(0, edge_src, flow_q)
    
    mass_conservation_loss = torch.mean(torch.clamp((node_inflow - node_outflow) ** 2, max=50.0))

    friction_slope_sf = (friction_n ** 2) * (velocity ** 2) / (pos_depth_src ** (4.0 / 3.0))
    energy_head_residual = (pos_depth_dst - pos_depth_src) + (grade * length) - (torch.clamp(friction_slope_sf, max=5.0) * length)
    momentum_conservation_loss = torch.mean(torch.clamp(energy_head_residual ** 2, max=50.0))

    total_loss = (
        mse_loss + 
        (lambda_phys * non_neg_penalty) + 
        (0.05 * surcharge_penalty) + 
        (lambda_mass * mass_conservation_loss) + 
        (lambda_momentum * momentum_conservation_loss)
    )

    return total_loss, mse_loss, mass_conservation_loss, momentum_conservation_loss

def active_uncertainty_sampling(model, data, num_mc_samples=20):
    model.eval()
    mc_preds = []
    
    x_norm = (data.x - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-6)
    edge_norm = (data.edge_attr - data.edge_attr.mean(dim=0)) / (data.edge_attr.std(dim=0) + 1e-6)
    
    with torch.no_grad():
        for _ in range(num_mc_samples):
            pred_sample = model(x_norm, data.edge_index, edge_norm, force_dropout=True)
            mc_preds.append(pred_sample)
            
    mc_tensor = torch.stack(mc_preds, dim=0)
    node_variance = torch.var(mc_tensor, dim=0).squeeze()
    
    return node_variance

def train_active_pinn():
    print("1. Loading combined multi-region PyTorch Geometric dataset...")
    data = torch.load("bengaluru_pyg_dataset.pt", weights_only=False)

    x_mean, x_std = data.x.mean(dim=0), data.x.std(dim=0)
    data.x = (data.x - x_mean) / (x_std + 1e-6)

    edge_attr_mean, edge_attr_std = data.edge_attr.mean(dim=0), data.edge_attr.std(dim=0)
    data.edge_attr = (data.edge_attr - edge_attr_mean) / (edge_attr_std + 1e-6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware compute device: {device}")

    model = PhysicsInformedGNN(in_channels=5, hidden_channels=64, out_channels=1).to(device)
    data = data.to(device)

    print("\n2. Executing Physics-Informed Active Sampling (Monte Carlo Dropout Uncertainty)...")
    uncertainty_scores = active_uncertainty_sampling(model, data, num_mc_samples=20)
    high_uncertainty_indices = torch.topk(uncertainty_scores, k=int(0.20 * len(data.x))).indices
    print(f"   - Identified top {len(high_uncertainty_indices)} high-variance nodes for active targeted physics weighting.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=600, eta_min=1e-5)

    print("\n3. Starting Saint-Venant Physics-Informed GNN Training Loop...")
    model.train()

    for epoch in range(1, 601):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        
        total_loss, mse, mass_loss, mom_loss = compute_saint_venant_pinn_loss(out, data.y, data)
        
        uncertainty_boost = torch.mean((out[high_uncertainty_indices] - data.y[high_uncertainty_indices]) ** 2)
        final_loss = total_loss + 0.2 * uncertainty_boost

        final_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Total PINN Loss: {final_loss.item():.6f} | MSE: {mse.item():.6f} | Mass Loss: {mass_loss.item():.6f} | Mom Loss: {mom_loss.item():.6f}")

    torch.save(model.state_dict(), "pinn_gnn_model.pth")
    print("\nSaint-Venant Physics-Informed GNN training complete! Saved to 'pinn_gnn_model.pth'.")

if __name__ == "__main__":
    train_active_pinn()
