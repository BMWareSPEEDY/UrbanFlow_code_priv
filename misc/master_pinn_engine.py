import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv

class MasterPhysicsGNN(nn.Module):
    """
    State-of-the-Art Deep Graph Attention Network (GATv2) with Residual Physics Blocks
    incorporating 8 hydraulic features and multi-head attention.
    """
    def __init__(self, in_channels=8, hidden_channels=128, out_channels=1):
        super(MasterPhysicsGNN, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, concat=True, edge_dim=2)
        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=2, concat=True, edge_dim=2)
        self.conv3 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=2, concat=False, edge_dim=2)
        self.conv4 = GATv2Conv(hidden_channels, hidden_channels, heads=1, concat=False, edge_dim=2)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels + in_channels, hidden_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        x_in = x
        x1 = F.elu(self.conv1(x, edge_index, edge_attr))
        x2 = F.elu(self.conv2(x1, edge_index, edge_attr))
        x3 = F.elu(self.conv3(x2, edge_index, edge_attr))
        x4 = F.elu(self.conv4(x3, edge_index, edge_attr))

        # Residual skip connection from raw spatial & hydraulic features
        x_cat = torch.cat([x4, x_in], dim=-1)
        out = F.relu(self.regressor(x_cat))
        return out

def compute_master_physics_loss(pred, target, data, g=9.81):
    """
    Uncompromising Comprehensive Physics Loss Engine:
    1. Huber & MSE Hydrodynamic Data Loss
    2. Saint-Venant 1D/2D Mass Conservation (Continuity Equation)
    3. Saint-Venant Momentum Conservation (Manning Friction Slope S_f)
    4. Bernoulli Energy Head Equation & Hydraulic Gradient Loss
    5. Non-negativity & Junction Surcharge Rim Saturation Penalties
    """
    # 1. Data Loss (Huber + MSE)
    huber_loss = F.huber_loss(pred, target, delta=0.10)
    mse_loss = F.mse_loss(pred, target)
    
    # Outfall Focal Weighting for high accumulation zones
    high_depth_mask = (target > 0.4).float()
    focal_loss = torch.mean(high_depth_mask * (pred - target) ** 2)

    # 2. Physical Non-Negativity & Rim Surcharge Saturation Penalties
    non_neg_penalty = torch.mean(F.relu(-pred) ** 2)
    surcharge_penalty = torch.mean(F.relu(pred - 3.0) ** 2)

    # Extract spatial edge dimensions
    edge_src, edge_dst = data.edge_index[0], data.edge_index[1]
    pos_depth_src = torch.clamp(pred[edge_src], min=1e-3)
    pos_depth_dst = torch.clamp(pred[edge_dst], min=1e-3)
    
    grade = data.edge_attr[:, 1:2]
    length = torch.clamp(data.edge_attr[:, 0:1], min=1.0)
    elev_src = data.x[edge_src, 2:3]
    elev_dst = data.x[edge_dst, 2:3]
    friction_n = torch.clamp(data.x[edge_src, 4:5], min=0.01)
    
    slope = torch.clamp(torch.abs(grade), min=1e-4)
    velocity = (1.0 / friction_n) * torch.sqrt(slope)
    flow_q = pos_depth_src * velocity

    # 3. Saint-Venant Mass Conservation (Continuity Equation)
    node_inflow = torch.zeros_like(pred)
    node_outflow = torch.zeros_like(pred)
    node_inflow.index_add_(0, edge_dst, flow_q)
    node_outflow.index_add_(0, edge_src, flow_q)
    mass_conservation_loss = torch.mean(torch.clamp((node_inflow - node_outflow) ** 2, max=25.0))

    # 4. Saint-Venant Momentum & Friction Slope Loss (S_f)
    friction_slope_sf = (friction_n ** 2) * (velocity ** 2) / (pos_depth_src ** (4.0 / 3.0))
    momentum_residual = (pos_depth_dst - pos_depth_src) + (grade * length) - (torch.clamp(friction_slope_sf, max=2.0) * length)
    momentum_conservation_loss = torch.mean(torch.clamp(momentum_residual ** 2, max=25.0))

    # 5. Bernoulli Energy Balance & Hydrostatic Pressure Head Equation
    energy_head_src = elev_src + pos_depth_src + ((velocity ** 2) / (2.0 * g))
    energy_head_dst = elev_dst + pos_depth_dst + ((velocity ** 2) / (2.0 * g))
    head_loss_friction = torch.clamp(friction_slope_sf, max=2.0) * length
    energy_residual = energy_head_src - energy_head_dst - head_loss_friction
    bernoulli_energy_loss = torch.mean(torch.clamp(energy_residual ** 2, max=25.0))

    # Total Master Uncompromising Physics Loss
    total_loss = (
        huber_loss +
        (0.4 * mse_loss) +
        (1.0 * focal_loss) +
        (0.2 * non_neg_penalty) +
        (0.05 * surcharge_penalty) +
        (0.08 * mass_conservation_loss) +
        (0.04 * momentum_conservation_loss) +
        (0.04 * bernoulli_energy_loss)
    )

    return total_loss, mse_loss, mass_conservation_loss, momentum_conservation_loss, bernoulli_energy_loss

def train_master_model():
    print("1. Loading 8-feature PyG dataset...", flush=True)
    data = torch.load("bengaluru_pyg_dataset.pt", weights_only=False)

    x_mean, x_std = data.x.mean(dim=0), data.x.std(dim=0)
    data.x = (data.x - x_mean) / (x_std + 1e-6)

    edge_attr_mean, edge_attr_std = data.edge_attr.mean(dim=0), data.edge_attr.std(dim=0)
    data.edge_attr = (data.edge_attr - edge_attr_mean) / (edge_attr_std + 1e-6)

    device = torch.device('cpu') # Rock solid CPU execution
    print(f"Hardware compute device: {device}", flush=True)

    model = MasterPhysicsGNN(in_channels=8, hidden_channels=128, out_channels=1).to(device)
    data = data.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800, eta_min=1e-5)

    print("\n2. Training Master Physics PINN (Saint-Venant + Bernoulli Equations, 800 Epochs)...", flush=True)
    model.train()

    best_mse = float('inf')

    for epoch in range(1, 801):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        
        total_loss, mse, mass_loss, mom_loss, energy_loss = compute_master_physics_loss(out, data.y, data)

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        mse_val = float(mse.detach().numpy())
        if mse_val < best_mse:
            best_mse = mse_val
            torch.save(model.state_dict(), "pinn_gnn_model.pth")

        if epoch % 50 == 0 or epoch == 1:
            tot_val = float(total_loss.detach().numpy())
            mass_val = float(mass_loss.detach().numpy())
            nrg_val = float(energy_loss.detach().numpy())
            print(f"Epoch {epoch:04d} | Total Loss: {tot_val:.6f} | MSE: {mse_val:.6f} | Mass: {mass_val:.6f} | Energy: {nrg_val:.6f} | Best MSE: {best_mse:.6f}", flush=True)

    print(f"\nMaster PINN Model Training Complete! Best MSE: {best_mse:.6f}. Saved to 'pinn_gnn_model.pth'.", flush=True)

if __name__ == "__main__":
    train_master_model()
