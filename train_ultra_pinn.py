import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATv2Conv

class UltraPhysicsGNN(nn.Module):
    def __init__(self, in_channels=8, hidden_channels=128, out_channels=1):
        super(UltraPhysicsGNN, self).__init__()
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

        # Skip connection from input features (elevation, accumulation, sink)
        x_cat = torch.cat([x4, x_in], dim=-1)
        out = F.relu(self.regressor(x_cat))
        return out

def ultra_pinn_loss(pred, target, data):
    # 1. Huber Loss (Smooth L1 Loss - less sensitive to extreme outfall outliers)
    huber_loss = F.huber_loss(pred, target, delta=0.15)
    
    # 2. MSE Loss
    mse_loss = F.mse_loss(pred, target)
    
    # 3. Outfall High-Depth Focal Penalty (weight high depth sink nodes more)
    high_depth_mask = (target > 0.5).float()
    focal_depth_loss = torch.mean(high_depth_mask * (pred - target) ** 2)
    
    # 4. Non-negativity penalty
    non_neg_penalty = torch.mean(F.relu(-pred) ** 2)

    total_loss = huber_loss + (0.5 * mse_loss) + (1.2 * focal_depth_loss) + (0.2 * non_neg_penalty)
    return total_loss, mse_loss, focal_depth_loss

def train_ultra_model():
    print("1. Loading enhanced 8-feature PyG dataset...")
    data = torch.load("bengaluru_pyg_dataset.pt", weights_only=False)

    x_mean, x_std = data.x.mean(dim=0), data.x.std(dim=0)
    data.x = (data.x - x_mean) / (x_std + 1e-6)

    edge_attr_mean, edge_attr_std = data.edge_attr.mean(dim=0), data.edge_attr.std(dim=0)
    data.edge_attr = (data.edge_attr - edge_attr_mean) / (edge_attr_std + 1e-6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware training device: {device}")

    model = UltraPhysicsGNN(in_channels=8, hidden_channels=128, out_channels=1).to(device)
    data = data.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    print("\n2. Training Deep 4-Layer GATv2 PINN with Focal Outfall Loss (1,000 Epochs)...")
    model.train()

    best_mse = float('inf')

    for epoch in range(1, 1001):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        
        total_loss, mse, focal_loss = ultra_pinn_loss(out, data.y, data)

        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        if mse.item() < best_mse:
            best_mse = mse.item()
            torch.save(model.state_dict(), "pinn_gnn_model.pth")

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | Total Loss: {total_loss.item():.6f} | MSE: {mse.item():.6f} | Focal Loss: {focal_loss.item():.6f} | Best MSE: {best_mse:.6f}")

    print(f"\nUltra PINN Model Training Complete! Best MSE: {best_mse:.6f}. Saved to 'pinn_gnn_model.pth'.")

if __name__ == "__main__":
    train_ultra_model()
