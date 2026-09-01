import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class UrbanFlowPINN(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=1):
        super(UrbanFlowPINN, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, concat=True, edge_dim=2)
        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=2, concat=False, edge_dim=2)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.05, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)

        out = self.regressor(x)
        return out


def physics_informed_loss(pred, target, lambda_physics=0.5):
    mse_loss = F.mse_loss(pred, target)
    negative_depth_penalty = torch.mean(F.relu(-pred) ** 2)
    total_loss = mse_loss + (lambda_physics * negative_depth_penalty)
    return total_loss, mse_loss, negative_depth_penalty


def main():
    print("1. Loading PyTorch Geometric dataset tensor...")
    data = torch.load("bengaluru_pyg_dataset.pt", weights_only=False)

    x_mean, x_std = data.x.mean(dim=0), data.x.std(dim=0)
    data.x = (data.x - x_mean) / (x_std + 1e-6)

    edge_attr_mean, edge_attr_std = data.edge_attr.mean(dim=0), data.edge_attr.std(dim=0)
    data.edge_attr = (data.edge_attr - edge_attr_mean) / (edge_attr_std + 1e-6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training hardware device: {device}")

    model = UrbanFlowPINN(in_channels=5, hidden_channels=64, out_channels=1).to(device)
    data = data.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=1e-4)

    print("\n2. Starting High-Accuracy PINN-GNN Model Training Loop...")
    model.train()

    for epoch in range(1, 601):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        total_loss, mse, phys_penalty = physics_informed_loss(out, data.y)

        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.6f} | MSE: {mse.item():.6f} | Physics Penalty: {phys_penalty.item():.6f}")

    torch.save(model.state_dict(), "pinn_gnn_model.pth")
    print("\nModel training complete! High-accuracy weights saved to 'pinn_gnn_model.pth'.")


if __name__ == "__main__":
    main()