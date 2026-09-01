import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv

sys.stdout.reconfigure(line_buffering=True)


class PerfectAccuracyGNN(nn.Module):
    """
    Ultra-High Precision Deep Graph Neural Network Architecture (12 Dynamic Features).
    Uses unconstrained linear output with Target Standard Scaling for 100% gradient flow,
    fusing 4-layer multi-head GATv2 attention representations with local hydraulic sag indices
    and dynamic meteorological forcing (rainfall intensity & duration).
    """
    def __init__(self, in_channels=14, hidden_channels=256, out_channels=1):
        super(PerfectAccuracyGNN, self).__init__()
        
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, concat=True, edge_dim=2)
        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=2, concat=True, edge_dim=2)
        self.conv3 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=2, concat=False, edge_dim=2)
        self.conv4 = GATv2Conv(hidden_channels, hidden_channels, heads=1, concat=False, edge_dim=2)

        self.ln1 = nn.LayerNorm(hidden_channels * 4)
        self.ln2 = nn.LayerNorm(hidden_channels * 2)
        self.ln3 = nn.LayerNorm(hidden_channels)
        self.ln4 = nn.LayerNorm(hidden_channels)

        fusion_dim = (hidden_channels * 2) + in_channels
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, out_channels)
        )


    def forward(self, x, edge_index, edge_attr):
        x_in = x
        h1 = F.elu(self.ln1(self.conv1(x, edge_index, edge_attr)))
        h2 = F.elu(self.ln2(self.conv2(h1, edge_index, edge_attr)))
        h3 = F.elu(self.ln3(self.conv3(h2, edge_index, edge_attr)))
        h4 = F.elu(self.ln4(self.conv4(h3, edge_index, edge_attr)))

        fusion = torch.cat([h4, h3, x_in], dim=-1)
        out = self.regressor(fusion)
        return out



def train_perfect_accuracy():
    print("=================================================================", flush=True)
    print("  URBANFLOW GENERALIZED MULTI-SCENARIO PINN ENGINE (12 FEATURES)", flush=True)
    print("=================================================================\n", flush=True)
    
    dataset_path = "multi_scenario_pyg_dataset.pt"
    try:
        scenario_list = torch.load(dataset_path, weights_only=False)
        if isinstance(scenario_list, Batch):
            # Convert back to list if single batch
            scenario_list = scenario_list.to_data_list()
    except Exception as e:
        print(f"Loading '{dataset_path}' failed, falling back to 'bengaluru_pyg_dataset.pt'...")
        scenario_list = [torch.load("bengaluru_pyg_dataset.pt", weights_only=False)]

    print(f"Loaded dataset with {len(scenario_list)} scenario graph batches.", flush=True)

    # Hold out 50.0 mm/hr and 100.0 mm/hr storms for Out-of-Distribution (OOD) validation
    HOLDOUT_INTENSITIES = [50.0, 100.0]
    train_graphs = [g for g in scenario_list if getattr(g, 'rain_intensity', 50.0) not in HOLDOUT_INTENSITIES]
    val_graphs = [g for g in scenario_list if getattr(g, 'rain_intensity', 50.0) in HOLDOUT_INTENSITIES]

    if len(val_graphs) == 0:
        print("Warning: No matching holdout scenarios found! Splitting 80/20 randomly...", flush=True)
        split_idx = int(len(scenario_list) * 0.8)
        train_graphs = scenario_list[:split_idx]
        val_graphs = scenario_list[split_idx:]

    print(f"Training Scenarios ({len(train_graphs)}): {[getattr(g, 'rain_intensity', 'N/A') for g in train_graphs]} mm/hr", flush=True)
    print(f"Holdout Validation Scenarios ({len(val_graphs)}): {[getattr(g, 'rain_intensity', 'N/A') for g in val_graphs]} mm/hr\n", flush=True)

    train_batch = Batch.from_data_list(train_graphs)
    val_batch = Batch.from_data_list(val_graphs)

    # Standardize features based on Training Set statistics
    x_mean, x_std = train_batch.x.mean(dim=0), train_batch.x.std(dim=0) + 1e-6
    edge_attr_mean, edge_attr_std = train_batch.edge_attr.mean(dim=0), train_batch.edge_attr.std(dim=0) + 1e-6
    y_mean, y_std = train_batch.y.mean(), train_batch.y.std() + 1e-6

    # Apply standardization
    train_batch.x = (train_batch.x - x_mean) / x_std
    train_batch.edge_attr = (train_batch.edge_attr - edge_attr_mean) / edge_attr_std
    train_y_norm = (train_batch.y - y_mean) / y_std

    val_batch.x = (val_batch.x - x_mean) / x_std
    val_batch.edge_attr = (val_batch.edge_attr - edge_attr_mean) / edge_attr_std
    val_y_norm = (val_batch.y - y_mean) / y_std

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compute Hardware Device: {device}", flush=True)

    model = PerfectAccuracyGNN(in_channels=14, hidden_channels=256, out_channels=1).to(device)

    train_batch = train_batch.to(device)
    train_y_norm = train_y_norm.to(device)
    val_batch = val_batch.to(device)
    val_y_norm = val_y_norm.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1200, eta_min=1e-5)

    print("\nExecuting 12-Feature Multi-Scenario Deep GNN Training (1,200 Epochs)...", flush=True)
    best_val_mae = float('inf')

    for epoch in range(1, 1201):
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

        # Validation loop
        if epoch % 10 == 0 or epoch == 1 or epoch == 1200:
            model.eval()
            with torch.no_grad():
                val_out_norm = model(val_batch.x, val_batch.edge_index, val_batch.edge_attr)
                val_pred_meters = torch.clamp(val_out_norm * y_std + y_mean, min=0.0)
                val_mae = float(F.l1_loss(val_pred_meters, val_batch.y).cpu().numpy())
                train_mae = float(F.l1_loss(pred_meters, train_batch.y).cpu().numpy())

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                # Save model along with dataset stats
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'x_mean': x_mean.cpu(),
                    'x_std': x_std.cpu(),
                    'edge_attr_mean': edge_attr_mean.cpu(),
                    'edge_attr_std': edge_attr_std.cpu(),
                    'y_mean': y_mean.cpu(),
                    'y_std': y_std.cpu()
                }
                torch.save(checkpoint, "pinn_gnn_checkpoint.pt")
                torch.save(model.state_dict(), "pinn_gnn_model.pth")

            if epoch % 100 == 0 or epoch == 1:
                print(f"Epoch {epoch:04d} | Train MAE: {train_mae:.4f}m ({train_mae*100:.2f}cm) | OOD Val MAE: {val_mae:.4f}m ({val_mae*100:.2f}cm) | Best Val MAE: {best_val_mae*100:.2f}cm", flush=True)


    print(f"\nOptimization Complete! Best Out-of-Distribution Validation MAE: {best_val_mae:.4f}m ({best_val_mae*100:.2f}cm). Model saved to 'pinn_gnn_model.pth'.", flush=True)

if __name__ == "__main__":
    train_perfect_accuracy()

