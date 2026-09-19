import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv


class FloodGNN(nn.Module):
    def __init__(self, in_channels=17, hidden_channels=128, out_channels=1):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, concat=True, edge_dim=2)
        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=2, concat=True, edge_dim=2)
        self.conv3 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=2, concat=False, edge_dim=2)
        self.ln1 = nn.LayerNorm(hidden_channels * 4)
        self.ln2 = nn.LayerNorm(hidden_channels * 2)
        self.ln3 = nn.LayerNorm(hidden_channels)
        self.reg = nn.Sequential(
            nn.Linear(hidden_channels + in_channels, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(64, out_channels),
        )

    def forward(self, x, edge_index, edge_attr):
        h1 = F.elu(self.ln1(self.conv1(x, edge_index, edge_attr)))
        h2 = F.elu(self.ln2(self.conv2(h1, edge_index, edge_attr)))
        h3 = F.elu(self.ln3(self.conv3(h2, edge_index, edge_attr)))
        fusion = torch.cat([h3, x], dim=-1)
        return self.reg(fusion)


def main():
    device = torch.device('cuda')
    data_list = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
    graphs = [g.clone() for g in data_list if g.city == 'bangalore']
    batch = Batch.from_data_list(graphs).to(device)
    print(f"nodes {batch.x.shape[0]} edges {batch.edge_index.shape[1]}", flush=True)

    x_mean, x_std = batch.x.mean(0), batch.x.std(0) + 1e-6
    y_mean, y_std = batch.y.mean(), batch.y.std() + 1e-6
    batch.x = (batch.x - x_mean) / x_std
    batch.edge_attr = (batch.edge_attr - batch.edge_attr.mean(0)) / (batch.edge_attr.std(0) + 1e-6)
    ty = (batch.y - y_mean) / y_std

    for h in [64, 128]:
        model = FloodGNN(in_channels=17, hidden_channels=h).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        model.train()
        t0 = time.time()
        n_epochs = 10
        for _ in range(n_epochs):
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = F.mse_loss(out, ty)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        dt = (time.time() - t0) / n_epochs
        print(f"hidden={h}: {dt:.2f}s/epoch | loss {loss.item():.4f}", flush=True)


if __name__ == "__main__":
    main()