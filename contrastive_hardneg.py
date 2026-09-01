"""Hard-Negative Topology-Preserving Contrastive Pre-Training for iter18.

Augmentations:
  1. Drop low-accumulation secondary edges (preserve high-accumulation drainage paths)
  2. Hard-negative mining: contrast topographically similar subgraphs with different outcomes

Encoder backbone matches iter17 FiLMGINE structure for clean warm-start into iter18.
SAGPool is NOT in the contrastive encoder -- it's added only in the fine-tuning architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv

dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_full31_pyg_dataset.pt", weights_only=False)

city_graphs = {}
for g in dl:
    key = (g.city, g.region)
    if key not in city_graphs and g.x[0, 15].item() == 150.0:
        city_graphs[key] = g

print(f"Loaded {len(city_graphs)} city-region graphs for hard-negative contrastive pre-training")


def topology_preserving_augment(g, p_edge_drop=0.15, p_feat_mask=0.2):
    """Drop low-accumulation edges (keep high-accumulation drainage paths)."""
    g_aug = g.clone()
    e = g_aug.edge_index.shape[1]

    acc_feat_idx = 7
    edge_acc = g_aug.x[g_aug.edge_index[0], acc_feat_idx]
    acc_thresh = edge_acc.quantile(0.5)
    drop_mask = (torch.rand(e) > p_edge_drop) | (edge_acc > acc_thresh)
    g_aug.edge_index = g_aug.edge_index[:, drop_mask]
    g_aug.edge_attr = g_aug.edge_attr[drop_mask]

    feat_mask = torch.rand(g_aug.x.shape) > p_feat_mask
    g_aug.x = g_aug.x * feat_mask
    return g_aug


def hard_negative_mining(graph_list, batch_size=8):
    """Create hard-negative pairs: similar elevation structure, different flood outcome."""
    pairs = []
    for g in graph_list:
        elev2 = g.x[:, 17].mean().item()
        y_mean = g.y.mean().item()
        best_hn = None
        best_dist = float('inf')
        for other in graph_list:
            if other is g:
                continue
            e2 = other.x[:, 17].mean().item()
            y2 = other.y.mean().item()
            elev_dist = abs(elev2 - e2)
            outcome_dist = abs(y_mean - y2)
            combined = elev_dist * 3 + outcome_dist
            if combined < best_dist:
                best_dist = combined
                best_hn = other
        if best_hn is not None:
            pairs.append((g, best_hn))
    return pairs


class ContrastiveGINE(nn.Module):
    """GINE encoder matching FiLMGINE backbone (5 layers, hidden=128)."""
    def __init__(self, in_c=31, edge_c=2, hidden=128, n_layers=5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GINEConv(nn.Linear(in_c if i == 0 else hidden, hidden), edge_dim=edge_c))
            self.lns.append(nn.LayerNorm(hidden))
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 64))

    def forward(self, x, edge_index, edge_attr):
        h = x
        for conv, ln in zip(self.convs, self.lns):
            h = F.elu(ln(conv(h, edge_index, edge_attr)))
        return h

    def project(self, h):
        return self.proj(h)


def info_nce_loss(z1, z2, temperature=0.07):
    batch_size = z1.size(0)
    device = z1.device
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(batch_size, device=device)
    loss = F.cross_entropy(sim, labels)
    loss += F.cross_entropy(sim.t(), labels)
    return loss * 0.5


graph_list = list(city_graphs.values())
hn_pairs = hard_negative_mining(graph_list)
print(f"Computed {len(hn_pairs)} hard-negative pairs")
print(f"Using {len(graph_list)} graphs for hard-negative contrastive pre-training")

device = torch.device('cuda')
model = ContrastiveGINE(in_c=31).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

N_EPOCHS = 150
BATCH_SIZE = 8

for ep in range(N_EPOCHS):
    model.train()
    random.shuffle(graph_list)
    total_loss = 0
    n_batches = 0

    for i in range(0, len(graph_list), BATCH_SIZE):
        batch_graphs = graph_list[i:i+BATCH_SIZE]
        if len(batch_graphs) < 2:
            continue

        view1 = [topology_preserving_augment(g) for g in batch_graphs]
        view2 = [topology_preserving_augment(g) for g in batch_graphs]

        if random.random() < 0.3:
            idx = random.randint(0, len(batch_graphs)-1)
            g_orig = batch_graphs[idx]
            hn = None
            for a, b in hn_pairs:
                if a is g_orig:
                    hn = b
                    break
            if hn is not None:
                view2[idx] = topology_preserving_augment(hn)

        batch1 = Batch.from_data_list(view1).to(device)
        batch2 = Batch.from_data_list(view2).to(device)

        opt.zero_grad()

        h1 = model(batch1.x, batch1.edge_index, batch1.edge_attr)
        h2 = model(batch2.x, batch2.edge_index, batch2.edge_attr)

        z1 = torch.stack([h1[batch1.batch == gi].mean(0) for gi in range(len(batch_graphs))])
        z2 = torch.stack([h2[batch2.batch == gi].mean(0) for gi in range(len(batch_graphs))])

        z1 = model.project(z1)
        z2 = model.project(z2)

        loss = info_nce_loss(z1, z2)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n_batches += 1

    if ep % 10 == 0:
        print(f"Epoch {ep}: loss = {total_loss/max(1,n_batches):.4f}")

torch.save({
    'model': model.state_dict(),
    'config': {'in_c': 31, 'hidden': 128, 'n_layers': 5}
}, r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\contrastive_hardneg_encoder.pt")
print("Saved hard-negative contrastive encoder")
