"""Graph Contrastive Pre-training (GraphCL style) on 50+ city road graphs.
Pre-trains GNN encoder on topology + local features, then fine-tunes on SWMM targets."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv
from torch_geometric.transforms import Compose
import copy

# Load all city graphs from the dataset
dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_full22_pyg_dataset.pt", weights_only=False)

# Collect unique graphs (one per region per intensity -> take I=150 as representative)
city_graphs = {}
for g in dl:
    key = (g.city, g.region)
    if key not in city_graphs and g.x[0, 15].item() == 150.0:
        city_graphs[key] = g

print(f"Loaded {len(city_graphs)} city-region graphs for contrastive pre-training")

# Graph augmentations for GraphCL
def augment_graph(g, p_edge_drop=0.2, p_feat_mask=0.2, p_node_drop=0.1):
    """Apply random augmentations: edge dropping, feature masking, node dropping."""
    g_aug = g.clone()
    n = g_aug.num_nodes
    e = g_aug.edge_index.shape[1]
    
    # Edge dropping
    if p_edge_drop > 0:
        mask = torch.rand(e) > p_edge_drop
        g_aug.edge_index = g_aug.edge_index[:, mask]
        g_aug.edge_attr = g_aug.edge_attr[mask]
    
    # Feature masking
    if p_feat_mask > 0:
        feat_mask = torch.rand(g_aug.x.shape) > p_feat_mask
        g_aug.x = g_aug.x * feat_mask
    
    # Node dropping
    if p_node_drop > 0:
        node_mask = torch.rand(n) > p_node_drop
        keep = node_mask.nonzero().squeeze()
        if len(keep) > 0:
            # Remap node indices
            old_to_new = {old.item(): new for new, old in enumerate(keep)}
            new_edge_index = g_aug.edge_index.clone()
            mask = torch.isin(g_aug.edge_index[0], keep) & torch.isin(g_aug.edge_index[1], keep)
            new_edge_index = g_aug.edge_index[:, mask]
            new_edge_index = torch.tensor([[old_to_new[idx.item()] for idx in new_edge_index[0]],
                                           [old_to_new[idx.item()] for idx in new_edge_index[1]]], dtype=torch.long)
            g_aug.edge_index = new_edge_index
            g_aug.edge_attr = g_aug.edge_attr[mask]
            g_aug.x = g_aug.x[keep]
    
    return g_aug


class ContrastiveGINE(nn.Module):
    def __init__(self, in_c, edge_c=2, hidden=128, n_layers=5, proj_dim=64):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GINEConv(nn.Linear(in_c if i == 0 else hidden, hidden), edge_dim=2))
        self.lns = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        
        # Projection head for contrastive loss
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, proj_dim)
        )
        
    def forward(self, x, edge_index, edge_attr):
        h = x
        for conv, ln in zip(self.convs, self.lns):
            h = F.elu(ln(conv(h, edge_index, edge_attr)))
        return h
    
    def project(self, h):
        return self.proj(h)


def info_nce_loss(z1, z2, temperature=0.1):
    """InfoNCE loss for contrastive learning with numerical stability."""
    batch_size = z1.size(0)
    device = z1.device
    
    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Similarity matrix
    sim = torch.mm(z1, z2.t()) / temperature  # B x B
    
    # Labels: positive pairs are on the diagonal
    labels = torch.arange(batch_size, device=device)
    
    loss = F.cross_entropy(sim, labels)
    loss += F.cross_entropy(sim.t(), labels)  # symmetric
    return loss * 0.5


# Prepare dataset for contrastive learning
graph_list = list(city_graphs.values())
print(f"Using {len(graph_list)} graphs for pre-training")

# Create data loaders
device = torch.device('cuda')
model = ContrastiveGINE(in_c=28).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

N_EPOCHS = 100
BATCH_SIZE = 16

for ep in range(N_EPOCHS):
    model.train()
    random.shuffle(graph_list)
    total_loss = 0
    n_batches = 0
    
    for i in range(0, len(graph_list), BATCH_SIZE):
        batch_graphs = graph_list[i:i+BATCH_SIZE]
        if len(batch_graphs) < 2:
            continue
        
        # Create two augmented views
        view1 = [augment_graph(g) for g in batch_graphs]
        view2 = [augment_graph(g) for g in batch_graphs]
        
        batch1 = Batch.from_data_list(view1).to(device)
        batch2 = Batch.from_data_list(view2).to(device)
        
        opt.zero_grad()
        
        # Forward pass
        h1 = model(batch1.x, batch1.edge_index, batch1.edge_attr)
        h2 = model(batch2.x, batch2.edge_index, batch2.edge_attr)
        
        # Global pooling (mean over nodes per graph)
        z1 = torch.stack([h1[batch1.batch == i].mean(0) for i in range(len(batch_graphs))])
        z2 = torch.stack([h2[batch2.batch == i].mean(0) for i in range(len(batch_graphs))])
        
        z1 = model.project(z1)
        z2 = model.project(z2)
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        loss = info_nce_loss(z1, z2)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n_batches += 1
    
    if ep % 10 == 0:
        print(f"Epoch {ep}: contrastive loss = {total_loss/n_batches:.4f}")

# Save pre-trained encoder
torch.save({
    'model': model.state_dict(),
    'config': {'in_c': 28, 'hidden': 128, 'n_layers': 5}
}, r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\contrastive_encoder.pt")
print("Saved contrastive pre-trained encoder")