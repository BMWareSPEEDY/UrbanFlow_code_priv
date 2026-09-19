import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import osmnx as ox
import pandas as pd
import numpy as np
from torch_geometric.nn import GATv2Conv

sys.stdout.reconfigure(line_buffering=True)


class ZeroToleranceHurdleGNN(nn.Module):
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
        return torch.where(gate_prob >= 0.40, reg_out, torch.zeros_like(reg_out))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}", flush=True)

    print("Loading PyG full dataset...", flush=True)
    g = torch.load("bengaluru_pyg_dataset.pt", weights_only=False).to(device)

    x_mean = g.x[:, :14].mean(dim=0)
    x_std = g.x[:, :14].std(dim=0).clamp(min=1e-5)
    e_mean = g.edge_attr.mean(dim=0)
    e_std = g.edge_attr.std(dim=0).clamp(min=1e-5)
    y_mean = g.y.mean()
    y_std = g.y.std().clamp(min=1e-5)

    x_norm = (g.x[:, :14] - x_mean) / x_std
    e_norm = (g.edge_attr - e_mean) / e_std
    targets = g.y.view(-1, 1)

    safe_mask = (targets < 0.08).float()
    wet_mask = (targets >= 0.02).float()
    crit_mask = (targets >= 0.30).float()
    total_nodes = len(targets)

    print(f"Dataset: {total_nodes:,} nodes | Safe (<8cm): {int(safe_mask.sum().item()):,} | Critical (>=30cm): {int(crit_mask.sum().item()):,}", flush=True)

    model = ZeroToleranceHurdleGNN(in_channels=14, hidden_channels=128, out_channels=1).to(device)
    if os.path.exists("zero_tolerance_gnn_checkpoint.pt"):
        ckpt = torch.load("zero_tolerance_gnn_checkpoint.pt", weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print("Warm-started from zero_tolerance_gnn_checkpoint.pt", flush=True)

    # Pre-load HSR spatial graph for monitoring
    print("Loading HSR Layout validation network...", flush=True)
    G = ox.load_graphml('bengaluru_complete_graph.graphml')
    node_list = list(G.nodes())
    num_hsr = len(node_list)
    elevs = [float(G.nodes[nid].get('elevation', 880.0)) for nid in node_list]
    min_elev, max_elev = min(elevs), max(elevs)
    elev_range = max(1.0, max_elev - min_elev)

    node_in_grades = {nid: [] for nid in node_list}
    node_out_grades = {nid: [] for nid in node_list}
    for u, v, k, data in G.edges(keys=True, data=True):
        grade = float(data.get('grade', 0.0))
        if u in node_out_grades: node_out_grades[u].append(grade)
        if v in node_in_grades: node_in_grades[v].append(grade)

    df_targets = pd.read_csv('swmm_groundtruth_targets.csv')
    target_lookup = {str(row['swmm_node_id']).replace('J_', ''): float(row['max_water_depth_m']) for _, row in df_targets.iterrows()}

    static_features_list, swmm_list = [], []
    for nid in node_list:
        d = G.nodes[nid]
        elev = float(d.get('elevation', 880.0))
        in_deg = float(G.in_degree(nid))
        out_deg = float(G.out_degree(nid))
        manning = float(d.get('manning_n', 0.024))
        rel_x = (float(d.get('x', 0.0)) - min([float(G.nodes[n].get('x', 0.0)) for n in node_list])) / max(1.0, max([float(G.nodes[n].get('x', 0.0)) for n in node_list]) - min([float(G.nodes[n].get('x', 0.0)) for n in node_list]))
        rel_y = (float(d.get('y', 0.0)) - min([float(G.nodes[n].get('y', 0.0)) for n in node_list])) / max(1.0, max([float(G.nodes[n].get('y', 0.0)) for n in node_list]) - min([float(G.nodes[n].get('y', 0.0)) for n in node_list]))
        delta_elev = (elev - min_elev) / elev_range
        rel_drop = (max_elev - elev) / elev_range
        accum_score = np.log1p(in_deg * 2.5 + (1.0 if out_deg == 0 else 0.0))
        is_sink = 1.0 if (rel_drop > 0.85 and in_deg >= 2) else 0.0
        in_grades = node_in_grades.get(nid, [0.0])
        out_grades = node_out_grades.get(nid, [0.0])
        max_in_grade = max(in_grades) if in_grades else 0.0
        min_out_grade = min(out_grades) if out_grades else 0.0
        sag_index = max(0.0, max_in_grade - min_out_grade) * max(1, in_deg)
        hydraulic_capacity = float(in_deg) / max(1.0, float(out_deg))
        swmm_list.append(target_lookup.get(str(nid), 0.0))
        static_features_list.append([rel_x, rel_y, delta_elev, rel_drop, manning, in_deg, out_deg, accum_score, is_sink, max_in_grade, sag_index, hydraulic_capacity, 50.0, 60.0])

    node_to_idx = {nid: idx for idx, nid in enumerate(node_list)}
    src_nodes, dst_nodes, edge_feats = [], [], []
    for u, v, k, data in G.edges(keys=True, data=True):
        src_nodes.append(node_to_idx[u])
        dst_nodes.append(node_to_idx[v])
        edge_feats.append([float(data.get('length', 10.0)), float(data.get('grade', 0.0))])

    edge_index_hsr = torch.tensor([src_nodes, dst_nodes], dtype=torch.long).to(device)
    edge_attr_hsr = torch.tensor(edge_feats, dtype=torch.float).to(device)
    edge_norm_hsr = (edge_attr_hsr - e_mean) / e_std
    full_x_hsr = torch.tensor(static_features_list, dtype=torch.float).to(device)
    swmm_hsr = np.array(swmm_list)

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

    print("\nStarting Severe Cloudburst Precision Training (200 Epochs)...", flush=True)
    t_start = time.time()
    best_score = float('inf')

    for ep in range(1, 201):
        model.train()
        optimizer.zero_grad()

        reg_out, gate_logit = model(x_norm, g.edge_index, e_norm, return_gate=True)
        preds_depth = torch.clamp(reg_out * y_std + y_mean, min=0.0)

        # 1. Classification Gate Loss
        gate_loss = F.binary_cross_entropy_with_logits(gate_logit, wet_mask)

        # 2. Critical Depth Loss (weighted by true magnitude)
        depth_weight = 1.0 + 35.0 * torch.clamp(targets, min=0.0, max=1.5)
        depth_loss = (wet_mask * depth_weight * F.smooth_l1_loss(preds_depth, targets, reduction='none', beta=0.02)).sum() / (wet_mask.sum() + 1e-6)

        # 3. Asymmetric False Alarm Penalty (150x quadratic penalty on safe nodes)
        false_alarm_penalty = (safe_mask * 150.0 * torch.square(torch.clamp(preds_depth - 0.02, min=0.0))).sum() / (safe_mask.sum() + 1e-6)

        total_loss = depth_loss + 3.5 * gate_loss + false_alarm_penalty

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if ep % 20 == 0 or ep == 200:
            model.eval()
            with torch.no_grad():
                # 50 mm/hr test
                full_x_50 = full_x_hsr.clone()
                full_x_50[:, 12] = 50.0
                r_50, g_50 = model((full_x_50 - x_mean)/x_std, edge_index_hsr, edge_norm_hsr, return_gate=True)
                p_50 = np.where(torch.sigmoid(g_50).cpu().numpy().ravel() >= 0.40, torch.clamp(r_50 * y_std + y_mean, min=0.0).cpu().numpy().ravel(), 0.0)
                fc_50 = int(np.sum((p_50 >= 0.30) & (swmm_hsr < 0.10)))
                mae_50 = np.mean(np.abs(p_50 - swmm_hsr)) * 100.0
                
                # 120 mm/hr test
                full_x_120 = full_x_hsr.clone()
                full_x_120[:, 12] = 120.0
                r_120, g_120 = model((full_x_120 - x_mean)/x_std, edge_index_hsr, edge_norm_hsr, return_gate=True)
                p_120 = np.where(torch.sigmoid(g_120).cpu().numpy().ravel() >= 0.40, torch.clamp(r_120 * y_std + y_mean, min=0.0).cpu().numpy().ravel(), 0.0)
                swmm_120 = swmm_hsr * (120.0 / 50.0)
                fc_120 = int(np.sum((p_120 >= 0.30) & (swmm_120 < 0.10)))
                mae_120 = np.mean(np.abs(p_120 - swmm_120)) * 100.0

                score = fc_50 * 3 + fc_120
                print(f"Epoch {ep:3d} | Loss: {total_loss.item():.4f} | 50mm FC: {fc_50:2d} (MAE: {mae_50:4.2f}cm) | 120mm FC: {fc_120:2d} (MAE: {mae_120:4.2f}cm)", flush=True)

                if score < best_score:
                    best_score = score
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
                        'best_fc_50': fc_50,
                        'best_fc_120': fc_120,
                        'best_mae_50': mae_50,
                        'best_mae_120': mae_120
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
                        'best_fc_50': fc_50,
                        'best_fc_120': fc_120,
                        'best_mae_50': mae_50,
                        'best_mae_120': mae_120
                    }, "pinn_gnn_checkpoint.pt")

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed / 60.0:.2f} minutes!", flush=True)
    print("Checkpoints saved successfully!", flush=True)


if __name__ == "__main__":
    main()
