"""Train HydroGINE-v5 with Bangalore & HSR balanced precision weighting.
Combines:
1. User Domain Insights: Bottleneck node 4x weighting + Margin Loss (|err| > 10cm).
2. Balanced Bangalore/HSR graph representation (3x gradient fidelity).
3. Zero coordinate leakage, 100% pure physics features.
"""
import torch, time, os, sys
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

dataset = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
base_ckpt = torch.load("hydro_gine_v5_bottleneck_opt.pt", weights_only=False)

from train_hydro_gine_v5_0 import HydroGINE_v5, MarginFocalLoss

model = HydroGINE_v5(in_c=32, edge_c=2, hidden=128, n_layers=6).to(device)
model.load_state_dict(base_ckpt['model'])

optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-5)
focal_criterion = MarginFocalLoss(gamma=2.0, alpha=0.40, margin=0.02)

x_mean = base_ckpt['x_mean'].to(device)
x_std = base_ckpt['x_std'].to(device)
ea_mean = base_ckpt['e_mean'].to(device)
ea_std = base_ckpt['e_std'].to(device)
yl_mean = float(base_ckpt['yl_mean'])
yl_std = float(base_ckpt['yl_std'])

bengaluru_regions = {'hsr', 'bellandur', 'whitefield', 'ecity', 'koramangala'}

epochs = 8
print(f"Starting Bangalore Precision Training for {epochs} epochs...")

for ep in range(1, epochs + 1):
    model.train()
    tot_loss = 0.0
    tot_margin = 0.0
    tot_graphs = 0
    t0 = time.time()
    
    perm = torch.randperm(len(dataset))
    for g_idx in perm:
        g = dataset[g_idx].to(device)
        optimizer.zero_grad()
        
        xn = (g.x - x_mean) / x_std
        ean = (g.edge_attr - ea_mean) / ea_std
        yl = torch.log1p(g.y)
        yl_norm = (yl - yl_mean) / yl_std
        
        cls_logits, d_out = model(xn, g.edge_index, ean)
        
        # User Domain Insight: Bottleneck weighting
        accum_s = g.x[:, 5]
        slope = torch.abs(g.x[:, 2])
        sink_d = g.x[:, 23]
        conv_def = g.x[:, 30]
        
        is_bottleneck = (accum_s >= 1.2) & (slope <= 0.025) & ((conv_def >= 0.5) | (sink_d >= 0.04))
        node_weights = torch.where(is_bottleneck, 4.0, 1.0)
        
        # Region balance: 3.0x multiplier on Bangalore / HSR graphs to match massive foreign pool
        reg_name = getattr(g, 'region', 'unknown')
        g_scale = 3.0 if reg_name in bengaluru_regions else 1.0
        
        # Classification loss
        c_target = (g.y >= 0.15).float()
        loss_cls = focal_criterion(cls_logits, c_target)
        
        # Depth regression loss
        pos_mask = (g.y > 0.02).squeeze()
        if pos_mask.sum() > 0:
            reg_err = F.smooth_l1_loss(d_out[pos_mask], yl_norm[pos_mask], reduction='none').squeeze()
            weighted_reg = (reg_err * node_weights[pos_mask]).mean()
            
            # Precision Margin Loss (penalize depth errors > 10 cm)
            p_m = torch.clamp(torch.expm1(d_out[pos_mask] * yl_std + yl_mean), min=0.0)
            y_m = g.y[pos_mask]
            err_m = torch.abs(p_m - y_m)
            margin_err = (F.relu(err_m - 0.10) ** 2) * node_weights[pos_mask]
            loss_margin = margin_err.mean()
            
            loss = (loss_cls * 2.0 + weighted_reg * 1.5 + loss_margin * 3.0) * g_scale
            tot_margin += loss_margin.item() * g_scale
        else:
            loss = loss_cls * 2.0 * g_scale
            
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        tot_loss += loss.item()
        tot_graphs += 1
        
    scheduler.step()
    dt = time.time() - t0
    avg_l = tot_loss / tot_graphs
    avg_m = tot_margin / max(1, tot_graphs)
    print(f"Epoch {ep:2d}/{epochs:2d} | Loss: {avg_l:.4f} | MarginLoss: {avg_m:.4f} | Time: {dt:.1f}s")

out_path = "hydro_gine_v5_bangalore_opt.pt"
save_dict = {
    'model': model.state_dict(),
    'x_mean': x_mean.cpu(),
    'x_std': x_std.cpu(),
    'e_mean': ea_mean.cpu(),
    'e_std': ea_std.cpu(),
    'yl_mean': yl_mean,
    'yl_std': yl_std,
    'epoch': epochs
}
torch.save(save_dict, out_path)
print(f"Saved optimized Bangalore precision model to {out_path}")
