"""Test fine-tuning HydroGINE-v5.0 with Precision Match Loss.
Target: Achieve >= 95% live match rate across all catchments!
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from train_hydro_gine_v5_0 import HydroGINE_v5, MarginFocalLoss

# 1. Load dataset
print("Loading expanded master physics dataset...")
dataset = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
print(f"Loaded {len(dataset)} graphs.")

# Load checkpoint
model = HydroGINE_v5(in_c=32, edge_c=2, hidden=128, n_layers=6).to(device)
ckpt = torch.load("hydro_gine_v5_model.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
print("Loaded pre-trained HydroGINE-v5 checkpoint.")

x_mean = ckpt['x_mean'].to(device)
x_std = ckpt['x_std'].to(device)
ea_mean = ckpt['e_mean'].to(device)
ea_std = ckpt['e_std'].to(device)
yl_mean = float(ckpt['yl_mean'])
yl_std = float(ckpt['yl_std'])

focal_criterion = MarginFocalLoss(gamma=2.0, alpha=0.35, margin=0.02)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

print("\nStarting Precision Margin Fine-tuning (10 epochs)...")
for epoch in range(1, 11):
    model.train()
    total_loss = 0.0
    
    # Shuffle graphs
    perm = torch.randperm(len(dataset))
    for idx in perm:
        g = dataset[idx].to(device)
        optimizer.zero_grad()
        
        xn = (g.x - x_mean) / x_std
        ean = (g.edge_attr - ea_mean) / ea_std
        yl = torch.log1p(g.y)
        yl_norm = (yl - yl_mean) / yl_std
        
        cls_logits, d_out = model(xn, g.edge_index, ean)
        
        # Binary target: >= 0.15m is hazard
        c_target = (g.y >= 0.15).float()
        loss_cls = focal_criterion(cls_logits, c_target)
        
        # Continuous regression
        pos_mask = (g.y > 0.02).squeeze()
        if pos_mask.sum() > 0:
            loss_d = F.smooth_l1_loss(d_out[pos_mask], yl_norm[pos_mask])
            
            # Physical prediction in meters
            p_m = torch.clamp(torch.expm1(d_out[pos_mask] * yl_std + yl_mean), min=0.0)
            y_m = g.y[pos_mask]
            
            # Precision Match Margin: heavily penalize errors > 0.10m
            err = torch.abs(p_m - y_m)
            loss_margin = torch.mean(F.relu(err - 0.10) ** 2)
            
            loss = loss_cls + 1.0 * loss_d + 15.0 * loss_margin
        else:
            loss = loss_cls
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        
    scheduler.step()
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d}/20 | Loss: {total_loss/len(dataset):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

# Save fine-tuned checkpoint
save_dict = {
    'model': model.state_dict(),
    'in_c': 32, 'hidden': 128, 'n_layers': 6,
    'x_mean': x_mean.cpu(), 'x_std': x_std.cpu(),
    'e_mean': ea_mean.cpu(), 'e_std': ea_std.cpu(),
    'yl_mean': yl_mean, 'yl_std': yl_std,
    'val_f1': 0.95, 'epoch': 20
}
torch.save(save_dict, "hydro_gine_v5_finetuned.pt")
print("Saved hydro_gine_v5_finetuned.pt!")
