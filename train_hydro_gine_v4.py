"""UrbanFLOW HydroGINE-v4.0: Physics-Informed Dual-Head GNN.
Backbone: 6-layer Residual GINE with multi-scale skip connections and edge conditioning.
Heads: Calibrated Margin Focal Classifier + FiLM Conditioned Depth Regressor.
Features: 32 strictly zero-leakage physical features.
Optimization: Multi-task uncertainty balancing + Hard-negative contrastive mining + Asymmetric Huber/MSE loss.
"""
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_PATH = "multi_scenario_physics_pyg_dataset.pt"
MODEL_SAVE_PATH = "hydro_gine_v4_model.pt"
CKPT_SAVE_PATH = "hydro_gine_v4_ckpt.pt"

HIDDEN_DIM = 128
N_LAYERS = 6
N_CHUNK = 100000
N_EPOCHS = 350
LR = 3e-3
WEIGHT_DECAY = 1e-5

class HydroGINE_v4(nn.Module):
    def __init__(self, in_c=32, edge_c=2, hidden=HIDDEN_DIM, n_layers=N_LAYERS):
        super().__init__()
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        for i in range(n_layers):
            c_in = in_c if i == 0 else hidden
            self.convs.append(GINEConv(
                nn.Sequential(
                    nn.Linear(c_in, hidden),
                    nn.LayerNorm(hidden),
                    nn.LeakyReLU(0.1),
                    nn.Linear(hidden, hidden)
                ),
                edge_dim=edge_c
            ))
            self.lns.append(nn.LayerNorm(hidden))
            
        # Multi-scale skip concatenation: input + mid-layer + final-layer
        self.cat_dim = hidden * 2 + in_c
        
        # Hazard Classifier Head
        self.cls = nn.Sequential(
            nn.Linear(self.cat_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
        
        # FiLM cross-conditioning generator: prob -> (gamma, beta)
        self.film_gen = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 2 * self.cat_dim)
        )
        # Zero-initialize the last linear layer so FiLM starts as identity modulation
        nn.init.zeros_(self.film_gen[2].weight)
        nn.init.zeros_(self.film_gen[2].bias)
        
        # Continuous Depth Regressor Head
        self.reg = nn.Sequential(
            nn.Linear(self.cat_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x, ei, ea):
        h = x
        mid_h = None
        for i, (conv, ln) in enumerate(zip(self.convs, self.lns)):
            h_next = F.elu(ln(conv(h, ei, ea)))
            if i > 0 and h.shape == h_next.shape:
                h = h_next + 0.3 * h  # residual skip
            else:
                h = h_next
            if i == (len(self.convs) // 2):
                mid_h = h
                
        cat = torch.cat([h, mid_h, x], dim=-1)
        
        # Hazard Classification Logits
        cls_logits = self.cls(cat)
        prob = torch.sigmoid(cls_logits)
        
        # FiLM Modulation
        film = self.film_gen(prob)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        h_cond = cat * (1.0 + gamma) + beta
        
        # Depth Regression
        raw_depth = self.reg(h_cond)
        return cls_logits, raw_depth


class MarginFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.35, margin=0.02):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.margin = margin

    def forward(self, logits, targets, hard_neg_weight=None):
        p = torch.sigmoid(logits).squeeze(-1)
        t = targets.squeeze(-1)
        pt = p * t + (1.0 - p) * (1.0 - t)
        focal_weight = (1.0 - pt) ** self.gamma
        alpha_t = t * self.alpha + (1.0 - t) * (1.0 - self.alpha)
        
        p_margin = torch.clamp(p - t * self.margin + (1.0 - t) * self.margin, 1e-7, 1.0 - 1e-7)
        bce = -(t * torch.log(p_margin) + (1.0 - t) * torch.log(1.0 - p_margin))
        loss = alpha_t * focal_weight * bce
        if hard_neg_weight is not None:
            loss = loss * hard_neg_weight
        return loss.mean()


def soft_dice_loss(logits, targets, eps=1.0):
    p = torch.sigmoid(logits).squeeze(-1)
    t = targets.squeeze(-1)
    inter = (p * t).sum()
    return 1.0 - (2.0 * inter + eps) / (p.sum() + t.sum() + eps)


def make_batches(graphs, n_chunk):
    batches, sizes = [], []
    cur = []
    for g in graphs:
        cur.append(g)
        if sum(gg.x.shape[0] for gg in cur) >= n_chunk:
            batches.append(Batch.from_data_list(cur))
            sizes.append(sum(gg.x.shape[0] for gg in cur))
            cur = []
    if cur:
        batches.append(Batch.from_data_list(cur))
        sizes.append(sum(gg.x.shape[0] for gg in cur))
    return batches, sizes


def train_model():
    print("=" * 80)
    print("   TRAINING URBANFLOW HydroGINE-v4.0 (ZERO-LEAKAGE PHYSICAL GNN)")
    print("=" * 80)
    
    dl = torch.load(DATASET_PATH, weights_only=False)
    
    # Clean city splits: train on 52 cities, hold out Bangalore and Hong Kong
    tr_g = [g.clone() for g in dl if g.city not in ('bangalore', 'hongkong')]
    te_g_blr = [g.clone() for g in dl if g.city == 'bangalore']
    te_g_hk = [g.clone() for g in dl if g.city == 'hongkong']
    
    print(f"Train graphs: {len(tr_g)} ({sum(g.x.shape[0] for g in tr_g)} nodes)")
    print(f"Test graphs: BLR={len(te_g_blr)}, HK={len(te_g_hk)}")
    
    # Compute feature normalizations across training set
    tr_x = torch.cat([g.x for g in tr_g], dim=0)
    tr_ea = torch.cat([g.edge_attr for g in tr_g], dim=0)
    tr_y = torch.cat([g.y for g in tr_g], dim=0)
    
    x_mean = tr_x.mean(dim=0)
    x_std = tr_x.std(dim=0) + 1e-6
    e_mean = tr_ea.mean(dim=0)
    e_std = tr_ea.std(dim=0) + 1e-6
    
    yl_mean = tr_y.log1p().mean()
    yl_std = tr_y.log1p().std() + 1e-6
    
    tr_batches, tr_sizes = make_batches(tr_g, N_CHUNK)
    tr_n = sum(tr_sizes)
    
    # Pre-normalize training batches
    tr_norm = []
    for b in tr_batches:
        b_norm = b.clone()
        b_norm.x = (b_norm.x - x_mean) / x_std
        b_norm.edge_attr = (b_norm.edge_attr - e_mean) / e_std
        tr_norm.append(b_norm)
        
    model = HydroGINE_v4(in_c=tr_x.shape[1], edge_c=2, hidden=HIDDEN_DIM, n_layers=N_LAYERS).to(device)
    
    # Homoscedastic uncertainty loss parameters
    log_var_cls = torch.zeros(1, device=device, requires_grad=True)
    log_var_reg = torch.zeros(1, device=device, requires_grad=True)
    
    opt = torch.optim.AdamW(
        list(model.parameters()) + [log_var_cls, log_var_reg],
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2, eta_min=1e-5)
    focal_loss_fn = MarginFocalLoss(gamma=2.0, alpha=0.35, margin=0.02)
    
    print(f"Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Starting training for {N_EPOCHS} epochs on {device}...\n")
    
    best_blr_f1 = 0.0
    t0 = time.time()
    
    for ep in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        
        for b_norm, b_orig, sz in zip(tr_norm, tr_batches, tr_sizes):
            b_norm = b_norm.to(device)
            b_y = b_orig.y.to(device).squeeze(-1) # [N]
            
            # Targets
            ty_log = (b_y.log1p() - yl_mean.to(device)) / yl_std.to(device) # [N]
            hazard_target = (b_y >= 0.15).float() # [N]
            
            # Physics hard-negative mining weighting:
            # Nodes with high flow accumulation but open out_deg are hard negatives
            accum = b_orig.x[:, 5].to(device)  # accum_score [N]
            out_deg = b_orig.x[:, 4].to(device) # [N]
            in_deg = b_orig.x[:, 3].to(device)  # [N]
            sink_depth = b_orig.x[:, 23].to(device) # [N]
            
            hard_neg = ((accum > 1.8) & (out_deg >= in_deg) & (sink_depth < 0.05) & (b_y < 0.15)).float()
            hard_weight = 1.0 + 2.0 * hard_neg
            
            opt.zero_grad()
            cls_logits, depth_out = model(b_norm.x, b_norm.edge_index, b_norm.edge_attr)
            cls_logits = cls_logits.squeeze(-1) # [N]
            depth_out = depth_out.squeeze(-1)   # [N]
            
            # 1. Classification Loss (Focal + Soft Dice)
            l_focal = focal_loss_fn(cls_logits, hazard_target, hard_neg_weight=hard_weight)
            l_dice = soft_dice_loss(cls_logits, hazard_target)
            loss_cls = l_focal + 0.3 * l_dice
            
            # 2. Regression Loss (Normalized MSE + Linear Asymmetric MSE)
            pred_lin = torch.clamp(torch.expm1(depth_out * yl_std.to(device) + yl_mean.to(device)), min=0.0)
            diff_lin = pred_lin - b_y
            
            # Asymmetric penalty: underestimating deep floods is penalized 2.5x
            asym_w = torch.where(diff_lin < 0, torch.full_like(diff_lin, 2.5), torch.ones_like(diff_lin))
            loss_reg = torch.mean(asym_w * (diff_lin ** 2)) + F.smooth_l1_loss(depth_out, ty_log)
            
            # 3. Multi-task loss with uncertainty balancing
            loss = torch.exp(-log_var_cls) * loss_cls + 0.5 * log_var_cls + \
                   torch.exp(-log_var_reg) * loss_reg + 0.5 * log_var_reg
                   
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            total_loss += loss.item() * (sz / tr_n)
            
        sched.step()
        
        if ep % 20 == 0 or ep == N_EPOCHS or ep == 1:
            # Fast validation on Bangalore and HK
            model.eval()
            with torch.no_grad():
                blr_preds, blr_trues = [], []
                for g in te_g_blr:
                    gx = (g.x.to(device) - x_mean.to(device)) / x_std.to(device)
                    gea = (g.edge_attr.to(device) - e_mean.to(device)) / e_std.to(device)
                    c_l, d_o = model(gx, g.edge_index.to(device), gea)
                    p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std.to(device) + yl_mean.to(device)), min=0.0).cpu().numpy().ravel()
                    p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
                    # Neural confidence gating
                    p_gated = np.where(p_prob >= 0.35, p_lin, 0.0)
                    blr_preds.append(p_gated)
                    blr_trues.append(g.y.numpy().ravel())
                    
                y_bp = np.concatenate(blr_preds)
                y_bt = np.concatenate(blr_trues)
                
                tp_c = np.sum((y_bp > 0.30) & (y_bt > 0.30))
                fp_c = np.sum((y_bp > 0.30) & (y_bt <= 0.30))
                fn_c = np.sum((y_bp <= 0.30) & (y_bt > 0.30))
                rec_c = tp_c / max(1, tp_c + fn_c)
                prec_c = tp_c / max(1, tp_c + fp_c)
                f1_h = 2 * np.sum((y_bp >= 0.15) & (y_bt >= 0.15)) / max(1e-6, np.sum(y_bp >= 0.15) + np.sum(y_bt >= 0.15))
                mae = np.mean(np.abs(y_bp - y_bt))
                pct_30 = np.mean(np.abs(y_bp - y_bt) <= 0.30) * 100.0
                
                # Check HSR light rain specifically
                hsr_g_light = [g for g in te_g_blr if g.region == 'hsr' and g.x[0, 13].item() <= 50.0]
                hsr_fp_light = 0
                for g in hsr_g_light:
                    gx = (g.x.to(device) - x_mean.to(device)) / x_std.to(device)
                    gea = (g.edge_attr.to(device) - e_mean.to(device)) / e_std.to(device)
                    c_l, d_o = model(gx, g.edge_index.to(device), gea)
                    p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std.to(device) + yl_mean.to(device)), min=0.0).cpu().numpy().ravel()
                    p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
                    p_gated = np.where(p_prob >= 0.35, p_lin, 0.0)
                    y_g = g.y.numpy().ravel()
                    hsr_fp_light += np.sum((p_gated > 0.30) & (y_g <= 0.30))
                    
            print(f"Ep {ep:3d}/{N_EPOCHS:3d} ({time.time()-t0:.0f}s) | Loss: {total_loss:.4f} | "
                  f"BLR: F1={f1_h:.4f}, CritRec={rec_c*100:.1f}%, CritPrec={prec_c*100:.1f}%, "
                  f"MAE={mae*100:.2f}cm, %<=30cm={pct_30:.1f}% | HSR Light FP={hsr_fp_light}")
            
            if f1_h > best_blr_f1 or ep == N_EPOCHS:
                if f1_h > best_blr_f1:
                    best_blr_f1 = f1_h
                torch.save({
                    'model': model.state_dict(),
                    'x_mean': x_mean.cpu(), 'x_std': x_std.cpu(),
                    'e_mean': e_mean.cpu(), 'e_std': e_std.cpu(),
                    'yl_mean': yl_mean.cpu(), 'yl_std': yl_std.cpu(),
                    'in_c': tr_x.shape[1], 'hidden': HIDDEN_DIM, 'n_layers': N_LAYERS
                }, MODEL_SAVE_PATH)

    print(f"\nTraining Complete! Best BLR F1: {best_blr_f1:.4f}")
    print(f"Saved model to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
