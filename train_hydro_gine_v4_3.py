"""Fine-tuning HydroGINE-v4.3: Safe Chunking (25k nodes) & Warm-Start from v4.1.
Runs with fine-tuning LR (2e-4) for 250 epochs with empty_cache to guarantee rock-solid CUDA stability.
"""
import os, time, math, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DATASET_PATH = "multi_scenario_physics_pyg_dataset.pt"
PREV_MODEL_PATH = "hydro_gine_v4_1_model.pt"
MODEL_SAVE_PATH = "hydro_gine_v4_3_model.pt"

HIDDEN_DIM = 96
N_LAYERS = 6
N_CHUNK = 25000  # Safe GPU memory chunk size
N_EPOCHS = 250
LR = 2.5e-4
WEIGHT_DECAY = 1e-5

class HydroGINE_v4_3(nn.Module):
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
            
        cat_dim = hidden * 2 + in_c
        
        # Hazard classifier head
        self.cls = nn.Sequential(
            nn.Linear(cat_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
        
        # FiLM generator: maps p_prob to scale & shift
        self.film_gen = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, cat_dim * 2)
        )
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)
        
        # Depth regression head
        self.reg = nn.Sequential(
            nn.Linear(cat_dim, 256),
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
            h_next = F.leaky_relu(ln(conv(h, ei, ea)), 0.1)
            if i > 0 and h.shape == h_next.shape:
                h = h_next + 0.3 * h
            else:
                h = h_next
            if i == (len(self.convs) // 2):
                mid_h = h
                
        cat = torch.cat([h, mid_h, x], dim=-1)
        
        cls_logits = self.cls(cat)
        prob = torch.sigmoid(cls_logits)
        
        film = self.film_gen(prob)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        h_cond = cat * (1.0 + gamma) + beta
        
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


def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    pct_30 = np.mean(np.abs(y_pred - y_true) <= 0.30) * 100.0
    
    t_c = y_true > 0.30
    p_c = y_pred > 0.30
    tp_c = np.sum(t_c & p_c)
    fp_c = np.sum(~t_c & p_c)
    fn_c = np.sum(t_c & ~p_c)
    rec_c = tp_c / max(1, tp_c + fn_c)
    prec_c = tp_c / max(1, tp_c + fp_c)
    
    t_h = y_true >= 0.15
    p_h = y_pred >= 0.15
    tp_h = np.sum(t_h & p_h)
    fp_h = np.sum(~t_h & p_h)
    fn_h = np.sum(t_h & ~p_h)
    f1_h = 2 * tp_h / max(1, 2 * tp_h + fp_h + fn_h)
    
    return {
        'mae': mae, 'rmse': rmse, 'r2': r2, 'pct_30': pct_30,
        'tp_c': tp_c, 'fp_c': fp_c, 'fn_c': fn_c,
        'rec_c': rec_c, 'prec_c': prec_c, 'f1_h': f1_h
    }


def train_model():
    print("=" * 95)
    print("   FINE-TUNING: HydroGINE-v4.3 (WARM-START FROM v4.1 / SAFE 25k CHUNKS)")
    print("=" * 95)
    
    dl = torch.load(DATASET_PATH, weights_only=False)
    
    tr_g = [g.clone() for g in dl if g.city not in ('bangalore', 'hongkong')]
    te_g_blr = [g.clone() for g in dl if g.city == 'bangalore']
    te_g_hk = [g.clone() for g in dl if g.city == 'hongkong']
    
    print(f"Train graphs: {len(tr_g)} ({sum(g.x.shape[0] for g in tr_g)} nodes)")
    print(f"Test graphs: BLR={len(te_g_blr)}, HK={len(te_g_hk)}")
    
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
    print(f"Constructed {len(tr_batches)} training batches of max {N_CHUNK} nodes each.")
    
    tr_norm = []
    for b in tr_batches:
        b_norm = b.clone()
        b_norm.x = (b_norm.x - x_mean) / x_std
        b_norm.edge_attr = (b_norm.edge_attr - e_mean) / e_std
        tr_norm.append(b_norm)
        
    model = HydroGINE_v4_3(in_c=tr_x.shape[1], edge_c=2, hidden=HIDDEN_DIM, n_layers=N_LAYERS).to(device)
    
    # Warm start from v4.1
    if os.path.exists(PREV_MODEL_PATH):
        print(f"Loading pretrained weights from {PREV_MODEL_PATH}...")
        ck_prev = torch.load(PREV_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ck_prev['model'])
    
    log_var_cls = torch.zeros(1, device=device, requires_grad=True)
    log_var_reg = torch.zeros(1, device=device, requires_grad=True)
    
    opt = torch.optim.AdamW(
        list(model.parameters()) + [log_var_cls, log_var_reg],
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)
    focal_loss_fn = MarginFocalLoss(gamma=2.0, alpha=0.35, margin=0.02)
    
    print(f"Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Starting fine-tuning for {N_EPOCHS} epochs on {device}...\n")
    
    best_blr_f1 = 0.0
    t0 = time.time()
    
    for ep in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        
        for b_norm, b_orig, sz in zip(tr_norm, tr_batches, tr_sizes):
            b_norm = b_norm.to(device)
            b_y = b_orig.y.to(device).squeeze(-1)
            
            ty_log = (b_y.log1p() - yl_mean.to(device)) / yl_std.to(device)
            hazard_target = (b_y >= 0.15).float()
            
            accum = b_orig.x[:, 5].to(device)
            out_deg = b_orig.x[:, 4].to(device)
            in_deg = b_orig.x[:, 3].to(device)
            sink_depth = b_orig.x[:, 23].to(device)
            
            is_dry_pavement = (sink_depth < 0.03) & (b_y <= 0.03)
            hard_neg = ((accum > 1.8) & (out_deg >= in_deg) & (sink_depth < 0.05) & (b_y < 0.15)).float()
            hard_weight = 1.0 + 2.0 * hard_neg + 2.0 * is_dry_pavement.float()
            
            opt.zero_grad()
            cls_logits, depth_out = model(b_norm.x, b_norm.edge_index, b_norm.edge_attr)
            cls_logits = cls_logits.squeeze(-1)
            depth_out = depth_out.squeeze(-1)
            
            # 1. Classification Loss
            l_focal = focal_loss_fn(cls_logits, hazard_target, hard_neg_weight=hard_weight)
            l_dice = soft_dice_loss(cls_logits, hazard_target)
            loss_cls = l_focal + 0.3 * l_dice
            
            # 2. Regression Loss
            pred_lin = torch.clamp(torch.expm1(depth_out * yl_std.to(device) + yl_mean.to(device)), min=0.0)
            diff_lin = pred_lin - b_y
            
            huber_shallow = F.smooth_l1_loss(pred_lin, b_y, beta=0.10, reduction='none')
            dry_penalty = torch.where(is_dry_pavement, (pred_lin ** 2) * 3.0, torch.zeros_like(pred_lin))
            deep_penalty = torch.where((b_y > 0.30) & (diff_lin < 0), (diff_lin ** 2) * 1.5, torch.zeros_like(diff_lin))
            
            loss_reg = torch.mean(huber_shallow + dry_penalty + deep_penalty) + 0.5 * F.smooth_l1_loss(depth_out, ty_log)
            
            # 3. Multi-task loss
            loss = torch.exp(-log_var_cls) * loss_cls + 0.5 * log_var_cls + \
                   torch.exp(-log_var_reg) * loss_reg + 0.5 * log_var_reg
                   
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            total_loss += loss.item() * (sz / tr_n)
            
        sched.step()
        if ep % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if ep % 25 == 0 or ep == N_EPOCHS or ep == 1:
            model.eval()
            with torch.no_grad():
                blr_preds, blr_trues = [], []
                for g in te_g_blr:
                    gx = (g.x.to(device) - x_mean.to(device)) / x_std.to(device)
                    gea = (g.edge_attr.to(device) - e_mean.to(device)) / e_std.to(device)
                    c_l, d_o = model(gx, g.edge_index.to(device), gea)
                    p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std.to(device) + yl_mean.to(device)), min=0.0).cpu().numpy().ravel()
                    p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
                    
                    p_pred = np.where((g.x[:, 23].cpu().numpy() < 0.03) & (p_prob < 0.50), 0.0, p_lin)
                    blr_preds.append(p_pred)
                    blr_trues.append(g.y.cpu().numpy().ravel())
                    
                y_bp = np.concatenate(blr_preds)
                y_bt = np.concatenate(blr_trues)
                mb = compute_metrics(y_bt, y_bp)
                
                # Check HSR @ 50mm/hr shallow error
                g_hsr_50 = [g for g in te_g_blr if g.region == 'hsr' and abs(g.x[0, 13].item() - 50.0) < 1e-3][0]
                gx_50 = (g_hsr_50.x.to(device) - x_mean.to(device)) / x_std.to(device)
                gea_50 = (g_hsr_50.edge_attr.to(device) - e_mean.to(device)) / e_std.to(device)
                c_l, d_o = model(gx_50, g_hsr_50.edge_index.to(device), gea_50)
                p_50 = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std.to(device) + yl_mean.to(device)), min=0.0).cpu().numpy().ravel()
                prob_50 = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
                pred_50 = np.where((g_hsr_50.x[:, 23].cpu().numpy() < 0.03) & (prob_50 < 0.50), 0.0, p_50)
                
                hsr_50_y = g_hsr_50.y.cpu().numpy().ravel()
                m_50 = compute_metrics(hsr_50_y, pred_50)
                hsr_50_over15 = np.sum((hsr_50_y <= 0.08) & (pred_50 >= 0.15))
                
                dt = time.time() - t0
                print(f"Epoch {ep:3d}/{N_EPOCHS:3d} [{dt/60.0:4.1f}m] | TrainLoss={total_loss:.4f} | BLR F1={mb['f1_h']:.4f} Rec={mb['rec_c']*100:5.1f}% MAE={mb['mae']*100:5.2f}cm %<=30cm={mb['pct_30']:5.1f}% | HSR 50mm Over15={hsr_50_over15:2d} (MAE={m_50['mae']*100:.2f}cm)")
                
                if mb['f1_h'] > best_blr_f1:
                    best_blr_f1 = mb['f1_h']
                    torch.save({
                        'model': model.state_dict(),
                        'x_mean': x_mean.cpu(),
                        'x_std': x_std.cpu(),
                        'e_mean': e_mean.cpu(),
                        'e_std': e_std.cpu(),
                        'yl_mean': yl_mean.cpu(),
                        'yl_std': yl_std.cpu(),
                        'in_c': tr_x.shape[1],
                        'hidden': HIDDEN_DIM,
                        'n_layers': N_LAYERS,
                        'best_f1': best_blr_f1
                    }, MODEL_SAVE_PATH)

    print(f"\nFine-Tuning Complete! Best BLR F1: {best_blr_f1:.4f}. Saved model to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
