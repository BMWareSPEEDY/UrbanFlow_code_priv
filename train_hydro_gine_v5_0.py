"""HydroGINE-v5.0 Training Engine
Trained on Expanded Master Physics Dataset (480 graphs, 2,350,544 nodes).
Features:
- Gravity-Directional Edge Routing
- Multi-Scale Residual GINE Backbone
- Zero-Bias Asymmetric Huber Loss for Shallow/Pavement Nodes (over-prediction 1.5x)
- Power-Law Deep-Valley Weighting (dep_d > 1.5m, 2nd-order ramp to 3x)
- Margin Focal Hazard Classifier
- Zero-Shift FiLM Depth Regressor
- Strict Valley Sink Formulations
"""
import os, sys, time, math, random
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GINEConv
from torch_geometric.data import Batch

sys.stdout.reconfigure(line_buffering=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training HydroGINE-v5.0 on device: {device}")

DATASET_PATH = "expanded_master_physics_dataset.pt"
OUT_MODEL_PATH = "hydro_gine_v5_5_model.pt"
WARM_START_PATH = "hydro_gine_v5_2_model.pt"

EPOCHS = 220
LR = 3e-4
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 128
N_LAYERS = 6
N_CHUNK = 25000  # Safe CUDA memory chunk

# Deep-Valley Tail-Aware Regression Loss (Item #8):
# log(1+y) targets amplify small log-domain errors into meter-scale spikes on
# deep basins after expm1 inversion. Scenario-conditional asymmetry:
# over-prediction penalized harder on DRY cells (roads/pavement false alarms),
# under-prediction penalized harder on FLOODED cells (deep-valley misses).
# PLUS a linear-depth match-hinge gated to flooded cells ONLY (y>0.15): it
# pushes deep valleys UP into the +-15cm band without touching dry cells
# (that global version inflated OVER massively in v5.1).
LOSS_ASSYM_UP = 1.5      # extra multiplier on positive (over-prediction) residuals on DRY/shallow cells
LOSS_ASSYM_UNDER = 2.0   # extra multiplier on negative (under-prediction) residuals on FLOODED cells
DEEP_W_STRENGTH = 3.0    # max power-law boost at the deepest basins
DEEP_W_THRESH = 1.5      # dep_d (m) above which the ramp engages
DEEP_W_RANGE = 2.5       # ramp width (m) to saturation
DEEP_W_POWER = 2.0       # power-law exponent on (dep_d - thresh)
MATCH_BAND_M = 0.12      # linear meters within which no flood-gated hinge penalty
MATCH_HINGE_STRENGTH = 2.0  # multiplier on flooded-gated match-hinge residuals

class GravityGINEConv(nn.Module):
    def __init__(self, in_c, out_c, edge_c=2):
        super().__init__()
        self.conv = GINEConv(
            nn.Sequential(
                nn.Linear(in_c, out_c),
                nn.LayerNorm(out_c),
                nn.LeakyReLU(0.1),
                nn.Linear(out_c, out_c)
            ),
            edge_dim=edge_c
        )
        self.ln = nn.LayerNorm(out_c)
        
    def forward(self, h, ei, ea):
        # Asymmetric gravity gate based on edge grade (col 1 of ea)
        # Downhill (grade <= 0) gets full flow (1.0), uphill gets attenuated flow
        grade = ea[:, 1:2]
        gravity_gate = torch.sigmoid(1.0 - 5.0 * F.relu(grade))
        ea_gated = ea * gravity_gate
        return F.leaky_relu(self.ln(self.conv(h, ei, ea_gated)), 0.1)


class HydroGINE_v5(nn.Module):
    def __init__(self, in_c=32, edge_c=2, hidden=HIDDEN_DIM, n_layers=N_LAYERS):
        super().__init__()
        self.convs = nn.ModuleList()
        
        for i in range(n_layers):
            c_in = in_c if i == 0 else hidden
            self.convs.append(GravityGINEConv(c_in, hidden, edge_c))
            
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
        for i, conv in enumerate(self.convs):
            h_next = conv(h, ei, ea)
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

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        targets_m = torch.where(targets == 1.0, 1.0 - self.margin, self.margin)
        p_t = torch.where(targets == 1.0, probs, 1.0 - probs)
        alpha_t = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * ((1.0 - p_t) ** self.gamma)
        bce = F.binary_cross_entropy_with_logits(logits, targets_m, reduction='none')
        return (focal_weight * bce).mean()


def train():
    print(f"Loading expanded master physics dataset from {DATASET_PATH}...")
    dataset = torch.load(DATASET_PATH, weights_only=False)
    print(f"Loaded {len(dataset)} graphs.")
    
    # Stratified Train/Val split: held-out Bangalore district graphs & international cities for validation
    val_regions = {'hsr', 'bellandur', 'london', 'newyork'}
    train_graphs = [g for g in dataset if getattr(g, 'region', '') not in val_regions]
    val_graphs = [g for g in dataset if getattr(g, 'region', '') in val_regions]
    
    print(f"Train graphs: {len(train_graphs)} | Val graphs: {len(val_graphs)}")
    
    # Compute normalization statistics
    all_x = torch.cat([g.x for g in train_graphs], dim=0)
    all_ea = torch.cat([g.edge_attr for g in train_graphs], dim=0)
    all_y = torch.cat([g.y for g in train_graphs], dim=0)
    
    x_mean = all_x.mean(dim=0, keepdim=True).to(device)
    x_std = (all_x.std(dim=0, keepdim=True) + 1e-6).to(device)
    e_mean = all_ea.mean(dim=0, keepdim=True).to(device)
    e_std = (all_ea.std(dim=0, keepdim=True) + 1e-6).to(device)
    
    all_yl = torch.log1p(torch.clamp(all_y, min=0.0))
    yl_mean = all_yl.mean().item()
    yl_std = (all_yl.std() + 1e-6).item()
    
    print(f"Features: in_c={x_mean.shape[1]} | edge_c={e_mean.shape[1]} | yl_mean={yl_mean:.4f} | yl_std={yl_std:.4f}")
    
    model = HydroGINE_v5(in_c=x_mean.shape[1], edge_c=e_mean.shape[1]).to(device)
    
    if os.path.exists(WARM_START_PATH):
        print(f"Loading warm-start checkpoint from {WARM_START_PATH}...")
        ck = torch.load(WARM_START_PATH, map_location=device, weights_only=False)
        try:
            # Transfer compatible weights
            m_sd = model.state_dict()
            for k, v in ck['model'].items():
                if k in m_sd and m_sd[k].shape == v.shape:
                    m_sd[k] = v
            model.load_state_dict(m_sd)
            print("Warm-start weights loaded successfully!")
        except Exception as e:
            print(f"Warm-start partial transfer: {e}")
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    focal_loss = MarginFocalLoss(gamma=2.0, alpha=0.35, margin=0.02)
    
    best_val_f1 = 0.0
    best_val_match = 0.0
    no_improve_evals = 0
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(train_graphs)
        
        total_loss_accum = 0.0
        n_batches = 0
        
        # Batch graphs into safe chunks
        current_chunk = []
        current_nodes = 0
        
        for g in train_graphs:
            current_chunk.append(g)
            current_nodes += g.x.shape[0]
            
            if current_nodes >= N_CHUNK:
                batch = Batch.from_data_list(current_chunk).to(device)
                optimizer.zero_grad()
                
                gx = (batch.x - x_mean) / x_std
                gea = (batch.edge_attr - e_mean) / e_std
                
                cls_logits, raw_depth = model(gx, batch.edge_index, gea)
                
                y_true = batch.y.to(device).squeeze(-1)
                y_log = torch.log1p(torch.clamp(y_true, min=0.0))
                y_norm = (y_log - yl_mean) / yl_std
                
                # Binary hazard label (>= 0.15m)
                y_hazard = (y_true >= 0.15).float().unsqueeze(-1)
                
                # 1. Margin Focal Loss for Classification
                loss_cls = focal_loss(cls_logits, y_hazard)
                
                # 2. Asymmetric Zero-Bias Huber Loss for Depth Regression
                depth_pred_log = raw_depth.squeeze(-1)
                diff = depth_pred_log - y_norm
                
                # Huber loss with delta = 0.10
                delta = 0.10
                abs_diff = torch.abs(diff)
                huber = torch.where(abs_diff < delta, 0.5 * (diff ** 2), delta * (abs_diff - 0.5 * delta))
                # Scenario-conditional asymmetry:
                #  - Over-prediction on DRY/shallow cells (y <= 0.15): penalized harder
                #    (the road/pavement false-alarms that killed old match rate).
                #  - Under-prediction on genuinely FLOODED cells (y > 0.15): penalized
                #    harder (the deep-valley misses: pred 0.55 vs swmm 1.0m).
                #  - Everywhere else: symmetric. This stops the blanket 1.5x over-penalty
                #    from pressing ALL deep predictions down toward the shallow mean.
                asym = torch.where(
                    (diff > 0.0) & (y_true <= 0.15),
                    LOSS_ASSYM_UP,
                    torch.where(
                        (diff < 0.0) & (y_true > 0.15),
                        LOSS_ASSYM_UNDER,
                        1.0
                    )
                )
                huber = huber * asym
                
                # Weights:
                # Critical flood nodes (>= 0.30m): 3.0x
                # Advisory flood nodes (0.15-0.30m): 2.0x
                # Dry pavement hard negatives: 3.0x
                # Shallow normal: 1.0x
                # Deep valleys (dep_d > 1.5m): power-law ramp up to 3.0x extra
                sink_d = batch.x[:, 23]
                dep_d = batch.x[:, 16]
                is_dry_pavement = (y_true <= 0.03) & (sink_d < 0.03)
                
                w = torch.ones_like(y_true)
                w = torch.where(y_true >= 0.30, 3.0, w)
                w = torch.where((y_true >= 0.15) & (y_true < 0.30), 2.0, w)
                w = torch.where(is_dry_pavement, 3.0, w)
                deep_w = 1.0 + DEEP_W_STRENGTH * torch.clamp(
                    (dep_d - DEEP_W_THRESH) / DEEP_W_RANGE, min=0.0, max=1.0
                ) ** DEEP_W_POWER
                w = w * deep_w
                
                loss_reg = (w * huber).mean()
                
                # 3. Flood-Gated Linear Match-Hinge Regression Loss:
                # Applied ONLY to genuinely flooded cells (y_true > 0.15). The
                # linear-depth residual beyond MATCH_BAND_M directly maps to the
                # production +-15cm match band, so this lifts deep valleys that
                # the log-space Huber alone saturates on (residual -0.3 to -1.2m
                # at swmm 0.5-3.0m). Dry cells are untouched, avoiding the OVER
                # inflation seen with the global hinge (v5.1).
                if MATCH_HINGE_STRENGTH > 0.0:
                    pred_lin = torch.clamp(torch.expm1(depth_pred_log * yl_std + yl_mean), min=0.0)
                    res_lin = pred_lin - y_true
                    hinge_mask = (y_true > 0.15)
                    match_hinge = F.relu(torch.abs(res_lin) - MATCH_BAND_M)
                    n_flooded = hinge_mask.sum().clamp(min=1)
                    loss_match = (MATCH_HINGE_STRENGTH * hinge_mask.float() * w * match_hinge).sum() / n_flooded
                else:
                    loss_match = torch.zeros((), device=device)
                
                total_loss = loss_cls + 1.5 * loss_reg + 0.8 * loss_match
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss_accum += total_loss.item()
                n_batches += 1
                
                current_chunk = []
                current_nodes = 0
                
        scheduler.step()
        
        # Validation Evaluation every 10 epochs
        if epoch % 10 == 0 or epoch == EPOCHS:
            model.eval()
            val_tp, val_fp, val_fn, val_tn = 0, 0, 0, 0
            val_errs = []
            val_risk = 0
            val_match = 0
            
            with torch.no_grad():
                for g in val_graphs:
                    gb = g.to(device)
                    gx = (gb.x - x_mean) / x_std
                    gea = (gb.edge_attr - e_mean) / e_std
                    
                    c_l, d_o = model(gx, gb.edge_index, gea)
                    p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
                    p_depth = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0).cpu().numpy().ravel()
                    
                    y_np = gb.y.cpu().numpy().ravel()
                    pred_haz = (p_prob >= 0.35)
                    true_haz = (y_np >= 0.15)
                    
                    val_tp += np.sum(pred_haz & true_haz)
                    val_fp += np.sum(pred_haz & ~true_haz)
                    val_fn += np.sum(~pred_haz & true_haz)
                    val_tn += np.sum(~pred_haz & ~true_haz)
                    
                    val_errs.append(np.abs(p_depth - y_np))
                    
                    risk = p_depth > 0.08
                    diff = p_depth[risk] - y_np[risk]
                    val_risk += int(np.sum(risk))
                    val_match += int(np.sum(np.abs(diff) < 0.15))
                    
            prec = val_tp / max(1, val_tp + val_fp)
            rec = val_tp / max(1, val_tp + val_fn)
            f1 = 2 * prec * rec / max(1e-6, prec + rec)
            all_e = np.concatenate(val_errs)
            mae_cm = np.mean(all_e) * 100.0
            pct_30 = np.mean(all_e <= 0.30) * 100.0
            val_match_rate = (val_match / max(1, val_risk)) * 100.0
            
            elapsed = (time.time() - start_time) / 60.0
            print(f"Epoch {epoch:3d}/{EPOCHS} [{elapsed:4.1f}m] | Loss: {total_loss_accum/max(1,n_batches):.4f} | Val F1: {f1:.4f} (P={prec*100:.1f}%, R={rec*100:.1f}%) | Val Match: {val_match_rate:.1f}% | MAE: {mae_cm:.2f}cm | <=30cm: {pct_30:.1f}%")
            
            if f1 > best_val_f1:
                best_val_f1 = f1

            if val_match_rate > best_val_match:
                best_val_match = val_match_rate
                no_improve_evals = 0
                print(f"  -> Saving new best model with Val Match = {best_val_match:.1f}% to {OUT_MODEL_PATH}...")
                torch.save({
                    'model': model.state_dict(),
                    'in_c': x_mean.shape[1],
                    'hidden': HIDDEN_DIM,
                    'n_layers': N_LAYERS,
                    'x_mean': x_mean.cpu(),
                    'x_std': x_std.cpu(),
                    'e_mean': e_mean.cpu(),
                    'e_std': e_std.cpu(),
                    'yl_mean': yl_mean,
                    'yl_std': yl_std,
                    'val_f1': f1,
                    'val_match_rate': best_val_match,
                    'epoch': epoch
                }, OUT_MODEL_PATH)
            else:
                no_improve_evals += 1
                if no_improve_evals >= 9:
                    print(f"  Early stop: no val match improvement for {no_improve_evals} evaluations (best {best_val_match:.1f}%).")
                    break
                
    print(f"\nHydroGINE-v5.0 Training Complete! Best Validation Match Rate: {best_val_match:.1f}% (best hazard F1: {best_val_f1:.4f})")

if __name__ == '__main__':
    train()
