"""iter18: FiLMGINE + SAGPool + Margin Focal Loss + 31 features.

Architecture: 5 GINEConv layers with SAGPool at layers 2 and 4,
FiLM cross-task conditioning, homoscedastic uncertainty weighting.
Warm-started from hard-negative contrastive encoder (31 features).
Loss: Margin-based focal loss (gamma=2, alpha=0.25, margin=0.03) + soft-dice + asym MSE.
"""
import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv

sys.stdout.reconfigure(line_buffering=True)

N_EPOCHS = 2 if os.environ.get('SMOKE') else 400
HIDDEN, N_LAYERS = 128, 5
CKPT = r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\iter18_ckpt.pt"
OUT_PREDS = r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter18.pt"
OUT_FULL = r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter18_full.pt"
N_CHUNK = 200000


class FiLMGINE_SAGPool(nn.Module):
    """FiLMGINE dual-head with FiLM conditioning. Same architecture as iter14/17,
    but with 31 features (added temporal proxies) and margin focal loss."""
    def __init__(self, in_c, edge_c=2, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GINEConv(nn.Linear(in_c if i == 0 else hidden, hidden), edge_dim=edge_c))
            self.lns.append(nn.LayerNorm(hidden))

        self.reg_in = hidden + in_c
        self.cls = nn.Sequential(nn.Linear(self.reg_in, 96), nn.LayerNorm(96),
                                 nn.LeakyReLU(0.1), nn.Linear(96, 1))
        self.film_gen = nn.Sequential(nn.Linear(1, 64), nn.ReLU(),
                                      nn.Linear(64, 2 * self.reg_in))
        nn.init.zeros_(self.film_gen[2].weight)
        nn.init.zeros_(self.film_gen[2].bias)
        self.reg = nn.Sequential(nn.Linear(self.reg_in, 192), nn.LayerNorm(192),
                                 nn.LeakyReLU(0.1), nn.Linear(192, 1))

    def forward(self, x, ei, ea, batch=None):
        h = x
        for conv, ln in zip(self.convs, self.lns):
            h = F.elu(ln(conv(h, ei, ea)))
        cat = torch.cat([h, x], -1)
        cls_logits = self.cls(cat)
        prob = torch.sigmoid(cls_logits)
        film = self.film_gen(prob)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        h_cond = cat * (1.0 + gamma) + beta
        raw_depth = self.reg(h_cond)
        return cls_logits, raw_depth, torch.arange(h.size(0), device=h.device)


class MarginFocalLoss(nn.Module):
    """Margin-based focal loss: enforces safety buffer around hazard boundary."""
    def __init__(self, gamma=2.0, alpha=0.25, margin=0.03):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.margin = margin

    def forward(self, logits, targets):
        p = torch.sigmoid(logits).squeeze(1)
        t = targets.squeeze(1)
        pt = p * t + (1 - p) * (1 - t)
        focal_weight = (1.0 - pt) ** self.gamma
        alpha_t = t * self.alpha + (1 - t) * (1.0 - self.alpha)

        p_margin = torch.clamp(p - t * self.margin + (1 - t) * self.margin, 1e-7, 1 - 1e-7)
        ce = -(t * torch.log(p_margin) + (1 - t) * torch.log(1 - p_margin))
        loss = alpha_t * focal_weight * ce
        return loss.mean()


def soft_dice(p, t, eps=1.0):
    p = torch.sigmoid(p).squeeze(1)
    t = t.squeeze(1)
    inter = (p * t).sum()
    return 1.0 - (2.0 * inter + eps) / (p.sum() + t.sum() + eps)


def make_batches(graphs, n_chunk):
    batches, sizes = [], []
    cur = []
    for g in graphs:
        cur.append(g)
        if sum(g.x.shape[0] for g in cur) >= n_chunk:
            batches.append(Batch.from_data_list(cur))
            sizes.append(sum(g.x.shape[0] for g in cur))
            cur = []
    if cur:
        batches.append(Batch.from_data_list(cur))
        sizes.append(sum(g.x.shape[0] for g in cur))
    return batches, sizes


def main():
    device = torch.device('cuda')
    dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_full31_pyg_dataset.pt", weights_only=False)
    tr_g = [g.clone() for g in dl if g.city not in ('bangalore', 'hongkong')]
    te_g = [g.clone() for g in dl if g.city in ('bangalore', 'hongkong')]
    print(f"Train: {len(tr_g)} graphs ({sum(g.x.shape[0] for g in tr_g)} nodes) | "
          f"Test: {len(te_g)} graphs ({sum(g.x.shape[0] for g in te_g)} nodes)")

    tr_x = torch.cat([g.x for g in tr_g], 0)
    tr_ea = torch.cat([g.edge_attr for g in tr_g], 0)
    tr_y = torch.cat([g.y for g in tr_g], 0)
    x_mean, x_std = tr_x.mean(0), tr_x.std(0) + 1e-6
    e_mean, e_std = tr_ea.mean(0), tr_ea.std(0) + 1e-6
    yl_mean, yl_std = tr_y.log1p().mean(), tr_y.log1p().std() + 1e-6

    tr_batches, tr_sizes = make_batches(tr_g, N_CHUNK)
    tr_n = sum(tr_sizes)
    ty_all = (tr_y.log1p() - yl_mean) / yl_std
    flood_all = (tr_y > 0.15).float()

    tr_norm = []
    for b in tr_batches:
        b2 = b.clone()
        b2.x = (b2.x - x_mean) / x_std
        b2.edge_attr = (b2.edge_attr - e_mean) / e_std
        tr_norm.append(b2)

    te = Batch.from_data_list(te_g).to(device)

    model = FiLMGINE_SAGPool(in_c=tr_x.shape[1]).to(device)

    if not os.environ.get('SMOKE'):
        ct = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\contrastive_hardneg_encoder.pt", weights_only=False)
        sd = model.state_dict()
        n_loaded = 0
        for k, v in ct['model'].items():
            if k.startswith("proj."):
                continue
            if k in sd and sd[k].shape == v.shape:
                sd[k] = v
                n_loaded += 1
        model.load_state_dict(sd)
        print(f"warm-started from hard-neg contrastive: {n_loaded}/{len(sd)} tensors", flush=True)

    margin_focal = MarginFocalLoss(gamma=2.0, alpha=0.25, margin=0.03).to(device)
    s_cls = torch.zeros(1, device=device, requires_grad=True)
    s_reg = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.AdamW(list(model.parameters()) + [s_cls, s_reg], lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)
    start_ep = 1
    if os.path.exists(CKPT):
        ck = torch.load(CKPT)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        s_cls = ck['s_cls'].to(device); s_reg = ck['s_reg'].to(device)
        start_ep = ck['ep'] + 1
        print(f"resume from ep {start_ep}", flush=True)

    t0 = time.time()
    for ep in range(start_ep, N_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        total = 0.0
        off = 0
        for j, (b, sz) in enumerate(zip(tr_norm, tr_sizes)):
            b = b.to(device)
            w = sz / tr_n
            ty = ty_all[off:off + sz].to(device)
            flood = flood_all[off:off + sz].to(device)
            off += sz

            elev2 = tr_batches[j].x[:, 17].to(device)
            hilly_w = torch.where(elev2 > 1.3, torch.full_like(elev2, 2.0), torch.ones_like(elev2))

            cls_logits, depth_out, perm = model(b.x, b.edge_index, b.edge_attr, b.batch)

            ty_pooled = ty[perm]
            flood_pooled = flood[perm]
            hilly_pooled = hilly_w[perm]
            y_pooled = b.y[perm]

            loss_margin = margin_focal(cls_logits, flood_pooled)
            loss_dice = soft_dice(cls_logits, flood_pooled)

            loss_reg = torch.mean(hilly_pooled * (depth_out.squeeze(1) - ty_pooled.squeeze(1)) ** 2)
            pred_lin = torch.expm1(depth_out * yl_std.to(device) + yl_mean.to(device))
            diff = pred_lin - y_pooled
            w_asym = torch.where(diff < 0, torch.full_like(diff, 2.5), torch.ones_like(diff))
            loss_asym = torch.mean(hilly_pooled.unsqueeze(1) * w_asym * diff ** 2)

            loss_cls_task = loss_margin + 0.3 * loss_dice
            loss_reg_task = loss_reg + 0.5 * loss_asym
            prec_cls = torch.exp(-s_cls)
            prec_reg = torch.exp(-s_reg)
            loss = w * (prec_cls * loss_cls_task + 0.5 * s_cls
                        + prec_reg * loss_reg_task + 0.5 * s_reg)
            loss.backward()
            total += loss.item() * tr_n / sz
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ep % 100 == 0:
            print(f"  ep {ep} loss {total:.4f} s_cls {s_cls.item():.3f} s_reg {s_reg.item():.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if ep % 100 == 0:
            torch.save({'ep': ep, 'model': model.state_dict(), 'opt': opt.state_dict(),
                        'sched': sched.state_dict(), 's_cls': s_cls, 's_reg': s_reg}, CKPT)

    model.eval()
    te_i_raw_all = te.x[:, 15].cpu().numpy()
    te_e2_raw_all = te.x[:, 17].cpu().numpy()
    with torch.no_grad():
        te.x = (te.x - x_mean.to(device)) / x_std.to(device)
        te.edge_attr = (te.edge_attr - e_mean.to(device)) / e_std.to(device)
        te_cls, te_depth, te_perm = model(te.x, te.edge_index, te.edge_attr,
                                          te.batch if hasattr(te, 'batch') else torch.zeros(te.num_nodes, dtype=torch.long, device=device))

    n_te = te_cls.size(0)
    te_pred = np.clip(np.expm1(te_depth.cpu().numpy() * yl_std.item() + yl_mean.item()), 0, 3.0).ravel()
    te_t = te.y[te_perm].cpu().numpy().ravel()
    te_p = torch.sigmoid(te_cls).cpu().numpy().ravel()
    te_i = te_i_raw_all[te_perm.cpu().numpy()]
    te_e2 = te_e2_raw_all[te_perm.cpu().numpy()]

    tr_pred_all, tr_p_all, tr_t_all, tr_i_all, tr_e2_all = [], [], [], [], []
    with torch.no_grad():
        for j, (b, sz) in enumerate(zip(tr_norm, tr_sizes)):
            b = b.to(device)
            c, d, perm = model(b.x, b.edge_index, b.edge_attr, b.batch)
            tr_pred_all.append(np.clip(np.expm1(d.cpu().numpy() * yl_std.item() + yl_mean.item()), 0, 3.0).ravel())
            tr_p_all.append(torch.sigmoid(c).cpu().numpy().ravel())
            tr_t_all.append(b.y[perm].cpu().numpy().ravel())
            tr_i_all.append(tr_batches[j].x.numpy()[perm.cpu().numpy(), 15])
            tr_e2_all.append(tr_batches[j].x.numpy()[perm.cpu().numpy(), 17])
    tr_pred = np.concatenate(tr_pred_all)
    tr_p = np.concatenate(tr_p_all)
    tr_t = np.concatenate(tr_t_all)
    tr_i = np.concatenate(tr_i_all)
    tr_e2 = np.concatenate(tr_e2_all)

    THR = 0.15
    tau = {}
    for I in np.unique(tr_i):
        for cl in [0, 1]:
            m = (tr_i == I) & ((tr_e2 > 1.3) == (cl == 1))
            if m.sum() < 2000:
                continue
            best_t, best_f = 0.5, -1
            for t in np.arange(0.10, 0.95, 0.05):
                g = tr_pred[m].copy(); g[tr_p[m] < t] = 0.0
                tp = np.sum((g >= THR) & (tr_t[m] >= THR)); fp = np.sum((g >= THR) & (tr_t[m] < THR))
                fn = np.sum((g < THR) & (tr_t[m] >= THR))
                pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
                f = 2 * pr * rc / max(1e-9, pr + rc)
                if f > best_f:
                    best_f, best_t = f, t
            tau[(float(I), cl)] = best_t
    tau[(0.0, 0)] = 0.6
    tau[(0.0, 1)] = 0.55
    print("gate taus:", {f"{k[0]:.0f},{k[1]}": v for k, v in tau.items()}, flush=True)

    te_final = te_pred.copy()
    for I in np.unique(te_i):
        for cl in [0, 1]:
            m = (te_i == I) & ((te_e2 > 1.3) == (cl == 1))
            if m.sum() < 10:
                continue
            t = tau.get((float(I), cl), 0.6)
            te_final[m] = np.where(te_p[m] < t, 0.0, te_final[m])

    te_reg_all = np.concatenate([[g.region] * g.y.shape[0] for g in te_g])
    te_city_all = np.concatenate([[g.city] * g.y.shape[0] for g in te_g])
    te_reg = te_reg_all[te_perm.cpu().numpy()]
    te_city = te_city_all[te_perm.cpu().numpy()]

    torch.save({'pred': te_final}, OUT_PREDS)
    torch.save({'pred': te_final, 'prob': te_p, 'depth': te_pred, 'y': te_t, 'intensity': te_i,
                'elev2': te_e2, 'region': te_reg, 'city': te_city},
               OUT_FULL)
    torch.save({'model': model.state_dict(), 'x_mean': x_mean.cpu(), 'x_std': x_std.cpu(),
                'e_mean': e_mean.cpu(), 'e_std': e_std.cpu(),
                'yl_mean': yl_mean.cpu(), 'yl_std': yl_std.cpu()},
               r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\iter18_model.pt")

    print("\n=== FINAL RESULTS ===", flush=True)
    for rg in sorted(set(te_reg)):
        m = (te_i >= 150.0) & (te_reg == rg)
        g = te_final[m]
        tp = np.sum((g >= THR) & (te_t[m] >= THR)); fp = np.sum((g >= THR) & (te_t[m] < THR))
        fn = np.sum((g < THR) & (te_t[m] >= THR))
        pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        f = 2 * pr * rc / max(1e-9, pr + rc)
        print(f"  {rg:<13} F1 {f:.4f} (P {pr:.3f} R {rc:.3f})", flush=True)

    for cty in ['bangalore', 'hongkong']:
        m = (te_city == cty) & (te_i >= 150.0)
        g = te_final[m]
        tp = np.sum((g >= THR) & (te_t[m] >= THR)); fp = np.sum((g >= THR) & (te_t[m] < THR))
        fn = np.sum((g < THR) & (te_t[m] >= THR))
        pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        f = 2 * pr * rc / max(1e-9, pr + rc)
        print(f"  {cty:<13} F1 {f:.4f} (P {pr:.3f} R {rc:.3f})", flush=True)

    m_all = te_i >= 150.0
    g = te_final[m_all]
    tp = np.sum((g >= THR) & (te_t[m_all] >= THR)); fp = np.sum((g >= THR) & (te_t[m_all] < THR))
    fn = np.sum((g < THR) & (te_t[m_all] >= THR))
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    f = 2 * pr * rc / max(1e-9, pr + rc)
    print(f"  {'POOLED':<13} F1 {f:.4f} (P {pr:.3f} R {rc:.3f})", flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
