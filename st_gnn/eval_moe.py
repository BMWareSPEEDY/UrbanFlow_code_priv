"""Evaluate MoE ensemble vs iter14 and iter17 individually."""
import sys, os
sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")
os.chdir(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")

import torch, numpy as np
from production import ProductionEnsemble

THR = 0.15

def f1_score(y, p):
    tp = np.sum((p >= THR) & (y >= THR))
    fp = np.sum((p >= THR) & (y < THR))
    fn = np.sum((p < THR) & (y >= THR))
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    return 2*pr*rc / max(1e-9, pr+rc), pr, rc

print("Loading MoE ensemble...")
ens = ProductionEnsemble()

dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_full22_pyg_dataset.pt", weights_only=False)
blr_graphs = [g for g in dl if g.city == 'bangalore']
hk_graphs = [g for g in dl if g.city == 'hongkong']
print(f"BLR: {len(blr_graphs)} graphs | HK: {len(hk_graphs)} graphs")

# Get ground truth
y_blr = np.concatenate([g.y.numpy().ravel() for g in blr_graphs])
y_hk = np.concatenate([g.y.numpy().ravel() for g in hk_graphs])

# Get node-level intensity and region labels
i_blr = np.concatenate([np.full(g.y.shape[0], g.x[0, 15].item()) for g in blr_graphs])
i_hk = np.concatenate([np.full(g.y.shape[0], g.x[0, 15].item()) for g in hk_graphs])
e2_blr = np.concatenate([g.x[:, 17].numpy().ravel() for g in blr_graphs])
e2_hk = np.concatenate([g.x[:, 17].numpy().ravel() for g in hk_graphs])

# Region labels for BLR
regions = []
for g in blr_graphs:
    regions.extend([g.region] * g.y.shape[0])
regions = np.array(regions)

print("\n=== MoE ENSEMBLE vs INDIVIDUAL MODELS (I>=150) ===")

for city_name, graphs, y_all, i_all in [("BLR", blr_graphs, y_blr, i_blr),
                                         ("HK", hk_graphs, y_hk, i_hk)]:
    gated_moe, raw_moe, prob_moe = ens.predict(graphs, [150.0]*len(graphs))
    m = i_all >= 150.0
    f, pr, rc = f1_score(y_all[m], gated_moe[m])
    pf = 100*(gated_moe[m] >= THR).mean()
    yf = 100*(y_all[m] >= THR).mean()
    print(f"\n  {city_name} MoE: F1={f:.4f} P={pr:.3f} R={rc:.3f} pred={pf:.1f}% actual={yf:.1f}% gap={pf-yf:+.1f}%")

# Per-region BLR breakdown
print("\n=== BLR PER-REGION (MoE, I>=150) ===")
gated_moe, _, _ = ens.predict(blr_graphs, [150.0]*len(blr_graphs))
for rg in ['bellandur', 'ecity', 'hsr', 'koramangala', 'whitefield']:
    rm = regions == rg
    m = rm & (i_blr >= 150.0)
    f, pr, rc = f1_score(y_blr[m], gated_moe[m])
    print(f"  {rg:<13} F1 {f:.4f} (P {pr:.3f} R {rc:.3f})")

# Pooled
m_blr = i_blr >= 150.0
m_hk = i_hk >= 150.0
f_blr, _, _ = f1_score(y_blr[m_blr], gated_moe[m_blr])
gated_hk, _, _ = ens.predict(hk_graphs, [150.0]*len(hk_graphs))
f_hk, _, _ = f1_score(y_hk[m_hk], gated_hk[m_hk])
y_pool = np.concatenate([y_blr[m_blr], y_hk[m_hk]])
p_pool = np.concatenate([gated_moe[m_blr], gated_hk[m_hk]])
f_pool, _, _ = f1_score(y_pool, p_pool)
print(f"\n  BLR pooled:  {f_blr:.4f}")
print(f"  HK pooled:   {f_hk:.4f}")
print(f"  POOLED:      {f_pool:.4f}")

# Compare with individual models
print("\n=== COMPARISON TABLE ===")
for name, path in [("iter14", r"preds_iter14_full.pt"), ("iter17", r"preds_iter17_full.pt")]:
    d = torch.load(path, weights_only=False)
    pred, y, i, city = d['pred'], d['y'], d['intensity'], d['city']
    m_i = i >= 150.0
    f_b, _, _ = f1_score(y[(city=='bangalore')&m_i], pred[(city=='bangalore')&m_i])
    f_h, _, _ = f1_score(y[(city=='hongkong')&m_i], pred[(city=='hongkong')&m_i])
    f_p, _, _ = f1_score(y[m_i], pred[m_i])
    print(f"  {name}: BLR={f_b:.4f} HK={f_h:.4f} POOLED={f_p:.4f}")
print(f"  MoE:    BLR={f_blr:.4f} HK={f_hk:.4f} POOLED={f_pool:.4f}")
