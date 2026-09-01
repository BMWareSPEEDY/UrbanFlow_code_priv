import sys, os
sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")
os.chdir(r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")
import torch, numpy as np
from production import ProductionEnsemble

print("Loading MoE ensemble (iter14 + iter17)...")
ens = ProductionEnsemble()
print(f"  Model14: in_c=26 | Model17: in_c=28 | Device: {ens.device}")

dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_full22_pyg_dataset.pt", weights_only=False)
blr_graphs = [g for g in dl if g.city == 'bangalore']
hk_graphs = [g for g in dl if g.city == 'hongkong']

THR = 0.15
def calc_f1(y, p):
    tp = np.sum((p >= THR) & (y >= THR))
    fp = np.sum((p >= THR) & (y < THR))
    fn = np.sum((p < THR) & (y >= THR))
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    return 2*pr*rc / max(1e-9, pr+rc), pr, rc

print("\n=== MoE ENSEMBLE FINAL RESULTS (I>=150) ===")
for city, graphs in [("BLR", blr_graphs), ("HK", hk_graphs)]:
    gated, raw, prob = ens.predict(graphs, [150.0]*len(graphs))
    y = np.concatenate([g.y.numpy().ravel() for g in graphs])
    i_arr = np.concatenate([np.full(g.y.shape[0], g.x[0, 15].item()) for g in graphs])
    m = i_arr >= 150.0
    f, pr, rc = calc_f1(y[m], gated[m])
    pf = 100*(gated[m] >= THR).mean(); yf = 100*(y[m] >= THR).mean()
    print(f"  {city}: F1={f:.4f} P={pr:.3f} R={rc:.3f} pred={pf:.1f}% actual={yf:.1f}%")

print("\n=== BLR PER-REGION ===")
gated_b, _, _ = ens.predict(blr_graphs, [150.0]*len(blr_graphs))
y_b = np.concatenate([g.y.numpy().ravel() for g in blr_graphs])
regions = np.concatenate([[g.region]*g.y.shape[0] for g in blr_graphs])
i_b = np.concatenate([np.full(g.y.shape[0], g.x[0, 15].item()) for g in blr_graphs])
m = i_b >= 150.0
for rg in ['bellandur', 'ecity', 'hsr', 'koramangala', 'whitefield']:
    rm = (regions == rg) & m
    f, _, _ = calc_f1(y_b[rm], gated_b[rm])
    print(f"  {rg:<13} F1 {f:.4f}")

gated_h, _, _ = ens.predict(hk_graphs, [150.0]*len(hk_graphs))
y_h = np.concatenate([g.y.numpy().ravel() for g in hk_graphs])
i_h = np.concatenate([np.full(g.y.shape[0], g.x[0, 15].item()) for g in hk_graphs])
m_h = i_h >= 150.0
p_pool = np.concatenate([gated_b[m], gated_h[m_h]])
y_pool = np.concatenate([y_b[m], y_h[m_h]])
fp, _, _ = calc_f1(y_pool, p_pool)
print(f"\n  POOLED: F1={fp:.4f}")

print("\n=== MULTI-INTENSITY TEST ===")
for I in [50.0, 150.0, 300.0]:
    g_b, _, _ = ens.predict(blr_graphs, [I]*len(blr_graphs))
    g_h, _, _ = ens.predict(hk_graphs, [I]*len(hk_graphs))
    i_b2 = np.concatenate([np.full(g.y.shape[0], g.x[0, 15].item()) for g in blr_graphs])
    i_h2 = np.concatenate([np.full(g.y.shape[0], g.x[0, 15].item()) for g in hk_graphs])
    m2 = i_b2 == I
    m2h = i_h2 == I
    if m2.sum() == 0 or m2h.sum() == 0:
        print(f"  I={I:5.0f}: no graphs at this intensity")
        continue
    fb, _, _ = calc_f1(y_b[m2], g_b[m2])
    fh, _, _ = calc_f1(y_h[m2h], g_h[m2h])
    print(f"  I={I:5.0f}: BLR={fb:.4f} (n={m2.sum()}) HK={fh:.4f} (n={m2h.sum()})")

print("\nPRODUCTION SMOKE TEST PASSED")
