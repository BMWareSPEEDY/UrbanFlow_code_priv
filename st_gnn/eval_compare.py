import torch, numpy as np

THR = 0.15

def f1(y, p, thr=THR):
    tp = np.sum((p >= thr) & (y >= thr))
    fp = np.sum((p >= thr) & (y < thr))
    fn = np.sum((p < thr) & (y >= thr))
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    return 2*pr*rc / max(1e-9, pr+rc), pr, rc

iters = {}
for name, path in [
    ('iter14', r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter14_full.pt"),
    ('iter15', r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter15_full.pt"),
    ('iter17', r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\preds_iter17_full.pt"),
]:
    try:
        d = torch.load(path, weights_only=False)
        iters[name] = d
    except:
        print(f"{name}: not found")

print("=== COMPARISON: ALL ITERATIONS (I>=150) ===")
for name, d in iters.items():
    pred, y, i, city, region = d['pred'], d['y'], d['intensity'], d['city'], d['region']
    m_i = i >= 150.0
    f_blr, _, _ = f1(y[(city=='bangalore')&m_i], pred[(city=='bangalore')&m_i])
    f_hk, _, _ = f1(y[(city=='hongkong')&m_i], pred[(city=='hongkong')&m_i])
    f_pool, _, _ = f1(y[m_i], pred[m_i])
    parts = []
    for rg in ['bellandur', 'ecity', 'hsr', 'koramangala', 'whitefield']:
        rm = np.array([region[k] == rg for k in range(len(region))])
        m = rm & m_i
        fv, _, _ = f1(y[m], pred[m])
        parts.append(f"{rg}={fv:.4f}")
    print(f"  {name}: BLR {f_blr:.4f} | HK {f_hk:.4f} | POOLED {f_pool:.4f}")
    sep = " | "
    print(f"    {sep.join(parts)}")

print()
print("=== PER-INTENSITY (iter17) ===")
d = iters['iter17']
pred, y, i, city = d['pred'], d['y'], d['intensity'], d['city']
for I in [20.0, 50.0, 80.0, 120.0, 150.0, 200.0, 250.0, 300.0]:
    m_b = (city == 'bangalore') & (i == I)
    m_h = (city == 'hongkong') & (i == I)
    f_b, _, _ = f1(y[m_b], pred[m_b])
    f_h, _, _ = f1(y[m_h], pred[m_h])
    f_p, _, _ = f1(y[m_b|m_h], pred[m_b|m_h])
    print(f"  I={I:5.0f}: BLR {f_b:.4f} | HK {f_h:.4f} | POOLED {f_p:.4f}")

print()
print("=== BLR PER-INTENSITY COMPARISON ===")
for name, d in iters.items():
    pred, y, i, city = d['pred'], d['y'], d['intensity'], d['city']
    parts = []
    for I in [150.0, 200.0, 250.0, 300.0]:
        m = (city == 'bangalore') & (i == I)
        fv, _, _ = f1(y[m], pred[m])
        parts.append(f"{I:.0f}={fv:.4f}")
    sep = " | "
    print(f"  {name}: {sep.join(parts)}")

print()
print("=== CALIBRATION (iter17, I>=150) ===")
d = iters['iter17']
pred, y, i, city = d['pred'], d['y'], d['intensity'], d['city']
prob = d['prob']
m_i = i >= 150.0
for c in ['bangalore', 'hongkong']:
    m = (city == c) & m_i
    pf = 100*(pred[m] >= THR).mean()
    yf = 100*(y[m] >= THR).mean()
    print(f"  {c}: pred_flood={pf:.1f}% actual_flood={yf:.1f}% gap={pf-yf:+.1f}%")
    for I in [150.0, 200.0, 250.0, 300.0]:
        mm = m & (i == I)
        if mm.sum() > 0:
            pf2 = 100*(pred[mm] >= THR).mean()
            yf2 = 100*(y[mm] >= THR).mean()
            print(f"    I={I:.0f}: pred={pf2:.1f}% actual={yf2:.1f}% gap={pf2-yf2:+.1f}%")
