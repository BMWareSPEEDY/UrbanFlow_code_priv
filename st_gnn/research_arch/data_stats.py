import torch
import numpy as np
import sys
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

dl = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)

INTENSITIES = [20.0, 50.0, 80.0, 120.0, 150.0, 200.0, 250.0, 300.0]

def y_stats(y, tag):
    y = np.asarray(y).ravel()
    n = len(y)
    out = dict(n=n, mean=y.mean(), std=y.std(), p50=np.percentile(y, 50),
               p90=np.percentile(y, 90), p99=np.percentile(y, 99), mx=y.max(),
               fgt05=np.mean(y > 0.5) * 100, fgt08=np.mean(y > 0.8) * 100,
               fgt12=np.mean(y > 1.2) * 100, frac0=np.mean(y < 0.01) * 100)
    return out

def fmt(d):
    return (f"n={d['n']:8d} mean={d['mean']:.3f} std={d['std']:.3f} P90={d['p90']:.3f} "
            f"P99={d['p99']:.3f} max={d['mx']:.3f} >0.5m={d['fgt05']:5.1f}% >0.8m={d['fgt08']:5.1f}% "
            f">1.2m={d['fgt12']:5.1f}% dry={d['frac0']:5.1f}%")

tr = defaultdict(list)
te = defaultdict(list)
per_city = defaultdict(list)

for g in dl:
    city = g.city
    I = float(g.rain_intensity)
    y = g.y.numpy().ravel()
    if city == 'bangalore':
        te[I].append(y)
    else:
        tr[I].append(y)
    per_city[city].append((I, y))

print("=== PER-INTENSITY y STATS: TRAIN (23 non-Bangalore cities) ===")
tr_tab = {}
for I in INTENSITIES:
    y = np.concatenate(tr[I]) if tr[I] else np.array([])
    d = y_stats(y, 'tr')
    tr_tab[I] = d
    print(f"  I={I:6.0f}: {fmt(d)}")

print("\n=== PER-INTENSITY y STATS: TEST (Bangalore) ===")
te_tab = {}
for I in INTENSITIES:
    y = np.concatenate(te[I]) if te[I] else np.array([])
    d = y_stats(y, 'te')
    te_tab[I] = d
    print(f"  I={I:6.0f}: {fmt(d)}")

print("\n=== IN-SAMPLE CHECK: test-vs-train depth distribution at each intensity ===")
for I in INTENSITIES:
    a, b = tr_tab[I], te_tab[I]
    if b['n'] == 0 or a['n'] == 0:
        continue
    ratio = b['mean'] / max(a['mean'], 1e-9)
    print(f"  I={I:6.0f}: test/train mean {ratio:5.2f}  | std {b['std']:.3f} vs {a['std']:.3f} | "
          f"P90 {b['p90']:.3f} vs {a['p90']:.3f} | >0.8m {b['fgt08']:.1f}% vs {a['fgt08']:.1f}%")

print("\n=== PER-CITY COVERAGE (train cities) ===")
print(f"{'city':<14}{'nodes':>8}{'mean@300':>9}{'P90@300':>9}{'>0.5@300':>9}{'>0.8@300':>9}{'dry@300':>9}{'mean_allI':>10}")
rows = []
for city, items in per_city.items():
    if city == 'bangalore':
        continue
    y300 = np.concatenate([y for I, y in items if I == 300.0])
    all_y = np.concatenate([y for _, y in items])
    rows.append((city, len(all_y), y300.mean(), np.percentile(y300, 90),
                 np.mean(y300 > 0.5) * 100, np.mean(y300 > 0.8) * 100,
                 np.mean(y300 < 0.01) * 100, all_y.mean()))
rows.sort(key=lambda r: -r[2])
for r in rows:
    print(f"{r[0]:<14}{r[1]:>8}{r[2]:>9.3f}{r[3]:>9.3f}{r[4]:>9.1f}{r[5]:>9.1f}{r[6]:>9.1f}{r[7]:>10.3f}")

print("\n=== BANGALORE per-region nodes ===")
for g in dl:
    if g.city == 'bangalore' and g.rain_intensity == 300.0:
        print(f"  {g.region:<14} {g.x.shape[0]:>7} nodes")

tot_tr = sum(len(v) for v in tr.values())
tot_te = sum(len(v) for v in te.values())
print(f"\nTOTAL train nodes: {tot_tr} | test nodes: {tot_te}")

# synthetic: same-feature ceiling check - variance of y explained by (intensity, mean-per-city)
print("\n=== FEATURE-CEILING PROBE: R2 of y explained by intensity + city-mean only (train) ===")
all_y = np.concatenate([y for v in tr.values() for y in v])
all_I = np.concatenate([np.full(len(y), I) for I in INTENSITIES for y in tr[I]])
pred = np.zeros_like(all_y)
for city, items in per_city.items():
    if city == 'bangalore':
        continue
    for I, y in items:
        m = (all_I == I)
        pred[m] = y.mean()  # will overwrite; fine for probe
# simpler: group mean per (city,intensity)
pred = np.zeros_like(all_y)
for city, items in per_city.items():
    if city == 'bangalore':
        continue
    for I, y in items:
        m = (all_I == I) & (np.concatenate([np.full(len(yy), c) for c, ii, yy in [(city, I, y)]]))
# skip complex; do per-intensity mean only
for I in INTENSITIES:
    m = all_I == I
    pred[m] = all_y[m].mean()
ss_res = np.sum((all_y - pred) ** 2)
ss_tot = np.sum((all_y - all_y.mean()) ** 2)
print(f"  R2 of intensity-mean-only predictor (train): {1 - ss_res / ss_tot:.4f}")