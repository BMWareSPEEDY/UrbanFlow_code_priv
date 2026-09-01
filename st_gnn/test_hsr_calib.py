"""CPU-only quick test: hsr-class post-hoc calibration via city clustering.
Fit per-cluster affine (slope/intercept) on 23 training cities using leave-one-city-out
at cluster level; apply matching cluster to Bangalore. No Bangalore labels used for fitting.
"""
import sys
import numpy as np
import torch
import json

BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"
sys.path.insert(0, BASE)

dl = torch.load(BASE + r"\multi_scenario_pyg_dataset.pt", weights_only=False)

# --- rebuild Bangalore preds + features in te_g order ---
te_g = [g for g in dl if g.city == 'bangalore']
y = np.concatenate([g.y.numpy().ravel() for g in te_g])
inten = np.concatenate([np.full(g.y.shape[0], float(g.rain_intensity)) for g in te_g])
x = np.concatenate([g.x.numpy() for g in te_g])
pred0 = np.load(BASE + r"\st_gnn\baseline_preds.npy") if False else torch.load(BASE + r"\st_gnn\baseline_preds.pt", weights_only=False)['pred']

# cluster features per graph: region-level stats of (elev_std2=17, imp=3, is_sink=8)
feat_names = {17: 'elev_std2', 3: 'imp', 8: 'is_sink', 19: 'surcharge', 20: 'path_cap', 0: 'rel_x', 9: 'max_in_grade'}
train_regions = {}
for g in dl:
    if g.city == 'bangalore':
        continue
    key = (g.city, g.region)
    xn = g.x.numpy()
    train_regions.setdefault(key, []).append(xn)

reg_stats = {}
for key, xs in train_regions.items():
    X = np.concatenate(xs)
    reg_stats[key] = {'elev_std2_mean': X[:, 17].mean(), 'imp_mean': X[:, 3].mean(),
                      'sink_rate': X[:, 8].mean()}

# simple rule-based clusters: flat-low-relief (elev_std2<1.0), hilly (>=1.0)
clusters = {'flat': [], 'hilly': []}
for key, s in reg_stats.items():
    clusters['hilly' if s['elev_std2_mean'] >= 1.0 else 'flat'].append(key)
print("hilly train regions:", len(clusters['hilly']), clusters['hilly'][:5])
print("flat train regions:", len(clusters['flat']))

# per-intensity affine calibration per cluster (fit on cluster cities' nodes)
def fit_calib(cities, ints):
    cal = {}
    for i in np.unique(ints):
        yc, pc = [], []
        for (c, r) in cities:
            for g in train_regions[(c, r)]:
                pass
        for (c, r) in cities:
            for xs in train_regions[(c, r)]:
                pass
    return cal

# simpler: fit per-cluster per-intensity affine on all non-Bangalore nodes
calib = {}
for cl in clusters:
    node_y, node_p = [], []
    node_int = []
    for (c, r) in clusters[cl]:
        for xs in train_regions[(c, r)]:
            node_y.append(xs.y.numpy().ravel())
            node_p.append(xs.y.numpy().ravel() * 0)  # placeholder - need model preds for train!
    # cannot get train preds without the model -> skip per-node fit; use per-region mean shift

# --- per-region mean-depth check: use SWMM y as reference ---
te_reg = [g.region for g in te_g]
reg_mean_y, reg_count = {}, {}
for rg, yy in zip(te_reg, y):
    reg_mean_y[rg] = reg_mean_y.get(rg, 0.0) + yy
    reg_count[rg] = reg_count.get(rg, 0) + 1
for rg in reg_mean_y:
    reg_mean_y[rg] /= reg_count[rg]
print("Bangalore region mean true depth:", {k: round(v, 3) for k, v in sorted(reg_mean_y.items())})

# --- train-side region mean depth per intensity (for comparison) ---
tr_reg_y = {}
for g in dl:
    if g.city == 'bangalore':
        continue
    key = (g.city, g.region)
    tr_reg_y.setdefault(key, []).append(g.y.numpy().ravel())
print("hilly train regions mean depth @I=300:", [
    (k[0], round(float(np.concatenate(vv)[inten if False else slice(None)]).mean() if False else np.concatenate([v for v in vv]).mean()), 3)
    for k, vv in list(tr_reg_y.items())[:3]
])
