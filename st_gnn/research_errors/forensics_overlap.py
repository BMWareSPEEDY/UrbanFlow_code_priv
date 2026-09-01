import os
import json
import numpy as np
import torch

BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"
OUT = os.path.join(BASE, "st_gnn", "research_errors", "results")

ds = torch.load(os.path.join(BASE, "multi_scenario_pyg_dataset.pt"), weights_only=False)
bp = torch.load(os.path.join(BASE, "st_gnn", "baseline_preds.pt"), weights_only=False)
pred_all = np.asarray(bp["pred"], dtype=np.float64)

FEAT_NAMES = ["rel_x","rel_y","rel_drop","imp","manning_n","in_deg","out_deg","accum_score",
    "is_sink","max_in_grade","sag_index","hydraulic_capacity","log_area","log_imp_area","dist_frac",
    "intensity","duration","elev_std2","dep_depth","surcharge","path_cap","path_hops"]

blg = [g for g in ds if g.city == "bangalore"]
blg.sort(key=lambda g: ds.index(g))
X = np.vstack([g.x.numpy() for g in blg])
Y = np.concatenate([g.y.numpy().ravel() for g in blg])
R = np.concatenate([[g.region] * g.num_nodes for g in blg])
I = np.concatenate([[float(g.rain_intensity)] * g.num_nodes for g in blg])
err = pred_all - Y
abs_err = np.abs(err)

# all-city node table for training-overlap analysis
def city_table(city):
    gs = [g for g in ds if g.city == city]
    gs.sort(key=lambda g: ds.index(g))
    return np.vstack([g.x.numpy() for g in gs]), np.concatenate([g.y.numpy().ravel() for g in gs])

train_cities = sorted({g.city for g in ds if g.city != "bangalore"})
train_tables = {c: city_table(c) for c in train_cities}
Xt = np.vstack([t[0] for t in train_tables.values()])
Yt = np.concatenate([t[1] for t in train_tables.values()])
n_t = Xt.shape[0]

col = {n: FEAT_NAMES.index(n) for n in FEAT_NAMES}
mE = I >= 150.0
hsr = R == "hsr"
nbl = ~hsr  # bangalore non-hsr

def frac(x, m, lo, hi):
    return float(np.mean((x[m] >= lo) & (x[m] < hi)))

report = {}

# ---------- A. imp x region interaction: does imp explain hsr bias? ----------
print("=== imp bin x region: bias / MAE at I>=150 ===")
rows = []
for rb in [0.0, 0.2, 0.35, 0.5]:
    for rg, name in [(hsr, "hsr"), (nbl, "blg_non_hsr")]:
        m = mE & rg & (X[:, col["imp"]] >= rb) & (X[:, col["imp"]] < rb + 0.15 + 1e-9)
        if m.sum() >= 100:
            rows.append(dict(bin=f"[{rb:.2f},{rb+0.15:.2f})", region=name, n=int(m.sum()),
                             mae=float(np.mean(abs_err[m])), bias=float(np.mean(err[m])),
                             mean_y=float(Y[m].mean()), mean_pred=float(pred_all[m].mean()),
                             acc10=float(np.mean(abs_err[m] <= 0.1))))
            print(f"  imp[{rb:.2f},{rb+0.15:.2f}) {name:12s} n={m.sum():6d} MAE={np.mean(abs_err[m]):.4f} bias={np.mean(err[m]):+.4f} acc10={np.mean(abs_err[m]<=0.1):.3f} mean_y={Y[m].mean():.3f} pred={pred_all[m].mean():.3f}")
report["imp_x_region"] = rows

# ---------- B. 2D failing regimes ----------
print("\n=== 2D regimes at I>=150 (regime -> bias/MAE/acc10, per region group) ===")
regimes = {
    "low_imp_low_grade": (X[:, col["imp"]] < 0.2) & (X[:, col["max_in_grade"]] < 0.02),
    "low_imp_any": (X[:, col["imp"]] < 0.2),
    "flat_grade_lt_0.01": (X[:, col["max_in_grade"]] < 0.01),
    "high_logimp_lowgrade": (X[:, col["log_imp_area"]] > 1.42) & (X[:, col["max_in_grade"]] < 0.02),
    "shallow_dep_le0": (X[:, col["dep_depth"]] <= 0.014),
    "surch_high_dep0": (X[:, col["surcharge"]] > 7.0) & (X[:, col["dep_depth"]] <= 0.014),
    "elev_flat_high": (X[:, col["elev_std2"]] > 1.4),
    "grade0_neg": X[:, col["max_in_grade"]] < 0.0,
    "sink_or_cap1": (X[:, col["hydraulic_capacity"]] == 1.0),
}
r2 = {}
for nm, mask in regimes.items():
    print(f"-- {nm} --")
    for name, m_rg in [("hsr", hsr), ("blg_non_hsr", nbl), ("all_blg", np.ones_like(hsr, dtype=bool))]:
        m = mE & m_rg & mask
        if m.sum() == 0:
            continue
        print(f"   {name:12s} n={m.sum():7d} MAE={np.mean(abs_err[m]):.4f} bias={np.mean(err[m]):+.4f} acc10={np.mean(abs_err[m]<=0.1):.3f} mean_y={Y[m].mean():.3f} pred={pred_all[m].mean():.3f}")
        r2.setdefault(nm, {})[name] = dict(n=int(m.sum()), mae=float(np.mean(abs_err[m])),
                                           bias=float(np.mean(err[m])), acc10=float(np.mean(abs_err[m] <= 0.1)),
                                           mean_y=float(Y[m].mean()), mean_pred=float(pred_all[m].mean()))
report["2d_regimes"] = r2

# ---------- C. training coverage of failing regimes ----------
def regime_on(Xx):
    return {
        "low_imp_low_grade": (Xx[:, col["imp"]] < 0.2) & (Xx[:, col["max_in_grade"]] < 0.02),
        "low_imp_any": Xx[:, col["imp"]] < 0.2,
        "flat_grade_lt_0.01": Xx[:, col["max_in_grade"]] < 0.01,
        "high_logimp_lowgrade": (Xx[:, col["log_imp_area"]] > 1.42) & (Xx[:, col["max_in_grade"]] < 0.02),
        "shallow_dep_le0": Xx[:, col["dep_depth"]] <= 0.014,
        "surch_high_dep0": (Xx[:, col["surcharge"]] > 7.0) & (Xx[:, col["dep_depth"]] <= 0.014),
        "elev_flat_high": Xx[:, col["elev_std2"]] > 1.4,
        "grade0_neg": Xx[:, col["max_in_grade"]] < 0.0,
        "sink_or_cap1": Xx[:, col["hydraulic_capacity"]] == 1.0,
    }

print("\n=== training-city coverage of Bangalore failing regimes (share of nodes) ===")
print(f"{'regime':24s} {'hsr':>9s} {'blg_nh':>9s} {'train':>9s}  + training cities top5")
cov = {}
for nm, mask in regime_on(X).items():
    f_hsr = float(np.mean(mask[hsr]))
    f_nh = float(np.mean(mask[nbl]))
    city_shares = []
    for c in train_cities:
        mc = regime_on(train_tables[c][0])[nm]
        city_shares.append((float(np.mean(mc)), c))
    city_shares.sort(reverse=True)
    f_t = float(sum(s * train_tables[c][0].shape[0] for s, c in city_shares) / n_t)
    top = " ".join(f"{c}:{s:.3f}" for s, c in city_shares[:5])
    print(f"{nm:24s} {f_hsr:9.3f} {f_nh:9.3f} {f_t:9.3f}  {top}")
    cov[nm] = dict(hsr=f_hsr, blg_non_hsr=f_nh, train=f_t,
                   top_cities={c: s for s, c in city_shares[:5]},
                   worst_cities={c: s for s, c in city_shares[-3:]})
report["train_coverage"] = cov

# ---------- D. per-city train imp / log_imp_area / surcharge summary ----------
print("\n=== per-training-city feature stats (nodes) ===")
stats = {}
for c in train_cities:
    Xc, _ = train_tables[c]
    stats[c] = dict(n=int(Xc.shape[0]), imp_mean=float(Xc[:, col["imp"]].mean()),
                    imp_q05=float(np.quantile(Xc[:, col["imp"]], 0.05)),
                    imp_q95=float(np.quantile(Xc[:, col["imp"]], 0.95)),
                    logimp_mean=float(Xc[:, col["log_imp_area"]].mean()),
                    surch_q95=float(np.quantile(Xc[:, col["surcharge"]], 0.95)),
                    y_mean=float(Yt_means.get(c, float("nan"))) if False else None)
Yc_means = {c: float(train_tables[c][1].mean()) for c in train_cities}
for c in train_cities:
    Xc, _ = train_tables[c]
    stats[c]["y_mean"] = Yc_means[c]
report["train_city_stats"] = stats
for c in train_cities:
    s = stats[c]
    print(f"  {c:14s} n={s['n']:6d} imp={s['imp_mean']:.3f}[{s['imp_q05']:.2f},{s['imp_q95']:.2f}] logimp={s['logimp_mean']:.2f} surch_q95={s['surch_q95']:.1f} mean_y={s['y_mean']:.2f}")

# ---------- E. where do extreme over-preds live vs training ----------
mbig = mE & (pred_all > 2.5)
print(f"\n=== pred>2.5m nodes (n={mbig.sum()}): feature means vs hsr vs training ===")
for f in ["imp", "log_imp_area", "max_in_grade", "sag_index", "dep_depth", "surcharge", "path_cap", "elev_std2", "hydraulic_capacity", "is_sink", "in_deg"]:
    c = col[f]
    print(f"  {f:18s} overpred={X[mbig, c].mean():7.3f}  hsr_all={X[hsr, c].mean():7.3f}  blg_nh={X[nbl, c].mean():7.3f}  train={Xt[:, c].mean():7.3f}")
report["overpred_node_means"] = {f: dict(overpred=float(X[mbig, col[f]].mean()),
                                          hsr=float(X[hsr, col[f]].mean()),
                                          blg_nh=float(X[nbl, col[f]].mean()),
                                          train=float(Xt[:, col[f]].mean()))
                                  for f in ["imp","log_imp_area","max_in_grade","sag_index","dep_depth","surcharge","path_cap","elev_std2","hydraulic_capacity","is_sink","in_deg"]}

# region mix of overpreds
from collections import Counter
print("region mix of pred>2.5m:", Counter(R[mbig]))

# ---------- F. deep under-pred regime: log_imp_area & dep for y>0.8 ----------
mdeep = mE & (Y > 0.8)
print(f"\n=== deep nodes (y>0.8, n={mdeep.sum()}): feature means vs training deep nodes ===")
print(f"  {'feature':18s} {'blg_deep':>10s} {'train_deep':>10s}")
deep_report = {}
for f in ["imp", "log_imp_area", "max_in_grade", "sag_index", "dep_depth", "surcharge", "path_cap", "elev_std2", "log_area", "accum_score"]:
    c = col[f]
    tr = np.quantile(Xt[Yt > 0.8, c], 0.5) if (Yt > 0.8).sum() else float("nan")
    print(f"  {f:18s} {X[mdeep, c].mean():10.3f} {tr:10.3f}")
    deep_report[f] = dict(blg_deep=float(X[mdeep, c].mean()), train_deep_med=float(tr))
report["deep_node_feats"] = deep_report

with open(os.path.join(OUT, "forensics_overlap.json"), "w") as f:
    json.dump(report, f, indent=1, default=float)
print("\nwrote", os.path.join(OUT, "forensics_overlap.json"))
