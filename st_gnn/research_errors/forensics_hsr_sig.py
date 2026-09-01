import os
import json
import numpy as np
import torch
from collections import Counter

BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"
OUT = os.path.join(BASE, "st_gnn", "research_errors", "results")

ds = torch.load(os.path.join(BASE, "multi_scenario_pyg_dataset.pt"), weights_only=False)
bp = torch.load(os.path.join(BASE, "st_gnn", "baseline_preds.pt"), weights_only=False)
pred_all = np.asarray(bp["pred"], dtype=np.float64)

FEAT_NAMES = ["rel_x","rel_y","rel_drop","imp","manning_n","in_deg","out_deg","accum_score",
    "is_sink","max_in_grade","sag_index","hydraulic_capacity","log_area","log_imp_area","dist_frac",
    "intensity","duration","elev_std2","dep_depth","surcharge","path_cap","path_hops"]
col = {n: FEAT_NAMES.index(n) for n in FEAT_NAMES}

blg = [g for g in ds if g.city == "bangalore"]
blg.sort(key=lambda g: ds.index(g))
X = np.vstack([g.x.numpy() for g in blg])
Y = np.concatenate([g.y.numpy().ravel() for g in blg])
R = np.concatenate([[g.region] * g.num_nodes for g in blg])
I = np.concatenate([[float(g.rain_intensity)] * g.num_nodes for g in blg])
err = pred_all - Y

def city_table(city):
    gs = [g for g in ds if g.city == city]
    gs.sort(key=lambda g: ds.index(g))
    return np.vstack([g.x.numpy() for g in gs]), np.concatenate([g.y.numpy().ravel() for g in gs])

train_cities = sorted({g.city for g in ds if g.city != "bangalore"})
train_tables = {c: city_table(c) for c in train_cities}
Xt = np.vstack([t[0] for t in train_tables.values()])
n_t = Xt.shape[0]

report = {}
mE = I >= 150.0
hsr = R == "hsr"
nbl = ~hsr

# ---------- A. within-hsr: over-predicted vs well-predicted nodes ----------
h_over = hsr & mE & (err > 0.5)
h_ok = hsr & mE & (np.abs(err) < 0.15)
print(f"hsr I>=150: n={hsr.sum()}, over-pred(>+0.5)={h_over.sum()}, well-pred(<0.15)={h_ok.sum()}")
print(f"{'feature':18s} {'hsr_over':>10s} {'hsr_ok':>10s} {'train_p50':>10s} {'train_p90':>10s} {'train_p99':>10s} {'blg_nh':>10s}")
rows = []
for f in FEAT_NAMES:
    c = col[f]
    tp50, tp90, tp99 = [float(np.quantile(Xt[:, c], q)) for q in (0.5, 0.9, 0.99)]
    rows.append(dict(feature=f, hsr_over=float(X[h_over, c].mean()), hsr_ok=float(X[h_ok, c].mean()),
                     train_p50=tp50, train_p90=tp90, train_p99=tp99, blg_nh=float(X[nbl, c].mean())))
    print(f"{f:18s} {X[h_over, c].mean():10.3f} {X[h_ok, c].mean():10.3f} {tp50:10.3f} {tp90:10.3f} {tp99:10.3f} {X[nbl, c].mean():10.3f}")
report["hsr_over_vs_ok"] = rows

# ---------- B. hsr overpred signature vs training overlap ----------
sig = (X[:, col["elev_std2"]] > 1.3) & (X[:, col["max_in_grade"]] < 0.02) & (X[:, col["sag_index"]] < 0.05)
print("\n=== signature (elev_std2>1.3 & grade<0.02 & sag<0.05) ===")
for name, m in [("hsr", hsr), ("blg_nh", nbl)]:
    mm = sig & m
    print(f"  {name:8s} share={np.mean(sig[m]):.3f} n={mm.sum()} bias_I150={np.mean(err[mm & mE]) if mm.sum() else float('nan'):+.3f}")
for c in train_cities:
    Xc, _ = train_tables[c]
    mc = (Xc[:, col["elev_std2"]] > 1.3) & (Xc[:, col["max_in_grade"]] < 0.02) & (Xc[:, col["sag_index"]] < 0.05)
    print(f"  train {c:14s} share={np.mean(mc):.4f} n={mc.sum()}")
train_sig = (Xt[:, col["elev_std2"]] > 1.3) & (Xt[:, col["max_in_grade"]] < 0.02) & (Xt[:, col["sag_index"]] < 0.05)
print(f"  TRAIN total share={np.mean(train_sig):.4f} n={train_sig.sum()}")
report["signature"] = dict(
    hsr_share=float(np.mean(sig[hsr])), blg_nh_share=float(np.mean(sig[nbl])),
    train_share=float(np.mean(train_sig)), train_n=int(train_sig.sum()),
    hsr_overpred_in_sig=float(np.mean(err[sig & hsr & mE])))

# elev_std2 alone per hsr bin and bias
print("\n=== hsr bias by elev_std2 bin (I>=150) ===")
for lo, hi in [(0, 0.9), (0.9, 1.3), (1.3, 1.8), (1.8, 3), (3, 1e9)]:
    m = hsr & mE & (X[:, col["elev_std2"]] >= lo) & (X[:, col["elev_std2"]] < hi)
    m2 = nbl & mE & (X[:, col["elev_std2"]] >= lo) & (X[:, col["elev_std2"]] < hi)
    print(f"  [{lo},{hi}) hsr: n={m.sum():6d} bias={np.mean(err[m]):+.3f} acc10={np.mean(np.abs(err[m])<=0.1):.3f} | blg_nh: n={m2.sum():6d} bias={np.mean(err[m2]):+.3f}")

# ---------- C. dynamic range / shrinkage check ----------
m = mE
slope = np.polyfit(Y[m], pred_all[m], 1)
print(f"\n=== shrinkage: pred = a*y+b at I>=150: a={slope[0]:.3f} b={slope[1]:+.3f} (a<1 => range compression)")
print(f"pred std={pred_all[m].std():.3f} y std={Y[m].std():.3f}")
print(f"corr={np.corrcoef(Y[m], pred_all[m])[0,1]:.3f}")
print(f"share of pred==0: {np.mean(pred_all[m]==0):.4f}; share y==0: {np.mean(Y[m]==0):.4f}")
deep_m = m & (Y > 0.8)
print(f"deep under-pred rate by intensity: ", {iv: float(np.mean(err[deep_m & (I==iv)])) for iv in [150,200,250,300]})
report["shrinkage"] = dict(slope=slope[0], intercept=slope[1], pred_std=float(pred_all[m].std()),
                           y_std=float(Y[m].std()), corr=float(np.corrcoef(Y[m], pred_all[m])[0,1]))
report["deep_bias_by_intensity"] = {iv: float(np.mean(err[deep_m & (I == iv)])) for iv in [150, 200, 250, 300]}

# ---------- D. hsr graph-level node stats ----------
print("\n=== hsr graphs ===")
for g in blg:
    if g.region == "hsr":
        print(f"  I={g.rain_intensity:g} dur={g.rain_duration:g} nodes={g.num_nodes}")
print("\n=== node counts per region ===")
print(Counter(R))

# ---------- E. hsr bias vs imp LOW-confirm: bias among hsr with imp>0.5 ----------
mh = hsr & mE & (X[:, col["imp"]] > 0.5)
print(f"\nhsr imp>0.5 nodes at I>=150: n={mh.sum()} bias={np.mean(err[mh]):+.3f} MAE={np.mean(np.abs(err[mh])):.3f} mean_y={Y[mh].mean():.3f} elev_std2={X[mh, col['elev_std2']].mean():.3f}")

with open(os.path.join(OUT, "forensics_hsr_sig.json"), "w") as f:
    json.dump(report, f, indent=1, default=float)
print("\nwrote", os.path.join(OUT, "forensics_hsr_sig.json"))
