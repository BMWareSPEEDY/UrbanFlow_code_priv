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
col = {n: FEAT_NAMES.index(n) for n in FEAT_NAMES}

blg = [g for g in ds if g.city == "bangalore"]
blg.sort(key=lambda g: ds.index(g))
X = np.vstack([g.x.numpy() for g in blg])
Y = np.concatenate([g.y.numpy().ravel() for g in blg])
R = np.concatenate([[g.region] * g.num_nodes for g in blg])
I = np.concatenate([[float(g.rain_intensity)] * g.num_nodes for g in blg])
err = pred_all - Y
abs_err = np.abs(err)

def city_table(city):
    gs = [g for g in ds if g.city == city]
    gs.sort(key=lambda g: ds.index(g))
    return np.vstack([g.x.numpy() for g in gs]), np.concatenate([g.y.numpy().ravel() for g in gs])

train_cities = sorted({g.city for g in ds if g.city != "bangalore"})
train_tables = {c: city_table(c) for c in train_cities}
Xt = np.vstack([t[0] for t in train_tables.values()])
Yt = np.concatenate([t[1] for t in train_tables.values()])

report = {}
mE = I >= 150.0
hsr = R == "hsr"
nbl = ~hsr
tot = abs_err[mE].sum()

# 1. is_sink / dep / grade stats
for name, m in [("hsr", hsr), ("blg_nh", nbl)]:
    print(f"{name}: is_sink={np.mean(X[m, col['is_sink']]):.4f} dep_mean={X[m, col['dep_depth']].mean():.3f} "
          f"grade_mean={X[m, col['max_in_grade']].mean():.4f} sag_mean={X[m, col['sag_index']].mean():.3f}")
    report[name + "_stats"] = dict(is_sink=float(np.mean(X[m, col['is_sink']])),
                                   dep=float(X[m, col['dep_depth']].mean()),
                                   grade=float(X[m, col['max_in_grade']].mean()))
print(f"train: is_sink={np.mean(Xt[:, col['is_sink']]):.4f} dep_mean={Xt[:, col['dep_depth']].mean():.3f}")

# 2. sink nodes: share + error
for name, m in [("hsr", hsr), ("blg_nh", nbl)]:
    ms = m & mE & (X[:, col["is_sink"]] > 0.5)
    print(f"sink nodes {name} at I>=150: n={ms.sum()} ({np.mean(ms):.4f} of region) MAE={np.mean(abs_err[ms]):.3f} bias={np.mean(err[ms]):+.3f} "
          f"share_abserr={abs_err[ms].sum()/tot:.3f} mean_y={Y[ms].mean():.3f} pred={pred_all[ms].mean():.3f}")

# 3. deep-nodes and dry->wet aggregates
mdeep = mE & (Y > 0.8)
mdry = mE & (Y < 0.15)
mwet = mdry & (pred_all > 0.15)
print(f"deep y>0.8: n={mdeep.sum()} MAE={np.mean(abs_err[mdeep]):.3f} bias={np.mean(err[mdeep]):+.3f} share={abs_err[mdeep].sum()/tot:.3f}")
print(f"dry y<0.15: n={mdry.sum()} MAE={np.mean(abs_err[mdry]):.3f} bias={np.mean(err[mdry]):+.3f} share={abs_err[mdry].sum()/tot:.3f}")
print(f"dry->wet: n={mwet.sum()} MAE={np.mean(abs_err[mwet]):.3f} bias={np.mean(err[mwet]):+.3f} share={abs_err[mwet].sum()/tot:.3f} mean_y={Y[mwet].mean():.3f}")
print(f"hsr dry->wet: n={(hsr & mwet).sum()} of hsr dry={(hsr & mdry).sum()}")
report["agg"] = dict(deep=dict(n=int(mdeep.sum()), mae=float(np.mean(abs_err[mdeep])), bias=float(np.mean(err[mdeep])), share=float(abs_err[mdeep].sum()/tot)),
                     dry=dict(n=int(mdry.sum()), mae=float(np.mean(abs_err[mdry])), bias=float(np.mean(err[mdry])), share=float(abs_err[mdry].sum()/tot)),
                     dry_wet=dict(n=int(mwet.sum()), mae=float(np.mean(abs_err[mwet])), bias=float(np.mean(err[mwet])), share=float(abs_err[mwet].sum()/tot)))

# 4. impact estimate: what if hsr bias removed (shift by -mean bias, clip 0)
p_fix = np.clip(pred_all[hsr & mE] - np.mean(err[hsr & mE]), 0, None)
tot_abs_now = abs_err[mE].sum()
tot_abs_fix = tot_abs_now - abs_err[hsr & mE].sum() + np.abs(p_fix - Y[hsr & mE]).sum()
print(f"MAE now={np.mean(abs_err[mE]):.4f}; after hsr debias: {tot_abs_fix/np.sum(mE):.4f} ({(1-tot_abs_fix/tot_abs_now)*100:.1f}% lower abs err)")
report["debias_sim"] = dict(mae_now=float(np.mean(abs_err[mE])), mae_after=float(tot_abs_fix / np.sum(mE)))

# 5. deep underpred: what if deep bias shifted (add 0.36 to deepest decile, cap none)
mdd = mE & (Y >= np.quantile(Y[mE], 0.9))
b = np.mean(err[mdd])
p2 = pred_all[mdd] - b  # remove under-bias
tot2 = tot_abs_now - abs_err[mdd].sum() + np.abs(p2 - Y[mdd]).sum()
print(f"MAE after deep-decile debias: {tot2/np.sum(mE):.4f} ({(1-tot2/tot_abs_now)*100:.1f}% lower)")
report["deep_debias_sim"] = dict(mae_after=float(tot2 / np.sum(mE)))

# 6. where do pred>2.5 live: sink/dep profile
mb = mE & (pred_all > 2.5)
print(f"pred>2.5: n={mb.sum()} share of these is_sink={np.mean(X[mb, col['is_sink']]):.3f} dep>0={np.mean(X[mb, col['dep_depth']] > 0):.3f} "
      f"grade<0.02={np.mean(X[mb, col['max_in_grade']] < 0.02):.3f} sag<0.1={np.mean(X[mb, col['sag_index']] < 0.1):.3f} "
      f"y<0.15={np.mean(Y[mb] < 0.15):.3f}")

# 7. e2e: fraction of abs err from sink nodes at I>=150
msink = mE & (X[:, col["is_sink"]] > 0.5)
print(f"sink nodes all-blg at I>=150: n={msink.sum()} share_abserr={abs_err[msink].sum()/tot:.3f} MAE={np.mean(abs_err[msink]):.3f} bias={np.mean(err[msink]):+.3f}")

with open(os.path.join(OUT, "forensics_impact.json"), "w") as f:
    json.dump(report, f, indent=1, default=float)
print("\nwrote", os.path.join(OUT, "forensics_impact.json"))
