import os
import json
import numpy as np
import torch

BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"
OUT = os.path.join(BASE, "st_gnn", "research_errors", "results")
os.makedirs(OUT, exist_ok=True)

ds = torch.load(os.path.join(BASE, "multi_scenario_pyg_dataset.pt"), weights_only=False)
bp = torch.load(os.path.join(BASE, "st_gnn", "baseline_preds.pt"), weights_only=False)
pred_all = np.asarray(bp["pred"], dtype=np.float64)

FEAT_NAMES = [
    "rel_x", "rel_y", "rel_drop", "imp", "manning_n", "in_deg", "out_deg",
    "accum_score", "is_sink", "max_in_grade", "sag_index", "hydraulic_capacity",
    "log_area", "log_imp_area", "dist_frac", "intensity", "duration", "elev_std2",
    "dep_depth", "surcharge", "path_cap", "path_hops",
]

blg = [g for g in ds if g.city == "bangalore"]
blg.sort(key=lambda g: ds.index(g))
X = np.vstack([g.x.numpy() for g in blg])
Y = np.concatenate([g.y.numpy().ravel() for g in blg])
R = np.concatenate([[g.region] * g.num_nodes for g in blg])
I = np.concatenate([[float(g.rain_intensity)] * g.num_nodes for g in blg])
assert X.shape[0] == pred_all.shape[0] == Y.shape[0], (X.shape, pred_all.shape, Y.shape)
n_nodes = X.shape[0]

err = pred_all - Y
abs_err = np.abs(err)


def metrics(p, y):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mae = np.mean(np.abs(p - y))
    bias = float(np.mean(p - y))
    acc10 = float(np.mean(np.abs(p - y) <= 0.1))
    wet = y > 0.15
    pwet = p > 0.15
    tp = np.sum(wet & pwet)
    fp = np.sum(~wet & pwet)
    fn = np.sum(wet & ~pwet)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(mae=float(mae), bias=float(bias), acc10=float(acc10), f1=float(f1))


mask_ext = I >= 150.0
print(f"Bangalore nodes total={n_nodes}, I>=150 nodes={mask_ext.sum()}, intensities={np.unique(I)}")

# sanity check against baseline_report per-intensity +-10cm
sanity = {}
for iv in np.unique(I):
    m = I == iv
    sanity[str(iv)] = float(np.mean(np.abs(err[m]) <= 0.1))
with open(os.path.join(OUT, "sanity_acc10_vs_report.json"), "w") as f:
    json.dump(sanity, f, indent=1)
print("acc10 by intensity (report: 150->0.580, 200->0.457, 250->0.402, 300->0.367):", sanity)

report = {}

# ---------- 1. per-region at I>=150 ----------
regions = {}
for r in np.unique(R):
    m = (R == r) & mask_ext
    if m.sum() == 0:
        continue
    regions[r] = metrics(pred_all[m], Y[m])
    regions[r]["n"] = int(m.sum())
    regions[r]["share_abs_err"] = float(abs_err[m].sum() / abs_err[mask_ext].sum())
    regions[r]["mean_y"] = float(Y[m].mean())
    regions[r]["mean_pred"] = float(pred_all[m].mean())
report["per_region_I150plus"] = regions
print("\n=== per-region I>=150 ===")
for k, v in regions.items():
    print(f"{k:14s} n={v['n']:7d} MAE={v['mae']:.4f} bias={v['bias']:+.4f} acc10={v['acc10']:.3f} F1={v['f1']:.3f} mean_y={v['mean_y']:.3f} share_abs_err={v['share_abs_err']:.3f}")

# ---------- 1b. per region per intensity acc10 ----------
r_by_i = {}
for r in np.unique(R):
    for iv in np.unique(I):
        m = (R == r) & (I == iv)
        if m.sum() == 0:
            continue
        r_by_i.setdefault(r, {})[str(iv)] = float(np.mean(np.abs(err[m]) <= 0.1))
report["region_x_intensity_acc10"] = r_by_i
print("\n=== acc10 region x intensity ===")
print("region  " + " ".join(f"{iv:>7}" for iv in np.unique(I)))
for r, d in r_by_i.items():
    print(f"{r:14s}" + " ".join(f"{d.get(str(iv), float('nan')):7.3f}" for iv in np.unique(I)))

# ---------- 2. feature-conditional bins at I>=150 ----------
feat_bins = {
    "imp": np.array([0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.01]),
    "log_imp_area": None,  # quantile bins
    "max_in_grade": np.array([-0.5, 0.0, 0.01, 0.02, 0.04, 0.08, 1.0]),
    "sag_index": None,
    "accum_score": None,
    "elev_std2": None,
    "dep_depth": None,
    "surcharge": None,
    "path_cap": None,
    "in_deg": np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 1e9]),
    "out_deg": np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 1e9]),
}
feat_report = {}
print("\n=== feature-conditional bins (I>=150) ===")
for fn in feat_bins:
    col = X[:, FEAT_NAMES.index(fn)]
    edges = feat_bins[fn]
    if edges is None:
        qs = np.quantile(col[mask_ext], [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        qs = np.unique(qs)
        edges = np.concatenate([[-np.inf], qs[1:-1], [np.inf]])
    idx = np.digitize(col, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        m = (idx == b) & mask_ext
        if m.sum() < 50:
            continue
        mm = metrics(pred_all[m], Y[m])
        rows.append(dict(
            bin=f"[{edges[b]:.3g},{edges[b+1]:.3g})", n=int(m.sum()),
            mae=mm["mae"], bias=mm["bias"], acc10=mm["acc10"], f1=mm["f1"],
            mean_y=float(Y[m].mean()), mean_pred=float(pred_all[m].mean()),
        ))
    feat_report[fn] = rows
    print(f"\n-- {fn} --")
    for row in rows:
        print(f"  {row['bin']:>22s} n={row['n']:7d} MAE={row['mae']:.4f} bias={row['bias']:+.4f} acc10={row['acc10']:.3f} F1={row['f1']:.3f} mean_y={row['mean_y']:.3f} mean_pred={row['mean_pred']:.3f}")

# ---------- 3. error vs true depth deciles ----------
ydc = {}
m3 = mask_ext
q_edges = np.quantile(Y[m3], np.arange(0, 1.01, 0.1))
q_edges[-1] += 1e-9
idx = np.digitize(Y, q_edges) - 1
print("\n=== error by y-decile (I>=150) ===")
print("decile  range          n      MAE    bias   acc10   F1   mean_pred")
for d in range(10):
    m = (idx == d) & m3
    if m.sum() == 0:
        continue
    mm = metrics(pred_all[m], Y[m])
    ydc[f"dec{d}"] = dict(
        yrange=[float(q_edges[d]), float(q_edges[d + 1])], n=int(m.sum()),
        mae=mm["mae"], bias=mm["bias"], acc10=mm["acc10"], f1=mm["f1"],
        mean_y=float(Y[m].mean()), mean_pred=float(pred_all[m].mean()),
    )
    print(f"  d{d}  [{q_edges[d]:.3f},{q_edges[d+1]:.3f}) n={m.sum():6d}  {mm['mae']:.4f}  {mm['bias']:+.4f}  {mm['acc10']:.3f}  {mm['f1']:.3f}  {pred_all[m].mean():.3f}")
# deepest 10%
m_deep = m3 & (Y >= q_edges[9])
print(f"deepest 10% (y>={q_edges[9]:.3f}): n={m_deep.sum()}, MAE={np.mean(abs_err[m_deep]):.4f}, bias={np.mean(err[m_deep]):+.4f}, mean_y={Y[m_deep].mean():.3f}, mean_pred={pred_all[m_deep].mean():.3f}")
report["y_deciles"] = ydc

# ---------- 4. outliers ----------
print("\n=== extreme over-predictions ===")
outl = {}
for iv in [150.0, 200.0, 250.0, 300.0]:
    m = (I == iv)
    cnt = int(np.sum(pred_all[m] > 2.5))
    outl[str(iv)] = dict(n_over2_5=int(np.sum(pred_all[m] > 2.5)), max_pred=float(pred_all[m].max()),
                         max_err=float(err[m].max()), n_over1_5=int(np.sum(pred_all[m] > 1.5)))
    print(f"I={iv:g}: pred>2.5m: {cnt}, pred>1.5m: {np.sum(pred_all[m] > 1.5)}, max pred {pred_all[m].max():.3f}, max err {err[m].max():.3f}")
report["outliers"] = outl

k = int(np.argmax(pred_all[mask_ext]))
print("\nmax over-pred node:")
for i, nm in enumerate(FEAT_NAMES):
    print(f"  {nm:18s} = {X[k, i]:.4g}")
print(f"  region={R[k]} intensity={I[k]} y={Y[k]:.4f} pred={pred_all[k]:.4f} err={err[k]:+.4f}")

# top-20 over-preds
print("\ntop-20 over-pred nodes (I>=150):")
topk = np.argsort(err[mask_ext])[::-1][:20]
for j in topk:
    kk = np.where(mask_ext)[0][j]
    print(f"  region={R[kk]:12s} I={I[kk]:g} y={Y[kk]:.3f} pred={pred_all[kk]:.3f} err={err[kk]:+.3f} imp={X[kk,3]:.3f} logimp={X[kk,13]:.3f} grade={X[kk,9]:.4f} sag={X[kk,10]:.4f} dep={X[kk,18]:.3f} surch={X[kk,19]:.3f} cap={X[kk,11]:.2e}")

# ---------- 5. hsr imperviousness structure ----------
print("\n=== imp / structure by region (all intensities) ===")
reg_stats = {}
for r in np.unique(R):
    m = R == r
    reg_stats[r] = dict(
        mean_imp=float(X[m, 3].mean()), imp_std=float(X[m, 3].std()),
        imp_q05=float(np.quantile(X[m, 3], 0.05)), imp_q95=float(np.quantile(X[m, 3], 0.95)),
        mean_y=float(Y[m].mean()), mean_logimp=float(X[m, 13].mean()),
        mean_sag=float(X[m, 10].mean()), mean_dep=float(X[m, 18].mean()),
    )
    print(f"{r:14s} imp mean={X[m,3].mean():.3f} std={X[m,3].std():.3f} q05={np.quantile(X[m,3],0.05):.3f} q95={np.quantile(X[m,3],0.95):.3f} mean_y={Y[m].mean():.3f}")
report["region_structure"] = reg_stats

# bias by region across intensities
print("\n=== bias by region x intensity ===")
b_r_i = {}
for r in np.unique(R):
    for iv in np.unique(I):
        m = (R == r) & (I == iv)
        b_r_i.setdefault(r, {})[str(iv)] = float(np.mean(err[m]))
    print(f"{r:14s} " + " ".join(f"{b_r_i[r][str(iv)]:+.4f}" for iv in np.unique(I)))
report["bias_region_x_intensity"] = b_r_i

# ---------- 7. error contribution fractions ----------
m3 = mask_ext
tot = float(abs_err[m3].sum())
frac = {}
frac["hsr"] = float(abs_err[(R == "hsr") & m3].sum() / tot)
frac["deep_nodes_y>0.8"] = float(abs_err[m3 & (Y > 0.8)].sum() / tot)
dry = m3 & (Y < 0.15)
wet_pred = dry & (pred_all > 0.15)
frac["dry_nodes_y<0.15"] = float(abs_err[dry].sum() / tot)
frac["dry_mispredicted_as_wet_count"] = int(wet_pred.sum())
frac["dry_mispredicted_as_wet_abs_err_share"] = float(abs_err[wet_pred].sum() / tot)
frac["deep_share_of_nodes"] = float((m3 & (Y > 0.8)).sum() / m3.sum())
frac["dry_share_of_nodes"] = float(dry.sum() / m3.sum())
print("\n=== contribution at I>=150 ===")
print(f"hsr abs-err share: {frac['hsr']:.3f}")
print(f"deep (y>0.8) abs-err share: {frac['deep_nodes_y>0.8']:.3f} (node share {frac['deep_share_of_nodes']:.3f})")
print(f"dry (y<0.15) abs-err share: {frac['dry_nodes_y<0.15']:.3f} (node share {frac['dry_share_of_nodes']:.3f})")
print(f"dry mispredicted wet: {frac['dry_mispredicted_as_wet_count']} nodes, abs-err share {frac['dry_mispredicted_as_wet_abs_err_share']:.3f}")
report["contributions_I150plus"] = frac

# under-prediction concentration: among y>0.3, what fraction have bias<0
m_under = m3 & (Y > 0.3)
print(f"\nunder-pred rate for y>0.3: {np.mean(err[m_under] < 0):.3f}, for y>0.8: {np.mean(err[m3 & (Y > 0.8)] < 0):.3f}")
report["underpred_rate"] = dict(y_gt_03=float(np.mean(err[m_under] < 0)), y_gt_08=float(np.mean(err[m3 & (Y > 0.8)] < 0)))

with open(os.path.join(OUT, "forensics_main.json"), "w") as f:
    json.dump(report, f, indent=1, default=float)
print("\nwrote", os.path.join(OUT, "forensics_main.json"))
