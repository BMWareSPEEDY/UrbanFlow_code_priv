"""Generate prediction data for the UrbanFLOW paper figures (> 50 mm/hr, 16-catchment set)."""
import json
import numpy as np
import torch

DATA = torch.load("expanded_master_physics_dataset.pt", map_location="cpu", weights_only=False)
by50 = {}
for g in DATA:
    if float(g.rain_intensity) == 50.0:
        by50.setdefault((g.city, g.region), g)

CITIES = json.load(open("benchmark_data_full.json"))
KEYS = [c["key"] for c in CITIES["50_mmhr"]]

def grab(key):
    cand = []
    for k, g in by50.items():
        if k[0] == key or k[1] == key:
            cand.append(g)
    if not cand:
        raise RuntimeError(f"no graph for key={key}")
    cand.sort(key=lambda g: -g.x.shape[0])
    seen = set()
    out = []
    for g in cand:
        pid = id(g.edge_index)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(g)
    return out[0]

from production_v4 import EnsembleFloodPredictorV4

ens = EnsembleFloodPredictorV4(device="cpu")
rows = []
for key in KEYS:
    g = grab(key)
    pred, raw, prob = ens.predict(g, 50.0, 60.0)
    rows.append({
        "key": key,
        "nodes": g.x.shape[0],
        "y": g.y.squeeze(1).numpy().astype(np.float32),
        "p": pred.astype(np.float32),
        "prob": prob.astype(np.float32),
        "rel_x": g.x[:, 0].numpy().astype(np.float32),
        "rel_y": g.x[:, 1].numpy().astype(np.float32),
        "edge_index": g.edge_index.numpy(),
    })
    mae = np.mean(np.abs(rows[-1]["y"] - rows[-1]["p"])) * 100
    print(f"{key:>12} nodes={rows[-1]['nodes']:>6} mae={mae:.2f}cm", flush=True)

torch.save(rows, "fig_pred_data.pt")
tot_n = sum(r["nodes"] for r in rows)
y = np.concatenate([r["y"] for r in rows])
p = np.concatenate([r["p"] for r in rows])
e = np.abs(y - p)
tp = ((y >= 0.15) & (p >= 0.15)).sum()
fp = ((y < 0.15) & (p >= 0.15)).sum()
fn = ((y >= 0.15) & (p < 0.15)).sum()
prec = tp / (tp + fp)
rec = tp / (tp + fn)
f1 = 2 * prec * rec / (prec + rec)
print(f"TOTAL nodes={tot_n}")
print(f"GLOBAL mae={e.mean()*100:.2f}cm rmse={np.sqrt((e**2).mean())*100:.2f}cm p15={(e<=0.15).mean()*100:.1f}% p30={(e<=0.30).mean()*100:.1f}%")
print(f"HAZARD F1={f1*100:.1f}%  prec={prec*100:.1f}%  rec={rec*100:.1f}%  dry_fraction={(y<0.15).mean()*100:.1f}%")