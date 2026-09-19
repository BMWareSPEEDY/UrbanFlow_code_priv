"""Reproduce the 50 mm/hr expanded-ensemble evaluation from committed artifacts.

Loads datasets/expanded_master_physics_dataset.pt (the live SWMM ground-truth
targets at 50 mm/hr / 60 min) and runs the production EnsembleFloodPredictorV4
(v5.10 + v5.11 + v5.0 + Bangalore-opt, weights 0.80/0.30/0.05/0.10 normalized to
0.64/0.24/0.04/0.08) on each catchment graph. Writes both data files consumed by
downstream documents and benchmarks:

  data/regime50_expanded_ensemble_full_table.json           (7-col vector)
  data/regime50_expanded_full_table_16city.json             (full cell table)
"""
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from core.production_v4 import EnsembleFloodPredictorV4

THR = 0.15

CITY_REGION = {
    "hsr": ("bangalore", "hsr"),
    "bellandur": ("bangalore", "bellandur"),
    "whitefield": ("bangalore", "whitefield"),
    "ecity": ("bangalore", "ecity"),
    "koramangala": ("bangalore", "koramangala"),
    "tokyo": ("tokyo", "tokyo"),
    "hongkong": ("hongkong", "hongkong"),
    "singapore": ("singapore", "singapore"),
    "london": ("london", "london"),
    "paris": ("paris", "paris"),
    "nyc": ("newyork", "nyc"),
    "chicago": ("chicago", "chicago"),
    "berlin": ("berlin", "berlin"),
    "bangkok": ("bangkok", "bangkok"),
    "mumbai": ("mumbai", "mumbai"),
    "delhi": ("delhi", "delhi"),
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = EnsembleFloodPredictorV4(device=device)
    dl = torch.load("datasets/expanded_master_physics_dataset.pt", weights_only=False)

    vec = {}
    rows = {}
    for reg, (city, region) in CITY_REGION.items():
        g = next(
            x
            for x in dl
            if x.city == city
            and x.region == region
            and abs(float(getattr(x, "rain_intensity", 0.0)) - 50.0) < 0.01
        )
        y = g.y.cpu().numpy().ravel()
        p, _, _ = predictor.predict(g, 50.0, 60.0)
        yb, pb = y >= THR, p >= THR
        tp = int(np.sum(yb & pb))
        fp = int(np.sum((~yb) & pb))
        fn = int(np.sum(yb & (~pb)))
        n = int(len(y))
        prec = tp / (tp + fp) * 100 if tp + fp else 0.0
        rec = tp / (tp + fn) * 100 if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) else 0.0
        acc = (n - fp - fn) / n * 100
        mae = float(np.abs(p - y).mean() * 100)
        rmse = float(np.sqrt(((p - y) ** 2).mean()) * 100)
        ad = np.abs(p - y) * 100
        le10 = float(np.mean(ad <= 10.0) * 100)
        le15 = float(np.mean(ad <= 15.0) * 100)
        le30 = float(np.mean(ad <= 30.0) * 100)
        p90 = float(np.percentile(ad, 90))
        p95 = float(np.percentile(ad, 95))
        mx = float(ad.max())
        swmm = int(np.sum(yb))
        gnn = int(np.sum(pb))
        vec[reg] = [
            swmm, gnn, round(f1, 1), round(rec, 1), round(prec, 1),
            round(mae, 2), round(rmse, 2),
        ]
        rows[reg] = {
            "nodes": n, "swmm": swmm, "gnn": gnn, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 1), "recall": round(rec, 1),
            "f1": round(f1, 1), "accuracy": round(acc, 1),
            "mae_cm": round(mae, 2), "rmse_cm": round(rmse, 2),
            "pct_le10_cm": round(le10, 1), "pct_le15_cm": round(le15, 1),
            "pct_le30_cm": round(le30, 1),
            "p90_cm": round(p90, 1), "p95_cm": round(p95, 1),
            "max_err_cm": round(mx, 1),
        }
        print(
            f"{reg:11s}: N={n:5d} SWMM={swmm:5d} GNN={gnn:5d} TP={tp:5d} "
            f"FP={fp:4d} FN={fn:5d} P={prec:5.1f} R={rec:5.1f} F1={f1:5.1f} "
            f"Acc={acc:5.1f} MAE={mae:6.2f} RMSE={rmse:6.2f}"
        )

    json.dump(vec, open("data/regime50_expanded_ensemble_full_table.json", "w"), indent=1)
    json.dump(rows, open("data/regime50_expanded_full_table_16city.json", "w"), indent=1)
    print("saved data/regime50_expanded_ensemble_full_table.json")
    print("saved data/regime50_expanded_full_table_16city.json")


if __name__ == "__main__":
    main()