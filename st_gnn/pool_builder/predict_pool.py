import argparse
import os
import sys

import numpy as np
import torch
from torch_geometric.data import Batch

from pool_common import (acquire_lock, load_data, compute_stats, release_lock, predict,
                         UrbanPoolNet)

BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"
OUT = os.path.join(BASE, "st_gnn", "preds_pool_v1.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="checkpoint .pt file from train_pool.py")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    acquire_lock()
    try:
        ck = torch.load(args.ckpt, weights_only=False)
        cfg = ck["cfg"]
        ratios = [float(r) for r in cfg["ratios"].split(",")] if isinstance(cfg["ratios"], str) else list(cfg["ratios"])
        model = UrbanPoolNet(22, 2, cfg["hidden"], cfg["n_local"], ratios, cfg["head"])
        model.load_state_dict(ck["model"])
        model = model.cuda().eval()

        tr_g, te_g = load_data()
        x_mean, x_std, e_mean, e_std, yl_mean, yl_std = compute_stats(tr_g, torch.device("cuda"))
        te = Batch.from_data_list(te_g).cuda()
        te.x = (te.x - x_mean) / x_std
        te.edge_attr = (te.edge_attr - e_mean) / e_std
        pred = predict(model, te, yl_mean, yl_std)
        assert pred.shape[0] == 99488, pred.shape
        torch.save({"pred": pred}, args.out)
        print(f"saved {args.out} n={len(pred)} best_mae={ck.get('best_mae')}")
    finally:
        release_lock()


if __name__ == "__main__":
    main()