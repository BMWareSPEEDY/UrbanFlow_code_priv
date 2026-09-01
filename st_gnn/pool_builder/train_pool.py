import argparse
import os
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Batch

from pool_common import (acquire_lock, evaluate, fmt_eval, load_data, make_city_batches,
                         release_lock, compute_stats, step_loss, UrbanPoolNet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n_local", type=int, default=2)
    ap.add_argument("--ratios", default="0.6")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--cities", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--head", type=int, default=192)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ratios = [float(r) for r in args.ratios.split(",")]
    device = torch.device("cuda")
    ckpt = os.path.join(os.path.dirname(__file__), f"ckpt_{args.tag}.pt")
    preds_file = os.path.join(os.path.dirname(__file__), f"preds_{args.tag}.pt")

    acquire_lock()
    t0_all = time.time()
    try:
        tr_g, te_g = load_data()
        x_mean, x_std, e_mean, e_std, yl_mean, yl_std = compute_stats(tr_g, device)
        tr_batches, cities = make_city_batches(tr_g, x_mean, x_std, e_mean, e_std, args.cities)
        print(f"cities: {cities}", flush=True)
        print(f"train nodes {sum(b.x.shape[0] for b in tr_batches)}, {len(tr_batches)} city batches", flush=True)

        te = Batch.from_data_list(te_g).to(device)
        te_raw_x = torch.cat([g.x for g in te_g], 0).numpy()
        te_y = torch.cat([g.y for g in te_g], 0).numpy()
        te.x = (te.x - x_mean) / x_std
        te.edge_attr = (te.edge_attr - e_mean) / e_std

        model = UrbanPoolNet(22, 2, args.hidden, args.n_local, ratios, args.head).to(device)
        nparam = sum(p.numel() for p in model.parameters())
        print(f"model params {nparam/1e6:.2f}M | config h{args.hidden} n_local{args.n_local} ratios{ratios}", flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
        start_ep = 1
        best_mae = 1e9
        if os.path.exists(ckpt):
            ck = torch.load(ckpt, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_ep = ck["ep"] + 1
            best_mae = ck.get("best_mae", 1e9)
            print(f"resume from ep {start_ep} (best_mae {best_mae:.4f})", flush=True)

        for ep in range(start_ep, args.epochs + 1):
            model.train()
            order = torch.randperm(len(tr_batches)).tolist()
            loss_sum = 0.0
            t0 = time.time()
            for j in order:
                opt.zero_grad()
                loss = step_loss(model, tr_batches[j], yl_mean, yl_std)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                loss_sum += loss.item()
            sched.step()
            spd = len(tr_batches) / max(1e-9, time.time() - t0)
            if ep % args.eval_every == 0 or ep == args.epochs:
                res, _ = evaluate(model, te, te_raw_x, te_y, yl_mean, yl_std, tr_batches)
                improved = res["mae"] < best_mae - 1e-6
                if improved:
                    best_mae = res["mae"]
                torch.save({"ep": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "best_mae": best_mae,
                            "x_mean": x_mean, "x_std": x_std, "e_mean": e_mean, "e_std": e_std,
                            "yl_mean": yl_mean, "yl_std": yl_std, "cfg": vars(args)},
                           ckpt)
                print(f"ep {ep} loss {loss_sum:.3f} ({spd:.1f} b/s) | {fmt_eval(res)} "
                      f"{'*BEST' if improved else ''} | {(time.time()-t0_all)/60:.0f}min", flush=True)
            elif ep % 5 == 0:
                print(f"ep {ep} loss {loss_sum:.3f} ({spd:.1f} b/s)", flush=True)

        res, pred = evaluate(model, te, te_raw_x, te_y, yl_mean, yl_std, tr_batches)
        torch.save({"pred": pred}, preds_file)
        print(f"FINAL {fmt_eval(res)}", flush=True)
        for i, d in sorted(res["per_int"].items()):
            print(f"  I={i:6.1f}: MAE {d['mae']:.4f} R2 {d['r2']:.4f} +-10cm {d['p10']:.1f}% F1 {d['f1']:.4f} bias {d['bias']:+.4f}", flush=True)
        print(f"preds saved {preds_file} (n={len(pred)}), total {(time.time()-t0_all)/60:.0f}min", flush=True)
    finally:
        release_lock()


if __name__ == "__main__":
    main()