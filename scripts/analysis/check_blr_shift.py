import sys
import numpy as np
import torch
from torch_geometric.data import Batch

sys.stdout.reconfigure(line_buffering=True)

fnames = ['rel_x', 'rel_y', 'rel_drop', 'imp', 'manning_n', 'in_deg', 'out_deg',
          'accum_score', 'is_sink', 'max_in_grade', 'sag_index', 'hydraulic_capacity',
          'log_area', 'log_imp_area', 'dist_frac', 'intensity', 'duration']


def main():
    dl = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
    blr = [g for g in dl if g.city == 'bangalore']
    others = [g for g in dl if g.city != 'bangalore']
    blr_b = Batch.from_data_list(blr)
    oth_b = Batch.from_data_list(others)

    print(f"Bangalore graphs: {len(blr)} (nodes {blr_b.x.shape[0]}), others: {len(others)} (nodes {oth_b.x.shape[0]})")

    print(f"\n{'feature':<16} {'blr mean':>10} {'oth mean':>10} {'blr std':>10} {'oth std':>10} {'shift(sd)':>10}")
    for i, f in enumerate(fnames):
        bx = blr_b.x[:, i].numpy()
        ox_ = oth_b.x[:, i].numpy()
        bm, om = bx.mean(), ox_.mean()
        bs, os_ = bx.std(), ox_.std()
        pooled = (bx.std() + ox_.std()) / 2
        shift = (bm - om) / max(1e-9, pooled)
        print(f"{f:<16} {bm:>10.4f} {om:>10.4f} {bs:>10.4f} {os_:>10.4f} {shift:>+10.2f}")

    # Per-intensity mean depth comparison
    print("\nPer-intensity mean depth (m):")
    print(f"{'I':>8} {'blr':>10} {'others':>10}")
    for i in sorted(set(blr_b.x[:, 15].tolist())):
        bm = blr_b.y[blr_b.x[:, 15] == i].mean().item()
        om = oth_b.y[oth_b.x[:, 15] == i].mean().item()
        print(f"{i:>8.1f} {bm:>10.3f} {om:>10.3f}")

    # Per Bangalore region
    print("\nPer Bangalore region stats:")
    for reg in sorted({g.region for g in blr}):
        gs = [g for g in blr if g.region == reg]
        x = torch.cat([g.x for g in gs])
        y = torch.cat([g.y for g in gs])
        print(f"  {reg:<12} nodes {x.shape[0]:>6} mean_depth {y.mean().item():.3f} "
              f"imp_mean {x[:,3].mean().item():.3f} log_area_mean {x[:,12].mean().item():.3f} "
              f"rel_drop_mean {x[:,2].mean().item():.3f}")


if __name__ == "__main__":
    main()