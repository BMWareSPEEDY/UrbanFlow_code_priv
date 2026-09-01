import sys
import numpy as np
import torch
from torch_geometric.data import Batch

sys.stdout.reconfigure(line_buffering=True)


def r2(t, p):
    t = np.ravel(t); p = np.ravel(p)
    return 1 - np.sum((t - p) ** 2) / max(1e-9, np.sum((t - t.mean()) ** 2))


def main():
    dl = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
    cities = sorted({g.city for g in dl})
    feats = ['rel_drop', 'in_deg', 'out_deg', 'accum_score', 'is_sink',
             'max_in_grade', 'sag_index', 'log_area', 'log_imp_area', 'dist_frac']
    fnames = ['rel_x', 'rel_y', 'rel_drop', 'imp', 'manning_n', 'in_deg', 'out_deg',
              'accum_score', 'is_sink', 'max_in_grade', 'sag_index', 'hydraulic_capacity',
              'log_area', 'log_imp_area', 'dist_frac', 'intensity', 'duration']
    fmap = {f: i for i, f in enumerate(fnames)}
    print("Feature-target correlations per city (intensity held constant is not possible, but shows consistency):")
    print(f"{'city':<12}" + "".join(f"{f:>12}" for f in feats) + f"{'n_nodes':>10}")
    for c in cities:
        gs = [g for g in dl if g.city == c]
        x = torch.cat([g.x for g in gs]).numpy()
        y = torch.cat([g.y for g in gs]).numpy().ravel()
        if y.std() < 1e-9:
            continue
        row = []
        for i, f in enumerate(feats):
            idx = list(dl[0].feature_names).index(f) if hasattr(dl[0], 'feature_names') else None
            break
        row = []
        for f in feats:
            idx = fmap[f]
            col = x[:, idx]
            if col.std() < 1e-9:
                row.append(0.0)
            else:
                row.append(float(np.corrcoef(col, y)[0, 1]))
        print(f"{c:<12}" + "".join(f"{v:>12.3f}" for v in row) + f"{len(y):>10}")


if __name__ == "__main__":
    main()