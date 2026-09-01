import sys
import numpy as np
import torch
from torch_geometric.data import Batch

sys.stdout.reconfigure(line_buffering=True)


def main():
    dl = torch.load("multi_scenario_pyg_dataset.pt", weights_only=False)
    # feature index map
    f = {'rel_x':0,'rel_y':1,'rel_drop':2,'imp':3,'manning_n':4,'in_deg':5,'out_deg':6,
         'accum_score':7,'is_sink':8,'max_in_grade':9,'sag_index':10,'hydraulic_capacity':11,
         'log_area':12,'log_imp_area':13,'dist_frac':14,'intensity':15,'duration':16}

    print("Physics-feature correlation with depth (log-log and raw):")
    print(f"{'city':<13} {'log(inflow/sqrt(grade))':>22} {'intensity*area':>16} {'area/sqrt(g)':>14} {'n':>7}")
    for city in sorted({g.city for g in dl}):
        gs = [g for g in dl if g.city == city]
        x = torch.cat([g.x for g in gs]).numpy()
        y = torch.cat([g.y for g in gs]).numpy().ravel()

        inflow = x[:, f['intensity']] * np.exp(x[:, f['log_imp_area']])  # intensity * imperv area (ha)
        grade = np.abs(x[:, f['max_in_grade']]) + 1e-4
        g_sqrt = np.sqrt(grade)
        phys = inflow / g_sqrt

        safe = y > 0.01
        if safe.sum() < 100:
            continue
        c1 = np.corrcoef(np.log1p(phys[safe]), np.log1p(y[safe]))[0, 1]
        c2 = np.corrcoef(np.log1p(inflow[safe]), np.log1p(y[safe]))[0, 1]
        c3 = np.corrcoef(np.log1p(np.exp(x[safe, f['log_area']]) / g_sqrt[safe]), np.log1p(y[safe]))[0, 1]
        print(f"{city:<13} {c1:>22.4f} {c2:>16.4f} {c3:>14.4f} {safe.sum():>7}")


if __name__ == "__main__":
    main()