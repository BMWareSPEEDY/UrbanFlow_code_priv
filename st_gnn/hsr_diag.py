import torch
import numpy as np

dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_pyg_dataset.pt", weights_only=False)
regs = ['hsr', 'bellandur', 'whitefield', 'ecity', 'koramangala']
for I in [200.0, 300.0]:
    print(f"=== I={I:.0f} ===")
    print(f"{'region':<12} {'mean_y':>7} {'imp':>6} {'accimp_ha':>8} {'max_in_g':>9} {'elev2':>6} {'sag':>6} {'sink%':>6} {'pathcap':>8} {'hops':>6} {'outdeg':>6}")
    for rg in regs:
        ys, xs = [], []
        for g in dl:
            if g.region == rg and abs(g.rain_intensity - I) < 1e-6:
                ys.append(g.y.numpy().ravel())
                xs.append(g.x.numpy())
        y = np.concatenate(ys)
        x = np.concatenate(xs)
        print(f"{rg:<12} {y.mean():7.3f} {x[:,3].mean():6.3f} {np.expm1(x[:,13]).mean():8.2f} "
              f"{x[:,9].mean():9.4f} {x[:,17].mean():6.2f} {x[:,10].mean():6.3f} {x[:,8].mean()*100:6.1f} "
              f"{np.expm1(x[:,20]).mean():8.2f} {x[:,21].mean():6.1f} {x[:,6].mean():6.2f}")