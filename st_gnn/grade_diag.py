import torch
import numpy as np

dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_full22_pyg_dataset.pt", weights_only=False)
regs = ['hsr', 'bellandur', 'tokyo', 'paris', 'singapore', 'london', 'newyork', 'varanasi', 'rajkot', 'agra']
print(f"{'region':<12} {'elev2':>6} {'flood%':>6} {'|grade|':>8} {'max_in_g':>8} {'grade>0%':>8} {'outdeg':>6} {'accimp':>7}")
for rg in regs:
    for g in dl:
        if g.region == rg and abs(g.rain_intensity - 300) < 1e-6:
            x = g.x.numpy()
            y = g.y.numpy().ravel()
            ea = g.edge_attr.numpy()
            e2 = x[:, 17]
            g_pos = (ea[:, 1] > 0).mean() * 100
            print(f"{rg:<12} {e2.mean():6.2f} {(y>=0.15).mean()*100:6.1f} {np.abs(ea[:,1]).mean():8.4f} "
                  f"{x[:,9].mean():8.4f} {g_pos:8.1f} {x[:,6].mean():6.2f} {np.expm1(x[:,13]).mean():7.2f}")
            break