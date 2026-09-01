import numpy as np
import torch
import osmnx as ox
import pandas as pd
from physics_lib import CityPhysics

BASE = r'D:\CODES\PYTHON_CODES\UrbanFLOW'
dl = torch.load(f'{BASE}\\multi_scenario_pyg_dataset.pt', weights_only=False)
dl_by = {(g.region, int(g.rain_intensity)): g for g in dl}

for reg, f in [('hyderabad', 'city_hyderabad_graph.graphml'),
               ('surat', 'city_surat_graph.graphml')]:
    G = ox.load_graphml(f'{BASE}\\{f}')
    cp = CityPhysics(G)
    g = dl_by[(reg, 300)]
    y = g.y.numpy().ravel()
    print(f'--- {reg} @300 ---')
    print(f'  pits: {cp.pit.sum()} / {cp.n_nodes}')
    for label, mask in [('pit', cp.pit), ('non-pit', ~cp.pit)]:
        print(f'  {label}: n={mask.sum()} mean_y={y[mask].mean():.3f} '
              f'frac>0.5={np.mean(y[mask]>0.5):.3f} frac>=2.99={np.mean(y[mask]>=2.99):.3f}')
    # deciles of rim_depth for non-pits? rim_depth>0 means pit-ish (no downhill but...) actually rim>0 any node
    print(f'  corr(y, rim_depth)={np.corrcoef(y, cp.rim_depth)[0,1]:+.3f} '
          f'corr(y, qcap_all)={np.corrcoef(y, np.log1p(cp.qcap_all))[0,1]:+.3f}')
    print(f'  corr(elev, acc_area)={np.corrcoef(cp.elev, np.log1p(cp.acc_area))[0,1]:+.3f} '
          f'corr(elev, y)={np.corrcoef(cp.elev, y)[0,1]:+.3f}')
    # deepest 30 nodes: what are they?
    idx = np.argsort(-y)[:30]
    print('  deepest 30 nodes:')
    for i in idx[:15]:
        print(f'    y={y[i]:.3f} elev={cp.elev[i]:.1f} pit={cp.pit[i]} '
              f'rim={cp.rim_depth[i]:.2f} acc_area={cp.acc_area[i]:.1f}ha '
              f'qcap={cp.qcap_all[i]:.2f} imp={cp.imp[i]:.2f} '
              f'down_n={len(cp.nn_all[i])}')
    # shallowest 10 among non-pits with big acc area
    idx2 = np.argsort(-cp.acc_area)[:20]
    print('  top-20 by catchment (undirected):')
    for i in idx2:
        print(f'    acc_area={cp.acc_area[i]:.0f}ha y={y[i]:.3f} elev={cp.elev[i]:.1f} pit={cp.pit[i]} qcap={cp.qcap_all[i]:.2f}')