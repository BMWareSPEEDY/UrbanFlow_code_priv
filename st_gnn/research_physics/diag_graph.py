import osmnx as ox
import numpy as np
import torch
import pandas as pd

BASE = r'D:\CODES\PYTHON_CODES\UrbanFLOW'
regions = ['hyderabad', 'chennai', 'surat', 'hsr']

for reg in regions:
    f = 'bengaluru_complete_graph.graphml' if reg == 'hsr' else f'city_{reg}_graph.graphml'
    G = ox.load_graphml(f"{BASE}\\{f}")
    ups = downs = flat = 0
    grade_sign_ok = True
    for u, v, k, d in G.edges(keys=True, data=True):
        if u == v:
            continue
        g = float(d.get('grade', 0.0))
        eu = float(G.nodes[u].get('elevation', 880.0))
        ev = float(G.nodes[v].get('elevation', 880.0))
        L = max(float(d.get('length', 10.0)), 1e-6)
        calc = (ev - eu) / L
        if abs(g - calc) > 1e-6 and abs(g + calc) > 1e-6:
            grade_sign_ok = False
        if g > 1e-6:
            ups += 1
        elif g < -1e-6:
            downs += 1
        else:
            flat += 1
    tot = ups + downs + flat
    print(f"{reg}: nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
          f"uphill(grade>0)={ups/tot:.3f} downhill={downs/tot:.3f} flat={flat/tot:.3f} "
          f"grade==(ev-eu)/L: {grade_sign_ok}")

    dl = torch.load(f"{BASE}\\multi_scenario_pyg_dataset.pt", weights_only=False)
    g300 = [g for g in dl if g.region == reg and int(g.rain_intensity) == 300][0]
    y = g300.y.numpy().ravel()
    nodes = list(G.nodes())
    elev = np.array([float(G.nodes[n].get('elevation', 880.0)) for n in nodes])
    imp = np.array([float(G.nodes[n].get('impervious_ratio', 0.2)) for n in nodes])
    # old directed accumulation (mirror create_combined_dataset)
    SUBC = 0.5
    acc = {n: SUBC * imp[i] for i, n in enumerate(nodes)}
    out_nb = {n: [] for n in nodes}
    for u, v, k, d in G.edges(keys=True, data=True):
        if u in out_nb and u != v:
            out_nb[u].append(v)
    ordered = sorted(nodes, key=lambda n: float(G.nodes[n].get('elevation', 880.0)), reverse=True)
    for u in ordered:
        for v in out_nb[u]:
            acc[v] += acc[u]
    acc_arr = np.array([acc[n] for n in nodes])
    print(f"  elev: min={elev.min():.1f} max={elev.max():.1f}; corr(y,elev)={np.corrcoef(y, elev)[0,1]:.3f}")
    print(f"  corr(y, directed_acc_imp)={np.corrcoef(y, acc_arr)[0,1]:.3f} "
          f"corr(log1p(y), log1p(acc_imp))={np.corrcoef(np.log1p(y), np.log1p(acc_arr))[0,1]:.3f}")
    print(f"  corr(y, imp)={np.corrcoef(y, imp)[0,1]:.3f}")
