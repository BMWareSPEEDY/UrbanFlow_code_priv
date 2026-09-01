import sys
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)
BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"
dl = torch.load(BASE + r"\multi_scenario_pyg_dataset.pt", weights_only=False)
for idx in [0, 2, 8]:
    g = dl[idx]
    x = g.x.numpy()
    ei = g.edge_index.numpy()
    n = x.shape[0]
    rel_drop = x[:, 2]
    log_area = x[:, 12]
    out = np.argsort(rel_drop)
    for rev in [False, True]:
        out_list = [[] for _ in range(n)]
        for i in range(ei.shape[1]):
            u, v = int(ei[0, i]), int(ei[1, i])
            if u == v:
                continue
            a, b = (v, u) if rev else (u, v)
            out_list[a].append(b)
        acc = np.full(n, 0.5)
        for u in out:
            c = acc[u]
            for v in out_list[u]:
                acc[v] += c
        corr = np.corrcoef(np.log1p(acc), log_area)[0, 1]
        err = np.abs(np.log1p(acc) - log_area).max()
        print(f"graph {idx} rev={rev}: corr={corr:.5f} max_err={err:.3f}", flush=True)
