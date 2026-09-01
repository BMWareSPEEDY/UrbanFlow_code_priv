import sys
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)
BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"
dl = torch.load(BASE + r"\multi_scenario_pyg_dataset.pt", weights_only=False)
g = dl[0]  # hsr I=20
x = g.x.numpy()
ei = g.edge_index.numpy()
n = x.shape[0]

in_deg = x[:, 5]
out_deg = x[:, 6]
print("directed?", "in_deg==out_deg for all:", bool((in_deg == out_deg).all()))
print("max |in-out|:", np.abs(in_deg - out_deg).max())
rel_drop = x[:, 2]
# fraction of edges going downhill in edge direction (u -> v)
downhill = np.mean(rel_drop[ei[1]] > rel_drop[ei[0]])
same = np.mean(rel_drop[ei[1]] == rel_drop[ei[0]])
print(f"edges downhill along edge direction: {downhill:.3f}, same-elev: {same:.3f}")

# self loops?
nself = int((ei[0] == ei[1]).sum())
print("self loops:", nself)

# replicate original accumulation exactly; find worst-mismatch nodes
out_list = [[] for _ in range(n)]
for i in range(ei.shape[1]):
    u, v = int(ei[0, i]), int(ei[1, i])
    if u != v:
        out_list[u].append(v)
acc = np.full(n, 0.5)
order = np.argsort(rel_drop)
for u in order:
    a = acc[u]
    for v in out_list[u]:
        acc[v] += a
log_area = x[:, 12]
err = np.abs(np.log1p(acc) - log_area)
worst = np.argsort(err)[-10:][::-1]
for w in worst:
    print(f"node {w}: log_area={log_area[w]:.3f} my={np.log1p(acc[w]):.3f} rel_drop={rel_drop[w]:.3f} "
          f"in={in_deg[w]:.0f} out={out_deg[w]:.0f} n_in_edges={np.sum(ei[1]==w)} n_out_edges={np.sum(ei[0]==w)}")

# correlation of rel_drop with in/out degree (is low elevation correlated with high in_deg?)
print("corr(rel_drop, in_deg):", np.corrcoef(rel_drop, in_deg)[0, 1].round(4))
print("corr(rel_drop, out_deg):", np.corrcoef(rel_drop, out_deg)[0, 1].round(4))
