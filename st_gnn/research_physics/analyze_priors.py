import os
import numpy as np
import torch
import osmnx as ox
import pandas as pd
from physics_lib import CityPhysics, qfull


def rankdata(a):
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    return ranks


def spearmanr(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]

BASE = r'D:\CODES\PYTHON_CODES\UrbanFLOW'
REGS = ['hyderabad', 'chennai', 'surat', 'hsr']
FILES = {'hsr': 'bengaluru_complete_graph.graphml'}
FILES.update({r: f'city_{r}_graph.graphml' for r in REGS if r != 'hsr'})
INTENSITIES = [20.0, 50.0, 80.0, 120.0, 150.0, 200.0, 250.0, 300.0]


def old_manning_prior(cp, intensity):
    """Reproduce check_manning_prior.py: directed (traffic-direction) accumulation,
    steepest outgoing conduit by |grade|, q = 0.9*I*acc_imp*1e4/3.6e6."""
    G = cp.G
    nodes = cp.nodes
    N = cp.n_nodes
    acc_imp_dir = np.zeros(N)
    out_steep = np.zeros(N)
    out_n = np.zeros(N)
    out_grade = np.zeros(N)
    for j in range(N):
        acc_imp_dir[j] = 0.5 * cp.imp[j]
    out_nb = [[] for _ in range(N)]
    for u, v, k, d in G.edges(keys=True, data=True):
        if u != v:
            out_nb[cp.idx[u]].append(cp.idx[v])
    order = np.argsort(-cp.elev)
    for u in order:
        for v in out_nb[u]:
            acc_imp_dir[v] += acc_imp_dir[u]
    for e in range(cp.U.size):
        u = cp.U[e]
        g = abs(cp.GR[e])
        if g > out_grade[u]:
            out_grade[u] = g
            out_n[u] = cp.NARR[e]
    d = np.zeros(N)
    for j in range(N):
        q = intensity * acc_imp_dir[j] * 1e4 * 0.9 / 3.6e6
        n = max(out_n[j], 1e-4)
        s = max(out_grade[j], 1e-6)
        if q <= 0:
            continue
        # manning bisection h in [0,3]
        lo, hi = 0.0, 3.0
        for _ in range(40):
            h = 0.5 * (lo + hi)
            A = h; P = 1.0 + 2.0 * h
            Qh = (1.0 / n) * A * (A / P) ** (2.0 / 3.0) * np.sqrt(s)
            if Qh < q:
                lo = h
            else:
                hi = h
        d[j] = 0.5 * (lo + hi)
    return d


def main():
    dl = torch.load(os.path.join(BASE, 'multi_scenario_pyg_dataset.pt'), weights_only=False)
    dl_by = {(g.region, int(g.rain_intensity)): g for g in dl}

    physics = {}
    for r in REGS:
        G = ox.load_graphml(os.path.join(BASE, FILES[r]))
        physics[r] = CityPhysics(G)

    prior_fns = {
        'undir_acc_imp_log': lambda cp, I: np.log1p(cp.acc_imp),
        'undir_acc_area_log': lambda cp, I: np.log1p(cp.acc_area),
        'drain_ratio_log': lambda cp, I: np.log1p(np.maximum(cp.q_catch(I) / (cp.qcap_all + 1e-12), 0.0)),
        'depth_peak': lambda cp, I: cp.depth_peak(I),
        'depth_final': lambda cp, I: cp.depth_final(I),
        'depth_norm_only': lambda cp, I: cp.depth_norm_only(I),
        'depth_ratio_model': lambda cp, I: cp.depth_ratio_model(I),
        'depth_all_peak': lambda cp, I: cp.depth_all_peak(I),
        'depth_all_final': lambda cp, I: cp.depth_all_final(I),
        'depth_spill_fill': lambda cp, I: cp.depth_spill_fill(),
        'depth_volume_fill': lambda cp, I: cp.depth_volume_fill(I),
        'depth_minimax': lambda cp, I: cp.depth_minimax(),
        'depth_minimax_vol': lambda cp, I: cp.depth_minimax_vol(I),
        'log_qcatch': lambda cp, I: np.log1p(cp.q_catch(I)),
        'log_qcap_all': lambda cp, I: np.log1p(cp.qcap_all),
        'log_local_qrun': lambda cp, I: np.log1p(cp.q_run_local(I)),
        'rim_depth': lambda cp, I: np.log1p(cp.rim_depth),
        'tc_sink': lambda cp, I: cp.tc_sink,
        'sink_drop': lambda cp, I: np.log1p(cp.sink_drop),
        'sink_grad': lambda cp, I: cp.sink_grad,
        'old_manning': old_manning_prior,
        'x_log_area(directed)': lambda cp, I: None,  # filled from dataset
        'x_log_imp_area(directed)': lambda cp, I: None,
    }

    rows = []
    for pname, fn in prior_fns.items():
        for I in INTENSITIES:
            ys, ps = [], []
            for r in REGS:
                g = dl_by[(r, int(I))]
                y = g.y.numpy().ravel()
                cp = physics[r]
                if pname == 'x_log_area(directed)':
                    p = g.x.numpy()[:, 12]
                elif pname == 'x_log_imp_area(directed)':
                    p = g.x.numpy()[:, 13]
                else:
                    p = np.asarray(fn(cp, I), dtype=float)
                ys.append(y); ps.append(p)
            y = np.concatenate(ys); p = np.concatenate(ps)
            mask = np.isfinite(p) & np.isfinite(y)
            y, p = y[mask], p[mask]
            if p.std() == 0 or len(y) < 100:
                rows.append(dict(prior=pname, I=I, n=len(y), corr_y=np.nan,
                                 corr_log=np.nan, spearman=np.nan, r2_log=np.nan))
                continue
            ly, lp = np.log1p(y), np.log1p(np.maximum(p, 0.0))
            corr_y = np.corrcoef(y, p)[0, 1]
            corr_log = np.corrcoef(ly, lp)[0, 1]
            sp = spearmanr(y, p)
            r2 = corr_log ** 2
            rows.append(dict(prior=pname, I=I, n=len(y), corr_y=corr_y,
                             corr_log=corr_log, spearman=sp, r2_log=r2))

    df = pd.DataFrame(rows)
    df.to_csv('prior_correlations.csv', index=False)

    def fmt(v):
        return '   --' if (v is None or (isinstance(v, float) and np.isnan(v))) else f'{v:+.3f}'

    # summary table: for key intensities
    print(f"{'prior':<26} {'I300 log':>8} {'I300 y':>8} {'I300 sp':>8} {'I150 log':>8} {'I20 log':>8} {'R2@300':>8}")
    for pname in prior_fns:
        d300 = df[(df.prior == pname) & (df.I == 300.0)].iloc[0]
        d150 = df[(df.prior == pname) & (df.I == 150.0)].iloc[0]
        d20 = df[(df.prior == pname) & (df.I == 20.0)].iloc[0]
        print(f"{pname:<26} {fmt(d300.corr_log):>8} {fmt(d300.corr_y):>8} {fmt(d300.spearman):>8} "
              f"{fmt(d150.corr_log):>8} {fmt(d20.corr_log):>8} {fmt(d300.r2_log):>8}")

    # per-city for the best priors at I=300
    print('\nPer-city at I=300 (corr log1p prior vs log1p y):')
    top = ['depth_final', 'depth_peak', 'drain_ratio_log', 'undir_acc_imp_log', 'old_manning', 'rim_depth']
    print(f"{'prior':<26}", *[f'{r:>10}' for r in REGS])
    for pname in top:
        fn = prior_fns[pname]
        vals = []
        for r in REGS:
            g = dl_by[(r, 300)]
            y = g.y.numpy().ravel()
            cp = physics[r]
            if pname in ('x_log_area(directed)', 'x_log_imp_area(directed)'):
                p = g.x.numpy()[:, 12]
            else:
                p = np.asarray(fn(cp, 300.0), dtype=float)
            m = np.isfinite(p) & np.isfinite(y)
            if np.std(p[m]) == 0:
                vals.append(float('nan'))
            else:
                vals.append(np.corrcoef(np.log1p(y[m]), np.log1p(np.maximum(p[m], 0.0)))[0, 1])
        print(f"{pname:<26}", *[f'{v:>10.3f}' if not np.isnan(v) else f'{"--":>10}' for v in vals])

    # why old prior anti-correlated: correlation of its components
    print('\nOld-prior autopsy at I=300 (hyderabad):')
    cp = physics['hyderabad']
    g = dl_by[('hyderabad', 300)]
    y = g.y.numpy().ravel()
    om = old_manning_prior(cp, 300.0)
    m = y > 0.01
    print(f"  old_manning vs y: corr={np.corrcoef(om, y)[0,1]:+.3f}, "
          f"corr(log)={np.corrcoef(np.log1p(om), np.log1p(y))[0,1]:+.3f}")
    print(f"  frac nodes prior at 3m cap: {np.mean(om >= 2.99):.3f}")
    acc = np.array([0.5 * cp.imp[j] for j in range(cp.n_nodes)])
    out_nb = [[] for _ in range(cp.n_nodes)]
    for u, v, k, d in cp.G.edges(keys=True, data=True):
        if u != v:
            out_nb[cp.idx[u]].append(cp.idx[v])
    for u in np.argsort(-cp.elev):
        for v in out_nb[u]:
            acc[v] += acc[u]
    print(f"  corr(log(acc_imp_directed), y)={np.corrcoef(np.log1p(acc), y)[0,1]:+.3f} "
          f"corr(log(acc_imp_directed), log(y))={np.corrcoef(np.log1p(acc), np.log1p(y))[0,1]:+.3f}")


if __name__ == '__main__':
    main()
