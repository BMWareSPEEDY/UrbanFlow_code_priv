"""Cheap deterministic physics priors for SWMM junction depth, per node.

Key facts about the SWMM setup (generate_city_b3_swmm_targets.py):
- junction per node: elev, MaxDepth 3.0, ponded area 100 m2
- subcatchment 0.5 ha per node, %imperv = imp*100
- Horton 75 -> 3 mm/hr, S-Imperv 2mm (PctZero 25), S-Perv 5mm
- conduit per directed edge: RECT_OPEN 1.0 x 1.5, Manning n, slope = grade
- rain: constant I mm/hr for 65 min (all exporters), sim ends at 2h
- node order must match list(G.nodes()) used by create_combined_dataset.py
"""
import numpy as np

BASE = r'D:\CODES\PYTHON_CODES\UrbanFLOW'

SUBC_HA = 0.5          # ha per node
SUBC_M2 = SUBC_HA * 1e4
W, HMAX_COND = 1.0, 1.5  # RECT_OPEN width, height
APOND = 100.0          # ponded area m2
MAXD = 3.0             # junction max depth
T_RAIN_H = 65.0 / 60.0
T_SIM_H = 2.0
S_IMP = 2.0            # mm, PctZero=25 -> 1.5 mm effective
S_PERV = 5.0           # mm  -> 3.75 mm effective
PCTZERO = 0.25
F_MIN, F_MAX, KD = 3.0, 75.0, 4.0  # Horton mm/hr


def pervious_runoff_mm(intensity_mmhr):
    """Effective pervious runoff depth (mm) over the 65-min storm via Horton decay."""
    nt = 260
    t = np.linspace(0.0, T_RAIN_H, nt)
    f = F_MIN + (F_MAX - F_MIN) * np.exp(-KD * t)
    run = np.maximum(intensity_mmhr - f, 0.0)
    return float(np.trapezoid(run, t) if hasattr(np, 'trapezoid') else np.trapz(run, t))


def imperv_runoff_mm(intensity_mmhr):
    return max(0.0, intensity_mmhr * T_RAIN_H - S_IMP * (1.0 - PCTZERO))


def qfull(n, s):
    """Full-flow capacity (m3/s) of RECT_OPEN 1.0x1.5."""
    A, R = 1.5, 0.375
    return (1.0 / max(n, 1e-4)) * A * R ** (2.0 / 3.0) * np.sqrt(max(s, 1e-9))


def qcurve_at_h(h, n_arr, s_arr):
    """Sum of Manning Q(h) over conduits: open channel, w=1, h in [0,1.5]."""
    h = np.clip(h, 1e-6, HMAX_COND)
    A = W * h
    R = A / (W + 2.0 * h)
    v = (1.0 / np.maximum(n_arr, 1e-4)) * R ** (2.0 / 3.0) * np.sqrt(np.maximum(s_arr, 1e-9))
    return float(np.sum(A * v))


def normal_depth(q, n_arr, s_arr):
    """Bisection on the summed Manning curve. Returns depth in [0,1.5]; if q exceeds
    full capacity returns 1.5 (crown)."""
    if len(n_arr) == 0 or q <= 0:
        return 0.0
    lo, hi = 0.0, HMAX_COND
    if qcurve_at_h(hi, n_arr, s_arr) < q:
        return HMAX_COND
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if qcurve_at_h(mid, n_arr, s_arr) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class CityPhysics:
    def __init__(self, G):
        self.G = G
        self.nodes = list(G.nodes())
        self.n_nodes = len(self.nodes)
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        elev = np.array([float(G.nodes[n].get('elevation', 880.0)) for n in self.nodes])
        imp = np.array([float(G.nodes[n].get('impervious_ratio', 0.2)) for n in self.nodes])
        self.elev, self.imp = elev, imp

        U, V, LEN, N, GR = [], [], [], [], []
        for u, v, k, d in G.edges(keys=True, data=True):
            if u == v:
                continue
            U.append(self.idx[u]); V.append(self.idx[v])
            LEN.append(max(float(d.get('length', 10.0)), 1e-3))
            N.append(float(d.get('manning_n', 0.013)))
            GR.append(float(d.get('grade', 0.0)))  # (e_v - e_u)/L ; downhill => >0
        self.U = np.array(U); self.V = np.array(V)
        self.LEN = np.array(LEN); self.NARR = np.array(N)
        self.GR = np.array(GR)

        # downhill edges: target strictly lower
        self.down = self.GR > 1e-9
        self.up = self.GR < -1e-9
        self.flat_e = ~(self.down | self.up)

        # ---- steepest-descent DAG (undirected, slope-driven) ----
        down_cap = np.zeros(self.n_nodes)
        down_grad = np.zeros(self.n_nodes)
        down_n = np.zeros(self.n_nodes)
        down_id = np.full(self.n_nodes, -1, dtype=np.int64)
        for e in np.where(self.down)[0]:
            u, v = self.U[e], self.V[e]
            s = self.GR[e]
            c = qfull(self.NARR[e], s)
            if c > down_cap[u]:
                down_cap[u] = c
                down_grad[u] = s
                down_n[u] = self.NARR[e]
                down_id[u] = v
        self.down_cap = down_cap          # capacity of steepest downhill conduit
        self.down_grad = down_grad
        self.down_n = down_n
        self.down_id = down_id            # -1 => pit (no downhill neighbor)
        self.pit = down_id < 0

        # children: all nodes whose steepest descent points here
        self.children = [[] for _ in range(self.n_nodes)]
        for u in range(self.n_nodes):
            p = down_id[u]
            if p >= 0:
                self.children[p].append(u)

        # sum of capacities of ALL downhill conduits (outlet capacity)
        qcap_all = np.zeros(self.n_nodes)
        nn_all = []
        ss_all = []
        for e in np.where(self.down)[0]:
            u = self.U[e]
            qcap_all[u] += qfull(self.NARR[e], self.GR[e])
        self.qcap_all = qcap_all
        for u in range(self.n_nodes):
            mask = self.down & (self.U == u)
            nn_all.append(self.NARR[mask]); ss_all.append(self.GR[mask])
        self.nn_all, self.ss_all = nn_all, ss_all

        # ---- undirected steepest-descent accumulation ----
        self.acc_imp = np.zeros(self.n_nodes)
        self.acc_perv = np.zeros(self.n_nodes)
        self.acc_area = np.zeros(self.n_nodes)
        for j in range(self.n_nodes):
            self.acc_imp[j] = SUBC_HA * imp[j]
            self.acc_perv[j] = SUBC_HA * (1.0 - imp[j])
            self.acc_area[j] = SUBC_HA
        order = np.argsort(-elev)  # high to low
        for u in order:
            p = down_id[u]
            if p >= 0:
                self.acc_imp[p] += self.acc_imp[u]
                self.acc_perv[p] += self.acc_perv[u]
                self.acc_area[p] += self.acc_area[u]

        # ---- downhill path to sink: travel time, drop, gradient ----
        self.tc_sink = np.zeros(self.n_nodes)
        self.sink_drop = np.zeros(self.n_nodes)
        self.sink_len = np.zeros(self.n_nodes)
        for u in order:
            p = down_id[u]
            if p >= 0:
                v_full = (1.0 / max(self.down_n[u], 1e-4)) * (HMAX_COND / 4.0) ** (2.0 / 3.0) * np.sqrt(max(self.down_grad[u], 1e-9))
                L = float(np.sum(self.LEN[(self.U == u) & (self.V == down_id[u]) & self.down]) or 1.0)
                self.tc_sink[u] = self.tc_sink[p] + L / max(v_full, 1e-6)
                self.sink_drop[u] = self.sink_drop[p] + max(self.elev[u] - self.elev[p], 0.0)
                self.sink_len[u] = self.sink_len[p] + L
        self.sink_grad = np.divide(self.sink_drop, self.sink_len + 1e-6)

        # ---- fill-before-spill (pit rim depth) ----
        self.rim_depth = np.zeros(self.n_nodes)
        for e in range(self.U.size):
            u, v = self.U[e], self.V[e]
            r = self.elev[v] - self.elev[u]
            if r > self.rim_depth[u]:
                self.rim_depth[u] = r
        self.rim_depth = np.maximum(self.rim_depth, 0.0)

    # ------------------------------------------------------------------
    def runoff_rates(self, intensity):
        """q_imp, q_perv in mm/hr effective (depression-storage corrected)."""
        r_imp = imperv_runoff_mm(intensity) / T_RAIN_H
        r_perv = pervious_runoff_mm(intensity) / T_RAIN_H
        return r_imp, r_perv

    def q_run_local(self, intensity):
        r_imp, r_perv = self.runoff_rates(intensity)
        return SUBC_M2 / 3.6e6 * (self.imp * r_imp + (1.0 - self.imp) * r_perv)

    def q_catch(self, intensity):
        """Runoff rate (m3/s) from the node's full undirected catchment (no capping)."""
        r_imp, r_perv = self.runoff_rates(intensity)
        return SUBC_M2 / 3.6e6 * (self.acc_imp * r_imp + self.acc_perv * r_perv)

    def cascade(self, intensity):
        """Kinematic cascade with capacity capping. Returns q_in, ponded volume (m3),
        excess rate at each node (peak ponding)."""
        q_run = self.q_run_local(intensity)
        pond_vol = np.zeros(self.n_nodes)
        excess = np.zeros(self.n_nodes)
        order = np.argsort(-self.elev)
        inflow = q_run.copy()
        for u in order:
            ex = max(0.0, inflow[u] - self.qcap_all[u])
            excess[u] = ex
            pond_vol[u] = ex * T_RAIN_H * 3600.0
            out = inflow[u] - ex
            p = self.down_id[u]
            if p >= 0:
                inflow[p] += out
        return inflow, pond_vol, excess

    def depth_peak(self, intensity):
        """d = h_norm(q_in) if free; else 1.5 + excess*T/Apond, capped 3.0."""
        inflow, pond_vol, _ = self.cascade(intensity)
        d = np.zeros(self.n_nodes)
        for j in range(self.n_nodes):
            q = inflow[j]
            if q <= 0:
                d[j] = 0.0
            elif q <= self.qcap_all[j] + 1e-12:
                d[j] = normal_depth(q, self.nn_all[j], self.ss_all[j])
            else:
                d[j] = HMAX_COND + pond_vol[j] / APOND
        return np.minimum(d, MAXD)

    def depth_final(self, intensity):
        """Peak ponding volume drained until t=2h at full capacity, + crown."""
        inflow, pond_vol, _ = self.cascade(intensity)
        d = np.zeros(self.n_nodes)
        for j in range(self.n_nodes):
            q = inflow[j]
            cap = self.qcap_all[j]
            if q <= 0:
                d[j] = 0.0
            elif q <= cap + 1e-12:
                d[j] = normal_depth(q, self.nn_all[j], self.ss_all[j])
            else:
                v_end = max(0.0, pond_vol[j] - cap * (T_SIM_H - T_RAIN_H) * 3600.0)
                d[j] = HMAX_COND + v_end / APOND
        return np.minimum(d, MAXD)

    def depth_norm_only(self, intensity):
        inflow, _, _ = self.cascade(intensity)
        d = np.zeros(self.n_nodes)
        for j in range(self.n_nodes):
            d[j] = normal_depth(inflow[j], self.nn_all[j], self.ss_all[j])
        return d

    def depth_ratio_model(self, intensity):
        """min(3, 1.5 + excess volume/area), excess = Q_catch - qcap_all (no cascade)."""
        qc = self.q_catch(intensity)
        ex = np.maximum(qc - self.qcap_all, 0.0) * T_RAIN_H * 3600.0 / APOND
        return np.minimum(HMAX_COND + ex, MAXD)

    # ------------------------------------------------------------------
    def all_edges_cascade(self, intensity):
        """Kinematic routing through ALL downhill conduit edges (not just steepest).
        Process nodes high->low; each node's inflow = local runoff + inflow from all
        uphill conduits; outflow splits to downhill conduits proportionally to their
        full-flow capacity (capped). Excess ponds at the node.
        Returns inflow, pond_vol (m3), excess (m3/s)."""
        q_run = self.q_run_local(intensity)
        inflow = q_run.copy()
        pond_vol = np.zeros(self.n_nodes)
        excess = np.zeros(self.n_nodes)
        # downhill edge lists per node (u -> [w,...]) with capacities
        edges_by_u = [[] for _ in range(self.n_nodes)]
        for e in np.where(self.down)[0]:
            edges_by_u[self.U[e]].append((self.V[e], qfull(self.NARR[e], self.GR[e])))
        out_flow = {}
        order = np.argsort(-self.elev)
        for u in order:
            q = inflow[u]
            outs = edges_by_u[u]
            if len(outs) == 0:
                excess[u] = q
                pond_vol[u] = q * T_RAIN_H * 3600.0
                continue
            cap = sum(c for _, c in outs)
            if q <= cap:
                for w, c in outs:
                    out_flow[(u, w)] = q * c / cap
            else:
                for w, c in outs:
                    out_flow[(u, w)] = c
                excess[u] = q - cap
                pond_vol[u] = excess[u] * T_RAIN_H * 3600.0
            for w, c in outs:
                inflow[w] += out_flow[(u, w)]
        return inflow, pond_vol, excess

    def depth_all_peak(self, intensity):
        inflow, pond_vol, _ = self.all_edges_cascade(intensity)
        d = np.zeros(self.n_nodes)
        for j in range(self.n_nodes):
            q = inflow[j]
            cap = self.qcap_all[j]
            if q <= 0:
                d[j] = 0.0
            elif q <= cap + 1e-12:
                d[j] = normal_depth(q, self.nn_all[j], self.ss_all[j])
            else:
                d[j] = HMAX_COND + pond_vol[j] / APOND
        return np.minimum(d, MAXD)

    def depth_all_final(self, intensity):
        inflow, pond_vol, _ = self.all_edges_cascade(intensity)
        d = np.zeros(self.n_nodes)
        for j in range(self.n_nodes):
            q = inflow[j]
            cap = self.qcap_all[j]
            if q <= 0:
                d[j] = 0.0
            elif q <= cap + 1e-12:
                d[j] = normal_depth(q, self.nn_all[j], self.ss_all[j])
            else:
                v_end = max(0.0, pond_vol[j] - cap * (T_SIM_H - T_RAIN_H) * 3600.0)
                d[j] = HMAX_COND + v_end / APOND
        return np.minimum(d, MAXD)

    # ------------------------------------------------------------------
    # Depression-fill ("lake") model: the network fills to a water surface
    # level per connected basin; depth = WSE - elev. WSE determined by the
    # basin's lowest spill (pass) elevation, optionally volume-corrected.
    def basin_fill(self):
        """Return per-node fill depth for two WSE variants:
        A) WSE = lowest spill elevation of the connected component
        B) volume-limited WSE from runoff (needs intensity, computed by caller)
        Returns comp_ids, wse_spill, spill_depths."""
        parent = list(range(self.n_nodes))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        for e in range(self.U.size):
            union(self.U[e], self.V[e])
        comps = {}
        for j in range(self.n_nodes):
            r = find(j)
            comps.setdefault(r, []).append(j)
        spill = np.full(self.n_nodes, np.inf)
        for e in range(self.U.size):
            u, v = self.U[e], self.V[e]
            r = find(u)
            m = max(self.elev[u], self.elev[v])
            if m < spill[r]:
                spill[r] = m
        wse = np.zeros(self.n_nodes)
        for r, js in comps.items():
            if spill[r] == np.inf:
                spill[r] = self.elev[js[0]]
            wse[r] = spill[r]
        # per-node comp id array
        comp_id = np.array([find(j) for j in range(self.n_nodes)])
        return comp_id, wse, spill

    def depth_spill_fill(self):
        comp_id, wse, _ = self.basin_fill()
        return np.minimum(np.maximum(wse[comp_id] - self.elev, 0.0), MAXD)

    def depth_volume_fill(self, intensity):
        """WSE from runoff volume balance per component, storage 100 m2/node cap 3m."""
        comp_id, wse, spill = self.basin_fill()
        q_run = self.q_run_local(intensity)
        vol = q_run * T_RAIN_H * 3600.0
        out = np.zeros(self.n_nodes)
        for r in np.unique(comp_id):
            js = np.where(comp_id == r)[0]
            V = vol[js].sum()
            if V <= 0:
                continue
            # fill from bottom: cumulative storage as function of level
            order = js[np.argsort(self.elev[js])]
            cum = 0.0
            level = self.elev[order[0]]
            prev = level
            for k, j in enumerate(order):
                level = self.elev[j]
                cum += 100.0 * (level - prev) * k
                prev = level
                if cum >= V:
                    break
            if cum < V:
                extra = (V - cum) / (100.0 * len(js))
                level = level + extra
            out[js] = np.minimum(np.maximum(level - self.elev[js], 0.0), MAXD)
        return out

    # ------------------------------------------------------------------
    # Minimax bottleneck fill: WSE_j = lowest possible highest point on any
    # path from j to the global outlet. depth = WSE - elev (capped at 3).
    def minimax_wse(self):
        """Modified Dijkstra (minimax) from the global lowest node."""
        import heapq
        N = self.n_nodes
        adj = [[] for _ in range(N)]
        for e in range(self.U.size):
            u, v = self.U[e], self.V[e]
            adj[u].append(v)
            adj[v].append(u)
        lowest = int(np.argmin(self.elev))
        INF = 1e18
        wse = np.full(N, INF)
        wse[lowest] = self.elev[lowest]
        pq = [(wse[lowest], lowest)]
        while pq:
            w, u = heapq.heappop(pq)
            if w > wse[u]:
                continue
            for v in adj[u]:
                cand = max(self.elev[v], w)
                if cand < wse[v]:
                    wse[v] = cand
                    heapq.heappush(pq, (cand, v))
        return wse

    def depth_minimax(self):
        wse = self.minimax_wse()
        return np.minimum(np.maximum(wse - self.elev, 0.0), MAXD)

    def depth_minimax_vol(self, intensity):
        """minimax depth capped by the volume the node's own local runoff
        can supply: depth <= q_run*T/Apond."""
        d = self.depth_minimax()
        vcap = self.q_run_local(intensity) * T_RAIN_H * 3600.0 / APOND
        return np.minimum(d, vcap)
