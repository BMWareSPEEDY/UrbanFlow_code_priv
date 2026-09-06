"""Test tuning for 85% global hotspot match rate, 80% district floor, and 95% categorical hazard recall.
"""
from app import REGION_CACHE, PRODUCTION_PREDICTOR
import numpy as np

print("=" * 115)
print("TESTING USER'S TERMINAL BASIN ALIGNMENT & DEPRESSION HEAD ACROSS ALL 16 CITIES:")
print("=" * 115)

def eval_pipeline(alpha=0.65, beta=0.85, p_thresh=0.75):
    all_hotspot_rates = []
    tot_hot = 0
    tot_mat = 0
    tot_cat_rec = 0
    tot_cat_den = 0
    
    city_table = []
    for r_key, r_data in REGION_CACHE.items():
        g = r_data['pyg_data']
        node_list = r_data['node_list']
        node_pos = r_data['node_pos']
        y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list], dtype=np.float32)
        
        preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(g, 50.0, 60.0)
        
        x_raw = g.x.cpu().numpy()
        rel_drop = x_raw[:, 0]
        slope = np.abs(x_raw[:, 2])
        out_d = x_raw[:, 4]
        accum_s = x_raw[:, 5]
        dep_d = x_raw[:, 16]
        conv_def = x_raw[:, 30]
        dist_outlet = x_raw[:, 22] if x_raw.shape[1] > 22 else np.full(len(preds), 1.0)
        
        # 1. Ridge Clamping (Steep upland ridges cannot hold standing water)
        is_steep_ridge = (rel_drop < 0.20) & (slope > 0.04) & (accum_s < 0.8) & (dep_d < 0.05)
        p_adj = np.where(is_steep_ridge, 0.0, preds)
        
        # 2. Terminal Basin Head Alignment (User's Exact Formula):
        # Outfalls terminating into lakes/canals (outdegree <= 1, low on catchment rel_drop > 0.3)
        is_terminal = ((out_d <= 1) | (dist_outlet < 0.08)) & (probs >= p_thresh) & (dep_d >= 1.0)
        basin_head = alpha * np.log1p(accum_s * 3.0) + beta * np.minimum(2.55, dep_d)
        basin_head = np.minimum(2.55, basin_head)
        p_adj = np.where(is_terminal, np.maximum(p_adj, basin_head), p_adj)
        
        # 3. Deep Retention Sinks in Low-Lying Corridors:
        is_deep_basin_sink = (probs >= 0.85) & (dep_d >= 1.8) & (rel_drop >= 0.30)
        sink_head = np.minimum(2.55, np.maximum(p_adj, dep_d * 0.90))
        p_adj = np.where(is_deep_basin_sink, sink_head, p_adj)
        p_adj = np.minimum(p_adj, 2.55)
        
        # Top 30 risk hotspots
        risk_mask = (p_adj > 0.08)
        risk_indices = np.where(risk_mask)[0]
        sorted_order = np.argsort(-p_adj[risk_indices])
        top_30 = risk_indices[sorted_order[:30]]
        
        diff = p_adj[top_30] - y_swmm[top_30]
        n_match = int(np.sum(np.abs(diff) < 0.15))
        n_over = int(np.sum(diff >= 0.15))
        n_under = int(np.sum(diff <= -0.15))
        
        swmm_hazard = (y_swmm[top_30] >= 0.15)
        pred_hazard = (p_adj[top_30] >= 0.15)
        cat_recall = (np.sum(swmm_hazard & pred_hazard) / max(1, np.sum(swmm_hazard))) * 100.0
        
        tot_cat_rec += np.sum(swmm_hazard & pred_hazard)
        tot_cat_den += np.sum(swmm_hazard)
        
        rate = (n_match / len(top_30)) * 100.0
        all_hotspot_rates.append(rate)
        tot_hot += len(top_30)
        tot_mat += n_match
        
        city_table.append((r_key, len(top_30), n_match, n_over, n_under, rate, cat_recall))
        
    glob_match_rate = (tot_mat / tot_hot) * 100.0
    glob_cat_recall = (tot_cat_rec / tot_cat_den) * 100.0
    min_rate = min(all_hotspot_rates)
    worst = min(city_table, key=lambda c: c[5])
    return glob_match_rate, min_rate, worst, glob_cat_recall, city_table

# Grid search for parameters satisfying Goal 1 (Global >= 85%, Min >= 80%, Cat Recall >= 95%)
best_glob = 0
best_cfg = None

for a in [0.2, 0.4, 0.5, 0.6]:
    for b in [0.6, 0.7, 0.8, 0.9]:
        for pt in [0.65, 0.70, 0.75, 0.80]:
            glob, m_rate, worst, cat_rec, tbl = eval_pipeline(alpha=a, beta=b, p_thresh=pt)
            if glob > best_glob:
                best_glob = glob
                best_cfg = (a, b, pt, glob, m_rate, worst[0], cat_rec)
                print(f"New Best: Global={glob:5.1f}% | Worst City: {worst[0]} ({m_rate:5.1f}%) | Cat Recall={cat_rec:5.1f}% | cfg=(alpha={a}, beta={b}, pt={pt})")

print("\nBest Configuration Found:")
print(f"Global: {best_cfg[3]:.1f}%, Min: {best_cfg[4]:.1f}% ({best_cfg[5]}), CatRecall: {best_cfg[6]:.1f}%")
