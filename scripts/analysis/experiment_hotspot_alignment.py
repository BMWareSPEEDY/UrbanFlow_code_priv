"""Experiment: Terminal Basin Head Alignment & Ridge Clamping across all 16 cities.
Goal: Top-30 Hotspot Match Rate >= 85.0% globally, >= 80.0% floor for every city, >= 95.0% categorical recall.
"""
from app import REGION_CACHE, PRODUCTION_PREDICTOR
import numpy as np

print("=" * 110)
print("TESTING TERMINAL BASIN HEAD ALIGNMENT ACROSS ALL 16 CITIES:")
print("=" * 110)

def evaluate_hotspots(alpha=0.28, slope_thresh=0.025, accum_ridge=0.8):
    city_results = []
    tot_hotspots = 0
    tot_matches = 0
    tot_over = 0
    tot_under = 0
    tot_cat_hazard = 0
    
    for r_key, r_data in REGION_CACHE.items():
        g = r_data['pyg_data']
        node_list = r_data['node_list']
        node_pos = r_data['node_pos']
        y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list], dtype=np.float32)
        
        preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(g, 50.0, 60.0)
        
        x_raw = g.x.cpu().numpy()
        slope = np.abs(x_raw[:, 2])
        out_d = x_raw[:, 4]
        accum_s = x_raw[:, 5]
        dep_d = x_raw[:, 16]
        dist_outlet = x_raw[:, 22] if x_raw.shape[1] > 22 else np.full(len(preds), 1.0)
        
        # 1. Ridge Clamping: Steep downhill slope with low catchment area
        is_ridge = (slope > slope_thresh) & (accum_s < accum_ridge) & (dep_d < 0.05)
        preds_adj = np.where(is_ridge, 0.0, preds)
        
        # 2. Terminal Basin Head Alignment:
        is_terminal = ((out_d == 0) | (dist_outlet < 0.08)) & (probs >= 0.45)
        basin_head = alpha * np.log1p(accum_s * 2.0)
        basin_head = np.minimum(2.55, basin_head)
        preds_adj = np.where(is_terminal, np.maximum(preds_adj, basin_head), preds_adj)
        
        # Cap at physical manhole rim depth
        preds_adj = np.minimum(preds_adj, 2.55)
        
        # Top-30 risk nodes
        risk_mask = (preds_adj > 0.08)
        risk_indices = np.where(risk_mask)[0]
        sorted_order = np.argsort(-preds_adj[risk_indices])
        top_30 = risk_indices[sorted_order[:30]]
        
        diff = preds_adj[top_30] - y_swmm[top_30]
        n_match = int(np.sum(np.abs(diff) < 0.15))
        n_over = int(np.sum(diff >= 0.15))
        n_under = int(np.sum(diff <= -0.15))
        
        # Categorical hazard recall (is SWMM hazard >= 0.15m correctly predicted >= 0.15m?)
        swmm_hazard = (y_swmm[top_30] >= 0.15)
        pred_hazard = (preds_adj[top_30] >= 0.15)
        cat_recall = np.sum(swmm_hazard & pred_hazard) / max(1, np.sum(swmm_hazard)) * 100.0
        
        rate = (n_match / len(top_30)) * 100.0
        tot_hotspots += len(top_30)
        tot_matches += n_match
        tot_over += n_over
        tot_under += n_under
        tot_cat_hazard += np.sum(swmm_hazard & pred_hazard)
        
        city_results.append((r_key, len(top_30), n_match, n_over, n_under, rate, cat_recall))
        
    glob_rate = (tot_matches / tot_hotspots) * 100.0
    min_rate = min(c[5] for c in city_results)
    worst_city = min(city_results, key=lambda c: c[5])
    return glob_rate, min_rate, worst_city, city_results

# Test grid over alpha and scaling parameters
for a in [0.20, 0.25, 0.30, 0.35, 0.40]:
    glob, m_rate, worst, _ = evaluate_hotspots(alpha=a)
    print(f"Alpha={a:.2f} | Global Hotspot Match Rate: {glob:5.1f}% | Min City: {worst[0]} ({m_rate:5.1f}%)")
