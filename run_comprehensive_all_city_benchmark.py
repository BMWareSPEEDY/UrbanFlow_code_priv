"""Full Comprehensive Benchmark of UrbanFLOW across All 16 Regional Networks.
Evaluates:
- Classification metrics: TP, FP, FN, TN, Precision, Recall, F1, Accuracy
- Depth regression metrics: MAE, RMSE, R2, 90th & 95th percentile error, % <= 10cm, % <= 15cm, % <= 30cm
- Active hotspot metrics: Top 30 risk nodes (MATCH, OVER, UNDER, Match Rate %)
- Multi-rainfall evaluations: 50 mm/hr (Standard) and 100 mm/hr (Cloudburst)
- Neural inference latency in milliseconds
"""
import torch, numpy as np, sys, time, json
from sklearn.metrics import r2_score

sys.stdout.reconfigure(line_buffering=True)
from app import REGION_CACHE, PRODUCTION_PREDICTOR

HAZARD_THRESHOLD = 0.15 # 15 cm threshold for flooded/critical node

def evaluate_network(r_key, r_data, intensity_mmhr=50.0, duration_min=60.0):
    g = r_data['pyg_data']
    node_list = r_data['node_list']
    node_pos = r_data['node_pos']
    y_swmm = np.array([node_pos[nid]['swmm_depth'] for nid in node_list])
    
    # Measure live neural inference latency
    t0 = time.perf_counter()
    preds, raw_p, probs = PRODUCTION_PREDICTOR.predict(g, intensity_mmhr, duration_min)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    num_nodes = len(node_list)
    num_edges = g.edge_index.shape[1] if hasattr(g, 'edge_index') and g.edge_index is not None else 0
    
    # 1. Classification Metrics (Hazard Threshold >= 0.15 m)
    y_true_binary = (y_swmm >= HAZARD_THRESHOLD).astype(int)
    y_pred_binary = (preds >= HAZARD_THRESHOLD).astype(int)
    
    tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))
    fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
    fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))
    tn = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
    
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 100.0
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 100.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    class_acc = ((tp + tn) / num_nodes) * 100.0
    
    # 2. Continuous Depth Regression Metrics (All Nodes)
    abs_errors = np.abs(preds - y_swmm)
    sq_errors = (preds - y_swmm) ** 2
    
    mae_cm = float(np.mean(abs_errors) * 100.0)
    rmse_cm = float(np.sqrt(np.mean(sq_errors)) * 100.0)
    p90_cm = float(np.percentile(abs_errors, 90) * 100.0)
    p95_cm = float(np.percentile(abs_errors, 95) * 100.0)
    max_err_cm = float(np.max(abs_errors) * 100.0)
    
    # Safe R2 computation
    if np.var(y_swmm) > 1e-6:
        r2 = float(r2_score(y_swmm, preds))
    else:
        r2 = 1.0 if mae_cm < 1.0 else 0.0
        
    pct_within_10cm = float(np.mean(abs_errors <= 0.10) * 100.0)
    pct_within_15cm = float(np.mean(abs_errors <= 0.15) * 100.0)
    pct_within_30cm = float(np.mean(abs_errors <= 0.30) * 100.0)
    
    # 3. Active Risk Hotspots (Top 30 Risk Nodes as displayed in Dashboard)
    risk_mask = (preds > 0.08)
    risk_indices = np.where(risk_mask)[0]
    sorted_order = np.argsort(-preds[risk_indices])
    top_30_indices = risk_indices[sorted_order[:30]]
    
    top30_match = 0
    top30_over = 0
    top30_under = 0
    for idx in top_30_indices:
        diff = preds[idx] - y_swmm[idx]
        if abs(diff) < 0.15:
            top30_match += 1
        elif diff > 0:
            top30_over += 1
        else:
            top30_under += 1
            
    num_hotspots = len(top_30_indices)
    top30_match_rate = (top30_match / max(1, num_hotspots)) * 100.0
    
    return {
        'region': r_key,
        'name': r_data.get('name', r_key),
        'nodes': num_nodes,
        'edges': num_edges,
        'latency_ms': latency_ms,
        'intensity_mmhr': intensity_mmhr,
        # Classification
        'swmm_critical': int(np.sum(y_true_binary == 1)),
        'gnn_critical': int(np.sum(y_pred_binary == 1)),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_acc': class_acc,
        # Regression
        'mae_cm': mae_cm,
        'rmse_cm': rmse_cm,
        'r2': r2,
        'p90_cm': p90_cm,
        'p95_cm': p95_cm,
        'max_err_cm': max_err_cm,
        'pct_10cm': pct_within_10cm,
        'pct_15cm': pct_within_15cm,
        'pct_30cm': pct_within_30cm,
        # Hotspots
        'hotspots': num_hotspots,
        'hotspot_match': top30_match,
        'hotspot_over': top30_over,
        'hotspot_under': top30_under,
        'hotspot_match_rate': top30_match_rate
    }

print("Running Full Benchmark across all 16 regions at 50 mm/hr (Standard) and 100 mm/hr (Cloudburst)...")
results_50 = {}
results_100 = {}

for r_key, r_data in REGION_CACHE.items():
    print(f"  Evaluating {r_key}...", end="", flush=True)
    res50 = evaluate_network(r_key, r_data, 50.0, 60.0)
    res100 = evaluate_network(r_key, r_data, 100.0, 60.0)
    results_50[r_key] = res50
    results_100[r_key] = res100
    print(f" done (50mm/hr: Match={res50['hotspot_match_rate']:.1f}%, MAE={res50['mae_cm']:.1f}cm, Latency={res50['latency_ms']:.1f}ms)")

# Save JSON results
with open("benchmark_results_all_cities.json", "w") as f:
    json.dump({'50_mmhr': results_50, '100_mmhr': results_100}, f, indent=2)

print("\nSaved full JSON benchmark data to benchmark_results_all_cities.json")
