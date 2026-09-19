"""Run comprehensive end-to-end benchmark across all 16 regions on the live HTTP server.
Generates full classification, regression, hotspot, and latency tables for:
- 50 mm/hr (Standard Design Storm)
- 100 mm/hr (Extreme Cloudburst)
"""
import requests, time, json, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
API_URL = "http://127.0.0.1:5000/api/predict"

REGIONS = [
    ('hsr', 'HSR Layout (Bengaluru)'),
    ('bellandur', 'Bellandur & ORR (Bengaluru)'),
    ('whitefield', 'Whitefield (Bengaluru)'),
    ('ecity', 'Electronic City (Bengaluru)'),
    ('koramangala', 'Koramangala & Indiranagar'),
    ('tokyo', 'Tokyo Metropolitan Catchment'),
    ('hongkong', 'Hong Kong Urban Basin'),
    ('singapore', 'Singapore Marina Catchment'),
    ('london', 'London Thames Catchment'),
    ('paris', 'Paris Seine Basin'),
    ('nyc', 'New York City Coastal Catchment'),
    ('chicago', 'Chicago Waterfront Catchment'),
    ('berlin', 'Berlin Spree Basin'),
    ('bangkok', 'Bangkok Chao Phraya Lowlands'),
    ('mumbai', 'Mumbai Coastal Floodplain'),
    ('delhi', 'Delhi Yamuna Floodplain')
]

HAZARD_THRESHOLD = 0.15 # 15 cm

def run_benchmark_for_intensity(intensity_mmhr):
    print(f"\nEvaluating all 16 regions at {intensity_mmhr:.0f} mm/hr...")
    results = []
    
    for r_key, r_name in REGIONS:
        payload = {
            'region': r_key,
            'rainfall_intensity': intensity_mmhr,
            'duration_min': 60.0
        }
        t0 = time.perf_counter()
        resp = requests.post(API_URL, json=payload)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        
        if resp.status_code != 200:
            print(f"  Error querying {r_key}: {resp.status_code}")
            continue
            
        data = resp.json()
        nodes = data.get('nodes', [])
        risk_nodes = data.get('risk_nodes', [])
        
        num_nodes = len(nodes)
        gnn_depths = np.array([n['gnn_depth'] for n in nodes], dtype=np.float32)
        swmm_depths = np.array([n['swmm_depth'] for n in nodes], dtype=np.float32)
        
        # Classification Metrics (at 0.15m flood threshold)
        y_true = (swmm_depths >= HAZARD_THRESHOLD).astype(int)
        y_pred = (gnn_depths >= HAZARD_THRESHOLD).astype(int)
        
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        
        precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 100.0
        recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 100.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        acc = ((tp + tn) / num_nodes) * 100.0
        
        # Continuous Depth Regression Metrics (cm)
        abs_err = np.abs(gnn_depths - swmm_depths)
        sq_err = (gnn_depths - swmm_depths) ** 2
        
        mae_cm = float(np.mean(abs_err) * 100.0)
        rmse_cm = float(np.sqrt(np.mean(sq_err)) * 100.0)
        p90_cm = float(np.percentile(abs_err, 90) * 100.0)
        p95_cm = float(np.percentile(abs_err, 95) * 100.0)
        max_cm = float(np.max(abs_err) * 100.0)
        
        pct_10cm = float(np.mean(abs_err <= 0.10) * 100.0)
        pct_15cm = float(np.mean(abs_err <= 0.15) * 100.0)
        pct_30cm = float(np.mean(abs_err <= 0.30) * 100.0)
        
        # Hotspots (Top 30 risk nodes)
        top30_match = 0
        top30_over = 0
        top30_under = 0
        for rn in risk_nodes:
            diff = rn['gnn_depth'] - rn['swmm_depth']
            if abs(diff) < 0.15:
                top30_match += 1
            elif diff > 0:
                top30_over += 1
            else:
                top30_under += 1
                
        n_hotspots = len(risk_nodes)
        top30_rate = (top30_match / max(1, n_hotspots)) * 100.0
        
        res = {
            'key': r_key,
            'name': r_name,
            'nodes': num_nodes,
            'latency_ms': lat_ms,
            'swmm_crit': int(np.sum(y_true == 1)),
            'gnn_crit': int(np.sum(y_pred == 1)),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'class_acc': acc,
            'mae_cm': mae_cm,
            'rmse_cm': rmse_cm,
            'p90_cm': p90_cm,
            'p95_cm': p95_cm,
            'max_cm': max_cm,
            'pct_10cm': pct_10cm,
            'pct_15cm': pct_15cm,
            'pct_30cm': pct_30cm,
            'hotspots': n_hotspots,
            'top30_match': top30_match,
            'top30_over': top30_over,
            'top30_under': top30_under,
            'top30_rate': top30_rate
        }
        results.append(res)
        print(f"  {r_name:<30s} | Nodes: {num_nodes:5d} | Recall: {recall:5.1f}% | FP: {fp:3d} | FN: {fn:3d} | MAE: {mae_cm:4.1f}cm | <=15cm: {pct_15cm:5.1f}% | Latency: {lat_ms:5.1f}ms")
        
    return results

res_50 = run_benchmark_for_intensity(50.0)
res_100 = run_benchmark_for_intensity(100.0)

# Save results JSON
with open("benchmark_data_full.json", "w") as f:
    json.dump({'50_mmhr': res_50, '100_mmhr': res_100}, f, indent=2)

print("\nSaved benchmark data to benchmark_data_full.json")
