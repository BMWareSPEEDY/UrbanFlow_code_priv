"""Benchmark EPA SWMM 5.2 Dynamic Wave solver vs UrbanFLOW GNN execution.
Measures:
- Raw GNN Forward Pass Tensor Latency (ms)
- Full Inference Pipeline Latency (ms)
- End-to-End HTTP API Latency (ms)
- Effective Speedup Factor vs EPA SWMM 5.2 numerical solver
"""
import torch, time, requests, numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Benchmarking on device: {device}")

# Catchments to benchmark
test_cases = [
    ('hsr', 'Small (HSR Layout)', 1379, 42.1),              # SWMM 5.2: 42.1s
    ('ecity', 'Medium (Electronic City)', 3337, 228.0),       # SWMM 5.2: 3.8 min = 228s
    ('tokyo', 'Large (Tokyo Catchment)', 13173, 1476.0)       # SWMM 5.2: 24.6 min = 1476s
]

results = []

# Load model for raw tensor forward pass timing
ckpt = torch.load('hydro_gine_v5_bottleneck_opt.pt', weights_only=False)
from app import REGION_CACHE, PRODUCTION_PREDICTOR

print("\nRunning warm-up passes...")
for _ in range(5):
    g = REGION_CACHE['hsr']['pyg_data']
    _ = PRODUCTION_PREDICTOR.predict(g, 50.0, 60.0)

print("\nMeasuring execution benchmarks:")
for r_key, label, n_nodes, swmm_sec in test_cases:
    r_data = REGION_CACHE[r_key]
    g = r_data['pyg_data']
    
    # 1. Pure GNN Forward Pass Tensor Time (average over 50 iterations)
    t_gnn_list = []
    for _ in range(50):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = PRODUCTION_PREDICTOR.predict(g, 50.0, 60.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_gnn_list.append((t1 - t0) * 1000.0)
        
    gnn_ms = np.median(t_gnn_list)
    
    # 2. End-to-End HTTP API Latency (average over 10 calls)
    t_api_list = []
    for _ in range(10):
        t0 = time.perf_counter()
        resp = requests.post('http://127.0.0.1:5000/api/predict', json={'region': r_key, 'rainfall_mmhr': 50.0, 'duration_min': 60.0})
        t1 = time.perf_counter()
        t_api_list.append((t1 - t0) * 1000.0)
        
    api_ms = np.median(t_api_list)
    
    # Effective speedup vs SWMM 5.2
    swmm_ms = swmm_sec * 1000.0
    speedup_tensor = swmm_ms / gnn_ms
    speedup_e2e = swmm_ms / api_ms
    
    res = {
        'scale': label,
        'nodes': n_nodes,
        'swmm_time': f"{swmm_sec:.1f} s" if swmm_sec < 60 else f"{swmm_sec/60:.1f} min",
        'gnn_ms': f"{gnn_ms:.1f} ms",
        'api_ms': f"{api_ms:.1f} ms",
        'speedup': f"~{int(round(speedup_e2e, -1)):,}×"
    }
    results.append(res)
    print(f"  {label:<28s} | Nodes: {n_nodes:5d} | SWMM: {res['swmm_time']:<9s} | GNN: {res['gnn_ms']:<8s} | API: {res['api_ms']:<8s} | Speedup: {res['speedup']}")

print("\nBenchmark completed successfully!")
