"""Diagnose errors on Koramangala, Whitefield, NYC, Chicago, and Berlin with bottleneck-opt model.
"""
import requests, numpy as np

for city in ['koramangala', 'whitefield', 'nyc', 'chicago', 'berlin']:
    resp = requests.post("http://127.0.0.1:5000/api/predict", json={'region': city, 'rainfall_intensity': 50.0, 'duration_min': 60.0}).json()
    r_nodes = resp['risk_nodes']
    diffs = [n['gnn_depth'] - n['swmm_depth'] for n in r_nodes]
    print(f"\n--- {city.upper()} ---")
    print(f"Mean error: {np.mean(diffs):+.3f}m | Median error: {np.median(diffs):+.3f}m | Std: {np.std(diffs):.3f}m")
    for i, n in enumerate(r_nodes[:10]):
        d = n['gnn_depth'] - n['swmm_depth']
        print(f"  Node #{n['id']}: GNN={n['gnn_depth']:.3f}m | SWMM={n['swmm_depth']:.3f}m | Diff={d:+.3f}m | Status={n.get('status')}")
