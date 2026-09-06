"""Verify live Flask API prediction across all international and domestic cities.
"""
import urllib.request
import json

cities = ['hsr', 'bellandur', 'whitefield', 'ecity', 'koramangala',
          'tokyo', 'hongkong', 'singapore', 'london', 'paris',
          'nyc', 'chicago', 'berlin', 'bangkok', 'mumbai', 'delhi']

print(f"{'City Key':<14s} | {'Status':<8s} | {'Total Nodes':<12s} | {'Flooded':<10s} | {'Max Depth':<12s} | {'Mean Depth':<12s}")
print("-" * 78)

for c in cities:
    req = urllib.request.Request(
        'http://127.0.0.1:5000/api/predict',
        data=json.dumps({'region': c, 'rain': 100, 'duration': 60, 'soil_moisture': 'dry'}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        m = res['metrics']
        print(f"{c:<14s} | {res['status']:<8s} | {m['total_nodes']:<12d} | {m['total_flooded_nodes']:<10d} | {m['max_depth_m']:<12.3f} | {m['avg_depth_m']:<12.4f} | {m['gnn_time_ms']:<6.1f} ms")
