"""Test app.py live prediction endpoint for HSR Layout and inspect the top risk nodes.
"""
import urllib.request, json

req = urllib.request.Request(
    'http://127.0.0.1:5000/api/predict',
    data=json.dumps({'region': 'hsr', 'rain': 50, 'duration': 60, 'soil_moisture': 'dry'}).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

try:
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print(f"Prediction Status: {res['status']}")
        print(f"Metrics: {res['metrics']}")
        print(f"\nTop 10 Risk Nodes in HSR Layout @ 50 mm/hr:")
        print(f"{'Rank':<5s} | {'Node ID':<10s} | {'GNN Depth (m)':<14s} | {'SWMM Depth (m)':<14s} | {'Risk Level':<12s} | {'Diff (cm)':<10s}")
        print("-" * 75)
        for r, node in enumerate(res['risk_nodes'][:10], 1):
            gnn = node['gnn_depth']
            swmm = node['swmm_depth']
            diff = (gnn - swmm) * 100.0
            print(f"{r:<5d} | {node['id']:<10s} | {gnn:<14.4f} | {swmm:<14.4f} | {node['risk_level']:<12s} | {diff:<+10.1f}")
except Exception as e:
    print(f"Error calling server: {e}")
