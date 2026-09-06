"""Diagnose the tail errors for all cities to understand exact under/over patterns.
"""
import requests, numpy as np

for reg in ['koramangala', 'whitefield', 'ecity', 'hsr', 'bangkok', 'paris', 'hongkong', 'chicago', 'delhi', 'mumbai']:
    r = requests.post('http://127.0.0.1:5000/api/predict', json={'region': reg, 'rainfall_intensity': 50.0, 'duration_min': 60.0}).json()
    rns = r['risk_nodes']
    diffs = [n['gnn_depth'] - n['swmm_depth'] for n in rns]
    matches = sum(1 for d in diffs if abs(d) < 0.15)
    overs = sum(1 for d in diffs if d >= 0.15)
    unders = sum(1 for d in diffs if d <= -0.15)
    mean_err = np.mean(diffs)
    mean_abs_err = np.mean(np.abs(diffs))
    
    # Check categorical recall (are all of them >= 0.15m in SWMM and GNN?)
    swmm_hazard = sum(1 for n in rns if n['swmm_depth'] >= 0.15)
    gnn_hazard = sum(1 for n in rns if n['gnn_depth'] >= 0.15)
    
    print(f"{reg:<14s} | M:{matches:2d} O:{overs:2d} U:{unders:2d} ({matches/len(rns)*100:5.1f}%) | MeanDiff:{mean_err:+.3f}m | MAE:{mean_abs_err:.3f}m | SWMM_Haz:{swmm_hazard}/30 | GNN_Haz:{gnn_hazard}/30")
