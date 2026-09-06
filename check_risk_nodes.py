import requests
for reg in ['koramangala', 'whitefield', 'hsr', 'tokyo', 'london']:
    r = requests.post('http://127.0.0.1:5000/api/predict', json={'region': reg, 'rainfall_intensity': 50.0, 'duration_min': 60.0}).json()
    rns = r['risk_nodes']
    matches = [n for n in rns if n['status'] == 'MATCH']
    over = [n for n in rns if n['status'] == 'OVER']
    under = [n for n in rns if n['status'] == 'UNDER']
    print(f"{reg:<12s} | Total: {len(rns)} | MATCH: {len(matches):2d} | OVER: {len(over):2d} | UNDER: {len(under):2d} | Rate: {len(matches)/max(1,len(rns))*100:5.1f}%")
    if reg == 'koramangala':
        print("  Koramangala sample top 5:")
        for i, n in enumerate(rns[:5]):
            print(f"    #{i+1} GNN={n['gnn_depth']:.3f}m SWMM={n['swmm_depth']:.3f}m diff={n['gnn_depth']-n['swmm_depth']:+.3f}m ({n['status']})")
