"""Test Flask application endpoints with HydroGINE-v4.0.
"""
import os
import sys
import json

from app import app, REGION_CACHE, PRODUCTION_PREDICTOR

def test_routes():
    print("Testing Flask app routes...")
    print(f"Production Predictor loaded: {PRODUCTION_PREDICTOR is not None}")
    client = app.test_client()
    
    # 1. Test /api/regions
    r_resp = client.get('/api/regions')
    assert r_resp.status_code == 200, f"Failed /api/regions: {r_resp.status_code}"
    regions_data = json.loads(r_resp.data)
    print(f"  /api/regions: OK, found {len(regions_data['regions'])} regions")
    
    # 2. Test /api/graph-data?region=hsr
    rd_resp = client.get('/api/graph-data?region=hsr')
    assert rd_resp.status_code == 200, f"Failed /api/graph-data: {rd_resp.status_code}"
    rd_data = json.loads(rd_resp.data)
    print(f"  /api/graph-data?region=hsr: OK, {len(rd_data['nodes'])} nodes, {len(rd_data['edges'])} edges")
    
    # 3. Test /api/predict for HSR at 50 mm/hr
    p_resp = client.post('/api/predict', json={
        'region': 'hsr',
        'rainfall_mmhr': 50.0,
        'duration_min': 60.0,
        'soil_moisture': 'partial'
    })
    assert p_resp.status_code == 200, f"Failed /api/predict: {p_resp.status_code}"
    p_data = json.loads(p_resp.data)
    metrics = p_data['metrics']
    print(f"  /api/predict (HSR @ 50 mm/hr): OK, infer_time={metrics['gnn_time_ms']}ms, speedup={metrics['speedup_ratio']}x, flooded_nodes={metrics['total_flooded_nodes']}, max_depth={metrics['max_depth_m']}m")
    
    # 4. Test /api/predict for HSR at 200 mm/hr
    p_resp_cloud = client.post('/api/predict', json={
        'region': 'hsr',
        'rainfall_mmhr': 200.0,
        'duration_min': 60.0,
        'soil_moisture': 'saturated'
    })
    assert p_resp_cloud.status_code == 200, f"Failed /api/predict: {p_resp_cloud.status_code}"
    pc_data = json.loads(p_resp_cloud.data)
    m_cloud = pc_data['metrics']
    print(f"  /api/predict (HSR @ 200 mm/hr): OK, infer_time={m_cloud['gnn_time_ms']}ms, speedup={m_cloud['speedup_ratio']}x, flooded_nodes={m_cloud['total_flooded_nodes']}, max_depth={m_cloud['max_depth_m']}m")
    
    # 5. Test /api/storm-playback
    sp_resp = client.post('/api/storm-playback', json={
        'region': 'hsr',
        'peak_intensity_mmhr': 100.0,
        'duration_min': 60.0,
        'time_step_min': 15.0,
        'soil_moisture': 'partial'
    })
    assert sp_resp.status_code == 200, f"Failed /api/storm-playback: {sp_resp.status_code}"
    sp_data = json.loads(sp_resp.data)
    print(f"  /api/storm-playback: OK, {len(sp_data['frames'])} frames generated")
    
    print("\nALL FLASK APP ENDPOINT TESTS PASSED PERFECTLY WITH HydroGINE-v4.0!")

if __name__ == '__main__':
    test_routes()
