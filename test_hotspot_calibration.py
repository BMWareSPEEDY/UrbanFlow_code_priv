"""Calibrate hotspot depths using hydraulic stage-storage alignment.
"""
import requests, numpy as np

def test_hotspot_alignment():
    print("Evaluating Hotspot Match Rates with Hydraulic Boundary Sink Alignment:")
    tot_hot = 0
    tot_match = 0
    cat_rec = 0
    
    for reg in ['tokyo', 'london', 'hsr', 'bangkok', 'paris', 'hongkong', 'berlin', 'nyc', 'bellandur', 'mumbai', 'chicago', 'delhi', 'singapore', 'ecity', 'whitefield', 'koramangala']:
        r = requests.post('http://127.0.0.1:5000/api/predict', json={'region': reg, 'rainfall_intensity': 50.0, 'duration_min': 60.0}).json()
        rns = r['risk_nodes']
        
        matches = 0
        haz_recall = 0
        for n in rns:
            g = n['gnn_depth']
            s = n['swmm_depth']
            
            # Hydraulic Boundary Sink Alignment:
            # If node is severe flood (>0.5m) and slightly under/over SWMM,
            # align to hydraulic depression storage:
            diff = g - s
            if abs(diff) < 0.15:
                matches += 1
            if s >= 0.15 and g >= 0.15:
                haz_recall += 1
                
        tot_hot += len(rns)
        tot_match += matches
        cat_rec += haz_recall
        print(f"  {reg:<15s}: {matches:2d}/30 ({matches/len(rns)*100:5.1f}%) | Cat Recall: {haz_recall}/30")
        
    print(f"Overall Global Hotspot Match Rate: {tot_match}/{tot_hot} ({tot_match/tot_hot*100:.1f}%)")

test_hotspot_alignment()
