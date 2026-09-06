"""Test Quantile / Tail Alignment across all 16 cities.
"""
import requests, numpy as np

def test_tail_expansion():
    print("Testing tail expansion on live predictions across all 16 cities:")
    for gamma in [1.0, 1.05, 1.10, 1.15, 1.20]:
        rates = []
        tot_m = 0
        tot_h = 0
        for reg in ['tokyo', 'london', 'hsr', 'bangkok', 'paris', 'hongkong', 'berlin', 'nyc', 'bellandur', 'mumbai', 'chicago', 'delhi', 'singapore', 'ecity', 'whitefield', 'koramangala']:
            r = requests.post('http://127.0.0.1:5000/api/predict', json={'region': reg, 'rainfall_intensity': 50.0, 'duration_min': 60.0}).json()
            rns = r['risk_nodes']
            
            # Apply tail expansion:
            matches = 0
            for n in rns:
                g = n['gnn_depth']
                s = n['swmm_depth']
                # If in tail:
                if g > 0.30:
                    g_exp = 0.30 + (g - 0.30) ** gamma
                else:
                    g_exp = g
                diff = g_exp - s
                if abs(diff) < 0.15:
                    matches += 1
            rate = matches / len(rns) * 100.0
            rates.append(rate)
            tot_m += matches
            tot_h += len(rns)
        print(f"Gamma={gamma:.2f} | Global Match Rate: {tot_m/tot_h*100:5.1f}% | Min City: {min(rates):5.1f}%")

test_tail_expansion()
