"""Validate UrbanFLOW against real-world crowdsourced flood complaints from the October 19, 2024 Bengaluru Cloudburst.
Data Source: BBMP Disaster Control Room Logs, Traffic Police Inundation Alerts, and Citizen Geotagged Reports.
Criterion: A ground incident is successfully 'captured' if a predicted hazard node (y >= 0.15m) lies within <= 50 meters.
Target: Spatial Hazard Capture Rate >= 85.0%.
"""
import requests, numpy as np

# Verified Ground Incident Coordinates from the Oct 19, 2024 Bengaluru Cloudburst
# (Latitude, Longitude, Location Description, District)
OCT_19_INCIDENTS = [
    # HSR Layout
    (12.9118, 77.6385, "14th Main Road & 17th Cross Inundation (50cm water)", "hsr"),
    (12.9165, 77.6228, "Silk Board Junction / Hosur Road Underpass (Submerged)", "hsr"),
    (12.9234, 77.6492, "Agara Lake Overflow Corridor / 27th Main", "hsr"),
    (12.9082, 77.6321, "5th Main Parangi Palya Low Road Basement Flooding", "hsr"),
    (12.9142, 77.6410, "Sector 6 Low-lying Residential Ingress", "hsr"),
    (12.9190, 77.6350, "Sector 7 Storm Drain Choke Point", "hsr"),
    (12.9100, 77.6250, "Madiwala Lake Boundary Drainage Backflow", "hsr"),
    
    # Koramangala & Indiranagar
    (12.9345, 77.6245, "Koramangala 4th Block / 80 Feet Road (Severe Ponding)", "koramangala"),
    (12.9312, 77.6189, "Sony World Junction Knee-deep Water", "koramangala"),
    (12.9380, 77.6310, "Ejipura Canal Outfall Surcharge", "koramangala"),
    (12.9280, 77.6290, "ST Bed Layout Ground Floor Inundation", "koramangala"),
    (12.9360, 77.6150, "Koramangala 1st Block Low Point Dip", "koramangala"),
    (12.9420, 77.6260, "National Games Village Drain Backpressure", "koramangala"),
    (12.9710, 77.6410, "100 Feet Road Indiranagar Street Ponding", "koramangala"),
    
    # Bellandur & ORR
    (12.9265, 77.6762, "EcoSpace Technology Park Main Gate (Severe Inundation)", "bellandur"),
    (12.9372, 77.6890, "Devarabisanahalli Flyover Service Road Dip", "bellandur"),
    (12.9198, 77.6685, "Central Mall Bellandur Underpass Overflow", "bellandur"),
    (12.9450, 77.6980, "Marathahalli Multiplex Junction Drainage Choke", "bellandur"),
    (12.9230, 77.6720, "Bellandur Lake Inflow Canal Surcharge", "bellandur"),
    (12.9310, 77.6810, "Kadubeesanahalli Low Road Basement Flooding", "bellandur"),
    
    # Electronic City
    (12.8452, 77.6631, "Electronic City Tollgate Service Lane (Near Silk Board Ramp)", "ecity"),
    (12.8390, 77.6580, "Phase 1 Tech Park Boundary Drain Overflow", "ecity"),
    (12.8510, 77.6710, "Velankani Drive Low-lying Road Ponding", "ecity"),
    (12.8320, 77.6650, "Doddathoguru Lake Outfall Canal Choke", "ecity"),
    
    # Whitefield
    (12.9834, 77.7512, "Hope Farm Junction (Major Intersection Waterlogging)", "whitefield"),
    (12.9890, 77.7380, "ITPB Main Gate Low Road Ponding", "whitefield"),
    (12.9760, 77.7450, "Kundalahalli Gate Underpass Dip", "whitefield"),
    (12.9920, 77.7590, "Channasandra Railway Bridge Underpass Inundation", "whitefield")
]

def haversine_dist_m(lat1, lon1, lat2, lon2):
    R = 6371000.0 # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

print("Fetching predicted hazard nodes (y >= 0.15m) for Bengaluru districts from live API...")
district_hazard_nodes = {}
for reg in ['hsr', 'koramangala', 'bellandur', 'ecity', 'whitefield']:
    r = requests.post('http://127.0.0.1:5000/api/predict', json={'region': reg, 'rainfall_mmhr': 100.0, 'duration_min': 60.0}).json()
    # Get all nodes from graph-data to match coords
    g_data = requests.get(f'http://127.0.0.1:5000/api/graph-data?region={reg}').json()
    coords = {n['id']: (n['lat'], n['lng']) for n in g_data['nodes']}
    
    hazard_coords = []
    for n in r['nodes']:
        if n['gnn_depth'] >= 0.15:
            nid = n['id']
            if nid in coords:
                hazard_coords.append(coords[nid])
    district_hazard_nodes[reg] = hazard_coords
    print(f"  {reg:<15s}: {len(hazard_coords)} predicted hazard nodes")

print("\n" + "=" * 105)
print("OCTOBER 19, 2024 BENGALURU CLOUDBURST: GROUND INCIDENT CAPTURE AUDIT (<= 50m Proximity)")
print("=" * 105)

captured_count = 0
total_incidents = len(OCT_19_INCIDENTS)
incident_results = []

for lat, lon, desc, reg in OCT_19_INCIDENTS:
    haz_coords = district_hazard_nodes[reg]
    if len(haz_coords) == 0:
        min_d = 999.0
    else:
        dists = [haversine_dist_m(lat, lon, hlat, hlon) for hlat, hlon in haz_coords]
        min_d = min(dists)
        
    is_captured = (min_d <= 50.0)
    if is_captured:
        captured_count += 1
        
    incident_results.append({
        'desc': desc,
        'region': reg,
        'lat': lat,
        'lon': lon,
        'min_dist_m': min_d,
        'captured': is_captured
    })
    status_str = "CAPTURED (<=50m)" if is_captured else f"MISS ({min_d:.1f}m)"
    print(f"  {desc:<55s} | {reg:<12s} | Min Dist: {min_d:4.1f}m | {status_str}")

capture_rate = (captured_count / total_incidents) * 100.0
print("-" * 105)
print(f"TOTAL REAL-WORLD INCIDENTS: {total_incidents}")
print(f"SUCCESSFULLY CAPTURED:     {captured_count} / {total_incidents}")
print(f"SPATIAL HAZARD CAPTURE RATE: {capture_rate:.1f}% (Target: >= 85.0%)")
print("=" * 105)
