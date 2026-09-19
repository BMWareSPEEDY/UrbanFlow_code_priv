"""Validate UrbanFLOW against documented flood-affected locations from the October 2024 Bengaluru rain events.

Data source: data/real_bengaluru_oct2024_incidents.json -- every incident is a location named in a
verifiable news report (The Hindu, Indian Express, New Indian Express, Times of India, Moneycontrol)
with source URL, publication date, and OSM-geocoded coordinates. No depths, incident counts, or
control-room records are asserted beyond what the cited sources report.

Criterion: a documented location is 'captured' if a predicted hazard node (gnn_depth >= 0.15 m)
lies within <= 50 meters when the model is run at the given storm intensity. The storm intensity is
an explicit assumption (the observed 24-hour accumulations are daily totals, not short-duration peak
rates), and the capture rate is computed live from the running model -- it is never hardcoded.

Usage:
    python validate_real_world_incidents.py [--rainfall 50.0] [--duration 60] [--api http://127.0.0.1:5000] [--event oct22_2024]
"""
import argparse
import json
import os
import requests
import numpy as np

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'data', 'real_bengaluru_oct2024_incidents.json')

CAPTURE_THRESHOLD_M = 0.15
CAPTURE_RADIUS_M = 50.0

def load_dataset():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Incident dataset not found: {DATA_FILE}")
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def haversine_dist_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

def fetch_hazard_nodes(api, region, rainfall_mmhr, duration_min):
    r = requests.post(f'{api}/api/predict',
                      json={'region': region, 'rainfall_mmhr': rainfall_mmhr,
                            'duration_min': duration_min},
                      timeout=120).json()
    if r.get('status') != 'success':
        raise RuntimeError(f"predict failed for {region}: {r.get('message')}")
    g = requests.get(f'{api}/api/graph-data?region={region}', timeout=120).json()
    coords = {n['id']: (n['lat'], n['lng']) for n in g['nodes']}
    hazard = []
    for n in r['nodes']:
        if n['gnn_depth'] >= CAPTURE_THRESHOLD_M:
            c = coords.get(n['id'])
            if c:
                hazard.append(c)
    return hazard

def compute_capture(api, rainfall_mmhr, duration_min, event=None, incidents=None, print_report=True):
    """Run the documented-incident capture audit against the live model API.

    Returns (rate_pct, result_dict) where result_dict has 'total_incidents',
    'captured_incidents', 'capture_rate_pct', 'incidents' (per-incident detail).
    `incidents` may be pre-filtered (e.g. filtered by event/region) before calling.
    """
    ds = load_dataset()
    if incidents is None:
        incidents = ds['incidents']
        if event:
            incidents = [i for i in incidents if event in i.get('event_id', [])]

    if print_report:
        print(f"Dataset: {DATA_FILE}")
        print(f"Evaluated incidents: {len(incidents)}  |  Storm intensity: {rainfall_mmhr:g} mm/hr, "
              f"{duration_min:g} min  |  Capture: gnn_depth >= {CAPTURE_THRESHOLD_M} m within "
              f"{CAPTURE_RADIUS_M:g} m")
        print(f"Storms evaluated: {', '.join(e['name'] for e in ds['events'])}\n")
        print("=" * 100)
        print("OCTOBER 2024 BENGALURU DOCUMENTED FLOOD LOCATIONS: MODEL CAPTURE AUDIT")
        print("=" * 100)

    district_hazard = {}
    captured_count = 0
    results = []
    for inc in incidents:
        reg = inc['region']
        if reg not in district_hazard:
            district_hazard[reg] = fetch_hazard_nodes(api, reg, rainfall_mmhr, duration_min)
            if print_report:
                print(f"  [model @ {rainfall_mmhr:g} mm/hr] {reg:<15s}: "
                      f"{len(district_hazard[reg])} predicted hazard nodes")
        haz = district_hazard[reg]
        if len(haz) == 0:
            min_d = 999.0
        else:
            min_d = min(haversine_dist_m(inc['lat'], inc['lon'], hlat, hlon) for hlat, hlon in haz)
        captured = bool(min_d <= CAPTURE_RADIUS_M)
        if captured:
            captured_count += 1
        results.append({**inc, 'min_dist_m': round(min_d, 1), 'captured': captured})
        if print_report:
            status = "CAPTURED (<=50m)" if captured else f"MISS ({min_d:.1f}m)"
            print(f"  {inc['id']:<36s} | {reg:<12s} | {min_d:5.1f}m | {status}")

    rate = 100.0 * captured_count / max(1, len(incidents))
    if print_report:
        print("-" * 100)
        print(f"DOCUMENTED LOCATIONS: {len(incidents)}")
        print(f"CAPTURED:             {captured_count} / {len(incidents)}")
        print(f"CAPTURE RATE:         {rate:.1f}%")
        print(f"INTENSITY ASSUMPTION: {rainfall_mmhr:g} mm/hr ({duration_min:g} min) - short-duration "
              "peak-rate assumption, not the published 24-hour accumulation")
        print("=" * 100)

    out = {
        'dataset': os.path.relpath(DATA_FILE),
        'storm_intensity_mmhr': rainfall_mmhr,
        'duration_min': duration_min,
        'capture_threshold_m': CAPTURE_THRESHOLD_M,
        'capture_radius_m': CAPTURE_RADIUS_M,
        'total_incidents': len(incidents),
        'captured_incidents': captured_count,
        'capture_rate_pct': round(rate, 1),
        'incidents': results,
    }
    return rate, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rainfall', type=float, default=100.0,
                    help='Storm intensity (mm/hr) at which the model is run. The Oct 2024 reports give '
                         '24-hour accumulations, not peak intensities; this value is the explicit '
                         'short-duration intensity assumption for the validation run.')
    ap.add_argument('--duration', type=float, default=60.0)
    ap.add_argument('--api', default='http://127.0.0.1:5000')
    ap.add_argument('--event', default=None,
                    help='Filter incidents to one event id (e.g. bengaluru_oct15_2024). Default: all.')
    args = ap.parse_args()

    ds = load_dataset()
    incidents = ds['incidents']
    if args.event:
        incidents = [i for i in incidents if args.event in i.get('event_id', [])]

    rate, out = compute_capture(args.api, args.rainfall, args.duration, incidents=incidents)

    out_file = 'real_world_validation_result.json'
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved detailed result to {out_file}")
    return rate


if __name__ == '__main__':
    main()