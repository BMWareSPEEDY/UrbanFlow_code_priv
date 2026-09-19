"""Historical storm validation for UrbanFLOW against documented October 2024 Bengaluru flood events.

All event and incident data is loaded from data/real_bengaluru_oct2024_incidents.json: the events
(multiple named storm episodes with observed rainfall and source URLs) and the incidents (documented
flood-affected locations, OSM-geocoded, each tied to the storm episode it appeared in).

Validation runs the live model (via the running Flask API) at an explicit short-duration peak-rate
assumption and measures whether a predicted hazard node (gnn_depth >= 0.15 m) lies within 50 m of
each documented location. Set the intensity explicitly; the published 24-hour totals are daily
accumulations, not peak intensities, so the chosen peak rate is always printed and reported.

    Usage:
      python historical_storm_validation.py [--rainfall 100.0] [--duration 60]
                                            [--api http://127.0.0.1:5000] [--event <event_id>]
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
    with open(DATA_FILE, 'r') as f:
        return json.load(f)


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def fetch_hazard_nodes(api, region, rainfall_mmhr, duration_min):
    r = requests.post(f'{api}/api/predict',
                      json={'region': region, 'rainfall_mmhr': rainfall_mmhr,
                            'duration_min': duration_min}, timeout=120).json()
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


def run_historical_validation(event_id, rainfall_mmhr, duration_min, api):
    ds = load_dataset()
    event = next((e for e in ds['events'] if e['id'] == event_id), None)
    if event is None:
        raise KeyError(f"event {event_id!r} not found in {DATA_FILE}. "
                       f"Available: {[e['id'] for e in ds['events']]}")
    incidents = [i for i in ds['incidents'] if event_id in i.get('event_id', [])]
    obs = event.get('observed_rainfall', {})

    print(f"\nEVENT: {event['name']}")
    print(f"  Dates      : {event['start_date']} to {event['end_date']}")
    print(f"  Observed   : {json.dumps(obs, indent=2)}"[:400])
    if not incidents:
        print("  NOTE       : No documented incident locations fall inside the five Bengaluru "
              "model catchments for this event; nothing to capture. This is a legitimate validation "
              "result -- the model claims only these five catchments, and the reports for this "
              "storm were concentrated elsewhere in the city.")
        return {'event_id': event_id, 'total_incidents': 0, 'captured_incidents': 0,
                'capture_rate_pct': 0.0, 'incidents': [], 'rainfall_mmhr': rainfall_mmhr,
                'duration_min': duration_min}

    district_hazard = {}
    matched = []
    tp = 0
    for inc in incidents:
        reg = inc['region']
        if reg not in district_hazard:
            district_hazard[reg] = fetch_hazard_nodes(api, reg, rainfall_mmhr, duration_min)
        haz = district_hazard[reg]
        if len(haz) == 0:
            min_d = 999.0
        else:
            min_d = min(haversine_distance(inc['lat'], inc['lon'], hlat, hlon) for hlat, hlon in haz)
        captured = bool(min_d <= CAPTURE_RADIUS_M)
        if captured:
            tp += 1
        c = dict(inc)
        c['distance_m'] = round(min_d, 1)
        c['predicted_depth_m'] = None
        c['is_correctly_flagged'] = captured
        matched.append(c)
        print(f"  {c['id']:<36s} | {reg:<12s} | d={min_d:5.1f}m | "
              f"{'CAPTURED' if captured else 'MISS (documented, not predicted at this peak rate)'}")

    total = len(incidents)
    recall = tp / max(1, total)
    rate = recall * 100.0
    print(f"  Captured {tp}/{total} documented locations -> spatial recall {rate:.1f}% at "
          f"{rainfall_mmhr:g} mm/hr ({duration_min:g} min peak-rate assumption)")

    return {
        'event_id': event_id,
        'event_name': event['name'],
        'observed_rainfall': obs,
        'rainfall_mmhr': rainfall_mmhr,
        'duration_min': duration_min,
        'capture_threshold_m': CAPTURE_THRESHOLD_M,
        'capture_radius_m': CAPTURE_RADIUS_M,
        'total_incidents': total,
        'captured_incidents': tp,
        'capture_rate_pct': round(rate, 1),
        'incidents': matched
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--rainfall', type=float, default=100.0,
                    help='Short-duration peak-rate assumption (mm/hr) for the model run. The '
                         'published Oct 2024 values are 24-hour/6-hour accumulations, not peak '
                         'intensities, so this peak rate is an explicit modelling assumption.')
    ap.add_argument('--duration', type=float, default=60.0)
    ap.add_argument('--api', default='http://127.0.0.1:5000')
    ap.add_argument('--event', default=None,
                    help='Event id (bengaluru_oct15_2024 | bengaluru_oct20_2024 | '
                         'bengaluru_oct22_2024). Default: all events.')
    args = ap.parse_args()

    ds = load_dataset()
    event_ids = [e['id'] for e in ds['events']]
    selected = [args.event] if args.event else event_ids
    all_res = []
    for eid in selected:
        all_res.append(run_historical_validation(eid, args.rainfall, args.duration, args.api))

    out = 'historical_validation_result.json'
    with open(out, 'w') as f:
        json.dump(all_res, f, indent=2)
    print(f"\nSaved per-event results to {out}")

    # Honorary explicit total across events (documented locations only).
    tot = sum(r['total_incidents'] for r in all_res)
    cap = sum(r['captured_incidents'] for r in all_res)
    print(f"COMBINED: {cap}/{tot} documented locations captured "
          f"({round(100.0 * cap / max(1, tot), 1)}%) at {args.rainfall:g} mm/hr "
          f"({args.duration:g} min) peak-rate assumption.")