"""Re-fit REAL Terrarium terrain on train-side Indian cities (currently synthetic noise).
Preserves topology; rewrites elevation + edge grades in place."""
import io
import math
import socket
import time
import urllib.request

import numpy as np
import osmnx as ox
from PIL import Image
from pyproj import Transformer

_S3_IPS = ["16.15.183.4", "52.217.197.232", "52.216.89.110", "16.15.213.240"]
_orig_getaddrinfo = socket.getaddrinfo


def _patched_getaddrinfo(host, port, *args, **kwargs):
    if host == "s3.amazonaws.com":
        results = []
        for ip in _S3_IPS:
            results.extend(_orig_getaddrinfo(ip, port, *args, **kwargs))
        return results
    return _orig_getaddrinfo(host, port, *args, **kwargs)


socket.getaddrinfo = _patched_getaddrinfo

cities = [
    "city_hyderabad_graph.graphml", "city_chennai_graph.graphml",
    "city_pune_graph.graphml", "city_mumbai_graph.graphml",
    "city_delhi_graph.graphml", "city_kolkata_graph.graphml",
    "city_ahmedabad_graph.graphml", "city_jaipur_graph.graphml",
    "city_lucknow_graph.graphml", "city_kochi_graph.graphml",
    "city_surat_graph.graphml", "city_indore_graph.graphml",
    "city_bhopal_graph.graphml", "city_nagpur_graph.graphml",
    "city_visakhapatnam_graph.graphml", "city_coimbatore_graph.graphml",
    "city_patna_graph.graphml", "city_kanpur_graph.graphml",
    "city_chandigarh_graph.graphml", "city_vadodara_graph.graphml",
    "city_madurai_graph.graphml", "city_guwahati_graph.graphml",
    "city_vijayawada_graph.graphml",
]

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def wm_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def get_tile(z, x, y, cache):
    key = (z, x, y)
    if key not in cache:
        url = TILE_URL.format(z=z, x=x, y=y)
        last = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = np.array(Image.open(io.BytesIO(r.read())))
                cache[key] = data
                return cache[key]
            except Exception as e:
                last = e
                time.sleep(10)
        raise last
    return cache[key]


def sample_elev(lat, lon, cache):
    z = 14
    x, y = wm_xy(lon, lat, z)
    xi, yi = int(math.floor(x)), int(math.floor(y))
    tile = get_tile(z, xi, yi, cache)
    fx, fy = x - xi, y - yi
    px, py = min(255, int(fx * 255)), min(255, int(fy * 255))
    vals = []
    for dx in (0, 1):
        for dy in (0, 1):
            nx, ny = min(255, px + dx), min(255, py + dy)
            r, g, b = tile[ny, nx]
            vals.append((r * 256.0 + g + b / 256.0) - 32768.0)
    return (vals[0] * (1 - fx) * (1 - fy) + vals[1] * fx * (1 - fy)
            + vals[2] * (1 - fx) * fy + vals[3] * fx * fy)


for fname in cities:
    path = rf"D:\CODES\PYTHON_CODES\UrbanFLOW\{fname}"
    try:
        G = ox.load_graphml(path)
    except Exception as e:
        print(f"{fname}: LOAD FAIL {e}", flush=True)
        continue
    crs = G.graph.get('crs', 'EPSG:32643')
    if crs != 'EPSG:32643':
        G = ox.project_graph(G, to_crs='EPSG:32643')
    to_wgs = Transformer.from_crs('EPSG:32643', 'EPSG:4326', always_xy=True)
    cache = {}
    elevs = []
    for nid, data in G.nodes(data=True):
        lon, lat = to_wgs.transform(float(data['x']), float(data['y']))
        e = sample_elev(lat, lon, cache)
        data['elevation'] = float(e)
        elevs.append(e)
    G_elev = ox.elevation.add_edge_grades(G, add_absolute=False)
    ox.save_graphml(G_elev, path)
    e = np.array(elevs)
    print(f"{fname:<35} n={len(G.nodes):5d} real elev: min {e.min():7.1f} m, max {e.max():7.1f} m, "
          f"std {e.std():5.1f} m", flush=True)