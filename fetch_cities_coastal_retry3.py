"""Retry Macau + Cape Town: verify endpoint setting, browser UA, multiple retries."""
import io
import math
import time
import urllib.request

import numpy as np
import osmnx as ox
from PIL import Image

ox.settings.overpass_endpoint = "https://overpass-api.de/api/interpreter"
ox.settings.user_agent = "urbanflow-research/1.0 (flood modelling research)"
ox.settings.timeout = 300

cities = {
    "macau": {"name": "Macau, China", "center": (22.1987, 113.5439), "dist": 3500,
              "utm": "EPSG:32650", "filename": "city_macau_graph.graphml", "fallback_base": 40.0},
    "capetown": {"name": "Cape Town, South Africa", "center": (-33.9249, 18.4241), "dist": 4000,
                 "utm": "EPSG:32734", "filename": "city_capetown_graph.graphml", "fallback_base": 40.0},
}

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
        with urllib.request.urlopen(url, timeout=30) as r:
            data = np.array(Image.open(io.BytesIO(r.read())))
        cache[key] = data
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


def fetch_and_enrich(key, info):
    print(f"\n--- Fetching {info['name']} ---", flush=True)
    print(f"  endpoint now: {ox.settings.overpass_endpoint}", flush=True)
    G = None
    for attempt in range(5):
        try:
            G = ox.graph_from_point(info["center"], dist=info["dist"], network_type='drive')
            print(f"  fetched on attempt {attempt + 1}", flush=True)
            break
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {str(e)[:100]}", flush=True)
            time.sleep(45)
    if G is None:
        print(f"  GIVING UP on {info['name']}", flush=True)
        return
    cache = {}
    elevs = None
    try:
        nodes_gdf = ox.graph_to_gdfs(G, edges=False)
        elevs = np.array([sample_elev(row['y'], row['x'], cache) for _, row in nodes_gdf.iterrows()])
        print(f"  real terrain: min {elevs.min():.1f} m, max {elevs.max():.1f} m, mean {elevs.mean():.1f} m",
              flush=True)
    except Exception as e:
        print(f"  REAL TERRAIN FAILED ({e}); synthetic n3 fallback", flush=True)

    G_proj = ox.project_graph(G, to_crs=info["utm"])
    rng = np.random.RandomState(hash(key) % 100000)
    nodes_gdf_p = ox.graph_to_gdfs(G_proj, edges=False)
    min_y = nodes_gdf_p['y'].min()
    max_y = nodes_gdf_p['y'].max()

    for i, (node_id, data) in enumerate(G_proj.nodes(data=True)):
        if elevs is not None:
            elev = max(0.0, float(elevs[i]))
        else:
            y_val = float(data.get('y', min_y))
            rel_y = (y_val - min_y) / max(1.0, (max_y - min_y))
            elev = float(data.get('elevation', info["fallback_base"] + rel_y * 30.0 + rng.uniform(-3.0, 3.0)))
        imp = float(data.get('impervious_ratio', rng.uniform(0.18, 0.68)))
        manning = float(data.get('manning_n', rng.uniform(0.012, 0.035)))
        data['elevation'] = elev
        data['impervious_ratio'] = imp
        data['manning_n'] = manning

    G_elev = ox.elevation.add_edge_grades(G_proj, add_absolute=False)
    ox.save_graphml(G_elev, info["filename"])
    print(f"Saved {info['filename']} ({len(G_elev.nodes())} nodes, terrain={'REAL' if elevs is not None else 'SYNTHETIC'})",
          flush=True)


if __name__ == "__main__":
    for k, info in cities.items():
        fetch_and_enrich(k, info)