"""Fetch well-known cities with REAL terrain: New York (Manhattan) + London.
Elevations sampled from Mapzen Terrarium tiles (AWS, no auth) at z14 (~7m/px).
Fallback to synthetic 'n3' profile only if tile fetch fails (clearly logged)."""
import io
import math
import os
import urllib.request
import zipfile

import numpy as np
import osmnx as ox
from PIL import Image

cities = {
    "nyc": {"name": "New York City (Manhattan), USA", "center": (40.7580, -73.9855),
            "dist": 4000, "utm": "EPSG:32618", "filename": "city_nyc_graph.graphml",
            "fallback_base": 10.0},
    "london": {"name": "London, UK", "center": (51.5074, -0.1278),
               "dist": 4000, "utm": "EPSG:32630", "filename": "city_london_graph.graphml",
               "fallback_base": 25.0},
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
    """Bilinear sample of Terrarium-encoded elevation (m) at z14."""
    z = 14
    x, y = wm_xy(lon, lat, z)
    xi, yi = int(math.floor(x)), int(math.floor(y))
    tile = get_tile(z, xi, yi, cache)
    fx, fy = x - xi, y - yi
    px = min(255, int(fx * 255))
    py = min(255, int(fy * 255))
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
    cache = {}
    try:
        G = ox.graph_from_point(info["center"], dist=info["dist"], network_type='drive')
        nodes_gdf = ox.graph_to_gdfs(G, edges=False)
        elevs = []
        for node_id, row in nodes_gdf.iterrows():
            elevs.append(sample_elev(row['y'], row['x'], cache))
        elevs = np.array(elevs)
        print(f"  real terrain sampled: min {elevs.min():.1f} m, max {elevs.max():.1f} m, "
              f"mean {elevs.mean():.1f} m", flush=True)
    except Exception as e:
        print(f"  REAL TERRAIN FAILED ({e}); falling back to synthetic n3 profile", flush=True)
        elevs = None

    G_proj = ox.project_graph(G, to_crs=info["utm"])
    rng = np.random.RandomState(hash(key) % 100000)
    nodes_gdf_p = ox.graph_to_gdfs(G_proj, edges=False)
    min_y = nodes_gdf_p['y'].min()
    max_y = nodes_gdf_p['y'].max()

    for i, (node_id, data) in enumerate(G_proj.nodes(data=True)):
        if elevs is not None:
            elev = float(elevs[i])
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