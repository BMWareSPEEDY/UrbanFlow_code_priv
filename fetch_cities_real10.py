"""Fetch 10 more well-known cities with REAL terrain (Terrarium tiles, z14).
Covers flat low-lying (jakarta, bangkok) to steep hills (sanfrancisco, taipei)."""
import io
import math
import urllib.request

import numpy as np
import osmnx as ox
from PIL import Image

cities = {
    "mumbai_r": {"name": "Mumbai, India", "center": (19.0760, 72.8777), "dist": 4000,
                 "utm": "EPSG:32643", "filename": "city_mumbai_r_graph.graphml", "fallback_base": 10.0},
    "jakarta": {"name": "Jakarta, Indonesia", "center": (-6.2088, 106.8456), "dist": 4000,
                "utm": "EPSG:32748", "filename": "city_jakarta_graph.graphml", "fallback_base": 8.0},
    "bangkok": {"name": "Bangkok, Thailand", "center": (13.7563, 100.5018), "dist": 4000,
                "utm": "EPSG:32647", "filename": "city_bangkok_graph.graphml", "fallback_base": 5.0},
    "istanbul": {"name": "Istanbul, Turkiye", "center": (41.0082, 28.9784), "dist": 4000,
                 "utm": "EPSG:32635", "filename": "city_istanbul_graph.graphml", "fallback_base": 60.0},
    "berlin": {"name": "Berlin, Germany", "center": (52.5200, 13.4050), "dist": 4000,
               "utm": "EPSG:32633", "filename": "city_berlin_graph.graphml", "fallback_base": 40.0},
    "chicago": {"name": "Chicago, USA", "center": (41.8781, -87.6298), "dist": 4000,
                "utm": "EPSG:32616", "filename": "city_chicago_graph.graphml", "fallback_base": 180.0},
    "sanfrancisco": {"name": "San Francisco, USA", "center": (37.7749, -122.4194), "dist": 4000,
                     "utm": "EPSG:32610", "filename": "city_sanfrancisco_graph.graphml", "fallback_base": 50.0},
    "seoul": {"name": "Seoul, South Korea", "center": (37.5665, 126.9780), "dist": 4000,
              "utm": "EPSG:32652", "filename": "city_seoul_graph.graphml", "fallback_base": 40.0},
    "sydney": {"name": "Sydney, Australia", "center": (-33.8688, 151.2093), "dist": 4000,
               "utm": "EPSG:32756", "filename": "city_sydney_graph.graphml", "fallback_base": 30.0},
    "taipei": {"name": "Taipei, Taiwan", "center": (25.0330, 121.5654), "dist": 4000,
               "utm": "EPSG:32651", "filename": "city_taipei_graph.graphml", "fallback_base": 20.0},
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
    cache = {}
    elevs = None
    try:
        G = ox.graph_from_point(info["center"], dist=info["dist"], network_type='drive')
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