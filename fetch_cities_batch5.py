import osmnx as ox
import numpy as np

# 6 new cities, 3 terrain profiles: terrain='n3' (double noise), 'wave' (mild waves),
# 'n2w' (medium noise + wave). Elevation is synthetic everywhere in this pipeline;
# widening the local-relief regime gives the model training exposure to high elev_std2.
cities = {
    "varanasi": {"name": "Varanasi, India", "center": (25.3200, 82.9900), "dist": 2500, "base_elev": 80.0, "terrain": "n3", "filename": "city_varanasi_graph.graphml"},
    "rajkot": {"name": "Rajkot, India", "center": (22.3000, 70.8000), "dist": 2500, "base_elev": 130.0, "terrain": "n3", "filename": "city_rajkot_graph.graphml"},
    "ludhiana": {"name": "Ludhiana, India", "center": (30.9000, 75.8500), "dist": 2500, "base_elev": 240.0, "terrain": "wave", "filename": "city_ludhiana_graph.graphml"},
    "ranthi": {"name": "Ranchi, India", "center": (23.3600, 85.3300), "dist": 2500, "base_elev": 650.0, "terrain": "wave", "filename": "city_ranchi_graph.graphml"},
    "agra": {"name": "Agra, India", "center": (27.1800, 78.0200), "dist": 2500, "base_elev": 170.0, "terrain": "n2w", "filename": "city_agra_graph.graphml"},
    "mysore": {"name": "Mysore, India", "center": (12.3000, 76.6400), "dist": 2500, "base_elev": 760.0, "terrain": "n2w", "filename": "city_mysore_graph.graphml"},
}


def terrain_elevation(terrain, rel_y, rng):
    base_slope = rel_y * 30.0
    if terrain == "n3":
        return base_slope + rng.uniform(-3.0, 3.0)
    if terrain == "wave":
        return base_slope + rng.uniform(-1.5, 1.5) + 3.0 * np.sin(rel_y * 15 * np.pi)
    if terrain == "n2w":
        return base_slope + rng.uniform(-2.0, 2.0) + 5.0 * np.sin(rel_y * 10 * np.pi)
    return base_slope + rng.uniform(-1.5, 1.5)


def fetch_and_enrich(key, info):
    print(f"\n--- Fetching {info['name']} ({info['terrain']}) ---", flush=True)
    try:
        G = ox.graph_from_point(info["center"], dist=info["dist"], network_type='drive')
        G_proj = ox.project_graph(G, to_crs="EPSG:32643")

        rng = np.random.RandomState(hash(key) % 100000)
        nodes_gdf = ox.graph_to_gdfs(G_proj, edges=False)
        min_y = nodes_gdf['y'].min()
        max_y = nodes_gdf['y'].max()
        base = info["base_elev"]

        for node_id, data in G_proj.nodes(data=True):
            y_val = float(data.get('y', min_y))
            rel_y = (y_val - min_y) / max(1.0, (max_y - min_y))
            elev = float(data.get('elevation', base + terrain_elevation(info["terrain"], rel_y, rng)))
            imp = float(data.get('impervious_ratio', rng.uniform(0.18, 0.68)))
            manning = float(data.get('manning_n', rng.uniform(0.012, 0.035)))
            data['elevation'] = elev
            data['impervious_ratio'] = imp
            data['manning_n'] = manning

        G_elev = ox.elevation.add_edge_grades(G_proj, add_absolute=False)
        ox.save_graphml(G_elev, info["filename"])
        print(f"Saved {info['filename']} ({len(G_elev.nodes())} nodes)", flush=True)
    except Exception as e:
        print(f"Error {info['name']}: {e}", flush=True)


if __name__ == "__main__":
    for k, info in cities.items():
        fetch_and_enrich(k, info)