import osmnx as ox
import numpy as np

cities = {
    "kolkata": {
        "name": "Salt Lake, Kolkata, India",
        "center": (22.5700, 88.4300),
        "dist": 2500,
        "base_elev": 6.0,
        "filename": "city_kolkata_graph.graphml",
    },
    "ahmedabad": {
        "name": "Navrangpura, Ahmedabad, India",
        "center": (23.0300, 72.5600),
        "dist": 2500,
        "base_elev": 55.0,
        "filename": "city_ahmedabad_graph.graphml",
    },
    "jaipur": {
        "name": "C-Scheme, Jaipur, India",
        "center": (26.9150, 75.7950),
        "dist": 2500,
        "base_elev": 432.0,
        "filename": "city_jaipur_graph.graphml",
    },
    "lucknow": {
        "name": "Hazratganj, Lucknow, India",
        "center": (26.8470, 80.9460),
        "dist": 2500,
        "base_elev": 123.0,
        "filename": "city_lucknow_graph.graphml",
    },
    "kochi": {
        "name": "Kochi, Ernakulam, India",
        "center": (9.9650, 76.2850),
        "dist": 2500,
        "base_elev": 5.0,
        "filename": "city_kochi_graph.graphml",
    },
}


def fetch_and_enrich(key, info):
    print(f"\n--- Fetching OSM graph for {info['name']} ---")
    try:
        G = ox.graph_from_point(info["center"], dist=info["dist"], network_type='drive')
        G_proj = ox.project_graph(G, to_crs="EPSG:32643")

        np.random.seed(hash(key) % 100000)
        nodes_gdf = ox.graph_to_gdfs(G_proj, edges=False)
        min_y = nodes_gdf['y'].min()
        max_y = nodes_gdf['y'].max()
        base = info["base_elev"]

        for node_id, data in G_proj.nodes(data=True):
            y_val = float(data.get('y', min_y))
            rel_y = (y_val - min_y) / max(1.0, (max_y - min_y))
            elev = float(data.get('elevation', base + (rel_y * 30.0) + np.random.uniform(-1.5, 1.5)))
            imp = float(data.get('impervious_ratio', np.random.uniform(0.18, 0.68)))
            manning = float(data.get('manning_n', np.random.uniform(0.012, 0.035)))
            data['elevation'] = elev
            data['impervious_ratio'] = imp
            data['manning_n'] = manning

        G_elev = ox.elevation.add_edge_grades(G_proj, add_absolute=False)
        ox.save_graphml(G_elev, info["filename"])
        print(f"Successfully saved {info['filename']} ({len(G_elev.nodes())} nodes, {len(G_elev.edges())} edges)!")
        return G_elev
    except Exception as e:
        print(f"Error fetching {info['name']}: {e}")
        return None


if __name__ == "__main__":
    for k, info in cities.items():
        fetch_and_enrich(k, info)