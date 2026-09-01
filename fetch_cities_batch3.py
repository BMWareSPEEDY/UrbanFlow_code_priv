import osmnx as ox
import numpy as np

cities = {
    "surat": {
        "name": "City Light, Surat, India",
        "center": (21.1700, 72.8300),
        "dist": 2500,
        "base_elev": 20.0,
        "filename": "city_surat_graph.graphml",
    },
    "indore": {
        "name": "Vijay Nagar, Indore, India",
        "center": (22.7450, 75.8950),
        "dist": 2500,
        "base_elev": 553.0,
        "filename": "city_indore_graph.graphml",
    },
    "bhopal": {
        "name": "MP Nagar, Bhopal, India",
        "center": (23.2550, 77.4050),
        "dist": 2500,
        "base_elev": 527.0,
        "filename": "city_bhopal_graph.graphml",
    },
    "nagpur": {
        "name": "Dharampeth, Nagpur, India",
        "center": (21.1500, 79.0800),
        "dist": 2500,
        "base_elev": 310.0,
        "filename": "city_nagpur_graph.graphml",
    },
    "visakhapatnam": {
        "name": "MVP Colony, Visakhapatnam, India",
        "center": (17.7300, 83.3000),
        "dist": 2500,
        "base_elev": 10.0,
        "filename": "city_visakhapatnam_graph.graphml",
    },
    "coimbatore": {
        "name": "Gandhipuram, Coimbatore, India",
        "center": (11.0200, 76.9700),
        "dist": 2500,
        "base_elev": 411.0,
        "filename": "city_coimbatore_graph.graphml",
    },
    "patna": {
        "name": "Boring Road, Patna, India",
        "center": (25.6100, 85.1300),
        "dist": 2500,
        "base_elev": 58.0,
        "filename": "city_patna_graph.graphml",
    },
    "kanpur": {
        "name": "Swaroop Nagar, Kanpur, India",
        "center": (26.4900, 80.2900),
        "dist": 2500,
        "base_elev": 126.0,
        "filename": "city_kanpur_graph.graphml",
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