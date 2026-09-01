import osmnx as ox
import numpy as np

cities = {
    "hyderabad": {
        "name": "Banjara Hills, Hyderabad, India",
        "center": (17.4156, 78.4347),
        "dist": 2500,
        "base_elev": 536.0,
        "filename": "city_hyderabad_graph.graphml",
    },
    "chennai": {
        "name": "T Nagar, Chennai, India",
        "center": (13.0400, 80.2400),
        "dist": 2500,
        "base_elev": 6.0,
        "filename": "city_chennai_graph.graphml",
    },
    "pune": {
        "name": "Shivajinagar, Pune, India",
        "center": (18.5300, 73.8500),
        "dist": 2500,
        "base_elev": 560.0,
        "filename": "city_pune_graph.graphml",
    },
    "mumbai": {
        "name": "Andheri, Mumbai, India",
        "center": (19.1197, 72.8467),
        "dist": 2500,
        "base_elev": 10.0,
        "filename": "city_mumbai_graph.graphml",
    },
    "delhi": {
        "name": "Connaught Place, New Delhi, India",
        "center": (28.6315, 77.2167),
        "dist": 2500,
        "base_elev": 216.0,
        "filename": "city_delhi_graph.graphml",
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