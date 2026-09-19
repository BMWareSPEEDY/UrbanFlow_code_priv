import osmnx as ox
import numpy as np

cities = {
    "kanpur": {
        "name": "Swaroop Nagar, Kanpur, India",
        "center": (26.4900, 80.2900),
        "dist": 2000,
        "base_elev": 126.0,
        "filename": "city_kanpur_graph.graphml",
    },
    "chandigarh": {
        "name": "Sector 22, Chandigarh, India",
        "center": (30.7300, 76.7800),
        "dist": 2500,
        "base_elev": 350.0,
        "filename": "city_chandigarh_graph.graphml",
    },
    "vadodara": {
        "name": "Alkapuri, Vadodara, India",
        "center": (22.3000, 73.1900),
        "dist": 2500,
        "base_elev": 35.0,
        "filename": "city_vadodara_graph.graphml",
    },
    "madurai": {
        "name": "Madurai Main, Madurai, India",
        "center": (9.9300, 78.1200),
        "dist": 2500,
        "base_elev": 100.0,
        "filename": "city_madurai_graph.graphml",
    },
    "guwahati": {
        "name": "GS Road, Guwahati, India",
        "center": (26.1800, 91.7500),
        "dist": 2500,
        "base_elev": 55.0,
        "filename": "city_guwahati_graph.graphml",
    },
    "vijayawada": {
        "name": "MG Road, Vijayawada, India",
        "center": (16.5100, 80.6500),
        "dist": 2500,
        "base_elev": 20.0,
        "filename": "city_vijayawada_graph.graphml",
    },
    "nashik": {
        "name": "College Road, Nashik, India",
        "center": (20.0000, 73.7900),
        "dist": 2500,
        "base_elev": 570.0,
        "filename": "city_nashik_graph.graphml",
    },
}


def fetch_and_enrich(key, info):
    print(f"\n--- Fetching OSM graph for {info['name']} ---", flush=True)
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
        print(f"Successfully saved {info['filename']} ({len(G_elev.nodes())} nodes, {len(G_elev.edges())} edges)!", flush=True)
        return G_elev
    except Exception as e:
        print(f"Error fetching {info['name']}: {e}", flush=True)
        return None


if __name__ == "__main__":
    for k, info in cities.items():
        fetch_and_enrich(k, info)