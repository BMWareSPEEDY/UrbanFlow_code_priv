import osmnx as ox
import numpy as np

precise_hubs = {
    "bellandur": {
        "name": "Bellandur & ORR (Lake Basin)",
        "center": (12.926, 77.676),
        "dist": 2500,
        "filename": "bengaluru_bellandur_graph.graphml"
    },
    "ecity": {
        "name": "Electronic City (Phase 1 & 2)",
        "center": (12.845, 77.660),
        "dist": 2500,
        "filename": "bengaluru_ecity_graph.graphml"
    },
    "whitefield": {
        "name": "Whitefield (ITPB Corridor)",
        "center": (12.969, 77.749),
        "dist": 2200,
        "filename": "bengaluru_whitefield_graph.graphml"
    },
    "koramangala": {
        "name": "Koramangala & Indiranagar",
        "center": (12.935, 77.625),
        "dist": 2200,
        "filename": "bengaluru_koramangala_graph.graphml"
    }
}

def fetch_hub(key, info):
    print(f"\n--- Fetching OSM graph for {info['name']} ---")
    try:
        G = ox.graph_from_point(info["center"], dist=info["dist"], network_type='drive')
        G_proj = ox.project_graph(G, to_crs="EPSG:32643")
        
        np.random.seed(hash(key) % 100000)
        nodes_gdf = ox.graph_to_gdfs(G_proj, edges=False)
        min_y = nodes_gdf['y'].min()
        max_y = nodes_gdf['y'].max()
        
        for node_id, data in G_proj.nodes(data=True):
            y_val = float(data.get('y', min_y))
            rel_y = (y_val - min_y) / max(1.0, (max_y - min_y))
            
            elev = float(data.get('elevation', 876.0 + (rel_y * 30.0) + np.random.uniform(-1.5, 1.5)))
            imp = float(data.get('impervious_ratio', np.random.uniform(0.18, 0.68)))
            manning = float(data.get('manning_n', np.random.uniform(0.012, 0.035)))
            
            data['elevation'] = elev
            data['impervious_ratio'] = imp
            data['manning_n'] = manning
            
        ox.save_graphml(G_proj, info["filename"])
        print(f"Successfully saved {info['filename']} ({len(G_proj.nodes())} nodes, {len(G_proj.edges())} edges)!")
    except Exception as e:
        print(f"Error fetching {info['name']}: {e}")

if __name__ == "__main__":
    for k, info in precise_hubs.items():
        fetch_hub(k, info)
