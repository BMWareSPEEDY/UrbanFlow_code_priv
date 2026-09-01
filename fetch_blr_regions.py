import osmnx as ox
import pandas as pd
import numpy as np

regions = {
    "hsr": "HSR Layout, Bengaluru, India",
    "bellandur": "Bellandur, Bengaluru, India",
    "whitefield": "Whitefield, Bengaluru, India",
    "ecity": "Electronic City, Bengaluru, India",
}

def fetch_and_enrich_graph(place_name, filename):
    print(f"\n--- Fetching graph for {place_name} ---")
    try:
        G = ox.graph_from_place(place_name, network_type='drive')
        G_proj = ox.project_graph(G, to_crs="EPSG:32643")
        
        # Enrich nodes with realistic elevation & impervious ratio if missing
        np.random.seed(42)
        nodes_gdf = ox.graph_to_gdfs(G_proj, edges=False)
        
        # Calculate bounding box & pseudo-topography (Bengaluru slopes south-east towards Bellandur lake basin)
        min_y = nodes_gdf['y'].min()
        max_y = nodes_gdf['y'].max()
        
        for node_id, data in G_proj.nodes(data=True):
            y_val = float(data.get('y', min_y))
            rel_y = (y_val - min_y) / max(1.0, (max_y - min_y))
            
            # Base elevation ~ 875m to 910m (low elevation towards south/lakes)
            elev = float(data.get('elevation', 878.0 + (rel_y * 28.0) + np.random.uniform(-1.5, 1.5)))
            imp = float(data.get('impervious_ratio', np.random.uniform(0.15, 0.65)))
            manning = float(data.get('manning_n', np.random.uniform(0.012, 0.035)))
            
            data['elevation'] = elev
            data['impervious_ratio'] = imp
            data['manning_n'] = manning
            
        ox.save_graphml(G_proj, filename)
        print(f"Successfully saved {filename} with {len(G_proj.nodes())} nodes and {len(G_proj.edges())} edges!")
        return G_proj
    except Exception as e:
        print(f"Error fetching {place_name}: {e}")
        return None

if __name__ == "__main__":
    for key, place in regions.items():
        fetch_and_enrich_graph(place, f"bengaluru_{key}_graph.graphml")
