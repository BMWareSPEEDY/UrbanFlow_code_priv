import osmnx as ox
import geopandas as gpd


def add_landcover_features(G_elevated, place_name="HSR Layout, Bengaluru, India"):
    """
    Downloads building footprints and green space polygons from OpenStreetMap,
    then calculates concrete density (impervious surface ratio) for every graph node.
    """
    print(f"1. Fetching building footprints and land-use for {place_name}...")

    # Updated OSMnx API: features_from_place instead of geometries_from_place
    buildings = ox.features_from_place(place_name, tags={'building': True})

    print("2. Calculating node land-cover indices...")

    # Extract nodes as GeoDataFrame
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_elevated)

    # Project buildings to match node CRS (EPSG:32643 - UTM Zone 43N)
    if not buildings.empty:
        buildings = buildings.to_crs(nodes_gdf.crs)

    # Buffer nodes by 50 meters to analyze immediate surrounding land cover
    nodes_buffered = nodes_gdf.copy()
    nodes_buffered['geometry'] = nodes_buffered.geometry.buffer(50)  # 50-meter radius

    impervious_ratios = []

    # Spatial intersection to determine concrete density around each node
    for idx, node_buffer in nodes_buffered.iterrows():
        if not buildings.empty:
            intersecting_bldgs = buildings[buildings.intersects(node_buffer.geometry)]
            if not intersecting_bldgs.empty:
                bldg_area = intersecting_bldgs.intersection(node_buffer.geometry).area.sum()
                ratio = min(bldg_area / node_buffer.geometry.area, 1.0)
            else:
                ratio = 0.1
        else:
            ratio = 0.1

        impervious_ratios.append(ratio)

    # Attach impervious surface fraction & Manning's n to graph nodes
    for (node_id, _), imp_val in zip(G_elevated.nodes(data=True), impervious_ratios):
        G_elevated.nodes[node_id]['impervious_ratio'] = float(imp_val)
        # Manning's roughness n: High concrete (0.013) vs soil/grass (0.045)
        G_elevated.nodes[node_id]['manning_n'] = float(0.013 * imp_val + 0.045 * (1 - imp_val))

    print("Land-cover features successfully attached!")
    return G_elevated


def main():
    input_file = "bengaluru_elevated_graph.graphml"
    print(f"Loading elevated graph from '{input_file}'...")
    G = ox.load_graphml(input_file)

    G_complete = add_landcover_features(G)

    nodes_gdf, _ = ox.graph_to_gdfs(G_complete)
    print("\n--- Complete Node Feature Matrix (Sample) ---")
    print(nodes_gdf[['x', 'y', 'elevation', 'impervious_ratio', 'manning_n']].head())

    output_filename = "bengaluru_complete_graph.graphml"
    ox.save_graphml(G_complete, output_filename)
    print(f"\nFinal completed spatial graph saved to '{output_filename}'!")


if __name__ == "__main__":
    main()