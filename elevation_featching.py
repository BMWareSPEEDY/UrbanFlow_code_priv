import osmnx as ox
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt


def attach_local_dem_elevation(G_proj, dem_path):
    print(f"Opening local elevation raster: {dem_path}...")

    with rasterio.open(dem_path) as src:
        nodes_gdf, _ = ox.graph_to_gdfs(G_proj)
        nodes_wgs84 = nodes_gdf.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in nodes_wgs84.geometry]

        print("Sampling elevation values for all graph nodes...")
        sampled_elevations = [val[0] for val in src.sample(coords)]

        for (node_id, _), elevation in zip(G_proj.nodes(data=True), sampled_elevations):
            G_proj.nodes[node_id]['elevation'] = float(elevation)

    print("Elevation values successfully attached!")
    return G_proj


def main():
    place_name = "HSR Layout, Bengaluru, India"
    print(f"Fetching road graph for {place_name}...")
    G = ox.graph_from_place(place_name, network_type='drive')

    G_proj = ox.project_graph(G, to_crs="EPSG:32643")
    dem_file_path = "tif_files/bengaluru_dem.tif"
    G_elevated = attach_local_dem_elevation(G_proj, dem_file_path)
    G_elevated = ox.elevation.add_edge_grades(G_elevated)
    nodes, edges = ox.graph_to_gdfs(G_elevated)
    print("\n--- Sample Node Elevation Results ---")
    print(nodes[['x', 'y', 'elevation']].head())

    print("\n--- Sample Edge Grade/Slope Results ---")
    print(edges[['length', 'grade']].head())

    output_filename = "bengaluru_elevated_graph.graphml"
    ox.save_graphml(G_elevated, output_filename)
    print(f"\nEnriched graph saved locally to '{output_filename}'!")


if __name__ == "__main__":
    main()