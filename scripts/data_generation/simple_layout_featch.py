import osmnx as ox
import matplotlib.pyplot as plt


def main():
    # 1. Define your target location
    place_name = "HSR Layout, Bengaluru, India"
    print(f"Downloading street network graph for: {place_name}...")
    G = ox.graph_from_place(place_name, network_type='drive')

    G_proj = ox.project_graph(G, to_crs="EPSG:32643")

    nodes, edges = ox.graph_to_gdfs(G_proj)

    print("\n--- Network Graph Summary ---")
    print(f"Total Nodes (Intersections): {len(nodes)}")
    print(f"Total Edges (Road Segments): {len(edges)}")
    print("\nSample Node Coordinates (UTM Meters):")
    print(nodes[['x', 'y']].head())

    fig, ax = ox.plot_graph(
        G_proj,
        node_color="red",
        node_size=15,
        edge_color="#333333",
        edge_linewidth=1,
        show=False,
        close=False
    )

    plt.title(f"Spatial Graph: {place_name}", fontsize=14)
    plt.savefig("hsr_layout_graph.png", dpi=300, bbox_inches="tight")
    print("\nGraph visualization saved to 'hsr_layout_graph.png'.")
    plt.show()


if __name__ == "__main__":
    main()