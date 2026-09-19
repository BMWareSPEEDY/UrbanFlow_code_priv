import osmnx as ox
import pandas as pd
from pyswmm import Simulation, Nodes


def export_graph_to_swmm_inp(G, inp_filename="hsr_50mm_storm.inp", rain_intensity_mmhr=50.0):
    """
    Generates a fully valid EPA SWMM .inp input file from the graph data.
    """
    print(f"Generating SWMM input file '{inp_filename}' for {rain_intensity_mmhr} mm/hr rainfall...")

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    # Safely find the lowest elevation node by casting elevation to float
    lowest_node_id = min(
        G.nodes(data=True),
        key=lambda x: float(x[1].get('elevation', 9999))
    )[0]

    with open(inp_filename, "w") as f:
        # 1. Title & Options
        f.write("[TITLE]\nUrbanFlow-GNN SWMM Simulation\n\n")
        f.write("[OPTIONS]\n")
        f.write("FLOW_UNITS CMS\n")
        f.write("INFILTRATION HORTON\n")
        f.write("FLOW_ROUTING DYNWAVE\n")
        f.write("START_DATE 01/01/2026\n")
        f.write("START_TIME 00:00:00\n")
        f.write("REPORT_START_DATE 01/01/2026\n")
        f.write("REPORT_START_TIME 00:00:00\n")
        f.write("END_DATE 01/01/2026\n")
        f.write("END_TIME 02:00:00\n")
        f.write("REPORT_STEP 00:01:00\n")
        f.write("WET_STEP 00:00:10\n")
        f.write("DRY_STEP 00:01:00\n\n")

        # 2. Evaporation & Report
        f.write("[EVAPORATION]\nCONSTANT 0.0\n\n")
        f.write("[REPORT]\nINPUT NO\nSUBCATCHMENTS ALL\nNODES ALL\nLINKS ALL\n\n")

        # 3. Rain Gage
        f.write("[RAINGAGES]\n")
        f.write("Gage1 INTENSITY 0:05 1.0 TIMESERIES RainTS\n\n")

        # 4. Time Series
        f.write("[TIMESERIES]\n")
        f.write(f"RainTS 00:00 {rain_intensity_mmhr}\n")
        f.write(f"RainTS 01:00 {rain_intensity_mmhr}\n")
        f.write("RainTS 01:05 0.0\n\n")

        # 5. Subcatchments (Cast impervious_ratio to float)
        f.write("[SUBCATCHMENTS]\n;;Name RainGage Outlet Area %Imperv Width Slope CurbLen\n")
        for node_id, data in G.nodes(data=True):
            raw_imp = data.get('impervious_ratio', 0.2)
            imp = float(raw_imp) * 100
            f.write(f"S_{node_id} Gage1 J_{node_id} 0.5 {imp:.1f} 50 0.5 0\n")
        f.write("\n")

        # 6. Subareas & Infiltration
        f.write("[SUBAREAS]\n;;Subcatchment N-Imperv N-Perv S-Imperv S-Perv PctZero RouteTo\n")
        for node_id in G.nodes():
            f.write(f"S_{node_id} 0.013 0.1 2.0 5.0 25 OUTLET\n")
        f.write("\n")

        f.write("[INFILTRATION]\n;;Subcatchment MaxRate MinRate Decay DryTime MaxInfil\n")
        for node_id in G.nodes():
            f.write(f"S_{node_id} 75.0 3.0 4.0 7.0 0\n")
        f.write("\n")

        # 7. Junctions (Cast elevation to float)
        f.write("[JUNCTIONS]\n;;Name Elevation MaxDepth InitDepth SurDepth Aponded\n")
        for node_id, data in G.nodes(data=True):
            elev = float(data.get('elevation', 880.0))
            f.write(f"J_{node_id} {elev:.2f} 3.0 0 0 100\n")
        f.write("\n")

        # 8. Outfalls
        f.write("[OUTFALLS]\n;;Name Elevation Type Stage Data Gated\n")
        lowest_elev = float(G.nodes[lowest_node_id].get('elevation', 870.0))
        outfall_elev = lowest_elev - 1.0
        f.write(f"Outfall_1 {outfall_elev:.2f} FREE\n\n")

        # 9. Conduits
        f.write("[CONDUITS]\n;;Name FromNode ToNode Length Roughness InOffset OutOffset\n")
        for u, v, k, data in G.edges(keys=True, data=True):
            if u != v:
                length = max(float(data.get('length', 10.0)), 1.0)
                n_val = float(data.get('manning_n', 0.013))
                f.write(f"C_{u}_{v}_{k} J_{u} J_{v} {length:.2f} {n_val:.3f} 0 0\n")
        # Connect lowest junction to outfall
        f.write(f"C_outfall J_{lowest_node_id} Outfall_1 10.00 0.013 0 0\n\n")

        # 10. Cross Sections
        f.write("[XSECTIONS]\n;;Link Shape Geom1 Geom2 Geom3 Geom4 Barrels\n")
        for u, v, k in G.edges(keys=True):
            if u != v:
                f.write(f"C_{u}_{v}_{k} RECT_OPEN 1.0 1.5 0 0 1\n")
        f.write("C_outfall RECT_OPEN 1.0 1.5 0 0 1\n\n")

    print("SWMM .inp file successfully created!")


def run_pyswmm_simulation(inp_filename="hsr_50mm_storm.inp"):
    print("Executing PySWMM simulation engine...")
    node_depths = {}

    with Simulation(inp_filename) as sim:
        for step in sim:
            pass

        for node in Nodes(sim):
            node_depths[node.nodeid] = node.depth

    print(f"Simulation complete! Tracked water depths across {len(node_depths)} SWMM nodes.")
    return node_depths


def main():
    input_file = "bengaluru_complete_graph.graphml"
    print(f"Loading complete spatial graph from '{input_file}'...")
    G = ox.load_graphml(input_file)

    export_graph_to_swmm_inp(G, inp_filename="hsr_50mm_storm.inp", rain_intensity_mmhr=50.0)

    max_depths = run_pyswmm_simulation("hsr_50mm_storm.inp")

    df_results = pd.DataFrame(list(max_depths.items()), columns=['swmm_node_id', 'max_water_depth_m'])
    df_results.to_csv("swmm_groundtruth_targets.csv", index=False)

    print("\n--- Sample SWMM Ground Truth Water Depths (Meters) ---")
    print(df_results.head(10))
    print("\nSaved ground truth targets to 'swmm_groundtruth_targets.csv'!")


if __name__ == "__main__":
    main()