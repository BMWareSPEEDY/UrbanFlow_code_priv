"""Extract 5-min interval depth time series from SWMM .out files for temporal ST-GNN training."""
import os
import struct
import numpy as np
from pyswmm import Simulation, Nodes

def extract_swmm_timeseries(inp_file, node_ids, dt=300, duration_hours=1.0):
    """Extract depth time series at dt-second intervals for given node IDs.
    Returns array of shape (T, N) where T = duration_hours*3600/dt.
    """
    T = int(duration_hours * 3600 / dt)
    with Simulation(inp_file) as sim:
        sim.step_advance(dt)
        depths = np.zeros((T, len(node_ids)), dtype=np.float32)
        node_map = {nid: Nodes(sim)[f"J_{nid}"] for nid in node_ids}
        
        for step in range(T):
            next(sim)  # Advance one time step
            for i, nid in enumerate(node_ids):
                depths[step, i] = node_map[nid].depth
        return depths

def run_swmm_temporal_for_region(region_name, graphml_path, intensities, durations, output_dir):
    """Run SWMM for all scenarios and save temporal depth arrays."""
    os.makedirs(output_dir, exist_ok=True)
    import osmnx as ox
    G = ox.load_graphml(graphml_path)
    nodes_list = list(G.nodes())
    node_to_idx = {nid: i for i, nid in enumerate(nodes_list)}
    
    for intensity, duration in zip(intensities, durations):
        # Use existing SWMM inp generator from generate_city_b5_swmm_targets.py pattern
        inp_file = f"temp_{region_name}_{intensity}.inp"
        # Generate inp file using existing export function
        export_graph_to_swmm_inp(G, inp_file, intensity)
        
        depths = extract_swmm_timeseries(inp_file, nodes_list, dt=300, duration_hours=duration/60.0)
        out_file = os.path.join(output_dir, f"{region_name}_I{intensity:.0f}_depths.npy")
        np.save(out_file, depths)
        print(f"Saved {out_file} shape={depths.shape}")
        os.remove(inp_file)

# Inline the export function to avoid circular imports
def export_graph_to_swmm_inp(G, inp_filename, rain_intensity_mmhr=50.0):
    lowest_node_id = min(G.nodes(data=True), key=lambda x: float(x[1].get('elevation', 9999)))[0]
    with open(inp_filename, "w") as f:
        f.write("[TITLE]\nUrbanFlow-GNN SWMM Regional Simulation\n\n")
        f.write("[OPTIONS]\nFLOW_UNITS CMS\nINFILTRATION HORTON\nFLOW_ROUTING DYNWAVE\n")
        f.write("START_DATE 01/01/2026\nSTART_TIME 00:00:00\n")
        f.write("REPORT_START_DATE 01/01/2026\nREPORT_START_TIME 00:00:00\n")
        f.write("END_DATE 01/01/2026\nEND_TIME 02:00:00\n")
        f.write("REPORT_STEP 00:01:00\nWET_STEP 00:00:10\nDRY_STEP 00:01:00\n\n")
        f.write("[EVAPORATION]\nCONSTANT 0.0\n\n")
        f.write("[REPORT]\nINPUT NO\nSUBCATCHMENTS ALL\nNODES ALL\nLINKS ALL\n\n")
        f.write("[RAINGAGES]\nGage1 INTENSITY 0:05 1.0 TIMESERIES RainTS\n\n")
        f.write("[TIMESERIES]\n")
        f.write(f"RainTS 00:00 {rain_intensity_mmhr:.1f}\n")
        f.write(f"RainTS 01:00 {rain_intensity_mmhr:.1f}\n")
        f.write("RainTS 01:05 0.0\n\n")
        f.write("[SUBCATCHMENTS]\n;;Name RainGage Outlet Area %Imperv Width Slope CurbLen\n")
        for node_id, data in G.nodes(data=True):
            raw_imp = float(data.get('impervious_ratio', 0.2)) * 100
            f.write(f"S_{node_id} Gage1 J_{node_id} 0.5 {raw_imp:.1f} 50 0.5 0\n")
        f.write("\n")
        f.write("[SUBAREAS]\n;;Subcatchment N-Imperv N-Perv S-Imperv S-Perv PctZero RouteTo\n")
        for node_id in G.nodes():
            f.write(f"S_{node_id} 0.013 0.1 2.0 5.0 25 OUTLET\n")
        f.write("\n")
        f.write("[INFILTRATION]\n;;Subcatchment MaxRate MinRate Decay DryTime MaxInfil\n")
        for node_id in G.nodes():
            f.write(f"S_{node_id} 75.0 3.0 4.0 7.0 0\n")
        f.write("\n")
        f.write("[JUNCTIONS]\n;;Name Elevation MaxDepth InitDepth SurDepth Aponded\n")
        for node_id, data in G.nodes(data=True):
            elev = float(data.get('elevation', 880.0))
            f.write(f"J_{node_id} {elev:.2f} 3.0 0 0 100\n")
        f.write("\n")
        f.write("[OUTFALLS]\n;;Name Elevation Type Stage Data Gated\n")
        lowest_elev = float(G.nodes[lowest_node_id].get('elevation', 870.0))
        outfall_elev = lowest_elev - 1.0
        f.write(f"Outfall_1 {outfall_elev:.2f} FREE\n\n")
        f.write("[CONDUITS]\n;;Name FromNode ToNode Length Roughness InOffset OutOffset InitFlow MaxFlow\n")
        for u, v, k, data in G.edges(keys=True, data=True):
            length = float(data.get('length', 10.0))
            manning = float(data.get('manning_n', 0.013))
            f.write(f"C_{u}_{v}_{k} J_{u} J_{v} {length:.1f} {manning:.3f} 0 0 0 0\n")
        f.write("\n")
        f.write("[XSECTIONS]\n;;Link Shape Geom1 Geom2 Geom3 Geom4 Barrels\n")
        for u, v, k, data in G.edges(keys=True, data=True):
            f.write(f"C_{u}_{v}_{k} CIRCULAR 0.5 0 0 0 1\n")

if __name__ == "__main__":
    # Quick test on one region
    import osmnx as ox
    G = ox.load_graphml(r"D:\CODES\PYTHON_CODES\UrbanFLOW\bengaluru_bellandur_graph.graphml")
    nodes_list = list(G.nodes())
    
    intensities = [150.0]
    durations = [30]
    
    for intensity, duration in zip(intensities, durations):
        inp_file = f"temp_test_{intensity}.inp"
        export_graph_to_swmm_inp(G, inp_file, intensity)
        depths = extract_swmm_timeseries(inp_file, nodes_list, dt=300, duration_hours=duration/60.0)
        print(f"Depths shape: {depths.shape} (T={depths.shape[0]}, N={depths.shape[1]})")
        print(f"Depth range: {depths.min():.4f} - {depths.max():.4f}")
        os.remove(inp_file)