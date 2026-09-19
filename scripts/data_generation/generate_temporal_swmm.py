"""Generate temporal SWMM depth targets for all regions and scenarios."""
import os
import numpy as np
import osmnx as ox
from pyswmm import Simulation, Nodes

intensities = [20.0, 50.0, 80.0, 120.0, 150.0, 200.0, 250.0, 300.0]
durations = [60, 60, 45, 30, 30, 45, 30, 30]

region_files = {
    'hsr': 'bengaluru_complete_graph.graphml',
    'bellandur': 'bengaluru_bellandur_graph.graphml',
    'whitefield': 'bengaluru_whitefield_graph.graphml',
    'ecity': 'bengaluru_ecity_graph.graphml',
    'koramangala': 'bengaluru_koramangala_graph.graphml',
    'hyderabad': 'city_hyderabad_graph.graphml',
    'chennai': 'city_chennai_graph.graphml',
    'pune': 'city_pune_graph.graphml',
    'mumbai': 'city_mumbai_graph.graphml',
    'delhi': 'city_delhi_graph.graphml',
    'kolkata': 'city_kolkata_graph.graphml',
    'ahmedabad': 'city_ahmedabad_graph.graphml',
    'jaipur': 'city_jaipur_graph.graphml',
    'lucknow': 'city_lucknow_graph.graphml',
    'kochi': 'city_kochi_graph.graphml',
    'surat': 'city_surat_graph.graphml',
    'indore': 'city_indore_graph.graphml',
    'bhopal': 'city_bhopal_graph.graphml',
    'nagpur': 'city_nagpur_graph.graphml',
    'visakhapatnam': 'city_visakhapatnam_graph.graphml',
    'coimbatore': 'city_coimbatore_graph.graphml',
    'patna': 'city_patna_graph.graphml',
    'kanpur': 'city_kanpur_graph.graphml',
    'chandigarh': 'city_chandigarh_graph.graphml',
    'vadodara': 'city_vadodara_graph.graphml',
    'madurai': 'city_madurai_graph.graphml',
    'guwahati': 'city_guwahati_graph.graphml',
    'vijayawada': 'city_vijayawada_graph.graphml',
    'varanasi': 'city_varanasi_graph.graphml',
    'rajkot': 'city_rajkot_graph.graphml',
    'ludhiana': 'city_ludhiana_graph.graphml',
    'ranchi': 'city_ranchi_graph.graphml',
    'agra': 'city_agra_graph.graphml',
    'mysore': 'city_mysore_graph.graphml',
    'mumbai_r': 'city_mumbai_r_graph.graphml',
    'jakarta': 'city_jakarta_graph.graphml',
    'bangkok': 'city_bangkok_graph.graphml',
    'istanbul': 'city_istanbul_graph.graphml',
    'berlin': 'city_berlin_graph.graphml',
    'chicago': 'city_chicago_graph.graphml',
    'sanfrancisco': 'city_sanfrancisco_graph.graphml',
    'seoul': 'city_seoul_graph.graphml',
    'sydney': 'city_sydney_graph.graphml',
    'taipei': 'city_taipei_graph.graphml',
    'lisbon': 'city_lisbon_graph.graphml',
    'vancouver': 'city_vancouver_graph.graphml',
    'rio': 'city_rio_graph.graphml',
    'naples': 'city_naples_graph.graphml',
}

OUTPUT_DIR = r"D:\CODES\PYTHON_CODES\UrbanFLOW\swmm_temporal"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def extract_swmm_timeseries(inp_file, node_ids, dt=300, duration_hours=1.0):
    T = int(duration_hours * 3600 / 300)
    with Simulation(inp_file) as sim:
        sim.step_advance(300)
        depths = np.zeros((T, len(node_ids)), dtype=np.float32)
        node_map = {nid: Nodes(sim)[f"J_{nid}"] for nid in node_ids}
        for step in range(T):
            next(sim)
            for i, nid in enumerate(node_ids):
                depths[step, i] = node_map[nid].depth
    return depths


def process_region(region, fname):
    path = rf"D:\CODES\PYTHON_CODES\UrbanFLOW\{fname}"
    if not os.path.exists(path):
        print(f"  SKIP {region}: file not found")
        return
    print(f"Processing {region} ({fname})...")
    G = ox.load_graphml(path)
    nodes_list = list(G.nodes())
    for intensity, duration in zip(intensities, durations):
        inp_file = os.path.join(OUTPUT_DIR, f"temp_{region}_{intensity}.inp")
        export_graph_to_swmm_inp(G, inp_file, intensity)
        T = int(duration * 60 / 5)
        try:
            depths = extract_swmm_timeseries(inp_file, nodes_list, dt=300, duration_hours=duration/60.0)
            out_file = os.path.join(OUTPUT_DIR, f"{region}_I{intensity:.0f}_depths.npy")
            np.save(out_file, depths)
            print(f"  {region} I={intensity:.0f}: {depths.shape}")
        except Exception as e:
            print(f"  ERROR {region} I={intensity:.0f}: {e}")
        finally:
            if os.path.exists(inp_file):
                os.remove(inp_file)


if __name__ == "__main__":
    print(f"Processing {len(region_files)} regions x {len(intensities)} scenarios...")
    for reg, fname in region_files.items():
        process_region(reg, fname)
    print("Done.")