import os
import time
import osmnx as ox
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from pyswmm import Simulation, Nodes

region_files = {
    'surat': 'city_surat_graph.graphml',
    'indore': 'city_indore_graph.graphml',
    'bhopal': 'city_bhopal_graph.graphml',
    'nagpur': 'city_nagpur_graph.graphml',
    'visakhapatnam': 'city_visakhapatnam_graph.graphml',
    'coimbatore': 'city_coimbatore_graph.graphml',
    'patna': 'city_patna_graph.graphml',
}

STORM_SCENARIOS = [
    {'intensity': 20.0, 'duration': 60},
    {'intensity': 50.0, 'duration': 60},
    {'intensity': 80.0, 'duration': 45},
    {'intensity': 120.0, 'duration': 30},
    {'intensity': 150.0, 'duration': 30},
    {'intensity': 200.0, 'duration': 45},
    {'intensity': 250.0, 'duration': 30},
    {'intensity': 300.0, 'duration': 30},
]


def export_graph_to_swmm_inp(G, inp_filename="temp_storm.inp", rain_intensity_mmhr=50.0):
    lowest_node_id = min(
        G.nodes(data=True),
        key=lambda x: float(x[1].get('elevation', 9999))
    )[0]

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

        f.write("[CONDUITS]\n;;Name FromNode ToNode Length Roughness InOffset OutOffset\n")
        for u, v, k, data in G.edges(keys=True, data=True):
            if u != v:
                length = max(float(data.get('length', 10.0)), 1.0)
                n_val = float(data.get('manning_n', 0.013))
                f.write(f"C_{u}_{v}_{k} J_{u} J_{v} {length:.2f} {n_val:.3f} 0 0\n")
        f.write(f"C_outfall J_{lowest_node_id} Outfall_1 10.00 0.013 0 0\n\n")

        f.write("[XSECTIONS]\n;;Link Shape Geom1 Geom2 Geom3 Geom4 Barrels\n")
        for u, v, k in G.edges(keys=True):
            if u != v:
                f.write(f"C_{u}_{v}_{k} RECT_OPEN 1.0 1.5 0 0 1\n")
        f.write("C_outfall RECT_OPEN 1.0 1.5 0 0 1\n\n")


def run_single_simulation(args):
    reg, s_idx, scen, G = args
    intensity = scen['intensity']
    inp_file = f"swmm_b3_{reg}_scen{s_idx}.inp"
    export_graph_to_swmm_inp(G, inp_filename=inp_file, rain_intensity_mmhr=intensity)

    node_depths = {}
    with Simulation(inp_file) as sim:
        for step in sim:
            pass
        for node in Nodes(sim):
            node_depths[node.nodeid] = node.depth

    if os.path.exists(inp_file):
        os.remove(inp_file)

    task_results = []
    for node_id, depth in node_depths.items():
        task_results.append({
            'region': reg,
            'scenario_idx': s_idx,
            'intensity_mmhr': intensity,
            'swmm_node_id': node_id,
            'max_water_depth_m': depth
        })
    print(f"  [OK] City '{reg}' scenario {s_idx} ({intensity} mm/hr) finished.", flush=True)
    return task_results


def main():
    print("=================================================================")
    print("  SWMM ENGINE - CITIES OUTSIDE BENGALURU (BATCH 3)")
    print("=================================================================\n", flush=True)

    print("Pre-loading city graphml topologies...", flush=True)
    graphs = {reg: ox.load_graphml(path) for reg, path in region_files.items()}

    tasks = []
    for reg, G in graphs.items():
        for s_idx, scen in enumerate(STORM_SCENARIOS):
            tasks.append((reg, s_idx, scen, G))

    n_cores = min(16, cpu_count())
    print(f"Launching {len(tasks)} SWMM runs across {n_cores} parallel processes...", flush=True)
    t0 = time.time()

    with Pool(processes=n_cores) as pool:
        results = pool.map(run_single_simulation, tasks)

    t1 = time.time()
    flattened = [item for sublist in results for item in sublist]
    df_results = pd.DataFrame(flattened)
    df_results.to_csv("swmm_city_b3_targets.csv", index=False)
    print(f"\nAll SWMM targets generated in {t1 - t0:.2f} seconds!")
    print(f"Saved {len(df_results)} targets to 'swmm_city_b3_targets.csv'.", flush=True)


if __name__ == "__main__":
    main()