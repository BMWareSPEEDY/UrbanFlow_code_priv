import os
import time
import osmnx as ox
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from pyswmm import Simulation, Nodes

region_files = {
    'hsr': 'bengaluru_complete_graph.graphml',
    'bellandur': 'bengaluru_bellandur_graph.graphml',
    'whitefield': 'bengaluru_whitefield_graph.graphml',
    'ecity': 'bengaluru_ecity_graph.graphml',
    'koramangala': 'bengaluru_koramangala_graph.graphml'
}

# Candidate storm scenarios (ranging from 20 mm/hr to 300 mm/hr extreme cloudbursts)
STORM_SCENARIOS = [
    {'intensity': 20.0, 'duration': 60, 'profile': 'constant'},
    {'intensity': 50.0, 'duration': 60, 'profile': 'peak'},        # Baseline
    {'intensity': 80.0, 'duration': 45, 'profile': 'peak'},
    {'intensity': 100.0, 'duration': 60, 'profile': 'cloudburst'}, # IMD cloudburst def (>=100 mm/hr, ~1 hr); Oct-2024 Bengaluru: GKVK 186.2mm single-day record
    {'intensity': 120.0, 'duration': 30, 'profile': 'cloudburst'},
    {'intensity': 150.0, 'duration': 30, 'profile': 'cloudburst'}, # Extreme cloudburst
    {'intensity': 200.0, 'duration': 45, 'profile': 'cloudburst'}, # Surcharge inflection point
    {'intensity': 250.0, 'duration': 30, 'profile': 'cloudburst'},
    {'intensity': 300.0, 'duration': 30, 'profile': 'cloudburst'}, # Severe street inundation
]

def generate_swmm_inp(reg_key, file_path, scenario_idx, intensity_mmhr, duration_min, profile):
    G = ox.load_graphml(file_path)
    lowest_node_id = min(
        G.nodes(data=True),
        key=lambda x: float(x[1].get('elevation', 9999))
    )[0]

    inp_filename = f"swmm_inp_{reg_key}_scen{scenario_idx}.inp"

    with open(inp_filename, "w") as f:
        f.write("[TITLE]\nUrbanFlow Parallel SWMM Extreme Storm Simulation\n\n")
        f.write("[OPTIONS]\nFLOW_UNITS CMS\nINFILTRATION HORTON\nFLOW_ROUTING DYNWAVE\n")
        f.write("START_DATE 01/01/2026\nSTART_TIME 00:00:00\n")
        f.write("REPORT_START_DATE 01/01/2026\nREPORT_START_TIME 00:00:00\n")
        f.write(f"END_DATE 01/01/2026\nEND_TIME 02:00:00\n")
        f.write("REPORT_STEP 00:01:00\nWET_STEP 00:00:05\nDRY_STEP 00:01:00\n\n")
        f.write("[EVAPORATION]\nCONSTANT 0.0\n\n")
        f.write("[REPORT]\nINPUT NO\nSUBCATCHMENTS ALL\nNODES ALL\nLINKS ALL\n\n")
        f.write("[RAINGAGES]\nGage1 INTENSITY 0:05 1.0 TIMESERIES RainTS\n\n")
        
        f.write("[TIMESERIES]\n")
        if profile == 'cloudburst':
            f.write(f"RainTS 00:00 {intensity_mmhr * 0.3:.1f}\n")
            f.write(f"RainTS 00:15 {intensity_mmhr * 1.5:.1f}\n") # Extreme peak cloudburst
            f.write(f"RainTS 00:30 {intensity_mmhr * 0.4:.1f}\n")
            f.write("RainTS 00:45 0.0\n\n")
        else:
            f.write(f"RainTS 00:00 {intensity_mmhr:.1f}\n")
            f.write(f"RainTS 01:00 {intensity_mmhr:.1f}\n")
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
            # Surcharge depth limit 3.0m
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

    return inp_filename

def run_single_swmm_task(task_args):
    reg_key, file_path, scen_idx, scen = task_args
    try:
        inp_file = generate_swmm_inp(reg_key, file_path, scen_idx, scen['intensity'], scen['duration'], scen['profile'])
        
        node_depths = {}
        with Simulation(inp_file) as sim:
            for step in sim:
                pass
            for node in Nodes(sim):
                node_depths[node.nodeid] = node.depth
                
        # Clean temp inp file
        if os.path.exists(inp_file):
            os.remove(inp_file)
            
        results = []
        for node_id, depth in node_depths.items():
            results.append({
                'region': reg_key,
                'scenario_idx': scen_idx,
                'intensity_mmhr': scen['intensity'],
                'swmm_node_id': node_id,
                'max_water_depth_m': depth
            })
        return results
    except Exception as e:
        print(f"Error in task ({reg_key}, scenario {scen_idx}): {e}")
        return []

def run_parallel_swmm_simulations():
    tasks = []
    scen_count = 0
    for reg_key, file_path in region_files.items():
        for s_idx, scen in enumerate(STORM_SCENARIOS):
            tasks.append((reg_key, file_path, s_idx, scen))
            scen_count += 1
            
    n_cores = min(16, cpu_count())
    print(f"\n=======================================================")
    print(f"  Parallel SWMM Multiprocessing Scenario Engine")
    print(f"  Total Scenarios: {scen_count} ({len(region_files)} regions x {len(STORM_SCENARIOS)} storms)")
    print(f"  Parallel Worker Pool Cores: {n_cores} CPU threads")
    print(f"=======================================================\n")
    
    t0 = time.time()
    with Pool(processes=n_cores) as pool:
        all_task_results = pool.map(run_single_swmm_task, tasks)
    t1 = time.time()
    
    flattened_results = [item for sublist in all_task_results for item in sublist]
    df_results = pd.DataFrame(flattened_results)
    df_results.to_csv("swmm_multi_scenario_targets.csv", index=False)
    
    print(f"\nParallel Simulation Complete! Elapsed Time: {t1 - t0:.2f} seconds.")
    print(f"Generated {len(df_results)} multi-scenario node targets saved to 'swmm_multi_scenario_targets.csv'.")
    return df_results

if __name__ == "__main__":
    run_parallel_swmm_simulations()
