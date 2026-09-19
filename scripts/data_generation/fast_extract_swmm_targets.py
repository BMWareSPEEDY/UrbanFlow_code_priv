import glob
import os
import re
import pandas as pd
from pyswmm import Output, Nodes

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

def extract():
    print("Extracting SWMM node depth targets from generated scenario files...")
    all_out_files = glob.glob("swmm_inp_*.out")
    print(f"Found {len(all_out_files)} SWMM output (.out) files.")
    
    results = []
    
    for out_path in all_out_files:
        # Match pattern swmm_inp_{region}_scen{idx}.out
        filename = os.path.basename(out_path)
        m = re.match(r"swmm_inp_([a-z_]+)_scen(\d+)\.out", filename)
        if not m:
            continue
            
        region = m.group(1)
        scen_idx = int(m.group(2))
        
        if scen_idx < len(STORM_SCENARIOS):
            intensity = STORM_SCENARIOS[scen_idx]['intensity']
        else:
            intensity = 50.0
            
        try:
            with Output(out_path) as out:
                # out.nodes is a dict mapping node_id -> Node object
                for node_id in out.nodes:
                    # Get max depth series or max depth value
                    depth_series = out.node_series(node_id, 0) # 0 is depth in PySWMM Output
                    max_depth = float(max(depth_series.values())) if depth_series else 0.0
                    results.append({
                        'region': region,
                        'scenario_idx': scen_idx,
                        'intensity_mmhr': intensity,
                        'swmm_node_id': node_id,
                        'max_water_depth_m': max_depth
                    })
        except Exception as e:
            # If binary .out reading fails, check if rpt file exists
            rpt_path = out_path.replace(".out", ".rpt")
            if os.path.exists(rpt_path):
                try:
                    with open(rpt_path, 'r') as f:
                        lines = f.readlines()
                    in_summary = False
                    for line in lines:
                        if "Node Depth Summary" in line:
                            in_summary = True
                            continue
                        if in_summary and line.strip().startswith("---"):
                            continue
                        if in_summary and line.strip() == "":
                            in_summary = False
                            continue
                        if in_summary:
                            parts = line.split()
                            if len(parts) >= 4:
                                node_id = parts[0]
                                try:
                                    max_d = float(parts[2])
                                    results.append({
                                        'region': region,
                                        'scenario_idx': scen_idx,
                                        'intensity_mmhr': intensity,
                                        'swmm_node_id': node_id,
                                        'max_water_depth_m': max_d
                                    })
                                except ValueError:
                                    pass
                except Exception as e2:
                    print(f"Error parsing {out_path} / {rpt_path}: {e2}")

    if len(results) > 0:
        df = pd.DataFrame(results)
        df.to_csv("swmm_multi_scenario_targets.csv", index=False)
        print(f"Extracted {len(df)} target records saved to 'swmm_multi_scenario_targets.csv'!")
        return True
    else:
        print("No targets extracted!")
        return False

if __name__ == "__main__":
    extract()
