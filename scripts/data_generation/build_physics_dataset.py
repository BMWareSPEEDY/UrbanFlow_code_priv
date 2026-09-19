"""Build strictly zero-leakage, physics-grounded PyG dataset for UrbanFLOW.
Eliminates rel_x and rel_y (columns 0 and 1) completely.
Adds physical hydrodynamic and conveyance features:
- deg_diff = in_deg - out_deg
- true_ponding_index = sink_depth * rainfall_vol / (out_deg + 0.5)
- conveyance_deficit = ln(1 + rainfall_vol * imp_area / (exp(path_cap) + 0.1))
- storm_water_mass = intensity * duration / 60
- dynamic_saturation = imp * (1 + 0.5 * ln(1 + intensity * duration / 1000))
- tc = Kirpich time-to-peak proxy
"""
import os
import sys
import numpy as np
import torch
from torch_geometric.data import Data

sys.stdout.reconfigure(line_buffering=True)

IN_PATH = "multi_scenario_full22_pyg_dataset.pt"
OUT_PATH = "multi_scenario_physics_pyg_dataset.pt"

def build_physics_dataset():
    print(f"Loading raw dataset from {IN_PATH}...")
    dl = torch.load(IN_PATH, weights_only=False)
    print(f"Loaded {len(dl)} graphs. Processing features...")
    
    physics_graphs = []
    
    for idx, g in enumerate(dl):
        # g.x has 28 features:
        # 0: rel_x (DROP - LEAKAGE)
        # 1: rel_y (DROP - LEAKAGE)
        # 2: rel_drop
        # 3: imp
        # 4: manning_n
        # 5: in_deg
        # 6: out_deg
        # 7: accum_score
        # 8: is_sink
        # 9: max_in_grade
        # 10: sag_index
        # 11: hydraulic_capacity
        # 12: log_area
        # 13: log_imp_area
        # 14: dist_frac
        # 15: intensity
        # 16: duration
        # 17: elev_std2
        # 18: dep_depth
        # 19: surcharge
        # 20: path_cap
        # 21: path_hops
        # 22: dist_outlet
        # 23: elev_above_outlet
        # 24: slope_outlet_ratio
        # 25: sink_depth
        # 26: inlet_cap
        # 27: surcharge_ratio
        
        x_raw = g.x.numpy()
        # Drop columns 0 and 1 (rel_x, rel_y) -> 26 physical features
        x_phys_26 = x_raw[:, 2:]
        
        rel_drop = x_phys_26[:, 0]
        imp = x_phys_26[:, 1]
        manning_n = x_phys_26[:, 2]
        in_deg = x_phys_26[:, 3]
        out_deg = x_phys_26[:, 4]
        accum_score = x_phys_26[:, 5]
        is_sink = x_phys_26[:, 6]
        max_in_grade = x_phys_26[:, 7]
        sag_index = x_phys_26[:, 8]
        hyd_cap = x_phys_26[:, 9]
        log_area = x_phys_26[:, 10]
        log_imp_area = x_phys_26[:, 11]
        dist_frac = x_phys_26[:, 12]
        intensity = x_phys_26[:, 13]
        duration = x_phys_26[:, 14]
        elev_std2 = x_phys_26[:, 15]
        dep_depth = x_phys_26[:, 16]
        surcharge = x_phys_26[:, 17]
        path_cap = x_phys_26[:, 18]
        path_hops = x_phys_26[:, 19]
        dist_outlet = x_phys_26[:, 20]
        elev_above_outlet = x_phys_26[:, 21]
        slope_outlet_ratio = x_phys_26[:, 22]
        sink_depth = x_phys_26[:, 23]
        inlet_cap = x_phys_26[:, 24]
        surcharge_ratio = x_phys_26[:, 25]
        
        # New Physics-Informed Derived Features:
        # 1. deg_diff: Topological convergence bottleneck
        deg_diff = in_deg - out_deg
        
        # 2. total_rain_mm: Storm total depth
        total_rain_mm = intensity * duration / 60.0
        
        # 3. dynamic_saturation: Soil/surface saturation proxy
        dyn_sat = imp * (1.0 + 0.5 * np.log1p(intensity * duration / 1000.0))
        
        # 4. true_ponding_index: Physical ponding trapped volume
        # High when sink_depth is positive and out_deg is low (cannot drain)
        true_ponding_index = np.log1p(sink_depth * total_rain_mm / (np.maximum(0.2, out_deg) + 0.3))
        
        # 5. conveyance_deficit: Inflow load vs Downstream pipe capacity
        inflow_load = np.expm1(log_imp_area) * total_rain_mm
        pipe_drain_cap = np.expm1(path_cap) + 0.1
        conveyance_deficit = np.log1p(inflow_load / pipe_drain_cap)
        
        # 6. depression_conveyance_ratio: Local depression relative to escape slope
        dep_escape_ratio = dep_depth / (np.maximum(0.005, np.abs(max_in_grade)) + 0.01)
        
        derived_feats = np.stack([
            deg_diff, total_rain_mm, dyn_sat, true_ponding_index, conveyance_deficit, dep_escape_ratio
        ], axis=1).astype(np.float32)
        
        # Combined feature matrix: 26 + 6 = 32 physical features
        x_combined = np.concatenate([x_phys_26, derived_feats], axis=1)
        
        g_new = Data(
            x=torch.tensor(x_combined, dtype=torch.float32),
            edge_index=g.edge_index.clone(),
            edge_attr=g.edge_attr.clone(),
            y=g.y.clone()
        )
        g_new.city = g.city
        g_new.region = g.region
        g_new.rain_intensity = getattr(g, 'rain_intensity', intensity[0])
        
        physics_graphs.append(g_new)
        
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(dl)} graphs: {g.city}/{g.region} x_shape={x_combined.shape}")
            
    print(f"Saving {len(physics_graphs)} physics-grounded graphs to {OUT_PATH}...")
    torch.save(physics_graphs, OUT_PATH)
    print("Dataset generation complete!")

if __name__ == '__main__':
    build_physics_dataset()
