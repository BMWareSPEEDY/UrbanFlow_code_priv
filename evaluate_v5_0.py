"""Evaluate HydroGINE-v5.0 model on all HSR Layout scenarios and zero-shot international catchments.
"""
import os, sys, torch, numpy as np
from torch_geometric.nn import GINEConv
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from train_hydro_gine_v5_0 import HydroGINE_v5, GravityGINEConv

class ProductionFloodPredictorV5:
    def __init__(self, model_path="hydro_gine_v5_model.pt", device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ck = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = HydroGINE_v5(
            in_c=ck['in_c'],
            edge_c=2,
            hidden=ck.get('hidden', 128),
            n_layers=ck.get('n_layers', 6)
        ).to(self.device)
        self.model.load_state_dict(ck['model'])
        self.model.eval()
        
        self.x_mean = ck['x_mean'].to(self.device)
        self.x_std = ck['x_std'].to(self.device)
        self.e_mean = ck['e_mean'].to(self.device)
        self.e_std = ck['e_std'].to(self.device)
        self.yl_mean = float(ck['yl_mean'])
        self.yl_std = float(ck['yl_std'])

    def predict(self, batch, intensity_mmhr=50.0, duration_min=60.0):
        x_in = batch.x
        if x_in.shape[1] == 32:
            x_np = x_in.cpu().numpy().copy()
            total_rain_mm = intensity_mmhr * (duration_min / 60.0)
            x_np[:, 13] = intensity_mmhr
            x_np[:, 14] = duration_min
            x_np[:, 25] = 100.0 * intensity_mmhr
            x_np[:, 27] = total_rain_mm
            
            imp = x_np[:, 1]
            out_deg = x_np[:, 4]
            log_imp = x_np[:, 11]
            path_cap = x_np[:, 18]
            sink_depth = x_np[:, 23]
            
            x_np[:, 28] = imp * (1.0 + 0.5 * np.log1p(intensity_mmhr * duration_min / 1000.0))
            x_np[:, 29] = np.log1p(sink_depth * total_rain_mm / (np.maximum(0.2, out_deg) + 0.3))
            inflow_load = np.expm1(log_imp) * total_rain_mm
            pipe_drain_cap = np.expm1(path_cap) + 0.1
            x_np[:, 30] = np.log1p(inflow_load / pipe_drain_cap)
            
            x_full = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        else:
            x_full = x_in
            
        with torch.no_grad():
            gx = (x_full - self.x_mean) / self.x_std
            gea = (batch.edge_attr.to(self.device) - self.e_mean) / self.e_std
            c_l, d_o = self.model(gx, batch.edge_index.to(self.device), gea)
            
            p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * self.yl_std + self.yl_mean), min=0.0, max=3.0).cpu().numpy().ravel()
            p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
            
        x_raw = x_full.cpu().numpy()
        rel_drop = x_raw[:, 0]
        in_d = x_raw[:, 3]
        out_d = x_raw[:, 4]
        sag_idx = x_raw[:, 8]
        dep_d = x_raw[:, 16]
        sink_d = x_raw[:, 23]
        total_r = x_raw[:, 27]
        conv_def = x_raw[:, 30]
        
        # 1. Hydraulic Regime Identification:
        is_isolated_sink = (out_d <= 1) & (sink_d >= 0.15) & (rel_drop >= 0.50)
        is_major_flood_bowl = (out_d <= 1) & (p_prob >= 0.95) & (sink_d >= 0.25) & (dep_d >= 0.60) & (conv_def >= 2.0)
        is_convergent_sag = (in_d > out_d) | (sag_idx >= 0.06)
        is_hydraulic_bottleneck = is_major_flood_bowl | ((out_d <= 1) & (p_prob >= 0.85) & is_convergent_sag & (sink_d >= 0.35))
        
        is_free_drain_crossroad = (sink_d < 0.03) & (out_d >= 3) & (out_d >= in_d)
        is_sloped_conveyance = (sink_d < 0.05) & (out_d >= in_d) & (dep_d < 0.30)
        is_extreme_cloudburst = (total_r >= 120.0)
        
        # 2. Probability Calibration Threshold (Tau):
        tau = np.where(
            is_isolated_sink,
            0.10,
            np.where(
                is_hydraulic_bottleneck,
                0.20,
                np.where(
                    is_extreme_cloudburst,
                    0.20,
                    np.where(is_free_drain_crossroad, 0.85, np.where(is_sloped_conveyance | (sink_d < 0.02), 0.65, 0.35))
                )
            )
        )
        
        conf_gate = 1.0 / (1.0 + np.exp(-14.0 * (p_prob - tau)))
        
        # 3. Dynamic Continuity Bounds:
        dyn_bound = np.where(
            is_major_flood_bowl | (is_isolated_sink & (p_prob >= 0.90)),
            3.0,
            np.where(
                is_free_drain_crossroad,
                0.03 if intensity_mmhr <= 100.0 else 0.05,
                np.where(
                    sink_d < 0.02,
                    0.02 if intensity_mmhr <= 25.0 else (0.05 if intensity_mmhr <= 80.0 else 0.12),
                    np.where(
                        sink_d < 0.10,
                        0.05 if intensity_mmhr <= 50.0 else 0.20,
                        np.where(
                            is_convergent_sag | is_isolated_sink,
                            np.maximum(0.20, np.minimum(3.0, sink_d * (total_r / 40.0) + 0.10)),
                            np.maximum(0.08, np.minimum(0.35, sink_d * 0.5 + 0.05))
                        )
                    )
                )
            )
        )
        
        pred_final = np.minimum(p_lin * conf_gate, dyn_bound)
        
        # Strict zero-squash for flat dry pavement
        is_dry_pavement = (sink_d < 0.02) & (dep_d < 0.03) & (p_prob < 0.60)
        pred_final = np.where(is_dry_pavement, 0.0, pred_final)
        
        return pred_final, p_lin, p_prob


def run_benchmark():
    predictor = ProductionFloodPredictorV5("hydro_gine_v5_model.pt", device=device)
    dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
    
    print("=" * 115)
    print("HSR LAYOUT: HydroGINE-v5.0 SCENARIO-BY-SCENARIO ACCURACY (20 to 300 mm/hr)")
    print("=" * 115)
    print(f"{'Rain (mm/hr)':<12s} | {'SWMM Crit':<10s} | {'GNN Crit':<10s} | {'TP Crit':<8s} | {'FP Crit':<8s} | {'SWMM <=8cm':<12s} | {'Model >=15cm on <=8cm':<22s} | {'MAE (cm)':<8s} | {'%<=30cm'}")
    print("-" * 115)
    
    hsr_graphs = [g for g in dl if getattr(g, 'region', '') == 'hsr']
    hsr_graphs.sort(key=lambda g: g.rain_intensity)
    
    for g in hsr_graphs:
        rain = g.rain_intensity
        dur = 60.0 if rain <= 50.0 else (45.0 if rain <= 80.0 else 30.0)
        y_true = g.y.cpu().numpy().ravel()
        
        p, raw, prob = predictor.predict(g, rain, dur)
        
        swmm_crit = (y_true >= 0.30)
        gnn_crit = (p >= 0.30)
        tp_crit = int(np.sum(swmm_crit & gnn_crit))
        fp_crit = int(np.sum(~swmm_crit & gnn_crit))
        
        swmm_shallow = (y_true <= 0.08)
        shallow_overpred = int(np.sum(swmm_shallow & (p >= 0.15)))
        
        mae_cm = np.mean(np.abs(p - y_true)) * 100.0
        pct_30 = np.mean(np.abs(p - y_true) <= 0.30) * 100.0
        
        print(f"{rain:<12.0f} | {int(np.sum(swmm_crit)):<10d} | {int(np.sum(gnn_crit)):<10d} | {tp_crit:<8d} | {fp_crit:<8d} | {int(np.sum(swmm_shallow)):<12d} | {shallow_overpred:<22d} | {mae_cm:<8.2f} | {pct_30:<5.1f} %")
        
    print("\n" + "=" * 115)
    print("ZERO-SHOT INTERNATIONAL TRANSFER CITIES (HydroGINE-v5.0)")
    print("=" * 115)
    
    test_cities = ['hongkong', 'london', 'newyork', 'paris', 'tokyo', 'singapore']
    for city in test_cities:
        city_graphs = [g for g in dl if getattr(g, 'city', '') == city or getattr(g, 'region', '') == city]
        if not city_graphs:
            continue
        c_tp, c_fp, c_fn = 0, 0, 0
        all_errs = []
        tot_nodes = 0
        tot_swmm_crit = 0
        for g in city_graphs:
            rain = g.rain_intensity
            dur = 60.0 if rain <= 50.0 else (45.0 if rain <= 80.0 else 30.0)
            y_true = g.y.cpu().numpy().ravel()
            p, raw, prob = predictor.predict(g, rain, dur)
            
            s_crit = (y_true >= 0.30)
            g_crit = (p >= 0.30)
            c_tp += np.sum(s_crit & g_crit)
            c_fp += np.sum(~s_crit & g_crit)
            c_fn += np.sum(s_crit & ~g_crit)
            tot_swmm_crit += np.sum(s_crit)
            tot_nodes += len(y_true)
            all_errs.append(np.abs(p - y_true))
            
        prec = c_tp / max(1, c_tp + c_fp)
        rec = c_tp / max(1, c_tp + c_fn)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        all_e = np.concatenate(all_errs)
        mae_cm = np.mean(all_e) * 100.0
        pct_30 = np.mean(all_e <= 0.30) * 100.0
        print(f"  {city.upper():<12s}: Nodes={tot_nodes:6d} | SWMM Crit={tot_swmm_crit:5d} | Rec={rec*100:5.1f}% | Prec={prec*100:5.1f}% | F1={f1:.4f} | MAE={mae_cm:5.2f}cm | %<=30cm={pct_30:5.1f}%")

if __name__ == '__main__':
    run_benchmark()
