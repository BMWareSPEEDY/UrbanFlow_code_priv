"""Production UrbanFLOW flood predictor v4.0.
Clean, strictly zero-leakage, physics-grounded GNN surrogate.
Model: HydroGINE-v4.0 (6-layer Multi-Scale Residual GINE + Margin Focal Classifier + FiLM Depth Regressor).
Features: 32 pure physical and hydraulic features (Zero coordinates / Zero spatial memorization).
Inference: Pure neural forward pass with Universal Physical Continuity Bounding.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv

THR_HAZARD = 0.15
THR_CRITICAL = 0.30

class HydroGINE_v4(nn.Module):
    def __init__(self, in_c=32, edge_c=2, hidden=128, n_layers=6):
        super().__init__()
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        for i in range(n_layers):
            c_in = in_c if i == 0 else hidden
            self.convs.append(GINEConv(
                nn.Sequential(
                    nn.Linear(c_in, hidden),
                    nn.LayerNorm(hidden),
                    nn.LeakyReLU(0.1),
                    nn.Linear(hidden, hidden)
                ),
                edge_dim=edge_c
            ))
            self.lns.append(nn.LayerNorm(hidden))
            
        self.cat_dim = hidden * 2 + in_c
        
        # Hazard Classifier Head
        self.cls = nn.Sequential(
            nn.Linear(self.cat_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
        
        # FiLM cross-task conditioning: prob -> (gamma, beta)
        self.film_gen = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 2 * self.cat_dim)
        )
        nn.init.zeros_(self.film_gen[2].weight)
        nn.init.zeros_(self.film_gen[2].bias)
        
        # Continuous Depth Regressor Head
        self.reg = nn.Sequential(
            nn.Linear(self.cat_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x, ei, ea):
        h = x
        mid_h = None
        for i, (conv, ln) in enumerate(zip(self.convs, self.lns)):
            h_next = F.elu(ln(conv(h, ei, ea)))
            if i > 0 and h.shape == h_next.shape:
                h = h_next + 0.3 * h
            else:
                h = h_next
            if i == (len(self.convs) // 2):
                mid_h = h
                
        cat = torch.cat([h, mid_h, x], dim=-1)
        cls_logits = self.cls(cat)
        prob = torch.sigmoid(cls_logits)
        
        film = self.film_gen(prob)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        h_cond = cat * (1.0 + gamma) + beta
        
        raw_depth = self.reg(h_cond)
        return cls_logits, raw_depth


class GravityGINEConv(nn.Module):
    def __init__(self, in_c, out_c, edge_c=2):
        super().__init__()
        self.conv = GINEConv(
            nn.Sequential(
                nn.Linear(in_c, out_c),
                nn.LayerNorm(out_c),
                nn.LeakyReLU(0.1),
                nn.Linear(out_c, out_c)
            ),
            edge_dim=edge_c
        )
        self.ln = nn.LayerNorm(out_c)
        
    def forward(self, h, ei, ea):
        grade = ea[:, 1:2]
        gravity_gate = torch.sigmoid(1.0 - 5.0 * F.relu(grade))
        ea_gated = ea * gravity_gate
        return F.leaky_relu(self.ln(self.conv(h, ei, ea_gated)), 0.1)


class HydroGINE_v5(nn.Module):
    def __init__(self, in_c=32, edge_c=2, hidden=128, n_layers=6):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            c_in = in_c if i == 0 else hidden
            self.convs.append(GravityGINEConv(c_in, hidden, edge_c))
            
        cat_dim = hidden * 2 + in_c
        self.cls = nn.Sequential(
            nn.Linear(cat_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
        self.film_gen = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, cat_dim * 2)
        )
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)
        self.reg = nn.Sequential(
            nn.Linear(cat_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x, ei, ea):
        h = x
        mid_h = None
        for i, conv in enumerate(self.convs):
            h_next = conv(h, ei, ea)
            if i > 0 and h.shape == h_next.shape:
                h = h_next + 0.3 * h
            else:
                h = h_next
            if i == (len(self.convs) // 2):
                mid_h = h
        cat = torch.cat([h, mid_h, x], dim=-1)
        cls_logits = self.cls(cat)
        prob = torch.sigmoid(cls_logits)
        film = self.film_gen(prob)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        h_cond = cat * (1.0 + gamma) + beta
        raw_depth = self.reg(h_cond)
        return cls_logits, raw_depth


class ProductionFloodPredictorV4:
    """Production flood prediction engine for UrbanFLOW."""
    
    def __init__(self, model_path="hydro_gine_v5_model.pt", device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(model_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, model_path)
            
        ck = torch.load(model_path, map_location=self.device, weights_only=False)
        if "convs.0.conv.nn.0.weight" in ck['model']:
            self.model = HydroGINE_v5(
                in_c=ck['in_c'],
                edge_c=2,
                hidden=ck.get('hidden', 128),
                n_layers=ck.get('n_layers', 6)
            ).to(self.device)
        else:
            self.model = HydroGINE_v4(
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

    def predict(self, graph_or_batch, intensity_mmhr, duration_min=60.0):
        """Pure neural inference with Universal Physical Continuity Bounding.
        Returns:
            gated_depth: Final physically bounded depth (numpy array)
            raw_depth: Continuous unconstrained regressor depth (numpy array)
            prob: Hazard probability (numpy array)
        """
        if isinstance(graph_or_batch, (list, tuple)):
            batch = Batch.from_data_list(graph_or_batch).to(self.device)
        else:
            batch = graph_or_batch.to(self.device)
            
        x_in = batch.x.clone()
        if x_in.shape[1] == 28:
            # 28 features (columns 0 and 1 are rel_x, rel_y) -> drop 0, 1 to get 26 physical features
            x_phys_26 = x_in[:, 2:].clone()
            x_np = x_phys_26.cpu().numpy()
            
            in_deg = x_np[:, 3]
            out_deg = x_np[:, 4]
            sink_depth = x_np[:, 23]
            dep_depth = x_np[:, 16]
            max_in_grade = x_np[:, 7]
            log_imp = x_np[:, 11]
            path_cap = x_np[:, 18]
            
            deg_diff = in_deg - out_deg
            total_rain_mm = intensity_mmhr * (duration_min / 60.0)
            dyn_sat = x_np[:, 1] * (1.0 + 0.5 * np.log1p(intensity_mmhr * duration_min / 1000.0))
            true_ponding = np.log1p(sink_depth * total_rain_mm / (np.maximum(0.2, out_deg) + 0.3))
            
            inflow_load = np.expm1(log_imp) * total_rain_mm
            pipe_drain_cap = np.expm1(path_cap) + 0.1
            conv_def = np.log1p(inflow_load / pipe_drain_cap)
            dep_escape = dep_depth / (np.maximum(0.005, np.abs(max_in_grade)) + 0.01)
            
            x_np[:, 13] = intensity_mmhr
            x_np[:, 14] = duration_min
            
            derived = np.stack([deg_diff, np.full_like(deg_diff, total_rain_mm), dyn_sat, true_ponding, conv_def, dep_escape], axis=1)
            x_full = torch.tensor(np.concatenate([x_np, derived], axis=1), dtype=torch.float32, device=self.device)
        elif x_in.shape[1] == 24:
            # 24 features from static feature builder in app.py
            # Map into 32 physics features
            x_np = x_in.cpu().numpy()
            num_nodes = x_np.shape[0]
            
            # 0:rel_x, 1:rel_y, 2:delta_elev, 3:rel_drop, 4:manning, 5:in_deg, 6:out_deg,
            # 7:accum_score, 8:is_sink, 9:max_in_grade, 10:sag_index, 11:hyd_cap,
            # 12:log_area, 13:log_imp_area, 14:dist_frac, 15:rain, 16:dur,
            # 17:elev_std2, 18:dep_depth, 19:surcharge, 20:path_cap, 21:path_hops, 22:dist_outlet, 23:imp
            rel_drop = x_np[:, 3]
            imp = x_np[:, 23]
            manning = x_np[:, 4]
            in_deg = x_np[:, 5]
            out_deg = x_np[:, 6]
            accum_score = x_np[:, 7]
            is_sink = x_np[:, 8]
            max_in_grade = x_np[:, 9]
            sag_index = x_np[:, 10]
            hyd_cap = x_np[:, 11]
            log_area = x_np[:, 12]
            log_imp = x_np[:, 13]
            dist_frac = x_np[:, 14]
            elev_std2 = x_np[:, 17]
            dep_depth = x_np[:, 18]
            surcharge = x_np[:, 19]
            path_cap = x_np[:, 20]
            path_hops = x_np[:, 21]
            dist_outlet = x_np[:, 22]
            elev_above_outlet = np.maximum(0.0, rel_drop * 10.0)
            slope_outlet_ratio = np.maximum(0.0, max_in_grade / (0.3 + elev_above_outlet))
            sink_depth = np.where(dep_depth >= 0.05, dep_depth, 0.0)
            inlet_cap = np.full(num_nodes, 0.12, dtype=np.float32)
            surcharge_ratio = np.full(num_nodes, 100.0, dtype=np.float32) * intensity_mmhr
            
            deg_diff = in_deg - out_deg
            total_rain_mm = intensity_mmhr * (duration_min / 60.0)
            dyn_sat = imp * (1.0 + 0.5 * np.log1p(intensity_mmhr * duration_min / 1000.0))
            true_ponding = np.log1p(sink_depth * total_rain_mm / (np.maximum(0.2, out_deg) + 0.3))
            inflow_load = np.expm1(log_imp) * total_rain_mm
            pipe_drain_cap = np.expm1(path_cap) + 0.1
            conv_def = np.log1p(inflow_load / pipe_drain_cap)
            dep_escape = dep_depth / (np.maximum(0.005, np.abs(max_in_grade)) + 0.01)
            
            intensity_arr = np.full(num_nodes, intensity_mmhr, dtype=np.float32)
            dur_arr = np.full(num_nodes, duration_min, dtype=np.float32)
            
            phys_32 = np.stack([
                rel_drop, imp, manning, in_deg, out_deg, accum_score, is_sink, max_in_grade,
                sag_index, hyd_cap, log_area, log_imp, dist_frac, intensity_arr, dur_arr,
                elev_std2, dep_depth, surcharge, path_cap, path_hops, dist_outlet,
                elev_above_outlet, slope_outlet_ratio, sink_depth, inlet_cap, surcharge_ratio,
                deg_diff, np.full(num_nodes, total_rain_mm, dtype=np.float32), dyn_sat, true_ponding, conv_def, dep_escape
            ], axis=1).astype(np.float32)
            x_full = torch.tensor(phys_32, dtype=torch.float32, device=self.device)
        elif x_in.shape[1] == 32:
            x_full = x_in.clone()
            x_np = x_full.cpu().numpy()
            num_nodes = x_np.shape[0]
            
            imp = x_np[:, 1]
            out_deg = x_np[:, 4]
            log_imp = x_np[:, 11]
            path_cap = x_np[:, 18]
            sink_depth = x_np[:, 23]
            
            total_rain_mm = intensity_mmhr * (duration_min / 60.0)
            x_np[:, 13] = intensity_mmhr
            x_np[:, 14] = duration_min
            x_np[:, 25] = 100.0 * intensity_mmhr
            x_np[:, 27] = total_rain_mm
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
        accum_s = x_raw[:, 5]
        sag_idx = x_raw[:, 8]
        log_imp = x_raw[:, 11]
        dep_d = x_raw[:, 16]
        sink_d = x_raw[:, 23]
        total_r = x_raw[:, 27]
        conv_def = x_raw[:, 30]
        
        # 1. Hydraulic Regime Identification:
        is_isolated_sink = (out_d <= 1) & (sink_d >= 0.15) & (rel_drop >= 0.50)
        is_major_flood_bowl = (out_d <= 1) & (p_prob >= 0.96) & (sink_d >= 0.25) & (dep_d >= 0.80) & (conv_def >= 2.2) & (rel_drop >= 0.50)
        is_convergent_sag = ((in_d > out_d) | (sag_idx >= 0.06)) & (rel_drop >= 0.40)
        is_hydraulic_bottleneck = is_major_flood_bowl | ((out_d <= 1) & (p_prob >= 0.85) & is_convergent_sag & (sink_d >= 0.35))
        
        # Upland slopes and high outflow crossroads drain freely
        is_upland_slope = (rel_drop < 0.45) & (out_d >= 1) & (sink_d < 0.10)
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
                    np.where(
                        is_upland_slope | is_free_drain_crossroad,
                        0.85,
                        np.where(is_sloped_conveyance | (sink_d < 0.02), 0.65, 0.35)
                    )
                )
            )
        )
        
        conf_gate = 1.0 / (1.0 + np.exp(-14.0 * (p_prob - tau)))
        raw_gated = p_lin * conf_gate
        
        # 3. Dynamic Hydrologic Continuity Bounds (UPCB v5):
        dyn_bound = np.where(
            is_major_flood_bowl | (is_isolated_sink & (p_prob >= 0.90)),
            3.0,
            np.where(
                is_free_drain_crossroad | is_upland_slope,
                0.02,
                np.where(
                    sink_d < 0.02,
                    0.02 if total_r <= 25.0 else (0.05 if total_r <= 80.0 else 0.12),
                    np.where(
                        sink_d < 0.10,
                        0.05 if total_r <= 50.0 else 0.20,
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
        
        # 4. Dry Pavement Clean-Up (Zero-out false standing water in sloped/dry channels):
        is_dry_pavement = (sink_d < 0.02) & (dep_d < 0.03) & (p_prob < 0.60)
        pred_final = np.where(is_dry_pavement, 0.0, pred_final)
        pred_final = np.where(is_upland_slope & (p_prob < 0.85) & (total_r <= 100.0), 0.0, pred_final)
        pred_final = np.where(is_free_drain_crossroad & (p_prob < 0.85) & (total_r <= 100.0), 0.0, pred_final)
        pred_final = np.where(pred_final < 0.02, 0.0, pred_final)
        
        return pred_final, p_lin, p_prob
