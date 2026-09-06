"""Test calibrated pipeline across all scenarios (20 to 300 mm/hr) and all 11 catchments.
"""
import torch, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from train_hydro_gine_v5_0 import HydroGINE_v5

def run_calibrated_benchmark():
    ck = torch.load("hydro_gine_v5_model.pt", map_location=device, weights_only=False)
    model = HydroGINE_v5(in_c=ck['in_c'], edge_c=2, hidden=ck.get('hidden', 128), n_layers=ck.get('n_layers', 6)).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    
    x_mean = ck['x_mean'].to(device)
    x_std = ck['x_std'].to(device)
    e_mean = ck['e_mean'].to(device)
    e_std = ck['e_std'].to(device)
    yl_mean = float(ck['yl_mean'])
    yl_std = float(ck['yl_std'])
    
    dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
    
    def predict_graph(g, rain, dur):
        gb = g.to(device)
        gx = (gb.x - x_mean) / x_std
        gea = (gb.edge_attr - e_mean) / e_std
        
        with torch.no_grad():
            c_l, d_o = model(gx, gb.edge_index, gea)
            p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
            p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0).cpu().numpy().ravel()
            
        x_raw = gb.x.cpu().numpy()
        rel_drop = x_raw[:, 0]
        in_d = x_raw[:, 3]
        out_d = x_raw[:, 4]
        accum_s = x_raw[:, 5]
        sag_idx = x_raw[:, 8]
        dep_d = x_raw[:, 16]
        sink_d = x_raw[:, 23]
        total_r = x_raw[:, 27]
        conv_def = x_raw[:, 30]
        
        # Surcharge & Valley Sink Identification
        is_choked_surcharge = (conv_def >= 0.7) | (accum_s >= 1.5) | ((dep_d >= 0.20) & (out_d <= in_d))
        is_deep_sink = (sink_d >= 0.08) & (rel_drop >= 0.40)
        is_valley_depression = (dep_d >= 0.05) & (rel_drop >= 0.40)
        is_convergent_sag = (in_d > out_d) | (sag_idx >= 0.03)
        
        # Upland Ridge Free Drainage
        is_ridge_crest = (rel_drop < 0.25) & (sink_d < 0.03) & (dep_d < 0.03)
        is_free_drain_slope = (sink_d < 0.02) & (dep_d < 0.02) & (out_d >= 2) & (~is_choked_surcharge)
        
        tau = np.where(
            is_deep_sink | is_valley_depression,
            0.15,
            np.where(
                is_choked_surcharge | is_convergent_sag,
                0.25,
                np.where(
                    is_ridge_crest | is_free_drain_slope,
                    0.75,
                    0.35
                )
            )
        )
        conf_gate = 1.0 / (1.0 + np.exp(-12.0 * (p_prob - tau)))
        
        mass_bound = np.where(
            is_deep_sink | is_valley_depression | is_choked_surcharge,
            3.0,
            np.where(
                is_ridge_crest,
                0.02,
                np.where(
                    is_free_drain_slope,
                    0.04 if total_r[0] <= 50.0 else 0.10,
                    np.maximum(0.15, sink_d * 1.5 + 0.08)
                )
            )
        )
        
        pred = np.minimum(p_lin * conf_gate, mass_bound)
        is_flat_dry = (sink_d < 0.02) & (dep_d < 0.02) & (p_prob < 0.50) & (~is_choked_surcharge)
        pred = np.where(is_flat_dry, 0.0, pred)
        pred = np.where(pred < 0.02, 0.0, pred)
        
        # Hydrostatic WSE Enveloping
        if hasattr(gb, 'edge_index') and gb.edge_index is not None and gb.edge_index.numel() > 0:
            ei = gb.edge_index
            src = ei[0].cpu().numpy()
            dst = ei[1].cpu().numpy()
            num_nodes = len(pred)
            
            elevs = - rel_drop * 20.0
            wse = elevs + pred
            
            backwater_dst = np.maximum(0.0, wse[src] - elevs[dst])
            backwater_src = np.maximum(0.0, wse[dst] - elevs[src])
            
            max_backwater = np.zeros(num_nodes, dtype=np.float32)
            np.maximum.at(max_backwater, dst, backwater_dst)
            np.maximum.at(max_backwater, src, backwater_src)
            
            is_protected = is_deep_sink | is_valley_depression | is_choked_surcharge
            pred[~is_protected] = np.minimum(
                pred[~is_protected],
                np.maximum(0.0, max_backwater[~is_protected])
            )
            pred = np.where(pred < 0.02, 0.0, pred)
            
        return pred
        
    print("=" * 125)
    print("HSR LAYOUT BENCHMARK (CALIBRATED PIPELINE):")
    print("=" * 125)
    print(f"{'Rainfall':<12s} | {'SWMM Crit':<10s} | {'Crit TP':<8s} | {'Crit FP':<8s} | {'Crit FN':<8s} | {'Crit Rec%':<10s} | {'Crit Prec%':<11s} | {'Depth MAE':<10s} | {'% <= 30cm'}")
    print("-" * 125)
    
    hsr_graphs = [g for g in dl if getattr(g, 'region', '') == 'hsr']
    hsr_graphs.sort(key=lambda g: g.rain_intensity)
    
    for g in hsr_graphs:
        rain = g.rain_intensity
        dur = 60.0 if rain <= 50.0 else (45.0 if rain <= 80.0 else 30.0)
        y_true = g.y.cpu().numpy().ravel()
        p = predict_graph(g, rain, dur)
        
        swmm_crit = (y_true >= 0.30)
        gnn_crit = (p >= 0.30)
        
        tp_c = int(np.sum(swmm_crit & gnn_crit))
        fp_c = int(np.sum(~swmm_crit & gnn_crit))
        fn_c = int(np.sum(swmm_crit & ~gnn_crit))
        
        rec_c = tp_c / max(1, tp_c + fn_c) * 100.0
        prec_c = tp_c / max(1, tp_c + fp_c) * 100.0
        mae_cm = np.mean(np.abs(p - y_true)) * 100.0
        pct30 = np.mean(np.abs(p - y_true) <= 0.30) * 100.0
        
        print(f"{rain:<6.0f} mm/hr | {int(np.sum(swmm_crit)):<10d} | {tp_c:<8d} | {fp_c:<8d} | {fn_c:<8d} | {rec_c:<9.1f}% | {prec_c:<10.1f}% | {mae_cm:<8.2f}cm | {pct30:<7.1f}%")
        
    print("\n" + "=" * 125)
    print("CROSS-CITY & INTERNATIONAL TRANSFER BENCHMARK (CALIBRATED PIPELINE):")
    print("=" * 125)
    print(f"{'City / Region':<16s} | {'Total Nodes':<12s} | {'Real Floods':<12s} | {'Detected':<10s} | {'False Alarms':<13s} | {'Missed':<8s} | {'Accuracy %':<11s} | {'Depth MAE':<10s} | {'% <= 30cm'}")
    print("-" * 125)
    
    all_regions = ['hsr', 'bellandur', 'whitefield', 'ecity', 'koramangala', 
                   'hongkong', 'singapore', 'tokyo', 'paris', 'newyork', 'london']
    
    for r in all_regions:
        r_graphs = [g for g in dl if getattr(g, 'region', '') == r or getattr(g, 'city', '') == r]
        if not r_graphs:
            continue
        tot_nodes, tot_swmm_f, tot_tp, tot_fp, tot_fn, tot_tn = 0, 0, 0, 0, 0, 0
        err_list = []
        for g in r_graphs:
            rain = g.rain_intensity
            dur = 60.0 if rain <= 50.0 else (45.0 if rain <= 80.0 else 30.0)
            y_true = g.y.cpu().numpy().ravel()
            p = predict_graph(g, rain, dur)
            
            sf = (y_true >= 0.15)
            gf = (p >= 0.15)
            
            tot_nodes += len(y_true)
            tot_swmm_f += int(np.sum(sf))
            tot_tp += int(np.sum(sf & gf))
            tot_fp += int(np.sum(~sf & gf))
            tot_fn += int(np.sum(sf & ~gf))
            tot_tn += int(np.sum(~sf & ~gf))
            err_list.append(np.abs(p - y_true))
            
        acc = (tot_tp + tot_tn) / max(1, tot_nodes) * 100.0
        all_e = np.concatenate(err_list)
        mae = np.mean(all_e) * 100.0
        pct30 = np.mean(all_e <= 0.30) * 100.0
        city_display = r.upper() if r in ['hsr', 'ecity'] else r.capitalize()
        print(f"{city_display:<16s} | {tot_nodes:<12d} | {tot_swmm_f:<12d} | {tot_tp:<10d} | {tot_fp:<13d} | {tot_fn:<8d} | {acc:<10.1f}% | {mae:<8.2f}cm | {pct30:<7.1f}%")

if __name__ == '__main__':
    run_calibrated_benchmark()
