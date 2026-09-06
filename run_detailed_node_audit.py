"""Generate comprehensive, user-friendly, detailed audit report on node detection accuracy.
Evaluates:
- True Positives (Correctly Detected Floods)
- False Positives (Falsely Flagged / False Alarms)
- False Negatives (Missed Floods)
- True Negatives (Correctly Identified Dry/Safe)
- Depth Accuracy Metrics (MAE, % <= 30cm)
Across multiple rainfall intensities and domestic + international catchments.
"""
import os, sys, torch, numpy as np

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from production_v4 import ProductionFloodPredictorV4

def run_detailed_audit():
    predictor = ProductionFloodPredictorV4("hydro_gine_v5_model.pt", device=device)
    dl = torch.load("expanded_master_physics_dataset.pt", weights_only=False)
    
    # -------------------------------------------------------------
    # 1. HSR LAYOUT: SCENARIO-BY-SCENARIO DETAILED NODE AUDIT
    # -------------------------------------------------------------
    print("=" * 125)
    print("                      URBANFLOW HYDROGINE-v5.1 DETAILED NODE ACCURACY AUDIT")
    print("                                   HSR LAYOUT (1,379 Total Nodes)")
    print("=" * 125)
    print(f"{'Rainfall':<12s} | {'Total':<6s} | {'SWMM Real':<10s} | {'Correctly':<10s} | {'False Alarm':<12s} | {'Missed':<8s} | {'Correctly':<10s} | {'Flood':<8s} | {'Detection':<10s} | {'Depth':<8s} | {'Within'}")
    print(f"{'Scenario':<12s} | {'Nodes':<6s} | {'Floods':<10s} | {'Detected(TP)':<10s} | {'(FP Alarms)':<12s} | {'(FN)':<8s} | {'Dry(TN)':<10s} | {'Prec %':<8s} | {'Recall %':<10s} | {'MAE(cm)':<8s} | {'±30cm %'}")
    print("-" * 125)
    
    hsr_graphs = [g for g in dl if getattr(g, 'region', '') == 'hsr']
    hsr_graphs.sort(key=lambda g: g.rain_intensity)
    
    for g in hsr_graphs:
        rain = g.rain_intensity
        dur = 60.0 if rain <= 50.0 else (45.0 if rain <= 80.0 else 30.0)
        y_true = g.y.cpu().numpy().ravel()
        p, raw, prob = predictor.predict(g, rain, dur)
        
        # Hazard definitions:
        # Flooded/Hazard: Depth >= 0.15m (15 cm)
        # Critical Flood: Depth >= 0.30m (30 cm)
        # Safe/Dry: Depth < 0.15m
        
        is_swmm_flood = (y_true >= 0.15)
        is_gnn_flood = (p >= 0.15)
        
        tp = int(np.sum(is_swmm_flood & is_gnn_flood))
        fp = int(np.sum(~is_swmm_flood & is_gnn_flood))
        fn = int(np.sum(is_swmm_flood & ~is_gnn_flood))
        tn = int(np.sum(~is_swmm_flood & ~is_gnn_flood))
        
        real_floods = int(np.sum(is_swmm_flood))
        tot = len(y_true)
        
        prec = (tp / max(1, tp + fp)) * 100.0
        rec = (tp / max(1, tp + fn)) * 100.0
        
        mae_cm = np.mean(np.abs(p - y_true)) * 100.0
        pct_30 = np.mean(np.abs(p - y_true) <= 0.30) * 100.0
        
        rain_label = f"{rain:.0f} mm/hr"
        print(f"{rain_label:<12s} | {tot:<6d} | {real_floods:<10d} | {tp:<10d} | {fp:<12d} | {fn:<8d} | {tn:<10d} | {prec:<7.1f}% | {rec:<9.1f}% | {mae_cm:<7.2f}cm | {pct_30:<6.1f}%")
        
    # -------------------------------------------------------------
    # 2. BREAKDOWN OF FALSE ALARMS (FALSE POSITIVES) & CRITICAL NODES
    # -------------------------------------------------------------
    print("\n" + "=" * 125)
    print("             CRITICAL FLOOD HAZARD (>= 30cm) PRECISION & SUPPRESSION BREAKDOWN")
    print("=" * 125)
    print(f"{'Rainfall':<12s} | {'SWMM Real':<11s} | {'GNN Flagged':<12s} | {'True Critical':<14s} | {'False Critical':<15s} | {'Critical':<10s} | {'Critical'}")
    print(f"{'Scenario':<12s} | {'Critical':<11s} | {'Critical':<12s} | {'Detected (TP)':<14s} | {'Alarms (FP)':<15s} | {'Recall %':<10s} | {'Precision %'}")
    print("-" * 125)
    
    for g in hsr_graphs:
        rain = g.rain_intensity
        dur = 60.0 if rain <= 50.0 else (45.0 if rain <= 80.0 else 30.0)
        y_true = g.y.cpu().numpy().ravel()
        p, raw, prob = predictor.predict(g, rain, dur)
        
        is_swmm_crit = (y_true >= 0.30)
        is_gnn_crit = (p >= 0.30)
        
        tp_crit = int(np.sum(is_swmm_crit & is_gnn_crit))
        fp_crit = int(np.sum(~is_swmm_crit & is_gnn_crit))
        fn_crit = int(np.sum(is_swmm_crit & ~is_gnn_crit))
        
        swmm_crit_cnt = int(np.sum(is_swmm_crit))
        gnn_crit_cnt = int(np.sum(is_gnn_crit))
        
        prec_c = (tp_crit / max(1, tp_crit + fp_crit)) * 100.0 if gnn_crit_cnt > 0 else 100.0
        rec_c = (tp_crit / max(1, tp_crit + fn_crit)) * 100.0 if swmm_crit_cnt > 0 else 100.0
        
        rain_label = f"{rain:.0f} mm/hr"
        print(f"{rain_label:<12s} | {swmm_crit_cnt:<11d} | {gnn_crit_cnt:<12d} | {tp_crit:<14d} | {fp_crit:<15d} | {rec_c:<9.1f}% | {prec_c:<10.1f}%")

    # -------------------------------------------------------------
    # 3. MULTI-CITY & INTERNATIONAL ZERO-SHOT GENERALIZATION AUDIT
    # -------------------------------------------------------------
    print("\n" + "=" * 125)
    print("                     CROSS-CITY & INTERNATIONAL NODE ACCURACY AUDIT")
    print("=" * 125)
    print(f"{'Catchment / City':<18s} | {'Total Nodes':<12s} | {'Real Floods':<12s} | {'Detected':<10s} | {'False Alarms':<13s} | {'Missed':<8s} | {'Accuracy %':<11s} | {'Depth MAE':<10s} | {'% <= 30cm'}")
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
            p, raw, prob = predictor.predict(g, rain, dur)
            
            sf = (y_true >= 0.15)
            gf = (p >= 0.15)
            
            tot_nodes += len(y_true)
            tot_swmm_f += int(np.sum(sf))
            tot_tp += int(np.sum(sf & gf))
            tot_fp += int(np.sum(~sf & gf))
            tot_fn += int(np.sum(sf & ~gf))
            tot_tn += int(np.sum(~sf & ~gf))
            
            err_list.append(np.abs(p - y_true))
            
        overall_acc = ((tot_tp + tot_tn) / max(1, tot_nodes)) * 100.0
        all_e = np.concatenate(err_list)
        mae = np.mean(all_e) * 100.0
        pct30 = np.mean(all_e <= 0.30) * 100.0
        
        city_display = r.upper() if r in ['hsr', 'ecity'] else r.capitalize()
        print(f"{city_display:<18s} | {tot_nodes:<12d} | {tot_swmm_f:<12d} | {tot_tp:<10d} | {tot_fp:<13d} | {tot_fn:<8d} | {overall_acc:<10.1f}% | {mae:<8.2f}cm | {pct30:<7.1f}%")

if __name__ == '__main__':
    run_detailed_audit()
