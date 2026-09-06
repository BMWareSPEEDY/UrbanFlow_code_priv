"""Deep comprehensive audit of all errors > 15cm across all scenarios and cities.
Identifies exactly where, why, and in what hydraulic regimes errors occur.
"""
import torch, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from production_v4 import ProductionFloodPredictorV4
from train_hydro_gine_v4 import HydroGINE_v4

def deep_comprehensive_audit():
    dl = torch.load("multi_scenario_physics_pyg_dataset.pt", weights_only=False)
    predictor = ProductionFloodPredictorV4("hydro_gine_v4_model.pt", device=device)
    
    print("=" * 115)
    print("            DEEP ERROR TAXONOMY: ALL NODES WITH ABSOLUTE ERROR > 15 CM")
    print("=" * 115)
    
    for region_name in ['hsr', 'bellandur', 'whitefield', 'ecity', 'koramangala', 'hongkong', 'tokyo']:
        r_graphs = [g for g in dl if (getattr(g, 'region', None) == region_name or getattr(g, 'city', None) == region_name)]
        if not r_graphs:
            continue
        
        all_errs = []
        all_y_true = []
        all_y_pred = []
        all_x = []
        
        for g in r_graphs:
            I = g.x[0, 13].item()
            dur = g.x[0, 14].item()
            y_t = g.y.cpu().numpy().ravel()
            p_f, p_raw, p_prob = predictor.predict(g, I, dur)
            
            err = np.abs(p_f - y_t)
            all_errs.append(err)
            all_y_true.append(y_t)
            all_y_pred.append(p_f)
            all_x.append(g.x.cpu().numpy())
            
        errs = np.concatenate(all_errs)
        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        x_all = np.concatenate(all_x, axis=0)
        
        n_total = len(errs)
        n_gt15 = np.sum(errs > 0.15)
        n_gt30 = np.sum(errs > 0.30)
        
        # Over-predictions (Model >> SWMM) vs Under-predictions (Model << SWMM)
        over_15 = np.sum((y_pred - y_true) > 0.15)
        under_15 = np.sum((y_true - y_pred) > 0.15)
        
        print(f"\nRegion: {region_name.upper():<14s} (Evaluated Nodes: {n_total:6d})")
        print(f"  Nodes with Error > 15 cm: {n_gt15:5d} ({n_gt15/n_total*100:4.1f}%) | Error > 30 cm: {n_gt30:5d} ({n_gt30/n_total*100:4.1f}%)")
        print(f"  Over-predictions (Model > SWMM + 15cm):  {over_15:5d} ({over_15/n_total*100:4.1f}%)")
        print(f"  Under-predictions (SWMM > Model + 15cm): {under_15:5d} ({under_15/n_total*100:4.1f}%)")
        print(f"  Overall MAE: {np.mean(errs)*100:5.2f} cm | Median Error: {np.median(errs)*100:5.2f} cm | % <= 15cm: {np.mean(errs <= 0.15)*100:4.1f}%")

        # Let's inspect the top 10 largest error cases in this region
        top_err_idx = np.argsort(errs)[::-1][:10]
        print(f"  Top 5 Largest Error Nodes in {region_name}:")
        print(f"  {'Rain (mm/hr)':<12s} | {'Pred Depth':<12s} | {'SWMM Depth':<12s} | {'Diff (m)':<10s} | {'Sink D':<8s} | {'In/Out':<7s} | {'ConvDef':<8s} | {'SagIdx'}")
        print("  " + "-" * 95)
        for idx in top_err_idx[:5]:
            I = x_all[idx, 13]
            p = y_pred[idx]
            t = y_true[idx]
            diff = p - t
            sd = x_all[idx, 23]
            ind = x_all[idx, 3]
            outd = x_all[idx, 4]
            conv = x_all[idx, 30]
            sag = x_all[idx, 8]
            print(f"  {I:<12.0f} | {p:<12.4f} | {t:<12.4f} | {diff:<+10.4f} | {sd:<8.4f} | {ind:.0f}/{outd:.0f}   | {conv:<8.4f} | {sag:.4f}")

if __name__ == '__main__':
    deep_comprehensive_audit()
