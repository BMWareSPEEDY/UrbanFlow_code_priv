"""Audit raw unconstrained neural model outputs vs post-processing clamped outputs.
See if the neural model HydroGINE-v5.0 itself knows where the floods are, but post-processing is killing recall!
"""
import torch, numpy as np, sys

sys.stdout.reconfigure(line_buffering=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from train_hydro_gine_v5_0 import HydroGINE_v5

def audit_neural_vs_postprocessing():
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
    hsr_50 = [g for g in dl if getattr(g, 'region', '') == 'hsr' and abs(g.rain_intensity - 50.0) < 1e-3][0].to(device)
    
    y_true = hsr_50.y.cpu().numpy().ravel()
    gx = (hsr_50.x - x_mean) / x_std
    gea = (hsr_50.edge_attr - e_mean) / e_std
    
    with torch.no_grad():
        c_l, d_o = model(gx, hsr_50.edge_index, gea)
        raw_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
        raw_depth = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0).cpu().numpy().ravel()
        
    swmm_crit = (y_true >= 0.30)
    swmm_flood = (y_true >= 0.15)
    
    print(f"Total nodes: {len(y_true)}")
    print(f"SWMM Critical (>= 0.30m): {np.sum(swmm_crit)}")
    print(f"SWMM Flood (>= 0.15m): {np.sum(swmm_flood)}")
    
    # Check RAW probability thresholds
    for thresh in [0.20, 0.30, 0.35, 0.40, 0.50, 0.60]:
        pred_flood = (raw_prob >= thresh)
        tp = np.sum(swmm_flood & pred_flood)
        fp = np.sum(~swmm_flood & pred_flood)
        fn = np.sum(swmm_flood & ~pred_flood)
        prec = tp / max(1, tp + fp) * 100
        rec = tp / max(1, tp + fn) * 100
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        print(f"Raw Prob >= {thresh:.2f}: TP={tp:3d} | FP={fp:3d} | FN={fn:3d} | Recall={rec:5.1f}% | Prec={prec:5.1f}% | F1={f1:.2f}%")
        
    print("\n--- RAW DEPTH PREDICTIONS (NO POST-PROCESSING HEURISTICS) ---")
    for d_thresh in [0.15, 0.20, 0.25, 0.30]:
        pred_d = (raw_depth >= d_thresh)
        tp = np.sum(swmm_crit & (raw_depth >= 0.30))
        fp = np.sum(~swmm_crit & (raw_depth >= 0.30))
        fn = np.sum(swmm_crit & (raw_depth < 0.30))
        mae = np.mean(np.abs(raw_depth - y_true)) * 100
        pct30 = np.mean(np.abs(raw_depth - y_true) <= 0.30) * 100
        print(f"Raw Depth Critical (>=0.30m): TP={tp:3d} | FP={fp:3d} | FN={fn:3d} | Crit Recall={tp/np.sum(swmm_crit)*100:5.1f}% | Crit Prec={tp/max(1,tp+fp)*100:5.1f}% | MAE={mae:.2f}cm | <=30cm={pct30:.1f}%")

if __name__ == '__main__':
    audit_neural_vs_postprocessing()
