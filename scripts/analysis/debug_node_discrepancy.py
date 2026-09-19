import torch
import numpy as np
import pandas as pd
import osmnx as ox
from train_perfect_accuracy_gnn import PerfectAccuracyGNN

region_files = {
    'hsr': 'bengaluru_complete_graph.graphml',
    'bellandur': 'bengaluru_bellandur_graph.graphml',
    'whitefield': 'bengaluru_whitefield_graph.graphml',
    'ecity': 'bengaluru_ecity_graph.graphml',
    'koramangala': 'bengaluru_koramangala_graph.graphml'
}

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / max(1e-6, ss_tot))

def debug_nodes():
    print("=================================================================")
    print("   URBANFLOW DEEP NODE-BY-NODE DISCREPANCY AUDIT (10-FEATURE GNN)")
    print("=================================================================\n")
    
    data = torch.load("bengaluru_pyg_dataset.pt", weights_only=False)
    y_mean = float(data.y.mean())
    y_std = float(data.y.std() + 1e-6)

    model = PerfectAccuracyGNN(in_channels=10, hidden_channels=128, out_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load("pinn_gnn_model.pth", map_location=device))
    model.eval()
    model.to(device)
    data = data.to(device)

    x_norm = (data.x - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-6)
    edge_norm = (data.edge_attr - data.edge_attr.mean(dim=0)) / (data.edge_attr.std(dim=0) + 1e-6)

    with torch.no_grad():
        out_norm = model(x_norm, data.edge_index, edge_attr=edge_norm).squeeze()
        preds = torch.clamp(out_norm * y_std + y_mean, min=0.0).cpu().numpy()
        
    targets = data.y.squeeze().cpu().numpy()

    overall_mae = np.mean(np.abs(preds - targets))
    overall_r2 = r2_score(targets, preds)
    print(f"OVERALL ALL-BENGALURU MAE:  {overall_mae:.4f} m ({overall_mae*100:.2f} cm)")
    print(f"OVERALL ALL-BENGALURU R²:   {overall_r2:.4f} ({overall_r2*100:.2f}% variance explained)\n")

    # Audit per region
    offset = 0
    for reg, file_path in region_files.items():
        G = ox.load_graphml(file_path)
        node_list = list(G.nodes())
        n_nodes = len(node_list)
        
        reg_preds = preds[offset : offset + n_nodes]
        reg_targets = targets[offset : offset + n_nodes]
        reg_diffs = np.abs(reg_preds - reg_targets)
        reg_r2 = r2_score(reg_targets, reg_preds)
        
        print(f"--- REGION: {reg.upper()} ({file_path}) ---")
        print(f"  Nodes: {n_nodes} | MAE: {np.mean(reg_diffs):.4f}m ({np.mean(reg_diffs)*100:.2f}cm) | R²: {reg_r2:.4f} | Max Diff: {np.max(reg_diffs):.4f}m")
        
        worst_local_indices = np.argsort(reg_diffs)[-5:]
        for idx in reversed(worst_local_indices):
            nid = node_list[idx]
            p_val = reg_preds[idx]
            t_val = reg_targets[idx]
            diff = reg_diffs[idx]
            print(f"    Node {str(nid):12s} | Target (SWMM): {t_val:.4f}m | Pred (GNN): {p_val:.4f}m | Diff: {diff:.4f}m ({diff*100:.2f}cm)")
        print()
        
        offset += n_nodes

if __name__ == "__main__":
    debug_nodes()
