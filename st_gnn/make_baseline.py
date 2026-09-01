import sys
import torch
import numpy as np
from torch_geometric.data import Batch

sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW")
sys.path.insert(0, r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn")

device = torch.device('cuda')
ck = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\blr_gine6_final.pt", weights_only=False)

from blr_gine6_test import GINE4

model = GINE4(in_c=22).to(device)
model.load_state_dict(ck['model'])
model.eval()

dl = torch.load(r"D:\CODES\PYTHON_CODES\UrbanFLOW\multi_scenario_pyg_dataset.pt", weights_only=False)
te_g = [g.clone() for g in dl if g.city == 'bangalore']
te = Batch.from_data_list(te_g).to(device)

with torch.no_grad():
    out = model((te.x - ck['x_mean'].to(device)) / ck['x_std'].to(device), te.edge_index,
                (te.edge_attr - ck['e_mean'].to(device)) / ck['e_std'].to(device))

pred = np.clip(np.expm1(out.cpu().numpy() * ck['yl_std'].item() + ck['yl_mean'].item()), 0, None).ravel()
torch.save({'pred': pred}, r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn\baseline_preds.pt")
print(f"baseline preds saved, n={len(pred)}")