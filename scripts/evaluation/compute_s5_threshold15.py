import torch, numpy as np, json, osmnx as ox, sys
sys.path.insert(0, 'scripts/training')
from train_hydro_gine_v4_3 import HydroGINE_v4_3

device = torch.device('cpu')
ck = torch.load('models/hydro_gine_v4_3_model.pt', map_location=device, weights_only=False)
model = HydroGINE_v4_3(in_c=ck['in_c'], edge_c=2, hidden=ck['hidden'], n_layers=ck['n_layers']).to(device)
model.load_state_dict(ck['model']); model.eval()
x_mean = ck['x_mean'].to(device); x_std = ck['x_std'].to(device)
e_mean = ck['e_mean'].to(device); e_std = ck['e_std'].to(device)
yl_mean = ck['yl_mean'].to(device); yl_std = ck['yl_std'].to(device)
dl = torch.load('datasets/multi_scenario_physics_pyg_dataset.pt', weights_only=False)
sw = json.load(open('data/regime100_literal_swmm_rows_16city.json'))

def predict(g, I=100.0, D=60.0):
    with torch.no_grad():
        gx = (g.x.to(device) - x_mean) / x_std
        gea = (g.edge_attr.to(device) - e_mean) / e_std
        c_l, d_o = model(gx, g.edge_index.to(device), gea)
        p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0, max=3.0).cpu().numpy().ravel()
        p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
    x = g.x.cpu().numpy()
    in_d = x[:, 3]; out_d = x[:, 4]; sag = x[:, 8]; dep_d = x[:, 16]; sink_d = x[:, 23]; conv = x[:, 30]
    total_r = I * (D / 60.0)
    is_iso = (out_d <= 1) & (sink_d >= 0.15)
    is_bowl = (out_d <= 1) & (p_prob >= 0.96) & (sink_d >= 0.25) & (dep_d >= 0.80) & (conv >= 2.2)
    is_conv = (in_d > out_d) | (sag >= 0.06)
    is_bneck = is_bowl | ((out_d <= 1) & (p_prob >= 0.85) & is_conv & (sink_d >= 0.35))
    is_free = (sink_d < 0.03) & (out_d >= 3) & (out_d >= in_d)
    is_slope = (sink_d < 0.05) & (out_d >= in_d) & (dep_d < 0.30)
    is_ext = (total_r >= 120.0)
    tau = np.where(is_iso, 0.10, np.where(is_bneck, 0.20, np.where(is_ext, 0.20,
        np.where(is_free, 0.85, np.where(is_slope | (sink_d < 0.02), 0.65, 0.35)))))
    conf = 1.0 / (1.0 + np.exp(-14.0 * (p_prob - tau)))
    raw = p_lin * conf
    light = np.where(is_iso, np.clip(0.30 + 0.35 * (total_r / 20.0) * sink_d, 0.30, 0.65), 0.02)
    mod = np.where(is_iso | is_bneck, np.clip(0.35 + 0.35 * (total_r / 100.0) * (1.0 + 0.3 * conv), 0.30, 2.50),
                   np.where(is_free, 0.03, np.where(is_slope, 0.01, 0.29)))
    heavy = np.where(is_free, 0.05, 3.0)
    db = np.where(total_r <= 25.0, light, np.where(total_r <= 100.0, mod, heavy))
    pred = np.minimum(raw, db)
    pred = np.where((sink_d < 0.02) & (p_prob < 0.60) & (total_r <= 80.0), 0.0, pred)
    pred = np.where(is_free & (p_prob < 0.85) & (total_r <= 100.0), 0.0, pred)
    pred = np.where(pred < 0.02, 0.0, pred)
    return pred

paths = {
    'hsr': 'graphs/bengaluru_complete_graph.graphml', 'bellandur': 'graphs/bengaluru_bellandur_graph.graphml',
    'whitefield': 'graphs/bengaluru_whitefield_graph.graphml', 'ecity': 'graphs/bengaluru_ecity_graph.graphml',
    'koramangala': 'graphs/bengaluru_koramangala_graph.graphml', 'tokyo': 'graphs/city_tokyo_graph.graphml',
    'hongkong': 'graphs/city_hongkong_graph.graphml', 'singapore': 'graphs/city_singapore_graph.graphml',
    'london': 'graphs/city_london_graph.graphml', 'paris': 'graphs/city_paris_graph.graphml',
    'newyork': 'graphs/city_nyc_graph.graphml', 'chicago': 'graphs/city_chicago_graph.graphml',
    'berlin': 'graphs/city_berlin_graph.graphml', 'bangkok': 'graphs/city_bangkok_graph.graphml',
    'mumbai': 'graphs/city_mumbai_graph.graphml', 'delhi': 'graphs/city_delhi_graph.graphml',
}
pairs = [
    ('hsr', 'bangalore', 'hsr'), ('bellandur', 'bangalore', 'bellandur'),
    ('whitefield', 'bangalore', 'whitefield'), ('ecity', 'bangalore', 'ecity'),
    ('koramangala', 'bangalore', 'koramangala'), ('tokyo', 'tokyo', 'tokyo'),
    ('hongkong', 'hongkong', 'hongkong'), ('singapore', 'singapore', 'singapore'),
    ('london', 'london', 'london'), ('paris', 'paris', 'paris'),
    ('newyork', 'newyork', 'nyc'), ('chicago', 'chicago', 'chicago'),
    ('berlin', 'berlin', 'berlin'), ('bangkok', 'bangkok', 'bangkok'),
    ('mumbai', 'mumbai', 'mumbai'), ('delhi', 'delhi', 'delhi'),
]
res = {}
for reg, city, region in pairs:
    engkey = reg if reg != 'newyork' else 'nyc'
    g = [x for x in dl if x.city == city and x.region == region and abs(x.x[0, 13].item() - 80.0) < 1e-3][0]
    G = ox.load_graphml(paths[reg]); order = list(G.nodes())
    d = {r['swmm_node_id']: float(r['max_water_depth_m']) for r in sw[engkey]}
    y = np.array([d.get(f'J_{n}', 0.0) for n in order], dtype=float)
    p = predict(g)
    thr = 0.15
    tp = int(np.sum((y > thr) & (p > thr))); fp = int(np.sum((y <= thr) & (p > thr))); fn = int(np.sum((y > thr) & (p <= thr)))
    prec = tp / (tp + fp) * 100 if tp + fp else 0; rec = tp / (tp + fn) * 100 if tp + fn else 0
    f1 = 2 * tp / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) else 0
    mae = np.abs(p - y).mean() * 100
    res[reg] = (len(order), int(np.sum(y > thr)), int(np.sum(p > thr)), tp, fp, fn, round(prec, 1), round(rec, 1), round(f1, 1), round(mae, 2))
    print(f"{reg:11s}: N={len(order)} SWMM={int(np.sum(y > thr))} MDL={int(np.sum(p > thr))} TP={tp} FP={fp} FN={fn} P={prec:.1f} R={rec:.1f} F1={f1:.1f} MAE={mae:.2f}")
json.dump(res, open('data/regime100_benchmark_s5_threshold15.json', 'w'), indent=1)
print("saved data/regime100_benchmark_s5_threshold15.json")