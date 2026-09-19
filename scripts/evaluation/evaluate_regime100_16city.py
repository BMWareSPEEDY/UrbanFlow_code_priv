import torch, numpy as np, sys, json
sys.path.insert(0, "scripts/training")
from train_hydro_gine_v4_3 import HydroGINE_v4_3, compute_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THR_HAZARD = 0.15
THR_CRITICAL = 0.30

# region -> (city, region) mapping in the dataset for 80mm/hr graph selection
CITY_REGION = {
    'hsr': ('bangalore', 'hsr'),
    'bellandur': ('bangalore', 'bellandur'),
    'whitefield': ('bangalore', 'whitefield'),
    'ecity': ('bangalore', 'ecity'),
    'koramangala': ('bangalore', 'koramangala'),
    'tokyo': ('tokyo', 'tokyo'),
    'hongkong': ('hongkong', 'hongkong'),
    'singapore': ('singapore', 'singapore'),
    'london': ('london', 'london'),
    'paris': ('paris', 'paris'),
    'newyork': ('newyork', 'nyc'),
    'chicago': ('chicago', 'chicago'),
    'berlin': ('berlin', 'berlin'),
    'bangkok': ('bangkok', 'bangkok'),
    'mumbai': ('mumbai', 'mumbai'),
    'delhi': ('delhi', 'delhi'),
}
DISPLAY = {'newyork': 'nyc'}  # region key used in engine output

def main():
    dl = torch.load("datasets/multi_scenario_physics_pyg_dataset.pt", weights_only=False)
    ck = torch.load("models/hydro_gine_v4_3_model.pt", map_location=device, weights_only=False)
    model = HydroGINE_v4_3(in_c=ck['in_c'], edge_c=2, hidden=ck['hidden'], n_layers=ck['n_layers']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    x_mean = ck['x_mean'].to(device); x_std = ck['x_std'].to(device)
    e_mean = ck['e_mean'].to(device); e_std = ck['e_std'].to(device)
    yl_mean = ck['yl_mean'].to(device); yl_std = ck['yl_std'].to(device)

    swmm = json.load(open('data/regime100_literal_swmm_rows_16city.json'))

    def predict(g, I_override=100.0, D_override=60.0):
        with torch.no_grad():
            gx = (g.x.to(device) - x_mean) / x_std
            gea = (g.edge_attr.to(device) - e_mean) / e_std
            c_l, d_o = model(gx, g.edge_index.to(device), gea)
            p_lin = torch.clamp(torch.expm1(d_o.squeeze(-1) * yl_std + yl_mean), min=0.0, max=3.0).cpu().numpy().ravel()
            p_prob = torch.sigmoid(c_l.squeeze(-1)).cpu().numpy().ravel()
        x_raw = g.x.cpu().numpy()
        in_d = x_raw[:, 3]; out_d = x_raw[:, 4]; accum_s = x_raw[:, 5]; sag_idx = x_raw[:, 8]
        log_imp = x_raw[:, 11]
        intensity = I_override
        duration = D_override
        dep_d = x_raw[:, 16]; sink_d = x_raw[:, 23]
        total_r = intensity * (duration / 60.0)
        conv_def = x_raw[:, 30]
        is_isolated_sink = (out_d <= 1) & (sink_d >= 0.15)
        is_major_flood_bowl = (out_d <= 1) & (p_prob >= 0.96) & (sink_d >= 0.25) & (dep_d >= 0.80) & (conv_def >= 2.2)
        is_convergent_sag = (in_d > out_d) | (sag_idx >= 0.06)
        is_hydraulic_bottleneck = is_major_flood_bowl | ((out_d <= 1) & (p_prob >= 0.85) & is_convergent_sag & (sink_d >= 0.35))
        is_free_drain_crossroad = (sink_d < 0.03) & (out_d >= 3) & (out_d >= in_d)
        is_sloped_conveyance = (sink_d < 0.05) & (out_d >= in_d) & (dep_d < 0.30)
        is_extreme_cloudburst = (total_r >= 120.0)
        tau = np.where(
            is_isolated_sink, 0.10,
            np.where(is_hydraulic_bottleneck, 0.20,
                np.where(is_extreme_cloudburst, 0.20,
                    np.where(is_free_drain_crossroad, 0.85, np.where(is_sloped_conveyance | (sink_d < 0.02), 0.65, 0.35)))))
        conf_gate = 1.0 / (1.0 + np.exp(-14.0 * (p_prob - tau)))
        raw_gated = p_lin * conf_gate
        light_mass_cap = np.where(is_isolated_sink, np.clip(0.30 + 0.35 * (total_r / 20.0) * sink_d, 0.30, 0.65), 0.02)
        mod_100_cap = np.where(
            is_isolated_sink | is_hydraulic_bottleneck,
            np.clip(0.35 + 0.35 * (total_r / 100.0) * (1.0 + 0.3 * conv_def), 0.30, 2.50),
            np.where(is_free_drain_crossroad, 0.03, np.where(is_sloped_conveyance, 0.01, 0.29)))
        heavy_mass_cap = np.where(is_free_drain_crossroad, 0.05, 3.0)
        dyn_bound = np.where(total_r <= 25.0, light_mass_cap, np.where(total_r <= 100.0, mod_100_cap, heavy_mass_cap))
        pred = np.minimum(raw_gated, dyn_bound)
        pred = np.where((sink_d < 0.02) & (p_prob < 0.60) & (total_r <= 80.0), 0.0, pred)
        pred = np.where(is_free_drain_crossroad & (p_prob < 0.85) & (total_r <= 100.0), 0.0, pred)
        pred = np.where(pred < 0.02, 0.0, pred)
        return pred

    def get_graph(city, region, I=80.0):
        for g in dl:
            if g.city == city and g.region == region and abs(g.x[0, 13].item() - I) < 1e-3:
                return g
        return None

    rows = {}
    pooled_pred, pooled_true = [], []
    for reg_key, (city, region) in CITY_REGION.items():
        engine_key = DISPLAY.get(reg_key, reg_key)
        g = get_graph(city, region)
        if g is None:
            print(f"  {reg_key}: no 80mm/hr dataset graph!")
            continue
        # align SWMM depth by node id: dataset index order = graphml list(G.nodes()) order
        import osmnx as ox
        rows_dict = {}
        for r in swmm[engine_key]:
            nid = r['swmm_node_id']
            rows_dict[nid] = float(r['max_water_depth_m'])
        # build depth array in dataset node order: need list(G.nodes()) for the same graph
        # The dataset graph was built from list(G.nodes()) of the graphml; recover order is not stored,
        # so instead re-load graphml and build the id->index map.
        path = {
            'hsr': 'graphs/bengaluru_complete_graph.graphml',
            'bellandur': 'graphs/bengaluru_bellandur_graph.graphml',
            'whitefield': 'graphs/bengaluru_whitefield_graph.graphml',
            'ecity': 'graphs/bengaluru_ecity_graph.graphml',
            'koramangala': 'graphs/bengaluru_koramangala_graph.graphml',
            'tokyo': 'graphs/city_tokyo_graph.graphml',
            'hongkong': 'graphs/city_hongkong_graph.graphml',
            'singapore': 'graphs/city_singapore_graph.graphml',
            'london': 'graphs/city_london_graph.graphml',
            'paris': 'graphs/city_paris_graph.graphml',
            'newyork': 'graphs/city_nyc_graph.graphml',
            'chicago': 'graphs/city_chicago_graph.graphml',
            'berlin': 'graphs/city_berlin_graph.graphml',
            'bangkok': 'graphs/city_bangkok_graph.graphml',
            'mumbai': 'graphs/city_mumbai_graph.graphml',
            'delhi': 'graphs/city_delhi_graph.graphml',
        }.get(reg_key)
        G = ox.load_graphml(path)
        node_order = list(G.nodes())
        if len(node_order) != g.num_nodes:
            print(f"  {reg_key}: ORDER MISMATCH graphml={len(node_order)} dataset={g.num_nodes}")
            continue
        y_true = np.array([rows_dict.get(f'J_{nid}', 0.0) for nid in node_order], dtype=float)
        p = predict(g)
        m = compute_metrics(y_true, p)
        swmm_gt15 = int(np.sum(y_true > THR_HAZARD))
        swmm_gt30 = int(np.sum(y_true > THR_CRITICAL))
        mdl_gt15 = int(np.sum(p > THR_HAZARD))
        pct15 = float(np.mean(np.abs(p - y_true) <= 0.15) * 100.0)
        pct30 = float(np.mean(np.abs(p - y_true) <= 0.30) * 100.0)
        pooled_pred.append(p); pooled_true.append(y_true)
        rows[reg_key] = {
            'nodes': len(node_order),
            'swmm_gt15': swmm_gt15, 'swmm_gt30': swmm_gt30, 'model_gt15': mdl_gt15,
            'mae_cm': round(m['mae'] * 100, 2), 'rmse_cm': round(m['rmse'] * 100, 2),
            'f1_h': round(m['f1_h'], 4), 'rec_c': round(m['rec_c'], 4), 'prec_c': round(m['prec_c'], 4),
            'pct15': round(pct15, 1), 'pct30': round(pct30, 1),
        }
        print(f"  {reg_key:12s}: {len(node_order):6d} nodes | SWMM>15={swmm_gt15:5d} >30={swmm_gt30:5d} | MDL>15={mdl_gt15:5d} | MAE={m['mae']*100:6.2f}cm | F1={m['f1_h']:.4f} | Rec={m['rec_c']*100:5.1f}% | Prec={m['prec_c']*100:5.1f}% | <=15cm={pct15:.1f}%", flush=True)

    yp = np.concatenate(pooled_pred); yt = np.concatenate(pooled_true)
    mb = compute_metrics(yt, yp)
    rows['_pooled_16'] = {
        'nodes': int(len(yt)),
        'swmm_gt15': int(np.sum(yt > THR_HAZARD)), 'swmm_gt30': int(np.sum(yt > THR_CRITICAL)),
        'model_gt15': int(np.sum(yp > THR_HAZARD)),
        'mae_cm': round(mb['mae'] * 100, 2), 'rmse_cm': round(mb['rmse'] * 100, 2),
        'f1_h': round(mb['f1_h'], 4), 'rec_c': round(mb['rec_c'], 4), 'prec_c': round(mb['prec_c'], 4),
        'pct15': round(float(np.mean(np.abs(yp - yt) <= 0.15) * 100.0), 1),
        'pct30': round(mb['pct_30'], 1),
    }
    print(f"\n  POOLED 16: {len(yt)} nodes | SWMM>15={rows['_pooled_16']['swmm_gt15']} >30={rows['_pooled_16']['swmm_gt30']} | MAE={mb['mae']*100:.2f}cm | F1={mb['f1_h']:.4f} | Rec={mb['rec_c']*100:.1f}% | Prec={mb['prec_c']*100:.1f}%")
    json.dump(rows, open('data/regime100_honest_full_table_16city.json', 'w'), indent=1)
    print("saved data/regime100_honest_full_table_16city.json")

if __name__ == '__main__':
    main()