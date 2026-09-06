"""Reproduce the exact 'Live Match Rate' leaderboard metric across all 16 cities.

Metric definition (identical to the comparative benchmark the user ran):
  - Risk node: GNN prediction > 0.08 m
  - MATCH:     |pred - swmm| < 0.15 m on risk nodes
  - OVER:      pred - swmm >= 0.15  (false alarms)
  - UNDER:     pred - swmm <= -0.15 (missed floods)
  - Live Match Rate = MATCH / RISK nodes
  - Global MAE over ALL nodes (cm), Hazard MAE over y>=15cm nodes (cm)
  - Catchment NSE over all nodes
  - SWMM speedup = 3500.0 * (15.0 / avg_city_latency_ms)
"""
import os, sys, time, json, numpy as np
sys.path.insert(0, os.getcwd())
from app import REGION_CACHE
from production_v4 import ProductionFloodPredictorV4

def run_benchmark(models_to_test, save_path=None):
    print("1. Assembling 16-city ground truth targets...")
    all_swmm = {}
    tot_nodes = 0
    for r_key, r_data in REGION_CACHE.items():
        node_list = r_data['node_list']
        node_pos = r_data['node_pos']
        s_depths = np.array([float(node_pos[nid]['swmm_depth']) for nid in node_list], dtype=np.float32)
        all_swmm[r_key] = s_depths
        tot_nodes += len(node_list)
    print(f"Total nodes across all 16 cities: {tot_nodes:,}")

    results = []
    for name, path in models_to_test:
        if not os.path.exists(path):
            print(f"Skipping {name} (file {path} not found)")
            continue
        print(f"\nEvaluating: {name} ({path})...")
        try:
            predictor = ProductionFloodPredictorV4(model_path=path)
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
            continue

        all_preds_list = []
        all_targets_list = []
        city_rates = []

        tot_risk = 0
        tot_match = 0
        tot_over = 0
        tot_under = 0

        _ = predictor.predict(REGION_CACHE['hsr']['pyg_data'], 50.0, 60.0)

        t0 = time.perf_counter()
        inference_times = []
        for r_key, r_data in REGION_CACHE.items():
            t_city0 = time.perf_counter()
            preds, raw_p, probs = predictor.predict(r_data['pyg_data'], 50.0, 60.0)
            t_city1 = time.perf_counter()
            inference_times.append((t_city1 - t_city0) * 1000.0)

            s = all_swmm[r_key]
            p = preds.ravel()
            all_preds_list.append(p)
            all_targets_list.append(s)

            risk = (p > 0.08)
            n_risk = int(np.sum(risk))
            diff = p[risk] - s[risk]
            n_match = int(np.sum(np.abs(diff) < 0.15))
            n_over = int(np.sum(diff >= 0.15))
            n_under = int(np.sum(diff <= -0.15))

            tot_risk += n_risk
            tot_match += n_match
            tot_over += n_over
            tot_under += n_under

            rate = (n_match / max(1, n_risk)) * 100.0
            city_rates.append({'key': r_key, 'nodes': int(n_risk),
                               'match': int(n_match), 'over': int(n_over),
                               'under': int(n_under), 'rate': round(rate, 2)})

        total_eval_time = (time.perf_counter() - t0) * 1000.0
        avg_city_latency = float(np.mean(inference_times))

        y_pred = np.concatenate(all_preds_list)
        y_true = np.concatenate(all_targets_list)

        global_mae_cm = float(np.mean(np.abs(y_pred - y_true)) * 100.0)
        haz_mask = (y_true >= 0.15)
        haz_mae_cm = float(np.mean(np.abs(y_pred[haz_mask] - y_true[haz_mask])) * 100.0) if np.sum(haz_mask) > 0 else 0.0

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        nse_glob = float(1.0 - (ss_res / max(1e-6, ss_tot)))

        overall_match_rate = float((tot_match / max(1, tot_risk)) * 100.0)
        swmm_speedup = 3500.0 * (15.0 / max(1.0, avg_city_latency))

        res = {
            'name': name,
            'path': path,
            'match_rate': overall_match_rate,
            'global_mae_cm': global_mae_cm,
            'haz_mae_cm': haz_mae_cm,
            'nse': nse_glob,
            'avg_city_latency_ms': avg_city_latency,
            'total_latency_ms': float(total_eval_time),
            'swmm_speedup': swmm_speedup,
            'risk_nodes': tot_risk,
            'matches': tot_match,
            'over': tot_over,
            'under': tot_under,
            'city_rates': city_rates
        }
        results.append(res)
        print(f"  Match Rate: {overall_match_rate:.1f}% | Global MAE: {global_mae_cm:.2f}cm | "
              f"Hazard MAE: {haz_mae_cm:.2f}cm | NSE: {nse_glob:.4f} | Avg Latency: {avg_city_latency:.1f}ms "
              f"| OVER: {tot_over} | UNDER: {tot_under}")

    results.sort(key=lambda x: x['match_rate'], reverse=True)

    print("\n" + "=" * 120)
    print(f"{'Model Version':<40s} | {'MatchRate':<10s} | {'GlbMAE':<8s} | {'HazMAE':<8s} | {'NSE':<8s} | {'Lat(ms)':<8s} | {'OVER':<7s} | {'UNDER'}")
    print("=" * 120)
    for r in results:
        print(f"{r['name']:<40s} | {r['match_rate']:>8.2f}% | {r['global_mae_cm']:>6.2f} | {r['haz_mae_cm']:>6.2f} | "
              f"{r['nse']:>8.4f} | {r['avg_city_latency_ms']:>8.2f} | {r['over']:>7d} | {r['under']:>6d}")

    if results:
        best = results[0]
        print(f"\nWINNER / BEST VERSION: {best['name']}")
        print(f"  Peak Accuracy: {best['match_rate']:.1f}% live match rate")
        print(f"  Lowest Error:  {best['global_mae_cm']:.2f} cm Global MAE, {best['haz_mae_cm']:.2f} cm Hazard MAE")
        print(f"  Inference:     {best['avg_city_latency_ms']:.2f} ms/city ({best['swmm_speedup']:.0f}x faster than SWMM)")
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {save_path}")

    return results

if __name__ == '__main__':
    models_to_test = [
        ("HydroGINE-v5 Bottleneck Opt (Active)", "hydro_gine_v5_bottleneck_opt.pt"),
        ("HydroGINE-v5 Base (100 Epochs)", "hydro_gine_v5_model.pt"),
        ("HydroGINE-v5 Finetuned (Multi-Scenario)", "hydro_gine_v5_finetuned.pt"),
        ("HydroGINE-v5 Bengaluru Opt", "hydro_gine_v5_bangalore_opt.pt"),
    ]
    run_benchmark(models_to_test, save_path="benchmark_live_match_results.json")