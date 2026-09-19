# UrbanFLOW — Production Set Marker

Everything below is the **current, live production** system. Everything else in the repo is
iteration / superseded / misc and is not part of the running app.

## How it runs

```
venv run python app.py        # the one entry point
```

## Production files (the complete runtime set)

| Path | Role |
|------|------|
| `app.py` | Flask entry point. The only file you run. Loads regions, models, datasets, benchmarks. |
| `templates/` | Web UI (index.html). |
| `static/` | CSS / JS / images for the UI. |
| `requirements.txt` | Environment. |
| `core/production_v4.py` | `ProductionFloodPredictorV4` + `EnsembleFloodPredictorV4` — the prediction engine. |
| `core/train_perfect_accuracy_gnn.py` | `PerfectAccuracyGNN` (legacy-ckpt fallback branch of app.py). |
| `core/train_zero_tolerance_gnn.py` | `ZeroToleranceHurdleGNN` (legacy-ckpt fallback branch). |
| `core/train_dual_stream_hydro_gnn.py` | `DualStreamHydroGNN` (legacy-ckpt fallback branch). |

## Production data (loaded at startup — verified at restructure)

| Asset | Location |
|-------|----------|
| Region graphs (16 global catchments) | `graphs/*.graphml` |
| Normalization stats + ground-truth targets | `datasets/bengaluru_pyg_dataset.pt`, `datasets/swmm_groundtruth_targets.csv` |
| Production model checkpoints | `models/hydro_gine_v5_10_model.pt` (w 0.80), `hydro_gine_v5_11_model.pt` (w 0.30), `hydro_gine_v5_bangalore_opt.pt` — ensembled by `core/production_v4.py`; fallback `hydro_gine_v5_bottleneck_opt.pt` |
| Real incidents (live validation) | `data/real_bengaluru_oct2024_incidents.json` |
| Whole-city physics + residual ensembles | `models/urbanflow_production_model.pt`, `models/ensemble_v5_10_model.pt`, `models/zero_tolerance_gnn_checkpoint.pt`, `models/regional_residual_calibrations.pt` |
| Benchmark endpoint data | `data/benchmarks/*.json` / `*.csv` |

> Production models live in `models/`; the 4 `core/` modules are the production-imported
> Python. Everything else (the ~250 scripts under `scripts/`, older `.pt` in `models/`,
> PDFs/docs) are iteration history — safe to archive or remove since the restructure.
