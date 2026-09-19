# UrbanFLOW: Comprehensive 16-City Scientific Benchmark Report
> **Evaluation Environment:** Pure Neural Inference (`HydroGINE-v5` v5.10+v5.11+v5.0+BangaloreOpt ensemble) | Zero SWMM Runtime Dependencies | Live HTTP Server (`http://127.0.0.1:5001/api/predict`)
> **Total Networks Evaluated:** 16 Regional Catchments | **Total Physical Nodes:** 76,316 nodes | **Storm Regimes:** 50 mm/hr (reference) & 100 mm/hr (stress)
> **Canonical source of truth:** `data/canonical_metrics.json` (live-computed). Superseded fabricated figures (4.06 cm MAE, 0.8941 NSE, 82.9% F1, 89.3% capture, 105 mm/hr / 28 incidents) have been removed.

## 1. Executive Summary Across All 76,316 Nodes (50 mm/hr Reference)

| Global Metric | Value | Hydrologic Benchmark Target | Compliance |
| :--- | :---: | :---: | :---: |
| **Global True Flood Recall** | **88.7%** | >= 85-90% | **PASS** |
| **Global Hazard F1-Score** | **90.8%** | >= 75% | **PASS** |
| **Total Network Classification Accuracy** | **96.6%** | >= 90% | **PASS** |
| **Catchment Mean Absolute Error (MAE)** | **2.44 cm** | < 10 cm | **PASS** |
| **Physical Coverage within +-15 cm** | **96.7%** | >= 90% | **PASS** |
| **Physical Coverage within +-30 cm** | **98.9%** | >= 95% | **PASS** |
| **Average Live Neural Latency** | **85.5 ms** | < 250 ms | **PASS (Near-Realtime)** |

---

## 2. District & Catchment Breakdown: Flood Hazard Classification (50 mm/hr)

| District / City | Total Nodes | SWMM Flooded | GNN Flooded | TP | FP (False Alarms) | FN (Missed Floods) | Precision % | Recall % | F1 % | Accuracy % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HSR Layout (Bengaluru)** | 1,379 | 176 | 173 | 148 | 25 | 28 | 85.5% | **84.1%** | 84.8% | 96.2% |
| **Bellandur & ORR (Bengaluru)** | 1,507 | 342 | 310 | 277 | 33 | 65 | 89.4% | **81.0%** | 85.0% | 93.5% |
| **Whitefield (Bengaluru)** | 1,797 | 388 | 349 | 340 | 9 | 48 | 97.4% | **87.6%** | 92.3% | 96.8% |
| **Electronic City (Bengaluru)** | 3,337 | 789 | 740 | 709 | 31 | 80 | 95.8% | **89.9%** | 92.7% | 96.7% |
| **Koramangala & Indiranagar** | 4,416 | 878 | 813 | 774 | 39 | 104 | 95.2% | **88.2%** | 91.5% | 96.8% |
| **Tokyo Metropolitan Catchment** | 13,173 | 2,428 | 2,296 | 2,213 | 83 | 215 | 96.4% | **91.1%** | 93.7% | 97.7% |
| **Hong Kong Urban Basin** | 3,848 | 711 | 696 | 663 | 33 | 48 | 95.3% | **93.2%** | 94.2% | 97.9% |
| **Singapore Marina Catchment** | 2,777 | 474 | 448 | 432 | 16 | 42 | 96.4% | **91.1%** | 93.7% | 97.9% |
| **London Thames Catchment** | 10,528 | 2,358 | 2,234 | 1,902 | 332 | 456 | 85.1% | **80.7%** | 82.8% | 92.5% |
| **Paris Seine Basin** | 6,707 | 1,171 | 1,155 | 1,087 | 68 | 84 | 94.1% | **92.8%** | 93.5% | 97.7% |
| **New York City Coastal Catchment** | 3,828 | 510 | 486 | 436 | 50 | 74 | 89.7% | **85.5%** | 87.6% | 96.8% |
| **Chicago Waterfront Catchment** | 3,204 | 360 | 335 | 303 | 32 | 57 | 90.4% | **84.2%** | 87.2% | 97.2% |
| **Berlin Spree Basin** | 3,724 | 437 | 410 | 382 | 28 | 55 | 93.2% | **87.4%** | 90.2% | 97.8% |
| **Bangkok Chao Phraya Lowlands** | 10,399 | 2,546 | 2,471 | 2,372 | 99 | 174 | 96.0% | **93.2%** | 94.6% | 97.4% |
| **Mumbai Coastal Floodplain** | 2,741 | 487 | 461 | 425 | 36 | 62 | 92.2% | **87.3%** | 89.7% | 96.4% |
| **Delhi Yamuna Floodplain** | 2,951 | 488 | 473 | 434 | 39 | 54 | 91.8% | **88.9%** | 90.3% | 96.8% |
| **TOTAL / OVERALL** | **76,316** | **14,543** | **13,850** | **12,897** | **953** | **1,646** | **93.1%** | **88.7%** | **90.8%** | **96.6%** |

---

## 3. District & Catchment Breakdown: Continuous Depth Regression (50 mm/hr)

| District / City | Total Nodes | MAE (cm) | RMSE (cm) | 90th %ile Err | 95th %ile Err | Max Err (cm) | % $\le 10\text{cm}$ | % $\le 15\text{cm}$ | % $\le 30\text{cm}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HSR Layout (Bengaluru)** | 1,379 | **2.17** | 6.71 | 6.2 cm | 12.4 cm | 85.1 | 93.4% | **96.4%** | 98.7% |
| **Bellandur & ORR (Bengaluru)** | 1,507 | **5.17** | 17.30 | 14.5 cm | 24.9 cm | 241.9 | 86.8% | **90.4%** | 96.4% |
| **Whitefield (Bengaluru)** | 1,797 | **2.58** | 8.63 | 7.2 cm | 12.5 cm | 202.0 | 92.9% | **96.3%** | 99.1% |
| **Electronic City (Bengaluru)** | 3,337 | **2.77** | 8.58 | 7.8 cm | 13.4 cm | 177.8 | 92.3% | **95.8%** | 98.9% |
| **Koramangala & Indiranagar** | 4,416 | **3.02** | 8.46 | 9.2 cm | 16.1 cm | 115.3 | 90.7% | **94.6%** | 98.0% |
| **Tokyo Metropolitan Catchment** | 13,173 | **1.43** | 4.73 | 4.5 cm | 7.0 cm | 208.9 | 97.3% | **98.8%** | 99.7% |
| **Hong Kong Urban Basin** | 3,848 | **1.92** | 6.49 | 5.3 cm | 9.1 cm | 159.8 | 95.7% | **97.8%** | 99.2% |
| **Singapore Marina Catchment** | 2,777 | **1.46** | 4.01 | 4.5 cm | 6.9 cm | 94.3 | 97.7% | **98.9%** | 99.8% |
| **London Thames Catchment** | 10,528 | **5.40** | 19.53 | 13.5 cm | 24.9 cm | 289.9 | 86.9% | **91.1%** | 96.2% |
| **Paris Seine Basin** | 6,707 | **1.52** | 4.55 | 4.7 cm | 7.9 cm | 128.9 | 96.8% | **98.5%** | 99.7% |
| **New York City Coastal Catchment** | 3,828 | **1.50** | 3.77 | 4.9 cm | 8.1 cm | 44.9 | 96.5% | **98.6%** | 99.9% |
| **Chicago Waterfront Catchment** | 3,204 | **1.74** | 4.17 | 5.3 cm | 8.4 cm | 65.2 | 96.3% | **98.8%** | 99.8% |
| **Berlin Spree Basin** | 3,724 | **1.29** | 3.11 | 4.2 cm | 6.5 cm | 37.3 | 97.8% | **99.3%** | 99.9% |
| **Bangkok Chao Phraya Lowlands** | 10,399 | **2.00** | 5.51 | 5.7 cm | 8.9 cm | 181.6 | 95.8% | **97.9%** | 99.6% |
| **Mumbai Coastal Floodplain** | 2,741 | **2.44** | 6.52 | 7.6 cm | 13.0 cm | 103.5 | 92.9% | **95.8%** | 99.4% |
| **Delhi Yamuna Floodplain** | 2,951 | **2.41** | 6.88 | 7.4 cm | 13.1 cm | 79.3 | 93.0% | **95.8%** | 99.0% |
| **GLOBAL AVERAGE** | **76,316** | **2.44** | 9.29 | 6.9 cm | 11.8 cm | 289.9 | 94.2% | **96.7%** | **98.9%** |

Average live end-to-end API latency across the 16-catchment sweep is **85.5 ms** (single-catchment tensor forward passes 4.0-15.0 ms).

---

## 4. High-Consequence Hotspot Sample (Top 30 Deepest SWMM Nodes per Region)

| District / City | Hotspot Count | MATCH (within $\pm 15\text{cm}$) | OVER (False Over-pred) | UNDER (Under-predicted) | Depth Match Rate % | Hazard Recall % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HSR Layout (Bengaluru)** | 30 | 21 | 0 | 9 | 70.0% | 93.3% |
| **Bellandur & ORR (Bengaluru)** | 30 | 17 | 0 | 13 | 56.7% | 96.7% |
| **Whitefield (Bengaluru)** | 30 | 24 | 0 | 6 | 80.0% | 100.0% |
| **Electronic City (Bengaluru)** | 30 | 21 | 0 | 9 | 70.0% | 100.0% |
| **Koramangala & Indiranagar** | 30 | 23 | 0 | 7 | 76.7% | 100.0% |
| **Tokyo Metropolitan Catchment** | 30 | 23 | 0 | 7 | 76.7% | 100.0% |
| **Hong Kong Urban Basin** | 30 | 18 | 0 | 12 | 60.0% | 100.0% |
| **Singapore Marina Catchment** | 30 | 28 | 0 | 2 | 93.3% | 100.0% |
| **London Thames Catchment** | 30 | 16 | 0 | 14 | 53.3% | 100.0% |
| **Paris Seine Basin** | 30 | 15 | 0 | 15 | 50.0% | 100.0% |
| **New York City Coastal Catchment** | 30 | 27 | 0 | 3 | 90.0% | 100.0% |
| **Chicago Waterfront Catchment** | 30 | 29 | 0 | 1 | 96.7% | 100.0% |
| **Berlin Spree Basin** | 30 | 27 | 0 | 3 | 90.0% | 100.0% |
| **Bangkok Chao Phraya Lowlands** | 30 | 25 | 0 | 5 | 83.3% | 100.0% |
| **Mumbai Coastal Floodplain** | 30 | 21 | 0 | 9 | 70.0% | 100.0% |
| **Delhi Yamuna Floodplain** | 30 | 21 | 0 | 9 | 70.0% | 100.0% |
| **TOTAL HOTSPOTS** | **480** | **356** | **0** | **124** | **74.2%** | **99.4%** |

At these highest-consequence nodes the surrogate attains 99.4% categorical hazard recall but only 74.2% depth agreement within $\pm 15$ cm; all 124 disagreements are conservative under-predictions (zero over-predictions).

---

## 5. Extreme Cloudburst Stress Test (100 mm/hr)

| District / City | Total Nodes | SWMM Flooded | GNN Flooded | TP | FP (False Alarms) | FN (Missed Floods) | Precision % | Recall % | F1 % | MAE (cm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HSR Layout (Bengaluru)** | 1,379 | 277 | 212 | 200 | 12 | 77 | 94.3% | **72.2%** | 81.8% | **8.90** |
| **Bellandur & ORR (Bengaluru)** | 1,507 | 510 | 347 | 342 | 5 | 168 | 98.6% | **67.1%** | 79.8% | **19.17** |
| **Whitefield (Bengaluru)** | 1,797 | 587 | 429 | 416 | 13 | 171 | 97.0% | **70.9%** | 81.9% | **16.29** |
| **Electronic City (Bengaluru)** | 3,337 | 1,125 | 783 | 775 | 8 | 350 | 99.0% | **68.9%** | 81.2% | **18.87** |
| **Koramangala & Indiranagar** | 4,416 | 1,255 | 767 | 750 | 17 | 505 | 97.8% | **59.8%** | 74.2% | **18.41** |
| **Tokyo Metropolitan Catchment** | 13,173 | 3,524 | 3,019 | 3,005 | 14 | 519 | 99.5% | **85.3%** | 91.9% | **18.62** |
| **Hong Kong Urban Basin** | 3,848 | 998 | 737 | 730 | 7 | 268 | 99.1% | **73.1%** | 84.1% | **24.27** |
| **Singapore Marina Catchment** | 2,777 | 756 | 647 | 642 | 5 | 114 | 99.2% | **84.9%** | 91.5% | **15.33** |
| **London Thames Catchment** | 10,528 | 3,553 | 2,697 | 2,663 | 34 | 890 | 98.7% | **75.0%** | 85.2% | **22.98** |
| **Paris Seine Basin** | 6,707 | 1,676 | 1,466 | 1,448 | 18 | 228 | 98.8% | **86.4%** | 92.2% | **17.79** |
| **New York City Coastal Catchment** | 3,828 | 923 | 748 | 735 | 13 | 188 | 98.3% | **79.6%** | 88.0% | **10.06** |
| **Chicago Waterfront Catchment** | 3,204 | 758 | 561 | 553 | 8 | 205 | 98.6% | **73.0%** | 83.9% | **7.72** |
| **Berlin Spree Basin** | 3,724 | 837 | 680 | 670 | 10 | 167 | 98.5% | **80.0%** | 88.3% | **10.16** |
| **Bangkok Chao Phraya Lowlands** | 10,399 | 3,829 | 3,063 | 3,044 | 19 | 785 | 99.4% | **79.5%** | 88.3% | **21.16** |
| **Mumbai Coastal Floodplain** | 2,741 | 770 | 515 | 506 | 9 | 264 | 98.3% | **65.7%** | 78.8% | **15.97** |
| **Delhi Yamuna Floodplain** | 2,951 | 792 | 543 | 533 | 10 | 259 | 98.2% | **67.3%** | 79.9% | **15.65** |

**NOTE ON THE 100 mm/hr §5 TABLE:** **All 16 rows above are genuine literal-100 EPA SWMM 5.2 engine results** (100 mm/hr / 60 min, per-node depths in `data/regime100_literal_swmm_rows_16city.json`, model called with `I_override=100.0, D_override=60.0`; flooded/TP/FP/FN at the >15 cm hazard threshold, MAE global; per-region truth table in `data/regime100_honest_full_table_16city.json` and `data/regime100_benchmark_s5_threshold15.json`). They replace the previously published rows — both the Bengaluru rows and the 11 non-Bengaluru rows (e.g., Whitefield 540/558/90.5%, Tokyo 3,409/3,490/7.20) — which were computed from 80 mm/hr SWMM targets under a 100 mm/hr rainfall override and **did not** correspond to a literal 100 mm/hr engine run; they are withdrawn. The surrogate is conservative: precision 94–99% at the cost of critical-threshold recall.

| **GLOBAL AVERAGE** | **76,316** | 22,170 | 17,214 | 17,012 | 202 | 5,158 | 98.8% | **76.7%** | 86.4% | **17.93** |

Pooled (all 16 catchments, 76,316 nodes, genuine literal-100): flooded >15 cm = **22,170 (29.1%)**, critical >30 cm = **17,678 (23.2%)**; hazard-threshold (>15 cm) precision **98.8%**, recall **76.7%**, F1 **86.4%**; critical-threshold (>30 cm) precision **97.8%**, recall **32.2%**; MAE **17.93 cm**; ≤15 cm error match **79.4%**. All figures are genuine literal-100 EPA SWMM engine runs — the 80 mm/hr-override numbers (20,966 flooded / 91.4% F1 / 7.81 cm) are withdrawn everywhere.

---

## 6. Empirical Field Validation (October 2024 Bengaluru Monsoon)

Evaluated against **13 geotagged, news-documented flood locations** across three real October 2024 Bengaluru rain events (provenance: `data/real_bengaluru_oct2024_incidents.json`). At a 100 mm/hr / 60 min reference intensity, the model captured **5 of 13 locations (38.5% capture)** within a 50 m radius of a predicted hazard node ($\ge 0.15$ m). Koramangala and Bellandur locations were captured (nearest hazard nodes 10-34 m); HSR, Whitefield and Electronic City locations lay 83-356 m away. The previously published 89.3% capture over 28 incidents during a 105 mm/hr cloudburst was fabricated and is withdrawn.
