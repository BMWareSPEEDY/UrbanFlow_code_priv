# UrbanFLOW: System Architecture & Model Specification Sheet

**Project:** UrbanFLOW (Autonomous Physics-Guided Graph Neural Network Surrogate for Hyper-Local Urban Flood Early Warning)  
**System Specification Document Version:** 5.8 (Production Camera-Ready)  
**Anonymity Status:** Strictly Anonymized for Competition Review (IRIS Anonymity Protocol Compliant)

---

## 1. Machine Learning & Graph Neural Operator Specification

### 1.1 Core Computational Framework
* **Deep Learning Framework:** PyTorch 2.x (`torch >= 2.1.0`) with CUDA 12.x acceleration.
* **Graph Neural Network Library:** PyTorch Geometric (`torch_geometric >= 2.4.0`) leveraging native sparse tensor operators (`torch_scatter`, `torch_sparse`).
* **Hardware Acceleration:** Native TensorFloat-32 (TF32) execution; FP16 mixed-precision inference pipeline for sub-10 ms forward passes.

### 1.2 Neural Backbone Architecture (HydroGINE-v5)
* **Message Passing Paradigm:** Edge-conditioned Graph Isomorphism Network (`GINEConv`) extended with gravitational slope gating:
  $$\mathbf{h}_v^{(k)} = \text{MLP}^{(k)}\left( (1 + \epsilon^{(k)})\mathbf{h}_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} \text{ReLU}\left( \mathbf{h}_u^{(k-1)} + \mathbf{\Theta}^{(k)} \mathbf{e}_{uv}^{\text{gated}} \right) \right)$$
* **Layer Depth:** 6 message passing layers ($K = 6$) providing a 6-hop topological receptive field (equivalent to ~350–600 meters of overland pipe conveyance).
* **Hidden Representation Dimension:** 192 channels across all intermediate latent layers.
* **Normalization & Regularization:** Post-convolution Layer Normalization (`nn.LayerNorm(192)`), Dropout ($p = 0.05$ to $0.15$), and multi-scale residual skip connections ($0.3 \cdot \mathbf{h}^{(k-1)}$).
* **Non-Linear Activations:** Exponential Linear Units (ELU) within message passing layers; LeakyReLU ($\alpha = 0.10$) within MLP readouts.

### 1.3 Decoupled FiLM Dual-Head Mechanism
* **Multi-Scale Latent Concatenation:**
  $$\mathbf{h}_{\text{cat}} = \left[ \mathbf{h}_v^{(6)} \,\|\, \mathbf{h}_v^{(3)} \,\|\, \mathbf{x}_v \right] \in \mathbb{R}^{192 + 192 + 32} = \mathbb{R}^{416}$$
* **Head 1 — Discrete Hazard Classifier:**
  $$z_{\text{cls}} = \text{MLP}_{\text{cls}}(\mathbf{h}_{\text{cat}}) \in \mathbb{R}^1, \quad p_{\text{hazard}} = \sigma(z_{\text{cls}})$$
  Predicts whether water depth will exceed the critical municipal hazard threshold ($\ge 0.15\text{ m}$).
* **Conditioning Generator — Feature-wise Linear Modulation (FiLM):**
  $$\left[ \boldsymbol{\gamma}, \boldsymbol{\beta} \right] = \text{MLP}_{\text{film}}(p_{\text{hazard}}) \in \mathbb{R}^{2 \times 416}$$
  Initialized with zero weights and biases to preserve clean identity conditioning at initialization.
* **Head 2 — Continuous Depth Regressor:**
  $$\mathbf{h}_{\text{cond}} = \mathbf{h}_{\text{cat}} \odot (1 + \boldsymbol{\gamma}) + \boldsymbol{\beta}$$
  $$\hat{y}_{\text{raw}} = \text{MLP}_{\text{reg}}(\mathbf{h}_{\text{cond}}) \in \mathbb{R}^1$$

### 1.4 Multi-Task Optimization Objective
* **Homoscedastic Task Uncertainty Balancing:**
  $$\mathcal{L}_{\text{total}} = \frac{1}{2\sigma_1^2} \mathcal{L}_{\text{focal}} + \frac{1}{2\sigma_2^2} \mathcal{L}_{\text{asym}} + \log(\sigma_1 \sigma_2)$$
* **Margin-Based Focal Loss (Classification):**
  $$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t) + \lambda \max(0, m - (z_{\text{cls}} - z_{\text{dry}}))$$
  Parameters: $\gamma = 2.0$, $\alpha = 0.25$, margin $m = 0.03$.
* **Asymmetric Huber Loss (Continuous Depth):**
  $$\mathcal{L}_{\text{asym}}(y, \hat{y}) = \begin{cases} \alpha (y - \hat{y})^2 & \text{if } y > \hat{y} \text{ (penalizing under-prediction)} \\ (y - \hat{y})^2 & \text{if } y \le \hat{y} \text{ (over-prediction)} \end{cases}$$
  Asymmetry weight $\alpha = 2.5$.

---

## 2. Input Physical Feature Dictionary (Zero Coordinate Memorization)

UrbanFLOW processes 32 scale-invariant physical and hydraulic features per node and 6 directional edge features. Crucially, raw spatial coordinates ($x, y$, latitude, longitude) are excluded from node embeddings to ensure strict zero-shot geographic transferability without coordinate overfitting.

### Node Feature Vector ($\mathbf{x} \in \mathbb{R}^{32}$)
1. `rel_drop`: Normalized elevation drop relative to catchment crest: $(z_{\max} - z_i) / \Delta z_{\text{relief}}$.
2. `impervious_ratio`: Urban surface runoff coefficient ($0.0 \le C_{\text{imp}} \le 1.0$).
3. `manning_n`: Overland Manning surface roughness coefficient (0.013 for asphalt to 0.040 for vegetated buffers).
4. `in_degree`: Topological incoming conduit count (flow convergence degree).
5. `out_degree`: Topological outgoing conduit count (drainage conveyance degree).
6. `accum_score`: Log-transformed hydraulic accumulation score: $\ln(1 + 2.0 \cdot \text{deg}_{\text{in}})$.
7. `is_sink`: Morphological depression indicator flag ($d_{\text{dep}} \ge 0.08\text{ m}$).
8. `max_in_grade`: Maximum incoming conduit gravitational slope ($\mathrm{m/m}$).
9. `sag_index`: Local parabolic sag index: $\max(0, S_{\text{in}} - S_{\text{out}}) \cdot \max(1, \text{deg}_{\text{in}})$.
10. `hydraulic_capacity`: Ratio of incoming to outgoing conduits ($\text{deg}_{\text{in}} / \max(1, \text{deg}_{\text{out}})$).
11. `log_area`: Natural logarithm of upstream contributing catchment surface area: $\ln(1 + A_{\text{acc}})$.
12. `log_imp_area`: Natural logarithm of effective impervious drainage area: $\ln(1 + A_{\text{acc}} \cdot C_{\text{imp}})$.
13. `dist_frac`: Normalized relative distance along the drainage network to the major terminal outfall.
14. `rainfall_intensity`: Dynamic event precipitation rate ($I$, in $\text{mm/hr}$).
15. `rainfall_duration`: Dynamic storm event duration ($t$, in $\text{minutes}$).
16. `elev_std`: Local elevation standard deviation within a 150-meter neighborhood window.
17. `depression_depth`: DEM sink depth below the lowest spill crest ($d_{\text{dep}}$ in meters).
18. `surcharge_proxy`: Conduit surcharge pressure head index under rational inflow peak.
19. `path_capacity`: Cumulative downstream conduit bottleneck capacity ($\ln(1 + Q_{\text{cap}})$).
20. `path_hops`: Minimum topological edge hops to the nearest natural drainage sink/outfall.
21. `dist_outlet`: Euclidean distance along flow path to the terminal basin boundary ($m$).
22. `elev_above_outlet`: Vertical relief elevation above the terminal receiving waterbody stage.
23. `slope_outlet_ratio`: Relative gradient ratio between local slope and regional base-level fall.
24. `sink_depth_filtered`: Clamped depression depth ($d_{\text{dep}}$ if $\ge 0.05\text{ m}$, else 0.0).
25. `inlet_capacity`: Manning-consistent road-class inlet capture capacity ($0.01$ to $0.50\text{ m}^3\text{/s}$).
26. `surcharge_ratio`: Dynamic ratio of incoming runoff to conduit capacity: $(A_{\text{acc}} / C_{\text{inlet}}) \cdot I$.
27. `deg_difference`: Convergence imbalance metric ($\text{deg}_{\text{in}} - \text{deg}_{\text{out}}$).
28. `total_rainfall_volume`: Cumulative precipitated volume depth: $I \cdot (t / 60)$ (mm).
29. `dynamic_saturation`: Effective soil and asphalt moisture saturation proxy.
30. `ponding_potential`: Theoretical ponding volume: $\ln(1 + d_{\text{dep}} \cdot I_{\text{tot}} / (\text{deg}_{\text{out}} + 0.3))$.
31. `conveyance_deficit`: Inflow loading versus conduit bottleneck deficit: $\ln(1 + Q_{\text{in}} / Q_{\text{pipe}})$.
32. `depression_escape_ratio`: Depression retention depth divided by exit escape slope: $d_{\text{dep}} / (|S_{\max}| + 0.01)$.

### Edge Feature Vector ($\mathbf{e} \in \mathbb{R}^6$)
1. `length`: Physical centerline conduit length ($L$, in meters).
2. `grade`: Longitudinal gravitational hydraulic slope ($S_0 = (z_u - z_v) / L$).
3. `capacity`: Maximum conduit carrying capacity derived from downstream road hierarchy.
4. `gravity_gate`: Directional gravity vector gating factor: $\sigma(1.0 - 5.0 \cdot \text{ReLU}(S_0))$.
5. `flow_dir_dx`: Unit vector horizontal component along street centerline.
6. `flow_dir_dy`: Unit vector vertical component along street centerline.

---

## 3. Inference Pipeline & Universal Physical Continuity Bounding

The end-to-end inference pipeline executes in four decoupled, sequential stages:

```
[Raw Graph & Storm Input (I, t)]
             │
             ▼
[Stage 1: Dynamic Feature Modulation (1.2 ms)]
  • Scales surcharge ratio & ponding potential by I · t
             │
             ▼
[Stage 2: Topological MoE Neural Operator (4.0–15.0 ms)]
  • Edge-conditioned message passing across 6 GINE layers
  • FiLM dual-head produces p_hazard and raw depth y_lin
             │
             ▼
[Stage 3: Universal Physical Continuity Bounding (2.1 ms)]
  • Confidence gating via sigmoid rail: conf_gate = σ(6 · (p_hazard - τ))
  • Hydraulic mass upper-bounding: min(y_lin · conf_gate, mass_bound)
  • High-slope dry conveyance pruning: zeroing dry nodes with S > 0.025
  • Inter-node Water Surface Elevation (WSE) backwater relaxation
             │
             ▼
[Stage 4: JSON / GeoJSON Serialization & API Delivery (15–45 ms)]
```

### 3.1 Confidence Gating
A dynamic activation rail scales raw neural predictions according to morphological regime thresholds ($\tau$):
$$\tau = \begin{cases} 0.15 & \text{if deep sink or valley depression} \\ 0.25 & \text{if choked conduit surcharge or convergent sag} \\ 0.75 & \text{if ridge crest, steep slope, or free-draining conveyance} \\ 0.35 & \text{default urban street junction} \end{cases}$$
$$g_{\text{conf}} = \frac{1}{1 + \exp\left(-6.0 \cdot (p_{\text{hazard}} - \tau)\right)}$$

### 3.2 Water Surface Elevation (WSE) Backwater Rail
To guarantee that backwater ponding obeys gravitational potential and cannot climb uphill across dry edges:
$$\text{WSE}_u = z_u + \hat{y}_u$$
$$\text{Backwater}_{v \leftarrow u} = \max(0, \text{WSE}_u - z_v)$$
$$\hat{y}_v^{\text{bounded}} = \min\left(\hat{y}_v^{\text{raw}} \cdot g_{\text{conf}}, \, \max(\text{Backwater}_{v}, \, d_{\text{dep}, v})\right)$$

---

## 4. Serving Infrastructure & REST API Specification

### 4.1 Microservice Specification
* **Server Framework:** Asynchronous Python WSGI/ASGI service using Flask and Gunicorn with gevent greenlets (or FastAPI/Uvicorn).
* **Process Model:** Multi-worker architecture with in-memory PyG graph caching (`REGION_CACHE`).
* **Graph Cache:** Pre-loaded PyG `Data` objects with pre-computed static feature matrices to eliminate runtime disk I/O.

### 4.2 Endpoint: Inundation Prediction (`/api/predict`)
* **HTTP Method:** `POST`
* **Content-Type:** `application/json`
* **Request Payload Schema:**
  ```json
  {
    "region": "hsr",
    "rainfall_mmhr": 75.0,
    "duration_min": 60.0
  }
  ```
* **Response Payload Schema:**
  ```json
  {
    "status": "success",
    "region": "hsr",
    "execution_time_ms": 34.2,
    "summary": {
      "total_nodes": 1379,
      "flooded_nodes": 184,
      "max_depth_m": 1.24,
      "mean_depth_m": 0.048,
      "critical_junctions": 28
    },
    "nodes": [
      {
        "id": "123456789",
        "lat": 12.9118,
        "lng": 77.6385,
        "gnn_depth": 0.482,
        "swmm_depth": 0.495,
        "hazard_level": "CRITICAL",
        "is_underpass": false,
        "confidence": 0.942
      }
    ]
  }
  ```

---

## 5. Summary Performance Benchmarks

| Parameter | Numerical Hydrodynamic Solver (EPA SWMM 5.2) | UrbanFLOW HydroGINE-v5 Surrogate |
| :--- | :---: | :---: |
| **Execution Paradigm** | 1D/2D St. Venant Differential Numerical Integration | Autonomous Graph Neural Operator Forward Pass |
| **Time-Step Stability** | CFL Condition Restricted ($\Delta t \le 0.5\text{ s}$) | Single-Shot Forward Pass ($\Delta t = \text{Event}$) |
| **Runtime (Small Catchment, ~1,400 nodes)** | 42.1 seconds | **32.0 milliseconds** (~1,310× faster) |
| **Runtime (Medium Catchment, ~3,300 nodes)** | 228.0 seconds (3.8 min) | **54.6 milliseconds** (~4,180× faster) |
| **Runtime (Large Catchment, ~13,200 nodes)**| 1,476.0 seconds (24.6 min) | **169.8 milliseconds** (~8,690× faster) |
| **Global Prediction Error (MAE, 50 mm/hr reference)** | Baseline Reference (0.0 cm) | **2.44 cm** across 76,316 nodes |
| **Global Hazard F1-Score ($\ge 0.15\text{ m}$)** | Baseline Reference (1.000) | **90.8%** (precision 93.1%, recall 88.7%) |
| **Catchment Nash-Sutcliffe Efficiency (NSE)**| Baseline Reference (1.000) | **0.9128** |
| **High-Consequence Hotspot Depth Agreement ($\pm 15\text{ cm}$)**| Baseline Reference (100%) | **74.2%** (356 / 480 deepest nodes; 99.4% categorical hazard recall) |
| **Real Cloudburst Emergency Capture Rate** | N/A (Cannot run in real time) | **38.5%** (5 / 13 geotagged documented October 2024 Bengaluru locations, $\le 50\text{ m}$ proximity) |
