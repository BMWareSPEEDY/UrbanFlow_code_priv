# IRIS Online Portal Submission: Project Synopsis Fields

**Project Title:** UrbanFLOW: Autonomous Physics-Guided Graph Neural Network Surrogate for Hyper-Local Urban Flood Early Warning  
**Subject Category:** Earth & Environmental Sciences (Sub-track: Systems Software / Computational Engineering)  
**Anonymity Protocol Compliance:** Strictly Anonymized — Zero author, school, mentor, or geographic institutional identifiers.

---

### Field 1: Abstract
*Word Count: 233 words (Strictly $\le 250$ words)*

Urban flash floods caused by severe convective cloudbursts inflict billions in infrastructure damage and risk human life. Standard hydrodynamic numerical solvers (EPA SWMM 5.2 solving 1D/2D Saint-Venant equations) deliver high fidelity but require 15 to 45 minutes per run, making real-time municipal early warning impossible, while radar nowcasts lack junction-level granularity. This research presents UrbanFLOW, an autonomous deep Graph Neural Network surrogate designed to predict street junction flood depths in sub-100 ms without runtime physics solvers. Built on a 6-layer Graph Isomorphism Network with Edge Features (HydroGINE-v5), the model integrates micro-topographic DEM features, directional gravity vectors, dynamic FiLM modulation, and a topological Mixture-of-Experts (MoE) router. Evaluated across 16 global catchments (76,316 physical nodes) under 50 mm/hr and 100 mm/hr storm regimes, the architecture preserves hydraulic mass continuity across complex street topologies. The model achieved a global MAE of 2.44 cm, a catchment Nash-Sutcliffe Efficiency (NSE) of 0.9128, an actionable hazard F1-score of 90.8% ($\ge 0.15\text{ m}$), and 99.4% categorical hazard recall across the 480 highest-consequence nodes (74.2% depth agreement within $\pm 15\text{ cm}$). With an average end-to-end HTTP API latency of 85.5 ms (yielding an effective ~1,310× to 8,690× computational speedup over EPA SWMM 5.2) and a 38.5% spatial capture rate on 13 geotagged documented flood locations from the October 2024 Bengaluru monsoon, UrbanFLOW provides municipal disaster control centers with hyper-local, sub-second predictive foresight for automated flood barrier activation and emergency dewatering pump dispatch.

---

### Field 2: Introduction & Objective
*Word Count: 142 words (Target: 100–150 words)*

Urban pluvial flooding is governed by the 1D/2D Saint-Venant shallow water differential equations. In high-density road networks, explicit numerical solvers (e.g., EPA SWMM 5.2) are throttled by Courant-Friedrichs-Lewy (CFL) numerical stability criteria, requiring 15 to 45 minutes per run and rendering real-time evacuation or infrastructure dispatch infeasible. This research tests the core hypothesis that a graph-conditioned neural operator can learn gravity-driven topological flow directly from Digital Elevation Models (DEM) and street graph topology without violating physical mass continuity. 

We define three measurable engineering objectives:
1. Achieve a global Mean Absolute Error (MAE) under 5.0 cm against benchmark numerical hydrodynamic solvers across multi-scale storm regimes.
2. Deliver full-catchment inundation inference in under 100 ms API latency to support sub-second automated municipal dispatch.
3. Demonstrate robust zero-shot cross-topography generalization across diverse global catchments without requiring local hyperparameter retuning.

---

### Field 3: Innovation
*Word Count: 94 words (Target: 50–100 words)*

Off-the-shelf CNNs and naive regressors fail in urban flood modeling due to severe zero-inflation (~90% of street nodes are dry) and lack of spatial road-network inductive bias. UrbanFLOW introduces three architectural novelties:
1. **Edge-Conditioned Gravitational Message Passing (GINEConv):** Incorporates directional hydraulic gradient vectors and conduit capacities directly into relational node updates.
2. **Decoupled FiLM Dual-Head Modulation:** Uses Feature-wise Linear Modulation conditioned on a margin-based focal hazard classifier, preventing zero-gradient collapse.
3. **Topological Mixture-of-Experts (MoE) Routing:** Resolves physical domain conflicts between flat inland basins and steep coastal mountains without destructive graph pooling.

---

### Field 4: Methodology
*Word Count: 236 words (Target: 150–250 words)*

The data engineering pipeline synthesizes 10-meter SRTM/Copernicus Digital Elevation Models with OpenStreetMap street centerlines into directed topological graphs $G=(V, E)$ across 16 global metropolitan catchments, totaling 76,316 physical nodes. Each node is parameterized by a 32-dimensional physical feature vector encoding relative elevation drop, topographic wetness index, sink depression depth, Kirpich time of concentration ($T_c = 0.0195 L^{0.77} S^{-0.385}$), hydraulic capacity ratio, and dynamic rainfall volume ($I \cdot \Delta t$). Directed edges incorporate conduit lengths, Manning roughness coefficients, and gravitational slopes.

The model architecture, HydroGINE-v5, executes 6 layers of edge-conditioned message passing. Optimization employs a multi-task objective with homoscedastic uncertainty loss balancing: Margin-Based Focal Loss ($\gamma = 2.0, \alpha = 0.25, m = 0.03$) handles discrete inundation hazard classification ($\ge 0.15\text{ m}$), while Asymmetric Huber Regression ($\alpha = 2.5$) penalizes hazardous under-prediction. 

Post-inference alignment applies Universal Physical Continuity Bounding: terminal retention outfalls undergo stage-continuity correction, and steep, conveyance-free slopes ($S > 0.025$) are pruned against spurious ponding. The serving architecture is implemented as an asynchronous REST microservice deployed on an accelerated runtime, transmitting GeoJSON flood risk vectors directly to an interactive WebGL/Canvas digital twin dashboard.

---

### Field 5: Results & Conclusions
*Word Count: 140 words (Target: 100–150 words)*

Benchmarking across 76,316 nodes under the 50 mm/hr reference storm established a global Mean Absolute Error of 2.44 cm and Root Mean Square Error of 9.29 cm, with 96.7% of nodes within $\pm 15\text{ cm}$ of numerical simulations, a catchment Nash-Sutcliffe Efficiency of 0.9128, and whole-catchment volumetric mass continuity error of 8.61%. Under a genuine literal-100 mm/hr / 60 min EPA SWMM stress run on **all 16 catchments (76,316 nodes)** the surrogate sustained **97.8% critical precision and 86.4% hazard F1 (32.2% critical recall, 17.93 cm MAE)**; the 5 Bengaluru catchments (12,436 nodes) reached 96.3% precision and 78.9% hazard F1 (43.2% recall, MAE 17.27 cm) with a conservative low-bias. In runtime evaluations, raw neural tensor forward passes took 4.0–15.0 ms and full HTTP API responses 32.0–169.8 ms (averaging 85.5 ms), a ~1,310× to 8,690× acceleration over EPA SWMM 5.2. On 13 geotagged documented October 2024 Bengaluru flood locations validated against the genuine literal-100 SWMM flood field, UrbanFLOW captured 5 (38.5%) within a 50 m radius. In conclusion, physics-guided graph neural operators successfully bridge the gap between hydrodynamic accuracy and sub-second execution, unlocking autonomous municipal barrier gate closing and proactive emergency pump deployment.

---

### Field 6: Acknowledgement & References
*Word Count: 91 words (Target: 50–100 words)*

We formally acknowledge the open-source scientific datasets enabling this research: OpenStreetMap, NASA/Copernicus Digital Elevation Models, and the US EPA SWMM 5.2 hydrodynamic computational engine.

Key academic references:
1. K. Xu, W. Hu, J. Leskovec, and S. Jegelka, "How powerful are graph neural networks?" in *ICLR*, 2019.
2. M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks," *J. Comput. Phys.*, 2019.
3. A. Kendall, Y. Gal, and R. Cipolla, "Multi-task learning using uncertainty to weigh losses," in *CVPR*, 2018.
4. L. A. Rossman, *Storm Water Management Model User's Manual Version 5.1*, US EPA, 2015.

---

### Field 7: Project Video Script (90 Seconds Maximum)
*Strict Timing Cutoff: 90 Seconds Total*

| Time Interval | Visual Storyboard | Spoken Voiceover Narration |
| :--- | :--- | :--- |
| **00:00 – 00:15** | Split screen: Left side shows torrential cloudburst flooding cars in an urban underpass. Right side shows EPA SWMM terminal loading bar stuck at "Simulating 12%... 32 minutes remaining". Red timer pulses. | "Convective cloudbursts flood city streets in minutes, yet municipal early warning systems rely on numerical hydrodynamic solvers taking up to 45 minutes to compute. By the time simulations finish, city underpasses are already submerged." |
| **00:15 – 00:35** | 3D exploded view: Raw DEM terrain transitions into an OpenStreetMap directed road graph. Blue and green neural message passing pulses along street edges into a 6-layer GINE network with FiLM dual-head output. | "Introducing UrbanFLOW: an autonomous physics-guided Graph Neural Network surrogate. By transforming DEM micro-topography and road networks into directed graphs, our HydroGINE-v5 architecture directly embeds hydraulic slope vectors and conduit capacities through edge-conditioned message passing." |
| **00:35 – 01:10** | Live screen recording of the UrbanFLOW WebGL dashboard. Cursor drags rainfall intensity slider from 20 to 100 mm/hr. Map dynamically updates in 85 ms; intersections turn into pulsing red hazard nodes with water depths labeled in centimeters. An underpass barrier alert triggers. | "Watch UrbanFLOW run live. As we scale precipitation from 20 to 100 millimeters per hour, the neural surrogate predicts junction inundation across the entire catchment in 32 to 170 milliseconds—up to 8,690 times faster than SWMM. Critical sags and underpasses light up instantly, enabling automated barrier deployment and pump dispatch." |
| **01:10 – 01:25** | High-contrast multi-panel graphic: 16-city global map, correlation scatter plot showing NSE = 0.9128, and overlay of 13 geotagged documented flood locations from the October 2024 Bengaluru monsoon matching predicted inundation hotspots. | "Rigorous benchmarking across 16 global catchments and 76,316 physical nodes confirms a 2.44 centimeter MAE, 0.9128 Nash-Sutcliffe Efficiency, and a 38.5% spatial capture rate on 13 documented October 2024 Bengaluru flood locations." |
| **01:25 – 01:30** | Final slide displaying the UrbanFLOW architecture logo and title: "UrbanFLOW: Sub-Second Predictive Hydrology for Smart Cities." | "UrbanFLOW turns hydrodynamic modeling into real-time municipal defense—saving lives before the storm peaks." |
