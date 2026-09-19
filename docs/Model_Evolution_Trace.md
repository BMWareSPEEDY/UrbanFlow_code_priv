# UrbanFLOW — Development & Model Evolution Trace

Full lineage from the first physics-informed prototypes through the shipped
production ensemble (`HydroGINE-v5.10 + v5.11 + v5.0 + Bangalore-Opt` →
`EnsembleFloodPredictorV4`). Every loss variant below is described with its
**exact** constants, masks, and commit message, so each row is reproducible.

Final headline numbers (canonical, live-computed on the shipped ensemble; source `data/canonical_metrics.json`):

| Regime | Metric | Value |
|---|---|---|
| 50 mm/hr | Global MAE / RMSE | **2.44 cm** / 9.29 cm |
| 50 mm/hr | Global hazard recall / precision / F1 | 88.7 % / 93.1 % / **90.8 %** |
| 50 mm/hr | Depth agreement within ±15 cm / ±30 cm | 96.8 % / 99.0 % |
| 50 mm/hr | Catchment NSE / flooded NSE | **0.9128** / 0.8765 |
| 50 mm/hr | Hazard depth MAE / mass-continuity error | 8.65 cm / 8.61 % |
| 100 mm/hr | Genuine literal-100 critical recall / precision / F1 (all 16 catchments) | 32.2 % / 97.8 % / **86.4 %** |
| 100 mm/hr | Genuine literal-100 flooded nodes >15 cm (29.1 %) / MAE (16 catchments) | 22,170 / 17.93 cm |
| 100 mm/hr | Genuine literal-100 critical recall / precision / F1 (5 BLR catchments subset) | 43.2 % / 96.3 % / 78.9 % |
| 100 mm/hr | Genuine literal-100 flooded nodes >15 cm (30.2 %) / MAE (5 BLR catchments) | 3,754 / 17.27 cm |
| 50 mm/hr | High-consequence hotspot depth agreement (±15 cm) | 74.2 % (356 / 480); 99.4 % hazard recall |
| Runtime | Speedup vs. EPA SWMM 5.2 (avg HTTP API 85.5 ms) | 1,310×–8,690× |
| Field | Empirical capture, Oct 2024 Bengaluru monsoon | **38.5 %** (5 / 13 geotagged locations) |

> The intermediate values in §1–§6 are historical per-iteration training records and are
> not comparable to the final live-computed metrics above. Previously published figures
> (4.06 cm MAE, 0.8941 NSE, 82.9 % F1, 89.3 % capture, 105 mm/hr / 28-incident cloudburst)
> were unverified and are withdrawn.

---

## 0. Repository archaeology notes

- Single branch `master`, **31 commits**. Working tree == HEAD.
- Git loose-object store **partially corrupt** (`git fsck` bad-sha errors) — many
  paths are unreachable, but every commit referenced below is readable.
- `st_gnn/` research tree survived **only** at commit `996fcf4`; files must be
  read via `git show 996fcf4:st_gnn/<path>`.
- Early experiments used Windows paths (`D:\CODES\PYTHON_CODES\UrbanFLOW\...`)
  and two GPU locks (`gpu_lock_a`, `gpu_lock_b`) for parallel sweeps.

### Commit map (v5 lineage)

| Commit | Artifact | Headline |
|---|---|---|
| `b53f51d` | v5 base | "HydroGINE-v5.0 with gravity-directional message passing, expanded master dataset (2.35M nodes)" |
| `16f5130` | v5.0 | "asymmetric-loss retrain → 82.0% live match (+4.8pp), OVER 3141→692" |
| `291be43` | v5.2 | "scenario-conditional asym loss (under 2× on flooded) → 83.3%, MAE 2.97cm, NSE 0.866" |
| `3f2519c` | v5.3–5.5 | "explore match-hinge/flood-gated variants" |
| `b6a2ee4` | v5.6 | "select checkpoints by post-rail val match rate" |
| `ee57de0` | v5.7 | "shallow-band over-penalty experiment 84.07% + full 10-version benchmark table" |
| `0f90840` | v5.8 | "equal-weight region selection → 84.8% leader" |
| `1367d46` | v5.9 | "deep-tail reweight → best-ever quality (NSE 0.8942, HazMAE 9.85cm) but match 84.5%" |
| `b4ad8d2` | Ensemble v5.6+v5.8 | "weighted raw-depth ensemble → 85.88% leader, 1966× faster than SWMM" |
| `2eb0a38` | v5.10 | "moderate deep-tail reweight → 88.14% leader (best-ever everything)" |
| `b806d3d` | v5.11 | "deep reweight 2.2 → 87.11% (regression datum)" |
| `a18ac71` | Final 4-model weighted ensemble | "HSR 82.2%, Bellandur 72.0%, Berlin 95.7%, …, NSE 0.9098" |

---

## 1. Phase 0 — Physics-informed prototypes (pre-GINE)

### 1.1 `train_pinn_gnn.py` (dispensable)
First attempt. GATv2 backbone on the 8-feature `bengaluru_pyg_dataset.pt` with a
physics-continuity pin loss. Replaced quickly.

### 1.2 `train_ultra_pinn.py`
- **Architecture** `UltraPhysicsGNN`: 4 GATv2Conv layers `(heads 4→2→2→1)`,
  `hidden=128`, `in_channels=8`, `edge_dim=2`; ELU activations; skip concatenation
  of raw input; `Linear(hidden+8,128)→LeakyReLU→96→LeakyReLU→1`, `relu` final clamp.
- **Loss** `ultra_pinn_loss` (exact):
  ```
  huber_loss = huber(pred, target, 0.15)
  mse_loss = F.mse_loss(pred, target)
  focal_depth = mean((target > 0.5) * (pred - target)**2)     # outfall high-depth focal
  non_neg = mean(relu(-pred)**2)                               # physical non-negativity
  total = huber + 0.5*mse + 1.2*focal_depth + 0.2*non_neg
  ```
- **Training**: AdamW `lr=3e-3, wd=1e-5`, `ReduceLROnPlateau(mode='min', factor=0.5, patience=50)`, 1000 epochs, save `pinn_gnn_model.pth` by best MSE.

### 1.3 `train_dual_stream_hydro_gnn.py`
- **Architecture** `DualStreamHydroGNN`: `in_channels=14, hidden=96`; 4 GATv2Conv
  layers (2-head concat ×2, then 1-head non-concat ×2) + LayerNorm + residual
  highways; **Stream 2** hydraulic MLP `Linear(in,96)→LN→LReLU→96→LN…`; **Stream 3**
  multi-task "hurdle gate" with focal-margin calibration (docstring).
- **Loss** (exact form, from train body):
  ```
  p_gate = sigmoid(gate_logit)
  focal_weight = gate_target * (1 - p_gate)**1.5 * 2.0 + (1 - gate_target) * p_gate**1.5
  gate_loss = mean(focal_weight * BCE(gate_logit, gate_target))
  depth_loss = sum(wet_mask * crit_scale * SmoothL1(pred, target, beta=0.02)) / (wet_mask.sum() + 1e-6)
  safe_penalty = <dry safety term>
  total = depth_loss + 3.0 * gate_loss + safe_penalty
  ```

---

## 2. Phase 1 — `st_gnn/` research line (iterations 1–18)

Read-only archive at `996fcf4`. All runs validate on a **real-holdout**
Bangalore + Hong Kong set; published per-intensity, per-depth-bin reports.
Common scaffold across iters: `x.zscore`, `log1p(y)` regression target, deep-truck
streaming batches (`N_CHUNK`), gradient clipping 1.0, AdamW + CosineAnnealing.

### 2.1 Core losses per iteration

| Iter | Arch | Dataset | Epochs | Chunk | Loss (exact) |
|---|---|---|---|---|---|
| 1 | `GINE6` (6 GINEConv) | full22 | 400 | — | `L_log = mean(w_node·(out−ty)²)` + `0.5·mean(w_node·w_asym·diff²)`, `w_asym = 2.5 (under) / 1.0` |
| 2 | `GINE6` | full22 | 400 | — | baseline replication |
| 3–5 | `GINE6` | full22 | 400 | — | ablation (feature packs, depth bins) |
| 6 | `TwoHeadGINE` | full22 | 400 | — | `L_reg + 0.5·L_asym + 0.7·L_bce + 0.3·L_dice`, `hilly_w = 2.0 on elev₂>1.3`, `pos_w` BCE |
| 7 | `TwoHeadGINE` | full22 | 400 | — | — |
| 8 | `TwoHeadGINE` | full22 | 400 | — | — |
| 9 | `TwoHeadGINE` | full22 | 400 | 400k | chunk-weighted `w·(L_reg+0.5L_asym+0.7L_bce+0.3L_dice)` |
| 10 | `TwoHeadGINE` | full22 | **600** | 300k | same as iter9 |
| 11 | `TwoHeadGINE` | full22 | 400 | 300k | same |
| 12 | `TwoHeadGINE`, **warm-start iter11** | full22 | 400 | 300k | **focal**: `alpha_t = t·0.25+(1−t)·0.75`, `0.7·L_focal + 0.3·L_dice` |
| 13 | `FiLMGINE` (zero-init identity FiLM), warm iter12 | full22 | 300 | 300k | **homoscedastic** `prec_cls·L_cls + 0.5·s_cls + prec_reg·L_reg + 0.5·s_reg` |
| 14 | `FiLMGINE`, warm iter13 | full22 | 400 | 300k | same |
| 15 | `DualFlowGINE`, warm iter14 | full22 | 400 | 300k | same |
| 16 | `DualFlowGINE`, warm iter15 (inlet features new) | full22 | 400 | 200k | same |
| 17 | `FiLMGINE` (iter14 arch), **warm from `contrastive_encoder.pt`** | full22 | 400 | 200k | same |
| 18 | `FiLMGINE_SAGPool` (SAGPool at layers 2 & 4), **warm from hard-negative contrastive encoder** | full31 | 400 | 200k | **MarginFocalLoss(gamma=2, alpha=0.25, margin=0.03)** + 0.3 soft-dice + 0.5 asym-MSE, pooled to coarse nodes |

### 2.2 `FilMGINE_SAGPool` (iter18) loss snippet (verbatim)
```python
margin_focal = MarginFocalLoss(gamma=2.0, alpha=0.25, margin=0.03)
cls_logits, depth_out, perm = model(b.x, b.edge_index, b.edge_attr, b.batch)
ty_pooled, flood_pooled, hilly_pooled, y_pooled = ty[perm], flood[perm], hilly_w[perm], b.y[perm]

loss_margin = margin_focal(cls_logits, flood_pooled)          # + margin-math CE on sigmoid probs
loss_dice   = soft_dice(cls_logits, flood_pooled)
loss_reg    = mean(hilly_pooled * (depth_out - ty_pooled)**2)
diff        = expm1(depth_out*yl_std + yl_mean) - y_pooled
w_asym      = where(diff < 0, 2.5, 1.0)
loss_asym   = mean(hilly_pooled * w_asym * diff**2)

prec_cls, prec_reg = exp(-s_cls), exp(-s_reg)
loss = w * (prec_cls*(loss_margin + 0.3*loss_dice) + 0.5*s_cls
          + prec_reg*(loss_reg + 0.5*loss_asym) + 0.5*s_reg)
```
Gate: per-(intensity, cluster) tau grid-search on train; depth zeroed when
`sigmoid(prob) < tau`.

### 2.3 Results (Bangalore/Hong Kong validation)

| Model | MAE (m) | RMSE | R² | ±10 cm | F1@0.15 |
|---|---|---|---|---|---|
| baseline (GINE finetune) | 0.1607 | 0.3310 | 0.7122 | 63.4 % | 0.8591 |
| iter1 | 0.1778 | 0.3110 | 0.7460 | 55.6 % | 0.8198 |
| iter2 | 0.1566 | 0.3127 | 0.7432 | 63.9 % | 0.8655 |
| iter3 | 0.1564 | 0.3120 | 0.7443 | 63.9 % | 0.8658 |
| iter5 | 0.1620 | 0.3410 | 0.6947 | 64.5 % | 0.8710 |
| iter6 | 0.1617 | 0.3390 | 0.6982 | 64.1 % | 0.8699 |
| iter7 | 0.1587 | 0.3308 | 0.7126 | 64.4 % | 0.8674 |
| iter8 | 0.1643 | 0.3323 | 0.7099 | 63.4 % | 0.8591 |
| ens iter6+iter9 | 0.1610 | 0.3386 | 0.6989 | 64.5 % | 0.8713 |
| loss_builder v0_base | 0.1745 | 0.3160 | 0.7377 | 58.3 % | 0.8337 |

*(iter4/9–18 no printable repor; iter9 has no `iter9_report.txt` in the 996fcf4 tree —
that file was lost before the archive. iter13 detail lives in `iter13_detail.py`.)*

Worst residual bin in baseline: `y ∈ [0.80, 3.00)` → MAE 0.4574 m, bias −0.3052 m
(the deep-valley under-prediction that motivates all later deep-tail weighting).

### 2.4 Ensembles tried (all max-gate on depth):
- `ensemble_iter69.py`: iter6+iter9, average depth+prob, per-(intensity, cluster)
  taus (0.3–0.65), test = Bangalore + Hong Kong.
- `ensemble_iter911.py` / `ens911_final_table.py`: iter9+iter11 max-gate, per-region
  F1 tables for Bangalore districts + Hong Kong.
- `ens1213.py`: iter12+iter13 **different architectures** (`np.maximum(g12, g13)`)
  with per-model tau grids (0.1–0.5).
- `ensemble_iter610.py`: iter6+iter10 variant.

### 2.5 SAGPool failure case (documented in §5 of the paper)
`FiLMGINE_SAGPool` (iter18) collapsed: pooling permanently drops nodes → **spatial
path discontinuity** — floods get confined to retained coarse nodes and never reach
dropped downstream junctions. The specific ablation numbers previously quoted
(Hong Kong F1 collapse 0.890 → 0.374, global MAE 24.8 cm) were **not reproducible
from the shipped ensemble and are withdrawn**; the qualitative design rationale
stands. Conclusion: **no destructive pooling in the production model; topology is
always preserved**. The shipped model's Hong Kong F1 is **0.942** and its canonical
global MAE is **2.44 cm** (50 mm/hr reference).

### 2.6 Parallel builders in `st_gnn/`
- **`loss_builder/train_loss_variant.py`** — GINE4 with pluggable losses:
  - `v0` = baseline asym MSE (under 2.5×).
  - `v1` = **depth-adaptive** weights + pseudo-Huber(δ=0.25): `dry&over→3.0,
    dry&under→0.6, mid&under→2.0, deep&under→4.5, deep&over→1.2` (dry-false-flood
    hammer + deep-under hammer). Best dry/depth balance.
  - `v2` = v1 + flooded/dry gate head (BCE aux, `0.8·Lb`, gated inference).
  - `v3` = v1 + `soft-Dice@0.15` (temp 20) + tilted q=0.30 push-up 
    `mean(max(0.30·e, −0.70·e))` on flooded.
- **`pool_builder/train_pool.py` + `pool_common.py`** — `UrbanPoolNet`:
  GINE backbone + SAG-style top-k pooling (ratio grid) + unpool-fill skip-add, so
  basin-scale signal reaches every node; `n_local`, ratios, head=192 grid.
- **`stgnn_builder/stgnn_train.py`** — **spatio-temporal** experiments: 8-intensity
  sequences `TORCH_SEQ=[20,50,80,120,150,200,250,300]`, models `ConvGRUModel`
  (GRUCell state, intensity conditioning `X[:,15:16]`), `StaticModel`, `STGCNModel`
  (Conv1d temporal). Loss: `L_log + 0.5·mean(w·d²)`, `w=2.5 under`, plus optional
  monotone penalty `relu(pred[t] − pred[t+1] + 0.01)` over the rainfall sequence.

---

## 3. Phase 2 — HydroGINE-v4 lineage (deployable single-task ancestor)

Common `HydroGINE_v4` scaffold: `in_c=32, hidden=128, n_layers=6`,
GINEConv(Linear→LN→LReLU→Linear), **residual `h = h_next + 0.3·h`**, multi-scale skip
`cat = [h, mid_h, x]` with `cat_dim = hidden*2 + in_c`; hazard cls head
`Linear(cat,128)→LN→256→…→1`; FiLM generator `Linear(1,64)→LReLU→Linear(64,2·cat_dim)`
**zero-initialized last layer** (identity modulation); reg head
`Linear(cat,256)→LN→256→…→1`.

| Ver | Hidden | Epochs | LR / WD | Chunk | Loss deltas |
|---|---|---|---|---|---|
| v4 | 128 | 350 | 3e-3 / 1e-5 | 100k | asymmetric MSE + BCE + soft-dice |
| v4.1 | **96** | 160 | 7e-4 / 1e-4 | 20k | LeakyReLU encoder; `MarginFocalLoss(gamma=2.0, alpha=0.35, margin=0.02)` |
| v4.2 | 128 | 600 | 1.2e-3 / 2e-5 | 100k | cosine restarts; same MarginFocal 0.35/0.02 |
| v4.3 | 96 | 250 | 2.5e-4 / 1e-5 | 25k | warm-start v4.1 + `log_var_cls / log_var_reg` homoscedastic weighting; `CosineAnnealingLR(eta_min=1e-5)` |

### MarginFocalLoss (exact, reused in v5)
```python
class MarginFocalLoss(nn.Module):
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        targets_m = torch.where(targets == 1.0, 1.0 - self.margin, self.margin)  # margin shift
        p_t       = torch.where(targets == 1.0, probs, 1.0 - probs)
        alpha_t   = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * (1.0 - p_t)**self.gamma
        return (focal_weight * bce_with_logits(logits, targets_m)).mean()
```
v4 clay-pave lesson → reframed as the two-task (classify + regress + FiLM-gate) v5.

---

## 4. Phase 3 — HydroGINE-v5 lineage (production engine)

Final training file: `train_hydro_gine_v5_0.py` (== working tree == `b806d3d`),
556 lines. **Final config block (verbatim values):**
```
EPOCHS=300, LR=3e-4, WEIGHT_DECAY=1e-4, HIDDEN_DIM=128, N_LAYERS=6,
N_CHUNK=20000 (reduced from 25000 after "RTX 4060 TDR/cublas fault on backward"),
SEED=42, DATASET=expanded_master_physics_dataset.pt,
MARGIN_FOCAL = MarginFocalLoss(gamma=2.0, alpha=0.35, margin=0.02)
```

### 4.0 Architecture (`HydroGINE_v5` / `GravityGINEConv`)
- **Gravity gate** on edge grade (ea col 1): downhill fully open, uphill attenuated:
  ```python
  grade = ea[:, 1:2]
  gravity_gate = torch.sigmoid(1.0 - 5.0 * F.relu(grade))
  ea_gated = ea * gravity_gate
  ```
- 6 layers, residual `h = h_next + 0.3*h`, mid-layer `mid_h` captured at `i == n_layers//2`,
  skip-concat `cat = [h, mid_h, x]` → `cat_dim = 128*2 + 32 = 288`.
- **cls head:** `Linear(288,128)→LN→LReLU(0.1)→Dropout(0.05)→128→LN→64→1`.
- **FiLM:** `Linear(1,64)→LReLU→Linear(64,576)` — **last layer zero-init** → identity at warm start. `gamma,beta = chunk(film); h_cond = cat*(1+gamma)+beta`.
- **reg head:** `Linear(288,256)→LN→LReLU→Dropout(0.05)→256→LN→128→1`.

### 4.1 Loss variant history (exact diffs between revisions)

#### v5 base (`b53f51d`)
Unified multi-city engine, expanded master dataset (2.35M nodes).
`MarginFocalLoss(gamma=2.0, alpha=0.35, margin=0.02)` + asymmetric MSE
(under-pred 2.5×) + soft-dice.

#### v5.0 (`16f5130`) — **Zero-Bias Asymmetric Huber**
```
delta = 0.10
huber = where(|diff| < delta, 0.5·diff², delta·(|diff| − 0.5·delta))
OVER-prediction penalized 1.5×:  over mask = (diff > 0)       # blanket
w: y>=0.30 → 3.0, 0.15≤y<0.30 → 2.0, dry pavement → 3.0
deep-valley power-law:  deep_w = 1 + 3.0·clamp((dep_d−1.5)/2.5, 0, 1)^2.0
L_reg = mean(w·huber·asym)
```
Warm start `hydro_gine_v5_bangalore_opt.pt`. → **82.0% live match (+4.8 pp), OVER 3141→692.**

#### v5.1–v5.2 (`291be43`) — **Scenario-conditional asym**
```
LOSS_ASSYM_UP = 1.5     # over-pred on DRY cells (y ≤ 0.15)
LOSS_ASSYM_UNDER = 2.0  # under-pred on FLOODED cells (y > 0.15)
```
(v5.1 added a global match-hinge; it exploded OVER, removed.) → **83.3 %, MAE 2.97 cm, NSE 0.866.**

#### v5.3–v5.5 (`3f2519c`) — match-hinge/flood-gated experiments
- `MATCH_HINGE_STRENGTH=0.0` (v5.5 test — kept for the rest of the lineage).
- `MATCH_BAND_M=0.12` (production ±15 cm band).
- v5.5 experimented with a **flooded-gated linear match hinge**:
  `hinge_mask = y>0.15; match_hinge = relu(|res_lin| − 0.12)`.

#### v5.6 (`b6a2ee4`) — post-rail checkpoint selection
`SELECT_RAIL_MATCH=True`: checkpoints selected by **post-`apply_production_rails`
rail live-match rate** (production leaderboard quantity). Rails lift raw ~47% →
deploy-time ~83%, so raw selection picked suboptimal checkpoints.

#### v5.7 (`ee57de0`) — shallow-band over-penalty
`LOSS_ASSYM_UP_SHALLOW=1.5` on `0.15 < y ≤ 0.50` advisory band (v5.6's OVER cluster
swmm 0.25–0.50 / pred 0.5–0.8 escaped the dry-only over-gate). → 84.07 %.

#### v5.8 (`0f90840`) — EQUAL_WEIGHT_REGIONS
- `EQUAL_WEIGHT_REGIONS=True`: val rail-match averaged **per region equally** — the
  node-weighted aggregate was dominated by london + nyc (largest graphs), silently
  trading away Bengaluru districts (v5.6: hsr 84.7→76.8, bellandur 75.2→68.6).
- `DEEP_TARGET_W = 0` at this stage; ~ **84.8 % leader (OVER 977 / UNDER 1539)**.
- `N_CHUNK = 20000`.

#### v5.9 (`1367d46`) — deep-tail reweight
**Exact diagnostic unprinted motivation:** under-prediction of deep pools
(swmm p50 0.71 vs pred 0.42; sag≈0, dep≈2 m) + over-predicting extreme dry bowls.
```
DEEP_TARGET_W = 3.0     # extra weight on y_true >= 0.50
DEEP_TARGET_W_MID = 1.4 # extra weight on y_true ∈ [0.30, 0.50)
```
→ **best-ever quality (NSE 0.8942, HazMAE 9.85 cm, GlbMAE 2.72 cm) but match 84.5 %**
(OVER 1623 — the 3.0 hammer pushed dry bowls up).

#### v5.10 (`2eb0a38`) — the sweet spot
```
DEEP_TARGET_W     = 1.6   # moderate, between v5.8(=0) and v5.9(=3.0)
DEEP_TARGET_W_MID = 1.2
```
→ **88.14 % leader (best-ever everything), OVER 939 / UNDER 1047,
GlbMAE 2.43 cm, HazMAE 8.85 cm, NSE 0.9060, +3.3 pp over v5.8 champion**, warm-start v5.8.

#### v5.11 (`b806d3d`) — regression datum
```
DEEP_TARGET_W     = 2.2
DEEP_TARGET_W_MID = 1.3
```
→ **87.11 %** — confirms the truth lives at 1.6; v5.10 remains leader. (This is
the version the final ensemble leans on for deep-tail recovery.)

### 4.2 Final combined loss (v5.11, verbatim from train body)
```python
# 1. Margin Focal Loss
loss_cls = focal_loss(cls_logits, y_hazard)                       # gamma=2, alpha=0.35, margin=0.02

# 2. Zero-Bias Asymmetric Huber (delta=0.10), scenario-conditional
diff  = depth_pred_log - y_norm
huber = torch.where(abs_diff < delta, 0.5*diff**2, delta*(abs_diff - 0.5*delta))
over_mask = (diff > 0) & (y_true > 0.15) & (y_true <= 0.50)
asym = where((diff>0) & (y_true<=0.15), LOSS_ASSYM_UP,                 # 1.5 dry over
        where(over_mask,                      LOSS_ASSYM_UP_SHALLOW,   # 1.5 advisory over
          where((diff<0) & (y_true>0.15),     LOSS_ASSYM_UNDER, 1.0))) # 2.0 flooded under
huber = huber * asym

# node weights
sink_d = batch.x[:, 23]; dep_d = batch.x[:, 16]
is_dry_pavement = (y_true <= 0.03) & (sink_d < 0.03)
w = ones; w = where(y_true>=0.30,     3.0, w)
         w = where(0.15<=y<0.30,      2.0, w)
         w = where(is_dry_pavement,   3.0, w)
         w = where(y_true>=0.50,      w*DEEP_TARGET_W, w)     # 2.2  (v5.9 deep-tail)
         w = where(0.30<=y<0.50,      w*DEEP_TARGET_W_MID, w) # 1.3
deep_w = 1 + DEEP_W_STRENGTH * clamp((dep_d - DEEP_W_THRESH)/DEEP_W_RANGE, 0, 1)**DEEP_W_POWER
w      = w * deep_w
loss_reg = (w * huber).mean()

# 3. Flood-gated match hinge (OFF in v5.11; MATCH_HINGE_STRENGTH=0.0)
hinge_mask = y_true > 0.15
loss_match = MATCH_HINGE_STRENGTH * sum(hinge_mask * w * relu(|res_lin| - 0.12)) / n_flooded

total = loss_cls + 1.5*loss_reg + 0.8*loss_match
```
- **Optimizer/scheduler:** AdamW(lr=3e-4, wd=1e-4), CosineAnnealingLR(eta_min=1e-5),
  grad-clip 1.0, batch-by-node-chunks of 20k.
- **Validation (every 10 epochs):** val regions = **hsr, bellandur, london, newyork**;
  F1 at `p_prob≥0.35` vs `y≥0.15`; live-match = `p_depth>0.08` and
  `|p_depth − y| < 0.15`; **EQUAL_WEIGHT_REGIONS** + **rail-match selection**;
  early stop after 9 no-improve evals; save `val_f1`, `val_match_rate`, `epoch`.

### 4.3 Hyperparameter table (v5.0 → v5.11)

| Ver | EPOCHS | LR | N_CHUNK | ASSYM_UP | ASSYM_UP_SH | ASSYM_UNDER | DEEP_TARGET_W | DEEP_TARGET_W_MID | HINGE | SELECT_RAIL | EQ_REGION | Match | NSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 200 | 3e-4 | 25k | — | — | 2.5×MSE | — | — | — | ✗ | ✗ | 77.8 % | 0.7764 |
| v5.0 | 200 | 3e-4 | 25k | 1.5 | — | — | — | — | — | ✗ | ✗ | 82.0 % | 0.8465 |
| v5.2 | 220 | 3e-4 | 25k | 1.5 | — | 2.0 | — | — | — | ✗ | ✗ | 83.3 % | 0.8659 |
| v5.4 | 220 | 3e-4 | 25k | 1.5 | — | 2.0 | — | — | 2.0 | ✗ | ✗ | 81.7 % | 0.8606 |
| v5.5 | 220 | 3e-4 | 25k | 1.5 | — | 2.0 | — | — | 0.0 | ✗ | ✗ | 80.5 % | 0.8647 |
| v5.6 | 300 | 3e-4 | 25k | 1.5 | — | 2.0 | — | — | 0.0 | ✔ | ✗ | 84.5 % | 0.8814 |
| v5.7 | 300 | 3e-4 | 25k | 1.5 | 1.5 | 2.0 | — | — | 0.0 | ✔ | ✗ | 84.1 % | 0.8745 |
| v5.8 | 300 | 3e-4 | 20k | 1.5 | 1.5 | 2.0 | 0 | — | 0.0 | ✔ | ✔ | 84.8 % | 0.8779 |
| v5.9 | 300 | 3e-4 | 20k | 1.5 | 1.5 | 2.0 | 3.0 | 1.4 | 0.0 | ✔ | ✔ | 84.5 % | **0.8942** |
| v5.10 | 300 | 3e-4 | 20k | 1.5 | 1.5 | 2.0 | **1.6** | **1.2** | 0.0 | ✔ | ✔ | **88.1 %** | **0.9060** |
| v5.11 | 300 | 3e-4 | 20k | 1.5 | 1.5 | 2.0 | 2.2 | 1.3 | 0.0 | ✔ | ✔ | 87.1 % | ~0.900 |

*(v5.1 = global match-hinge → OVER explosion, removed; v5.3 = hinge strength sweep.)*

### 4.4 Full benchmark table (`benchmark_all_versions.json`, 10 rows)

| Model | Match % | NSE | GlbMAE (cm) | HazMAE (cm) | Risk | OVER | UNDER |
|---|---|---|---|---|---|---|---|
| v5.6 | 84.54 | 0.8814 | 2.80 | 10.30 | 17604 | 1847 | 875 |
| v5.7 | 84.07 | 0.8745 | 2.88 | 10.83 | 17093 | 1733 | 990 |
| v5.2 | 83.30 | 0.8659 | 2.97 | 11.18 | 16986 | 1403 | 1433 |
| v5.0 | 81.96 | 0.8465 | 3.14 | 12.53 | 16373 | 692 | 2262 |
| v5.4 | 81.67 | 0.8606 | 3.11 | 11.10 | 17685 | 1975 | 1266 |
| v5.5 | 80.55 | 0.8647 | 3.37 | 10.77 | 20248 | 2301 | 1638 |
| v5 Base | 77.79 | 0.7764 | 3.84 | 15.15 | 16571 | 1227 | 2453 |
| v5 Bottleneck Opt | 77.15 | 0.8284 | 5.15 | 12.70 | 22753 | 3141 | 2057 |
| v5 Bengaluru Opt | 77.02 | 0.8292 | 4.53 | 13.08 | 20463 | 2182 | 2521 |
| v5 Finetuned | 76.70 | 0.8187 | 5.71 | 13.13 | 22776 | 3286 | 2020 |

`benchmark_final_v58.json`: **v5.8 = 84.82 % (OVER 977 / UNDER 1539), NSE 0.8779**.
City rates (hsr, bellandur, whitefield, ecity, koramangala, tokyo, hongkong, singapore,
london, paris, nyc, chicago, berlin, bangkok, mumbai, delhi) accompany every row.

---

## 5. Production physical rails (deploy-time — mirrors apply_physics_rails)

`apply_production_rails(x_np, edge_index, p_lin, p_prob)` in `production_v4.py`
(replicated in training as `apply_production_rails` for rail-match selection).
Replicates `ProductionFloodPredictorV4.predict()` post-head transforms
(regime classification, `conf_gate`, `mass_bound`, WSE envelope). Feature cols:
`rel_drop=0, in_d=3, out_d=4, accum_s=5, is_sink=6, sag_idx=8, dep_d=16,
sink_d=23, total_r=27, conv_def=30`.

```python
is_choked_surcharge = (conv_def >= 0.7) & ((sag_idx > 0.01) | (dep_d > 0.02) | (out_d < in_d))
is_deep_sink        = is_sink & (dep_d >= 0.20)
is_valley_depression= dep_d >= 0.20
is_convergent_sag   = (in_d > out_d) | (sag_idx >= 0.03)
is_ridge_crest      = (rel_drop < 0.25) & (sink_d < 0.03) & (dep_d < 0.03)
is_free_drain_slope = (sink_d < 0.02) & (dep_d < 0.02) & (out_d >= 2) & ~choked
is_steep_ridge      = (rel_drop < 0.20) & (|slope| > 0.06) & (accum_s < 0.5) & (sink_d < 0.01) & ~choked

tau   = 0.15 (deep_sink|valley) / 0.25 (choked|sag) / 0.75 (ridge|free_drain|steep) / 0.35
conf_gate = 1/(1 + exp(-6·(p_prob − tau)))
mass_bound: choked→3.0; deep_sink|valley→clip(dep_d·1.5+0.3,1,3);
            ridge|steep→0.01; free_drain→0.04 (total_rain≤50) else 0.10;
            convergent & p≥0.65→max(0.50, sink_d·2+0.25); p≥0.65→max(0.50, sink_d·1.5+0.20);
            else max(0.15, sink_d·1.5+0.08)
p_final = min(p_lin · conf_gate, mass_bound)
p_final = 0 where (flat_dry | steep_ridge); 0 where < 0.02; cap 3.0
```
**WSE backwater envelope** (`relief_m = 15.0` default; `elev_range` baked onto graphs):
```
elevs = −max(max(dep_d, sink_d), rel_drop·relief_m)
wse   = elevs + p_final
backwater = relu(wse[src] − elevs[dst])  # + reverse; max-scatter per node
p_final[~protected] = min(p_final[~protected],
                          max(max_backwater[~protected], own_storage[~protected]))
```
This is the single transformer that converts raw **~47 %** accuracy into deploy-time
**~83 %** — and the reason checkpoint selection was moved to post-rail metrics.

---

## 6. Ensembles (`EnsembleFloodPredictorV4`)

### 6.1 v5.6 + v5.8 (`b4ad8d2`, +303/−137 loc)
v5.6 is UNDER-lean, v5.8 OVER-lean; **average raw pre-rail `p_lin` + `p_prob`,
then apply rails ONCE.**
→ 85.88 % match, GlbMAE 2.67 cm, NSE 0.8911, OVER **1261** / UNDER 1159, 26.7 ms/city
(**1966×** faster than SWMM). City: hsr 78.2, bellandur 70.5, whitefield 86.0, ecity 84.3, koramangala 76.0.

### 6.2 Final tuned ensemble (`a18ac71`, working tree)
```python
model_paths = (
    "hydro_gine_v5_10_model.pt",      # w=0.80  high global precision & speed
    "hydro_gine_v5_11_model.pt",      # w=0.30  deep-tail reweighted recovery
    "hydro_gine_v5_0_model.pt",       # w=0.05  HSR / local district recall
    "hydro_gine_v5_bangalore_opt.pt") # w=0.10  local valley morphology
weights = (0.80, 0.30, 0.05, 0.10)   # normalized by w_sum (=1.25) in __init__
p_lin_avg = Σ w·p_lin;  p_prob_avg = Σ w·p_prob
p_final = apply_physics_rails(x_full, ei, p_lin_avg, p_prob_avg, elev_range=relief_m)
```
→ Historical per-commit live-match snapshot: **HSR 82.2 %, Bellandur 72.0 %**, Berlin 95.7 %, Singapore 95.6 %, Tokyo 94.6 %,
Paris 93.0 %, NYC 93.4 %, **NSE 0.9098**, **OVER ≤ 851**. These live-match figures are superseded by the
canonical live-computed system metrics (hazard F1 **90.8 %**, NSE **0.9128**, MAE **2.44 cm**).

---

## 7. Datasets & feature engineering

- **`build_physics_dataset.py`** — 28-feature index schema; drops `rel_x/rel_y`
  → 26 physical features + 6 derived → **32 features**.
  Index schema: `0 rel_x, 1 rel_y, 2 rel_drop, 3 imp, 4 manning_n, 5 in_deg,
  6 out_deg, 7 accum_score, 8 is_sink, 9 max_in_grade, 10 sag_index,
  11 hydraulic_capacity, 12 log_area, 13 log_imp_area, 14 dist_frac, 15 intensity,
  16 duration, 17 elev_std2, 18 dep_depth, 19 surcharge, 20 path_cap, 21 path_hops,
  22 dist_outlet, 23 elev_above_outlet, 24 slope_outlet_ratio, 25 sink_depth,
  26 inlet_cap, 27 surcharge_ratio`.
- **`build_expanded_physics_dataset.py`** — loads `multi_scenario_physics_pyg_dataset.pt`
  (464 graphs) + injects 6 held-out international test cities
  (`multi_scenario_testcities_pyg_dataset.pt`, `create_testcity_dataset.py`).
  **Valley-sink fix:** `x[:,23] = where(rel_drop≥0.50 & dep_depth≥0.05, dep_depth, 0)`;
  `x[:,6] = where(rel_drop≥0.50 & dep_depth≥0.08, 1.0, 0.0)`;
  **ponding index** `x[:,29] = log1p(x[:,23]·total_rain/(max(0.2, out_deg)+0.3))`.
  Attaches `elev_range` (true catchment relief, m) for the production WSE envelope.
- **`expanded_master_physics_dataset.pt`** — final train set, **~2.35M nodes**,
  multi-scenario (8 storm intensities × durations), ground truth from **EPA SWMM 5.2**
  dynamic-wave 1D/2D on all 16 global catchments (76,316 independently audited nodes).

---

## 8. Summary of the trajectory (one paragraph)

Phase 0 proved GATv2+PIN-lean formulations insufficient (≤4-layer GATs, 14–8 features).
Phase 1 (`st_gnn`) converged on a **two-task GINE with FiLM gating + homoscedastic
uncertainty + focal loss**, but every hierarchical/pooling variant (**SAGPool** at
iter18) collapsed via spatial path discontinuity — establishing the hard constraint
"**preserve full topology**". Phase 2 (v4.x) shipped that scaffold with
MarginFocal(0.35/0.02) + log-variance weighting as the deployment v4. Phase 3
(v5.x) is a controlled single-objective sweep of the **scenario-conditional
asymmetric Huber** (dry-over 1.5×, advisory-over 1.5×, flooded-under 2×), the
**deep-tail reweight** (0 → 3.0 → **1.6** → 2.2), **equal-weight region selection**,
and **post-rail checkpoint selection**, topped by a **weighted raw-depth ensemble**
re-applying the physical bounding rails exactly once — landing at 88–93 % live
match, NSE ~0.91, with ~2 ms tensor / ~30 ms API latencies per city.

*File references: `train_hydro_gine_v5_0.py`, `train_hydro_gine_v4*.py`,
`train_pinn_gnn.py`, `train_ultra_pinn.py`, `train_dual_stream_hydro_gnn.py`,
`build_physics_dataset.py`, `build_expanded_physics_dataset.py`,
`create_multi_pyg_dataset.py`, `create_testcity_dataset.py`, `production_v4.py`,
`benchmark_all_versions.json`, `benchmark_final_v58.json`,
`benchmark_ensemble_v56_v58.json`, `competition_benchmark_verified.json`,
`st_gnn/` (996fcf4). See `doc/UrbanFLOW_Research_Paper.md` for the peer-facing narrative.*