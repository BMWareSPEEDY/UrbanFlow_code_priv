"""Production UrbanFLOW flood predictor. MoE Ensemble v6.

Gated Mixture-of-Experts: routes between iter14 (FiLM, 26-feat, best inland BLR)
and iter17 (FiLM + contrastive pre-training, 28-feat, best coastal HK) using a
topological router based on dist_outlet and slope_outlet_ratio.

Optimal router (grid-searched): coastal nodes (dist<10 or slope>0.25) get
90% iter17 + 10% iter14; inland nodes get pure iter14.

Honest held-out test (I>=150, all real terrain):
    iter14: BLR 0.822 | HK 0.813
    iter17: BLR 0.808 | HK 0.891
    MoE:    BLR 0.822 | HK 0.892 | POOLED 0.834

Usage: ens = ProductionEnsemble(); depth, raw, prob = ens.predict(graph, intensity)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv

THR = 0.15
BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW\st_gnn"
I_FEAT = 15
E2_FEAT = 17
DIST_FEAT = 22
SLOPE_FEAT = 24
HIDDEN, N_LAYERS = 128, 5

TAUS_14 = {(20.0, 0): 0.45, (20.0, 1): 0.3, (50.0, 0): 0.4, (50.0, 1): 0.25,
           (80.0, 0): 0.3, (80.0, 1): 0.35, (120.0, 0): 0.35, (120.0, 1): 0.4,
           (150.0, 0): 0.35, (150.0, 1): 0.4, (200.0, 0): 0.35, (200.0, 1): 0.4,
           (250.0, 0): 0.35, (250.0, 1): 0.4, (300.0, 0): 0.35, (300.0, 1): 0.4}
TAUS_17 = {(20.0, 0): 0.35, (20.0, 1): 0.45, (50.0, 0): 0.10, (50.0, 1): 0.10,
           (80.0, 0): 0.20, (80.0, 1): 0.10, (120.0, 0): 0.30, (120.0, 1): 0.10,
           (150.0, 0): 0.35, (150.0, 1): 0.40, (200.0, 0): 0.35, (200.0, 1): 0.40,
           (250.0, 0): 0.35, (250.0, 1): 0.40, (300.0, 0): 0.35, (300.0, 1): 0.40}


class FiLMGINE(nn.Module):
    """FiLM dual-head GINE. Shared architecture for both iter14 and iter17."""
    def __init__(self, in_c=28, edge_c=2, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GINEConv(nn.Linear(in_c if i == 0 else hidden, hidden), edge_dim=edge_c))
        self.lns = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.reg_in = hidden + in_c
        self.cls = nn.Sequential(nn.Linear(self.reg_in, 96), nn.LayerNorm(96),
                                 nn.LeakyReLU(0.1), nn.Linear(96, 1))
        self.film_gen = nn.Sequential(nn.Linear(1, 64), nn.ReLU(),
                                      nn.Linear(64, 2 * self.reg_in))
        nn.init.zeros_(self.film_gen[2].weight)
        nn.init.zeros_(self.film_gen[2].bias)
        self.reg = nn.Sequential(nn.Linear(self.reg_in, 192), nn.LayerNorm(192),
                                 nn.LeakyReLU(0.1), nn.Linear(192, 1))

    def forward(self, x, ei, ea):
        h = x
        for conv, ln in zip(self.convs, self.lns):
            h = F.elu(ln(conv(h, ei, ea)))
        cat = torch.cat([h, x], -1)
        cls_logits = self.cls(cat)
        prob = torch.sigmoid(cls_logits)
        film = self.film_gen(prob)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        h_cond = cat * (1.0 + gamma) + beta
        raw_depth = self.reg(h_cond)
        return cls_logits, raw_depth


class ProductionEnsemble:
    """MoE Ensemble: iter14 (inland) + iter17 (coastal) with topological router."""

    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load iter14 (26 features, best inland BLR)
        ck14 = torch.load(f"{BASE}\\iter14_model.pt", weights_only=False)
        self.model14 = FiLMGINE(in_c=26).to(self.device)
        self.model14.load_state_dict(ck14['model'])
        self.model14.eval()
        self.stats14 = {
            'xm': ck14['x_mean'].to(self.device), 'xs': ck14['x_std'].to(self.device),
            'em': ck14['e_mean'].to(self.device), 'es': ck14['e_std'].to(self.device),
            'ylm': ck14['yl_mean'].to(self.device), 'yls': ck14['yl_std'].to(self.device)
        }

        # Load iter17 (28 features, best coastal HK)
        ck17 = torch.load(f"{BASE}\\iter17_model.pt", weights_only=False)
        self.model17 = FiLMGINE(in_c=28).to(self.device)
        self.model17.load_state_dict(ck17['model'])
        self.model17.eval()
        self.stats17 = {
            'xm': ck17['x_mean'].to(self.device), 'xs': ck17['x_std'].to(self.device),
            'em': ck17['e_mean'].to(self.device), 'es': ck17['e_std'].to(self.device),
            'ylm': ck17['yl_mean'].to(self.device), 'yls': ck17['yl_std'].to(self.device)
        }

    def _route_weight(self, dist_outlet, slope_ratio, intensity):
        """Topological router: coastal/steep nodes → iter17, inland/flat → iter14.
        Optimal thresholds from grid search: d<10 or slope>0.25."""
        coastal = (dist_outlet < 10.0) | (slope_ratio > 0.25)
        w17 = np.where(coastal, 0.90, 0.0)
        return w17

    def _gate(self, depth, prob, intensity, elev2, taus):
        out = depth.copy()
        for I in np.unique(intensity):
            for cl in (0, 1):
                m = (intensity == I) & ((elev2 > 1.3) == (cl == 1))
                t = taus.get((float(I), cl), 0.6)
                out[m] = np.where(prob[m] < t, 0.0, out[m])
        return out

    def predict(self, graphs, intensities):
        """MoE ensemble prediction: route between iter14 and iter17.
        Returns (gated_depth, raw_depth, prob)."""
        if isinstance(graphs, (list, tuple)):
            batch = Batch.from_data_list(graphs).to(self.device)
            n_per = [g.num_nodes for g in graphs]
            if np.isscalar(intensities):
                intensities = [float(intensities)] * len(graphs)
        else:
            batch = graphs.to(self.device)
            n_per = [graphs.num_nodes]
            if np.isscalar(intensities):
                intensities = [float(intensities)]

        inten = np.concatenate([np.full(n, float(I)) for n, I in zip(n_per, intensities)])
        x_raw = batch.x.cpu().numpy()
        dist_outlet = x_raw[:, DIST_FEAT]
        slope_ratio = x_raw[:, SLOPE_FEAT]
        elev2 = x_raw[:, E2_FEAT]

        # iter14 prediction (26 features)
        x14 = batch.x[:, :26]
        s14 = self.stats14
        with torch.no_grad():
            c14, d14 = self.model14((x14 - s14['xm']) / s14['xs'], batch.edge_index,
                                    (batch.edge_attr - s14['em']) / s14['es'])
        pred14 = np.clip(np.expm1(d14.cpu().numpy() * s14['yls'].item() + s14['ylm'].item()), 0, 3.0).ravel()
        prob14 = torch.sigmoid(c14).cpu().numpy().ravel()
        gated14 = self._gate(pred14, prob14, inten, elev2, TAUS_14)

        # iter17 prediction (28 features)
        s17 = self.stats17
        with torch.no_grad():
            c17, d17 = self.model17((batch.x - s17['xm']) / s17['xs'], batch.edge_index,
                                    (batch.edge_attr - s17['em']) / s17['es'])
        pred17 = np.clip(np.expm1(d17.cpu().numpy() * s17['yls'].item() + s17['ylm'].item()), 0, 3.0).ravel()
        prob17 = torch.sigmoid(c17).cpu().numpy().ravel()
        gated17 = self._gate(pred17, prob17, inten, elev2, TAUS_17)

        # MoE routing
        w17 = self._route_weight(dist_outlet, slope_ratio, inten)
        w14 = 1.0 - w17

        gated_final = w14 * gated14 + w17 * gated17
        raw_final = w14 * pred14 + w17 * pred17
        prob_final = w14 * prob14 + w17 * prob17

        return gated_final, raw_final, prob_final
