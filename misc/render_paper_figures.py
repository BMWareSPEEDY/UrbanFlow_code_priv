"""Render UrbanFLOW paper figures from fig_pred_data.pt and paper tables."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#505050",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelsize": 11,
    "legend.fontsize": 8.5,
    "axes.titlesize": 11,
    "figure.dpi": 200,
})

rows = torch.load("fig_pred_data.pt", map_location="cpu", weights_only=False)
Y = np.concatenate([r["y"] for r in rows])
P = np.concatenate([r["p"] for r in rows])
E = np.abs(Y - P)
mae = E.mean() * 100
rmse = np.sqrt((E**2).mean()) * 100
p15 = (E <= 0.15).mean() * 100
p30 = (E <= 0.30).mean() * 100
dry = (Y < 0.15).mean() * 100
N = Y.size

OUT = "UrbanFLOW_paper_fig_%s.png"

# ---------------------------------------------------------------- scatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(figsize=(5.4, 4.4))
hb = ax.hexbin(Y, P, gridsize=80, bins="log", cmap="cividis", mincnt=1)
mask = np.hypot(Y, P) > 0
lim = max(Y.max(), P.max(), 0.3)
ax.plot([0, lim], [0, lim], color="#111111", lw=1.2, label="1:1 line")
ax.plot([0, lim], [0.15, lim + 0.15], color="#d1495b", lw=0.9, ls="--", label="±15 cm band")
ax.plot([0, lim], [-0.15, lim - 0.15], color="#d1495b", lw=0.9, ls="--")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("SWMM 5.2 junction depth (m)")
ax.set_ylabel("UrbanFLOW predicted depth (m)")
ax.legend(loc="upper left", frameon=False)
txt = ("Production ensemble (Eq. 10)\n"
       f"N = {N:,} nodes\nMAE = {mae:.2f} cm\nRMSE = {rmse:.1f} cm\n"
       f"|err| ≤ 15 cm: {p15:.1f}%\n|err| ≤ 30 cm: {p30:.1f}%")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="26%", pad=0.35)
cb = fig.colorbar(hb, cax=cax)
cb.set_label("node count (log)", fontsize=9)
cax.text(0.5, 0.02, txt, transform=cax.transAxes, ha="center", va="bottom",
         fontsize=8, linespacing=1.35,
         bbox=dict(fc="white", ec="#cccccc", lw=0.6, boxstyle="round,pad=0.35"))
fig.tight_layout()
fig.savefig(OUT % "scatter", bbox_inches="tight")
plt.close(fig)
print("saved", OUT % "scatter")

# ---------------------------------------------------------------- speedup
cats = ["HSR Layout\n(1,379 nodes)", "Electronic City\n(3,337 nodes)", "Tokyo Metro\n(13,173 nodes)"]
swmm_s = [42.1, 228.0, 1476.0]
ufw_s = [0.0320, 0.0546, 0.1698]
speedup = [1310, 4180, 8690]

fig, ax = plt.subplots(figsize=(5.6, 3.6))
x = np.arange(3)
w = 0.34
b1 = ax.bar(x - w/2, swmm_s, w, color="#4c72b0", label="EPA SWMM 5.2")
b2 = ax.bar(x + w/2, ufw_s, w, color="#b0413e", label="UrbanFLOW (end-to-end HTTP API)")
ax.set_yscale("log")
ax.set_ylabel("runtime (s, log scale)")
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8)
ax.set_ylim(0.0008, 4000)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
for xi, s in zip(x, swmm_s):
    ax.annotate(f"{s:g} s", (xi - w/2, s), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=7.5)
for xi, s in zip(x, ufw_s):
    ax.annotate(f"{s*1000:g} ms", (xi + w/2, s), xytext=(0, -12), textcoords="offset points",
                ha="center", fontsize=7.5, color="#701b18")
for xi, sp in zip(x, speedup):
    ax.annotate(f"≈ {sp:,}×", (xi, 3300), ha="center", fontsize=9, fontweight="bold",
                color="#222222")
fig.tight_layout()
fig.savefig(OUT % "speedup", bbox_inches="tight")
plt.close(fig)
print("saved", OUT % "speedup")

# ---------------------------------------------------------------- ablation
# WITHDRAWN: the intermediate SAGPool / component ablation values
# (HK F1 0.374, MAE 24.8 cm, etc.) were not reproducible from the shipped
# ensemble and are withdrawn as unverified. No ablation figure is rendered;
# the qualitative SAGPool failure rationale remains in the paper text.
print("skipped ablation figure (withdrawn/unverified)")

# ---------------------------------------------------------------- zero-inflation histogram
vis = np.clip(Y, 0, 0.6)
bins = np.arange(0, 0.625, 0.025)
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.hist(vis, bins=bins, color="#8a6d3b", edgecolor="white", lw=0.3)
ax.set_yscale("log")
ax.set_xlabel("SWMM 5.2 junction depth (m), 50 mm/hr")
ax.set_ylabel("nodes (log scale)")
ax.axvline(0.15, color="#d1495b", ls="--", lw=1.1)
ax.annotate("hazard threshold\n0.15 m", (0.15, ax.get_ylim()[1] * 0.9), xytext=(0.18, ax.get_ylim()[1]*0.75),
            fontsize=8, color="#a33a48")
ax.text(0.98, 0.93, f"{dry:.1f}% of nodes dry (< 0.15 m)\nzero-inflated depth field",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
        bbox=dict(fc="white", ec="#cccccc", lw=0.6, boxstyle="round,pad=0.35"))
fig.tight_layout()
fig.savefig(OUT % "zeroinfl", bbox_inches="tight")
plt.close(fig)
print("saved", OUT % "zeroinfl")