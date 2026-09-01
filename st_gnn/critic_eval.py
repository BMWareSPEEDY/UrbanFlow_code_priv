"""Critic harness: scores model predictions vs SWMM DYNWAVE ground truth on held-out Bangalore.
Predictions file: .pt dict {'pred': np.ndarray (N_te,)} aligned with dataset order for city=='bangalore'.
Labels are never printed/returned by this harness -- only scores.
Usage: python critic_eval.py <preds_file.pt> <out_report.txt>"""
import sys
import numpy as np
import torch

BASE = r"D:\CODES\PYTHON_CODES\UrbanFLOW"


def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / max(1e-6, ss_tot)


def f1_hazard(y, p, thr=0.15):
    tp = np.sum((p >= thr) & (y >= thr))
    fp = np.sum((p >= thr) & (y < thr))
    fn = np.sum((p < thr) & (y >= thr))
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc), pr, rc


def load_te():
    dl = torch.load(BASE + r"\multi_scenario_pyg_dataset.pt", weights_only=False)
    te = [g for g in dl if g.city == 'bangalore']
    y = np.concatenate([g.y.numpy().ravel() for g in te])
    inten = np.concatenate([np.full(g.y.shape[0], float(g.rain_intensity)) for g in te])
    return y, inten


def main():
    preds_file, out_file = sys.argv[1], sys.argv[2]
    y, inten = load_te()
    ck = torch.load(preds_file, weights_only=False)
    p = np.clip(np.asarray(ck['pred'], dtype=float).ravel(), 0, None)
    assert p.shape == y.shape, f"pred {p.shape} != truth {y.shape}"
    mae = np.mean(np.abs(p - y))
    rmse = np.sqrt(np.mean((p - y) ** 2))
    p10 = np.mean(np.abs(p - y) <= 0.10) * 100
    p20 = np.mean(np.abs(p - y) <= 0.20) * 100
    f, pr, rc = f1_hazard(y, p)

    lines = [f"PREDICTIONS: {preds_file}",
             f"OVERALL: MAE {mae:.4f} m | RMSE {rmse:.4f} | R2 {r2(y,p):.4f} | "
             f"+-10cm {p10:.1f}% | +-20cm {p20:.1f}% | F1@0.15m {f:.4f} (P {pr:.3f} R {rc:.3f})"]
    lines.append("PER-INTENSITY:")
    for i in np.sort(np.unique(inten)):
        m = inten == i
        if m.sum() < 10:
            continue
        f_, _, _ = f1_hazard(y[m], p[m])
        lines.append(f"  I={i:6.1f}: MAE {np.mean(np.abs(p[m]-y[m])):.4f} | R2 {r2(y[m],p[m]):.4f} | "
                     f"+-10cm {np.mean(np.abs(p[m]-y[m])<=0.10)*100:.1f}% | F1@0.15 {f_:.4f}")

    lines.append("ERROR-BY-DEPTH-BIN (true depth):")
    bins = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 3.0)]
    err_contrib = {}
    for lo, hi in bins:
        m = (y >= lo) & (y < hi)
        if m.sum() == 0:
            continue
        lines.append(f"  y in [{lo:.2f},{hi:.2f}): n={m.sum():6d} | MAE {np.mean(np.abs(p[m]-y[m])):.4f} | "
                     f"bias {np.mean(p[m]-y[m]):+.4f}")
        err_contrib[f"[{lo:.2f},{hi:.2f})"] = np.sum(np.abs(p[m] - y[m]))

    # biggest single remaining error gap
    gap = max(err_contrib, key=err_contrib.get)
    lines.append(f"BIGGEST ERROR GAP: true-depth bin {gap} (contributes {err_contrib[gap]:.1f} m total abs error); "
                 f"overall total abs error {np.sum(np.abs(p-y)):.1f} m")
    hi_idx = int(np.argmax(np.abs(p - y)))
    lines.append(f"WORST NODE: idx {hi_idx}, intensity {inten[hi_idx]:.0f} mm/hr, "
                 f"true {y[hi_idx]:.3f} m, pred {p[hi_idx]:.3f} m")
    bias_hi = np.mean(p[inten >= 200] - y[inten >= 200])
    bias_lo = np.mean(p[inten < 120] - y[inten < 120])
    lines.append(f"BIAS: light (<120mm/hr) {bias_lo:+.4f} m | extreme (>=200mm/hr) {bias_hi:+.4f} m")

    txt = "\n".join(lines)
    print(txt, flush=True)
    with open(out_file, "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()