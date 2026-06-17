"""NF inference: posterior (Omega_m, sigma_8) over the 128^3 pool (real or synth).

    python -m nf.infer --config config/config.yaml --checkpoint ckpt [--data_mode real|synth]

Mode comes from nf.data.mode (or --data_mode). real = CAMELS LH 128^3 Mgas;
synth = FM-synth 128^3 sample dirs. Compare real- vs synth-trained NF (and
cross-eval) to quantify FM signal retention.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import yaml

from data import load_nf_pool, nf_mgas_stats
from .module import LitNFRegressor, MultiSuiteMgasDataset
from .predict import predict_with_uncertainty


def plot_truth_vs_pred(y_true, y_pred, y_std, out_path, param_names=None, dpi=150,
                       title=None):
    n = y_true.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4), squeeze=False)
    axes = axes.reshape(-1)
    for j in range(n):
        ax = axes[j]
        x, y, e = y_true[:, j], y_pred[:, j], y_std[:, j]
        rmse = float(np.sqrt(np.mean((x - y) ** 2)))
        r2 = 1.0 - np.sum((x - y) ** 2) / (np.sum((x - x.mean()) ** 2) + 1e-12)
        ax.errorbar(x, y, yerr=e, fmt="o", ms=3, alpha=0.5, elinewidth=0.4, capsize=0)
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        m = 0.05 * (hi - lo) if hi > lo else 0.1
        ax.plot([lo - m, hi + m], [lo - m, hi + m], "r--", lw=1)
        ax.set_title(f"{param_names[j] if param_names else f'p{j}'}  RMSE={rmse:.3f} R²={r2:.2f}", fontsize=9)
        ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data_mode", choices=["real", "synth"], default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    nf_cfg = cfg["nf"]
    nf_data = nf_cfg["data"]
    ic = nf_cfg.get("inference", {})
    if args.data_mode:
        nf_data["mode"] = args.data_mode

    ckpt = args.checkpoint or ic.get("checkpoint")
    assert ckpt, "no NF checkpoint"
    dev = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading {ckpt}  (data mode={nf_data.get('mode', 'real')})")
    model = LitNFRegressor.load_from_checkpoint(ckpt, map_location=dev).eval().to(dev)

    mgas_src, cosmo_all, flat = load_nf_pool(nf_data)
    mgas_mean, mgas_std = nf_mgas_stats(nf_data)
    ds = MultiSuiteMgasDataset(mgas_src, cosmo_all, flat, np.arange(len(flat)),
                               mgas_mean, mgas_std, augment=False)
    loader = DataLoader(ds, batch_size=ic.get("batch_size", 2), shuffle=False, num_workers=0)

    y_true, y_mean, y_std, aux = predict_with_uncertainty(
        model, loader, ic.get("n_posterior_samples", 2000), dev)

    out_root = Path(args.output_dir or ic.get("output_dir", "outputs/nf_inference"))
    out_root.mkdir(parents=True, exist_ok=True)
    tag = nf_data.get("mode", "real")
    np.savez(out_root / f"predictions_{tag}.npz",
             y_true=y_true, y_mean=y_mean, y_std=y_std, aux_pred=aux)
    title = f"NF whole-pool ({tag}, multi-suite N={len(flat)})"
    plot_truth_vs_pred(y_true, y_mean, y_std, out_root / f"truth_vs_pred_{tag}.png",
                       ic.get("param_names"), dpi=ic.get("dpi", 150), title=title)


if __name__ == "__main__":
    main()
