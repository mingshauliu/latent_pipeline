"""In-distribution truth-vs-pred for the multi-suite NF (Omega_m, sigma_8).

Mirrors ../upscaling/plot_nf_2param.py but on the latent_pipeline multi-suite
128^3 pool: encodes the held-out VAL split (same split/seed as training) and
plots truth vs posterior-mean with per-param RMSE / R^2.

    python plot_nf_2param.py --config config/config.yaml \
        --checkpoint nf_ck_real/best.ckpt --data_mode real \
        --out nf_indist_real.png

data_mode real|synth picks the eval pool (real CAMELS 256 or FM-synth 256).
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from data import load_nf_pool, nf_mgas_stats
from nf.module import LitNFRegressor, MultiSuiteNFDataModule
from nf.predict import predict_with_uncertainty


def pick_best_ckpt(ckpt_dir):
    paths = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    # prefer a non-`last` best by val/nll in the filename, else newest by mtime
    scored = []
    for p in paths:
        base = os.path.basename(p)
        if "last" in base:
            continue
        try:
            scored.append((float(base.split("-")[-1].replace(".ckpt", "")), p))
        except ValueError:
            pass
    if scored:
        return min(scored)[1]
    return max(paths, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data_mode", choices=["real", "synth"], default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["nf"]
    nf_data, nf_train = cfg["data"], cfg["training"]
    if args.data_mode:
        nf_data["mode"] = args.data_mode
    mode = nf_data.get("mode", "real")

    ckpt = args.checkpoint or pick_best_ckpt(nf_train.get("checkpoint_dir", "nf_checkpoints"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\nCkpt:   {ckpt}\nMode:   {mode}")

    model = LitNFRegressor.load_from_checkpoint(ckpt, map_location=device).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    mgas_src, cosmo_all, flat = load_nf_pool(nf_data)
    mgas_mean, mgas_std = nf_mgas_stats(nf_data)
    dm = MultiSuiteNFDataModule(
        mgas_src, cosmo_all, flat, mgas_mean, mgas_std,
        val_split=nf_data.get("val_split", 0.2),
        batch_size=cfg.get("inference", {}).get("batch_size", 2),
        num_workers=0, seed=nf_train.get("seed", 42))
    dm.setup()
    loader = DataLoader(dm.val_ds, batch_size=2, shuffle=False, num_workers=0)
    print(f"Val (held-out) samples: {len(dm.val_ds)}")

    n_post = cfg.get("inference", {}).get("n_posterior_samples", 2000)
    y_true, y_mean, y_std, _ = predict_with_uncertainty(model, loader, n_post, device)

    names = cfg.get("inference", {}).get("param_names") or [r"$\Omega_m$", r"$\sigma_8$"]
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    axes = np.atleast_1d(axes)
    rmse_per, r2_per = [], []
    for j, (ax, label) in enumerate(zip(axes, names)):
        x, yp, err = y_true[:, j], y_mean[:, j], y_std[:, j]
        rmse = float(np.sqrt(np.mean((x - yp) ** 2)))
        r2 = 1.0 - np.sum((x - yp) ** 2) / (np.sum((x - x.mean()) ** 2) + 1e-12)
        rmse_per.append(rmse); r2_per.append(r2)
        xmin, xmax = float(x.min()), float(x.max())
        m = 0.05 * (xmax - xmin) if xmax > xmin else 0.1
        line = np.linspace(xmin - m, xmax + m, 100)
        ax.errorbar(x, yp, yerr=err, fmt="o", ms=4, alpha=0.5, color="tab:blue",
                    elinewidth=0.5, capsize=1)
        ax.plot(line, line, "r--", lw=2)
        ax.set_xlabel("Truth"); ax.set_ylabel("Prediction")
        ax.set_title(f"{label}  RMSE={rmse:.4f}  R²={r2:.3f}")
        ax.set_xlim(xmin - m, xmax + m); ax.set_ylim(xmin - m, xmax + m)
        ax.grid(alpha=0.3)

    stat = "  ".join(f"{n}: RMSE={r:.4f}, R²={s:.3f}"
                     for n, r, s in zip(names, rmse_per, r2_per))
    plt.suptitle(f"NF in-distribution ({mode}, multi-suite val N={len(dm.val_ds)})\n{stat}",
                 fontsize=13)
    plt.tight_layout()
    out = args.out or f"nf_indist_{mode}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}\n{stat}")


if __name__ == "__main__":
    main()
