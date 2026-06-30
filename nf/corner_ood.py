"""Posterior CORNER plot for an NF OOD test (default held-out = Magneticum).

`nf.ood` only saves the posterior mean/std (Gaussian summary) -> can't corner from
its npz. This redraws the FULL posterior samples per cube (`flow.sample`) in PHYSICAL
(Omega_m, sigma_8) and makes a cosmology corner:

  --cube N  : single cube's posterior (n_samples draws) with truth crosshair.
  --cube -1 : POOLED corner of every cube's samples (aggregate OOD posterior) with all
              48 truths scattered on the joint panel -> shows the NF's posterior spread
              vs the Magneticum truth locus.

Reuses the exact OOD input pipeline (raw Mgas -> shared norm3d 'gas', physical-unit
flow draws), identical to nf.ood.

  python -m nf.corner_ood --config config/config.yaml \
      --checkpoint nf_ck_real/nf-752--5.4560.ckpt --tag real --cube -1
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

from data import nf_mgas_stats
from .module import LitNFRegressor, MultiSuiteMgasDataset
from .ood import load_heldout_pool, DEFAULT_HELDOUT


@torch.no_grad()
def draw_posteriors(model, loader, n_samples, dev):
    """-> samples (N_cubes, n_samples, P), truths (N_cubes, P)."""
    S, T = [], []
    for batch in loader:
        x, y = batch[0].to(dev), batch[1]
        summary, _ = model(x)
        s = model.flow.sample(summary, num_samples=n_samples).permute(1, 0, 2)  # (B,n,P)
        S.append(s.cpu().numpy())
        T.append(y.numpy())
    return np.concatenate(S, 0), np.concatenate(T, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", default="real")
    p.add_argument("--cube", type=int, default=-1,
                   help="cube index for a single-object posterior; -1 = pooled over all")
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--output_dir", default="nf_ood_test/outputs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--heldout_name", default=None)
    p.add_argument("--heldout_mgas", default=None)
    p.add_argument("--heldout_param", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    nf_cfg = cfg["nf"]; nf_data = nf_cfg["data"]; ic = nf_cfg.get("inference", {})
    ho = dict(nf_cfg.get("heldout", DEFAULT_HELDOUT))
    if args.heldout_name:  ho["name"] = args.heldout_name
    if args.heldout_mgas:  ho["mgas_path"] = args.heldout_mgas
    if args.heldout_param: ho["param_path"] = args.heldout_param
    names = ic.get("param_names") or [r"$\Omega_m$", r"$\sigma_8$"]

    dev = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.checkpoint} (tag={args.tag}) dev={dev}")
    model = LitNFRegressor.load_from_checkpoint(args.checkpoint, map_location=dev).eval().to(dev)
    for prm in model.parameters():
        prm.requires_grad_(False)

    mgas_src, cosmo_all, flat = load_heldout_pool(ho)
    mgas_mean, mgas_std = nf_mgas_stats(nf_data)
    ds = MultiSuiteMgasDataset(mgas_src, cosmo_all, flat, np.arange(len(flat)),
                               mgas_mean, mgas_std, augment=False)
    loader = DataLoader(ds, batch_size=ic.get("batch_size", 2), shuffle=False, num_workers=0)

    samples, truths = draw_posteriors(model, loader, args.n_samples, dev)  # (Nc,n,P),(Nc,P)
    name = ho.get("name", "heldout")
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)

    if args.cube >= 0:
        i = args.cube
        s, tr = samples[i], truths[i]
        fig = corner.corner(s, labels=names, bins=200, smooth=5, smooth1d=5,
                            show_titles=True, title_fmt=".3f", color="C0")
        fig.suptitle(f"{name} OOD posterior — {args.tag}-NF — cube {i} "
                     f"(truth Ωm={tr[0]:.3f}, σ8={tr[1]:.3f})", y=1.02)
        stem = f"corner_{name}_{args.tag}_cube{i}"
    else:
        pooled = samples.reshape(-1, samples.shape[-1])      # (Nc*n, P)
        fig = corner.corner(pooled, labels=names, color="C0", bins=200,
                            smooth=5, smooth1d=5,
                            levels=(0.6827, 0.955), plot_datapoints=False,
                            plot_density=False, fill_contours=True,
                            show_titles=True, title_fmt=".3f")
        fig.suptitle(f"{name} OOD pooled posterior — {args.tag}-NF "
                     f"(N={len(flat)} cubes × {args.n_samples} draws)", y=1.02)
        stem = f"corner_{name}_{args.tag}_pooled"

    png = out_root / f"{stem}.png"
    fig.savefig(png, dpi=ic.get("dpi", 150), bbox_inches="tight")
    np.savez(out_root / f"{stem}.npz", samples=samples, truths=truths,
             param_names=np.array(names))
    print(f"saved {png}\nsaved {out_root / (stem + '.npz')}")


if __name__ == "__main__":
    main()
