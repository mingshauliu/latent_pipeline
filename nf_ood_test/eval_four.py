"""Four truth-vs-pred plots for the IllustrisTNG-only NF (nf_ood_test/ckpt).

Same trained model evaluated on:
  1. IllustrisTNG  — IN-DISTRIBUTION (val split, held out from training)
  2. Astrid        — OOD (unseen feedback model)
  3. SIMBA         — OOD (unseen feedback model)
  4. Magneticum    — OOD held-out (4th feedback model, 48 cubes)

All inputs normalised identically (shared norm3d 'gas' log1p+z-score inside
MultiSuiteMgasDataset). Targets are physical (Omega_m, sigma_8); the flow
un-standardises its draws. Reuses nf.predict + nf.infer.plot_truth_vs_pred.

    python nf_ood_test/eval_four.py --checkpoint nf_ood_test/ckpt/nf-824--5.9440.ckpt
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from data import nf_mgas_stats
from nf.module import LitNFRegressor, MultiSuiteMgasDataset
from nf.predict import predict_with_uncertainty
from nf.infer import plot_truth_vs_pred

CAMELS = "/mnt/ceph/users/camels/PUBLIC_RELEASE/CMD/3D_grids/data/{s}/Grids_Mgas_{s}_LH_128_z=0.0.npy"
CAMELS_P = "/mnt/ceph/users/camels/PUBLIC_RELEASE/CMD/3D_grids/data/{s}/params_LH_{s}.txt"
MAGN = "/mnt/ceph/users/mliu1/CAMELS-L25n256/Grids_Mgas_Magneticum_LH_128_z=0.0.npy"
MAGN_P = "/mnt/ceph/users/mliu1/CAMELS-L25n256/params_LH_Magneticum.txt"


def pick_best_ckpt(ckpt_dir="nf_ood_test/ckpt"):
    """Lowest val/nll among nf-<ep>--<nll>.ckpt (training may still be running)."""
    best = bv = None
    for p in glob.glob(os.path.join(ckpt_dir, "nf-*.ckpt")):
        try:
            v = float(os.path.basename(p)[3:-5].split("-", 1)[1])
        except ValueError:
            continue
        if bv is None or v < bv:
            bv, best = v, p
    assert best, f"no nf-*.ckpt in {ckpt_dir}"
    return best


def val_indices(n_total, val_split, seed):
    """Reproduce MultiSuiteNFDataModule's val split (held-out from training)."""
    idx = np.random.RandomState(seed).permutation(n_total)
    n_val = max(1, int(n_total * val_split))
    return idx[:n_val]


def build_source(mgas_path, param_path, indices):
    mg = np.load(mgas_path, mmap_mode="r")
    pa = np.loadtxt(param_path).astype(np.float32)
    n = min(len(mg), len(pa))
    indices = np.asarray([i for i in indices if i < n])
    cosmo_all = pa[indices, :2].astype(np.float32)
    flat = [(0, int(i)) for i in indices]
    return [mg], cosmo_all, flat


def run_source(name, model, mgas_path, param_path, indices, mgas_mean, mgas_std,
               names, out_root, dev, n_post, dpi, in_dist):
    mgas_src, cosmo_all, flat = build_source(mgas_path, param_path, indices)
    ds = MultiSuiteMgasDataset(mgas_src, cosmo_all, flat, np.arange(len(flat)),
                               mgas_mean, mgas_std, augment=False)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    y_true, y_mean, y_std, aux = predict_with_uncertainty(model, loader, n_post, dev)

    rmse = np.sqrt(np.mean((y_true - y_mean) ** 2, axis=0))
    ss_res = np.sum((y_true - y_mean) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(0)) ** 2, axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    tag = "in-dist (val)" if in_dist else "OOD"
    title = f"{name} {tag} — TNG-only NF (N={len(flat)})"
    stem = f"four_{name}"
    np.savez(out_root / f"predictions_{stem}.npz",
             y_true=y_true, y_mean=y_mean, y_std=y_std, aux_pred=aux, rmse=rmse, r2=r2)
    plot_truth_vs_pred(y_true, y_mean, y_std, out_root / f"{stem}.png",
                       names, dpi=dpi, title=title)
    print(f"  {name:13s} N={len(flat):4d}  "
          f"Om: RMSE={rmse[0]:.4f} R2={r2[0]:+.3f} | s8: RMSE={rmse[1]:.4f} R2={r2[1]:+.3f}")
    return rmse, r2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="default: auto-pick lowest val/nll")
    p.add_argument("--config", default="nf_ood_test/config_tng.yaml")
    p.add_argument("--n_lh", type=int, default=200, help="samples per LH suite (TNG/Astrid/SIMBA)")
    p.add_argument("--n_post", type=int, default=2000)
    p.add_argument("--output_dir", default="nf_ood_test/outputs")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    nf_data = cfg["nf"]["data"]
    seed = cfg["nf"]["training"].get("seed", 42)
    val_split = nf_data.get("val_split", 0.2)
    dev = args.device if torch.cuda.is_available() else "cpu"
    names = [r"$\Omega_m$", r"$\sigma_8$"]

    ckpt = args.checkpoint or pick_best_ckpt()
    print(f"Loading {ckpt} on {dev}")
    model = LitNFRegressor.load_from_checkpoint(ckpt, map_location=dev).eval().to(dev)
    for prm in model.parameters():
        prm.requires_grad_(False)
    mgas_mean, mgas_std = nf_mgas_stats(nf_data)
    print(f"norm3d gas mean/std = {mgas_mean:.3f}/{mgas_std:.3f}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # IllustrisTNG IN-DIST = the held-out val split (not used in training)
    tng_val = val_indices(1000, val_split, seed)[:args.n_lh]
    print("\nSource summary:")
    run_source("IllustrisTNG", model, CAMELS.format(s="IllustrisTNG"),
               CAMELS_P.format(s="IllustrisTNG"), tng_val, mgas_mean, mgas_std,
               names, out_root, dev, args.n_post, 150, in_dist=True)
    # OOD LH suites: first n_lh cubes
    for s in ("Astrid", "SIMBA"):
        run_source(s, model, CAMELS.format(s=s), CAMELS_P.format(s=s),
                   np.arange(args.n_lh), mgas_mean, mgas_std,
                   names, out_root, dev, args.n_post, 150, in_dist=False)
    # OOD held-out Magneticum: all 48
    n_magn = len(np.load(MAGN, mmap_mode="r"))
    run_source("Magneticum", model, MAGN, MAGN_P, np.arange(n_magn),
               mgas_mean, mgas_std, names, out_root, dev, args.n_post, 150, in_dist=False)
    print(f"\n4 plots -> {out_root}/four_<suite>.png")


if __name__ == "__main__":
    main()
