"""NF out-of-distribution (OOD) test branch.

Evaluate a trained NF (real or synth) on a held-out feedback model NOT in the
training pool (IllustrisTNG, Astrid, SIMBA). Default held-out = **Magneticum**
(4th feedback model). This is the OOD generalisation test:

    real-NF on Magneticum  -> BASELINE (how well does cosmo inference from real
                              CAMELS Mgas transfer to an unseen feedback model?)
    synth-NF on Magneticum -> later comparison (does FM-synth Mgas preserve the
                              same OOD-transferable cosmo signal?)

The NF input pipeline (`MultiSuiteMgasDataset`) takes RAW Mgas and applies the
shared norm3d 'gas' log1p+z-score internally — identical to how the held-out
cache (`prep_magneticum_cache.py`) was built — so we feed the RAW Magneticum
cubes here and reuse the exact training-time normalisation (`data.nf_mgas_stats`).
Targets are the PHYSICAL (Omega_m, sigma_8); the flow un-standardises its draws
via the target_mean/std buffers stored in the checkpoint, so predictions come
back in physical units directly comparable to the truth.

    python -m nf.ood --config config/config.yaml \
        --checkpoint nf_ck_real/nf-752--5.4560.ckpt --tag real
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from data import nf_mgas_stats
from .module import LitNFRegressor, MultiSuiteMgasDataset
from .predict import predict_with_uncertainty
from .infer import plot_truth_vs_pred


# default held-out source (mirrors prep_magneticum_cache.py constants)
DEFAULT_HELDOUT = {
    "name": "Magneticum",
    "mgas_path": "/mnt/ceph/users/mliu1/CAMELS-L25n256/Grids_Mgas_Magneticum_LH_128_z=0.0.npy",
    "param_path": "/mnt/ceph/users/mliu1/CAMELS-L25n256/params_LH_Magneticum.txt",
    "n_cosmo": 2,
}


def load_heldout_pool(ho):
    """Open the raw held-out Mgas mmap + physical cosmo. Returns
    (mgas_src, cosmo_all, flat) shaped for MultiSuiteMgasDataset (single suite)."""
    n_cosmo = ho.get("n_cosmo", 2)
    mg = np.load(ho["mgas_path"], mmap_mode="r")
    pa = np.loadtxt(ho["param_path"]).astype(np.float32)
    n = min(len(mg), len(pa))
    cosmo_all = pa[:n, :n_cosmo].astype(np.float32)
    flat = [(0, j) for j in range(n)]
    print(f"  [OOD {ho.get('name', 'heldout')}] {n} cubes | physical cosmo "
          f"Om[{cosmo_all[:,0].min():.3f},{cosmo_all[:,0].max():.3f}] "
          f"s8[{cosmo_all[:,1].min():.3f},{cosmo_all[:,1].max():.3f}]")
    return [mg], cosmo_all, flat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", default="real",
                   help="label for outputs (which NF training produced the ckpt)")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    nf_cfg = cfg["nf"]
    nf_data = nf_cfg["data"]
    ic = nf_cfg.get("inference", {})
    ho = nf_cfg.get("heldout", DEFAULT_HELDOUT)

    dev = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.checkpoint}  (tag={args.tag})  device={dev}")
    model = LitNFRegressor.load_from_checkpoint(args.checkpoint, map_location=dev).eval().to(dev)
    for prm in model.parameters():
        prm.requires_grad_(False)

    mgas_src, cosmo_all, flat = load_heldout_pool(ho)
    # shared norm3d 'gas' stats — identical normalisation to NF training input
    mgas_mean, mgas_std = nf_mgas_stats(nf_data)
    ds = MultiSuiteMgasDataset(mgas_src, cosmo_all, flat, np.arange(len(flat)),
                               mgas_mean, mgas_std, augment=False)
    loader = DataLoader(ds, batch_size=ic.get("batch_size", 2), shuffle=False, num_workers=0)

    y_true, y_mean, y_std, aux = predict_with_uncertainty(
        model, loader, ic.get("n_posterior_samples", 2000), dev)

    names = ic.get("param_names") or [r"$\Omega_m$", r"$\sigma_8$"]
    rmse = np.sqrt(np.mean((y_true - y_mean) ** 2, axis=0))
    ss_res = np.sum((y_true - y_mean) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(0)) ** 2, axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    out_root = Path(args.output_dir or "outputs/nf_ood")
    out_root.mkdir(parents=True, exist_ok=True)
    name = ho.get("name", "heldout")
    stem = f"ood_{name}_{args.tag}"
    np.savez(out_root / f"predictions_{stem}.npz",
             y_true=y_true, y_mean=y_mean, y_std=y_std, aux_pred=aux,
             rmse=rmse, r2=r2, param_names=np.array(names))
    nf_label = {"real": "real-NF", "synth": "synth-NF"}.get(args.tag, f"{args.tag}-NF")
    title = f"{name} held-out OOD test — {nf_label} baseline (N={len(flat)})"
    plot_truth_vs_pred(y_true, y_mean, y_std, out_root / f"{stem}.png",
                       names, dpi=ic.get("dpi", 150), title=title)
    print(f"\nOOD ({name}, NF={args.tag}, N={len(flat)}):")
    for nm, rm, rr in zip(names, rmse, r2):
        print(f"  {nm}: RMSE={rm:.4f}  R2={rr:.3f}")


if __name__ == "__main__":
    main()
