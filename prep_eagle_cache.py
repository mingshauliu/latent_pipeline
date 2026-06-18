"""Pre-process + normalise the held-out Swift-EAGLE L25 (LH 128^3) Mgas cubes.

Swift-EAGLE is a feedback model NOT in the FM/NF training pool (IllustrisTNG,
Astrid, SIMBA) -> use it as an out-of-distribution held-out test (4th/5th
feedback model alongside Magneticum).

Source = the freshly voxelised per-cube grids from `voxelise_clean`:
    /mnt/home/mliu1/ceph/CAMELS-L25n256/Swift-EAGLE/Mgas_128_LH_{i}_z=0.0.npy
    (1000 cubes, 128^3, float64, raw Msun/h per voxel).

UNIT FIX: this voxelisation is low by exactly (128/25)^3 = 134.2177 relative to
the CAMELS public training grids (a cells-per-Mpc/h voxel-volume factor). We
multiply by VOL_FIX so the gas-mass scale matches the training distribution
(log1p mean ~22, matching norm3d 'gas' mean 22.003). Without it the cubes would
be ~5 log-units low -> a fake unit-OOD, not a feedback-OOD.

Two artifacts written, both with the SAME recipe as prep_magneticum_cache.py:
  1. RAW-corrected stacked array (training units) for `nf.ood` (which applies
     log1p + norm3d 'gas' z-score internally):
        <RAW_OUT>/Grids_Mgas_Swift-EAGLE_LH_128_z=0.0.npy  (N,128,128,128) f32
  2. Pre-normalised held-out cache (log1p + norm3d 'gas' z-score; cosmo
     z-scored with norm3d 'param'[:2]):
        <OUT>/Mgas_norm_Swift-EAGLE_LH_128_z=0.0.npy  (N,128,128,128) f32
        <OUT>/cosmo_Swift-EAGLE.npy                    (N,2)
        <OUT>/norm_meta.npz

    python prep_eagle_cache.py
"""

import argparse
import os
import numpy as np
from numpy.lib.format import open_memmap

SRC_DIR = "/mnt/home/mliu1/ceph/CAMELS-L25n256/Swift-EAGLE"
SRC_TMPL = "Mgas_128_LH_{i}_z=0.0.npy"
PARAM = "/mnt/ceph/users/mliu1/CAMELS-L25n256/EAGLE/params_LH_EAGLE.txt"
NORM3D = "/mnt/home/mliu1/ceph/norm3d.npy"
RAW_OUT = "/mnt/ceph/users/mliu1/CAMELS-L25n256"   # raw-corrected stacked for nf.ood
RAW_NAME = "Grids_Mgas_Swift-EAGLE_LH_128_z=0.0.npy"
OUT_DIR = "/mnt/ceph/users/mliu1/latent_pipeline_cache/L25_LH_128_norm_heldout_EAGLE"
GRID = 128
N_LH = 1000
VOL_FIX = (128.0 / 25.0) ** 3   # 134.2177 voxel-volume unit correction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--raw_out", default=RAW_OUT)
    ap.add_argument("--n", type=int, default=N_LH)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--no_raw", action="store_true", help="skip raw stacked array")
    args = ap.parse_args()

    norm = np.load(NORM3D, allow_pickle=True).item()
    mg_m, mg_s = float(norm["gas"]["mean"]), float(norm["gas"]["std"])
    cos_m = np.asarray(norm["param"]["mean"][:2], dtype=np.float32)
    cos_s = np.asarray(norm["param"]["std"][:2], dtype=np.float32)
    print(f"VOL_FIX = (128/25)^3 = {VOL_FIX:.4f}")
    print(f"norm3d gas mean/std = {mg_m:.4f}/{mg_s:.4f}")
    print(f"norm3d param[:2] (Om,s8) mean {cos_m} std {cos_s}")

    params = np.loadtxt(PARAM).astype(np.float32)
    n = min(args.n, len(params))
    os.makedirs(args.out, exist_ok=True)

    norm_path = os.path.join(args.out, "Mgas_norm_Swift-EAGLE_LH_128_z=0.0.npy")
    mg_norm = open_memmap(norm_path, mode="w+", dtype=np.float32, shape=(n, GRID, GRID, GRID))
    if not args.no_raw:
        os.makedirs(args.raw_out, exist_ok=True)
        raw_path = os.path.join(args.raw_out, RAW_NAME)
        mg_raw = open_memmap(raw_path, mode="w+", dtype=np.float32, shape=(n, GRID, GRID, GRID))
    print(f"\n[Swift-EAGLE] {n} Mgas cubes -> norm {os.path.basename(norm_path)}"
          + ("" if args.no_raw else f" + raw {RAW_NAME}"))

    s = q = 0.0
    for i in range(n):
        src = os.path.join(SRC_DIR, SRC_TMPL.format(i=i))
        cube = np.load(src).astype(np.float32) * np.float32(VOL_FIX)
        if not args.no_raw:
            mg_raw[i] = cube
        mgn = ((np.log1p(cube) - np.float32(mg_m)) / np.float32(mg_s)).astype(np.float32)
        mg_norm[i] = mgn
        s += float(mgn.mean()); q += float((mgn ** 2).mean())
        if (i + 1) % args.log_every == 0:
            print(f"  Swift-EAGLE {i+1}/{n}", flush=True)
    mg_norm.flush(); del mg_norm
    if not args.no_raw:
        mg_raw.flush(); del mg_raw

    cosmo = ((params[:n, :2] - cos_m) / cos_s).astype(np.float32)
    np.save(os.path.join(args.out, "cosmo_Swift-EAGLE.npy"), cosmo)
    mean, std = s / n, (q / n - (s / n) ** 2) ** 0.5
    print(f"  [Swift-EAGLE] Mgas_norm pool mean~{mean:+.3f} std~{std:.3f} | "
          f"cosmo_norm mean {cosmo.mean(0)} std {cosmo.std(0)}")

    np.savez(os.path.join(args.out, "norm_meta.npz"),
             mgas_mean=np.float32(mg_m), mgas_std=np.float32(mg_s),
             cosmo_mean=cos_m, cosmo_std=cos_s,
             vol_fix=np.float32(VOL_FIX),
             source=np.array(NORM3D),
             transform=np.array("*(128/25)^3; log1p+zscore(gas); cosmo zscore(param[:2])"),
             field=np.array("Swift-EAGLE Mgas held-out; voxelise_clean per-cube 128^3 float64"),
             suites=np.array(["Swift-EAGLE"]), counts=np.array([n]))
    print(f"\nDone. Swift-EAGLE held-out cache -> {args.out}")
    if not args.no_raw:
        print(f"      raw-corrected stacked    -> {os.path.join(args.raw_out, RAW_NAME)}")


if __name__ == "__main__":
    main()
