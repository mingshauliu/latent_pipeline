"""Pre-normalise the held-out Magneticum L25 (LH 128^3) Mgas cubes.

Magneticum is a 4th feedback model NOT in the FM/NF training pool (IllustrisTNG,
Astrid, SIMBA) -> use it as an out-of-distribution held-out test. Normalised with
the SAME recipe as the training cache (prep_cache.py): log1p + norm3d 'gas'
z-score, so it is directly comparable to the cached training distribution. Cosmo
(Omega_m, sigma_8) z-scored with norm3d 'param'[:2].

    python prep_magneticum_cache.py

Output:
    <OUT>/Mgas_norm_Magneticum_LH_128_z=0.0.npy   (N,128,128,128) f32
    <OUT>/cosmo_Magneticum.npy                     (N,2)
    <OUT>/norm_meta.npz
"""

import argparse
import os
import numpy as np
from numpy.lib.format import open_memmap

MGAS = "/mnt/ceph/users/mliu1/CAMELS-L25n256/Grids_Mgas_Magneticum_LH_128_z=0.0.npy"
PARAM = "/mnt/ceph/users/mliu1/CAMELS-L25n256/params_LH_Magneticum.txt"
NORM3D = "/mnt/home/mliu1/ceph/norm3d.npy"
OUT_DIR = "/mnt/ceph/users/mliu1/latent_pipeline_cache/L25_LH_128_norm_heldout"
GRID = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    norm = np.load(NORM3D, allow_pickle=True).item()
    mg_m, mg_s = float(norm["gas"]["mean"]), float(norm["gas"]["std"])
    cos_m = np.asarray(norm["param"]["mean"][:2], dtype=np.float32)
    cos_s = np.asarray(norm["param"]["std"][:2], dtype=np.float32)
    print(f"norm3d gas mean/std = {mg_m:.4f}/{mg_s:.4f}")
    print(f"norm3d param[:2] (Om,s8) mean {cos_m} std {cos_s}")

    os.makedirs(args.out, exist_ok=True)
    mg_in = np.load(MGAS, mmap_mode="r")
    params = np.loadtxt(PARAM).astype(np.float32)
    n = min(len(mg_in), len(params))
    out_path = os.path.join(args.out, "Mgas_norm_Magneticum_LH_128_z=0.0.npy")
    mg_out = open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, GRID, GRID, GRID))
    print(f"\n[Magneticum] {n} Mgas cubes -> {os.path.basename(out_path)}")

    s = q = 0.0
    for i in range(n):
        mgn = ((np.log1p(np.asarray(mg_in[i]).astype(np.float32)) - np.float32(mg_m))
               / np.float32(mg_s)).astype(np.float32)
        mg_out[i] = mgn
        s += float(mgn.mean()); q += float((mgn ** 2).mean())
        if (i + 1) % args.log_every == 0:
            print(f"  Magneticum {i+1}/{n}", flush=True)
    mg_out.flush(); del mg_out

    cosmo = ((params[:n, :2] - cos_m) / cos_s).astype(np.float32)
    np.save(os.path.join(args.out, "cosmo_Magneticum.npy"), cosmo)
    mean, std = s / n, (q / n - (s / n) ** 2) ** 0.5
    print(f"  [Magneticum] Mgas_norm pool mean~{mean:+.3f} std~{std:.3f} | "
          f"cosmo_norm mean {cosmo.mean(0)} std {cosmo.std(0)}")

    np.savez(os.path.join(args.out, "norm_meta.npz"),
             mgas_mean=np.float32(mg_m), mgas_std=np.float32(mg_s),
             cosmo_mean=cos_m, cosmo_std=cos_s,
             source=np.array(NORM3D),
             transform=np.array("log1p+zscore(gas); cosmo zscore(param[:2])"),
             field=np.array("Magneticum Mgas held-out; 128^3 raw float64 source"),
             suites=np.array(["Magneticum"]), counts=np.array([n]))
    print(f"\nDone. Magneticum held-out cache -> {args.out}")


if __name__ == "__main__":
    main()
