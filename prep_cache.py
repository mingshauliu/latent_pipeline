"""Pre-normalise the L25 (LH 128^3) Nbody + Mgas cubes for IllustrisTNG, Astrid,
SIMBA and write them to a systematic ceph cache.

Normalization (per /mnt/home/mliu1/ceph/norm3d.npy, log1p):
    Mgas_norm  = (log1p(Mgas) - norm['gas']['mean']) / norm['gas']['std']
    Nbody_norm = (log1p(Mtot / suite_div) - norm['nbody']['mean']) / norm['nbody']['std']
                 where suite_div = (128/25)^3 for Astrid (missing volume factor), 1 else
    cosmo      = (params[:, :2] - norm['param']['mean'][:2]) / norm['param']['std'][:2]

Streamed cube-by-cube via np.lib.format.open_memmap (low RAM). Output layout:

    <OUT>/
      Nbody_norm_{suite}_LH_128_z=0.0.npy   (1000,128,128,128) f32
      Mgas_norm_{suite}_LH_128_z=0.0.npy    (1000,128,128,128) f32
      cosmo_{suite}.npy                      (1000,2)  Omega_m, sigma_8
      norm_meta.npz                          stats + provenance

    python prep_cache.py                 # Nbody + Mgas + cosmo
    python prep_cache.py --no_mgas       # Nbody only (keep existing Mgas/cosmo)
"""

import argparse
import os
import numpy as np
from numpy.lib.format import open_memmap

MGAS_TMPL = "/mnt/ceph/users/camels/PUBLIC_RELEASE/CMD/3D_grids/data/{suite}/Grids_Mgas_{suite}_LH_128_z=0.0.npy"
NBODY_TMPL = "/mnt/home/camels/ceph/PUBLIC_RELEASE/CMD/3D_grids/data/Nbody/Grids_Mtot_Nbody_{suite}_LH_128_z=0.0.npy"
PARAM_TMPL = "/mnt/ceph/users/camels/PUBLIC_RELEASE/CMD/3D_grids/data/{suite}/params_LH_{suite}.txt"
NORM3D = "/mnt/home/mliu1/ceph/norm3d.npy"
OUT_DIR = "/mnt/ceph/users/mliu1/latent_pipeline_cache/L25_LH_128_norm"
GRID = 128


def normalise_mgas(raw, mean, std):
    """log1p then z-score with shared norm3d gas stats. float32 out."""
    return ((np.log1p(raw.astype(np.float32)) - np.float32(mean)) / np.float32(std)).astype(np.float32)


# Astrid's Mtot grid is missing the (N/L)^3 = (128/25)^3 volume factor -> raw values
# ~134.22x too large (confirmed: mean/Omega_m is 134.22x TNG/SIMBA, 0% scatter).
# Divide it back to TNG/SIMBA units, THEN apply the shared log1p + norm3d 'nbody'.
NBODY_SUITE_DIV = {"Astrid": (128.0 / 25.0) ** 3}   # = 134.217728; others default 1.0


def normalise_nbody(raw, div, mean, std):
    """Astrid (128/25)^3 unit correction -> log1p -> shared norm3d nbody z-score."""
    c = raw.astype(np.float32)
    if div != 1.0:
        c = c / np.float32(div)
    return ((np.log1p(c) - np.float32(mean)) / np.float32(std)).astype(np.float32)


def process_suite_nbody(suite, nb_m, nb_s, out_dir, log_every=100):
    nb_in = np.load(NBODY_TMPL.format(suite=suite), mmap_mode="r")
    n = len(nb_in)
    div = float(NBODY_SUITE_DIV.get(suite, 1.0))
    out_path = os.path.join(out_dir, f"Nbody_norm_{suite}_LH_128_z=0.0.npy")
    nb_out = open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, GRID, GRID, GRID))
    print(f"\n[{suite}] {n} Nbody cubes (raw/{div:g} -> log1p+norm3d) -> {os.path.basename(out_path)}")
    s = q = 0.0
    for i in range(n):
        nbn = normalise_nbody(np.asarray(nb_in[i]), div, nb_m, nb_s)
        nb_out[i] = nbn
        s += float(nbn.mean()); q += float((nbn ** 2).mean())
        if (i + 1) % log_every == 0:
            print(f"  {suite} nbody {i+1}/{n}", flush=True)
    nb_out.flush(); del nb_out
    mean, std = s / n, (q / n - (s / n) ** 2) ** 0.5
    print(f"  [{suite}] Nbody_norm pool mean~{mean:+.3f} std~{std:.3f} (div {div:g}) | wrote {out_path}")
    return n


def process_suite(suite, mg_m, mg_s, cos_m, cos_s, out_dir, log_every=100):
    mg_in = np.load(MGAS_TMPL.format(suite=suite), mmap_mode="r")
    params = np.loadtxt(PARAM_TMPL.format(suite=suite)).astype(np.float32)
    n = min(len(mg_in), len(params))
    out_path = os.path.join(out_dir, f"Mgas_norm_{suite}_LH_128_z=0.0.npy")
    mg_out = open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, GRID, GRID, GRID))
    print(f"\n[{suite}] {n} Mgas cubes -> {os.path.basename(out_path)}")

    s = q = 0.0
    for i in range(n):
        mgn = normalise_mgas(np.asarray(mg_in[i]), mg_m, mg_s)
        mg_out[i] = mgn
        s += float(mgn.mean()); q += float((mgn ** 2).mean())
        if (i + 1) % log_every == 0:
            print(f"  {suite} {i+1}/{n}", flush=True)
    mg_out.flush(); del mg_out
    # cosmo (Omega_m, sigma_8): normalise first 2 params with norm3d['param']
    cosmo = ((params[:n, :2] - cos_m) / cos_s).astype(np.float32)
    np.save(os.path.join(out_dir, f"cosmo_{suite}.npy"), cosmo)
    mean, std = s / n, (q / n - (s / n) ** 2) ** 0.5
    print(f"  [{suite}] Mgas_norm pool mean~{mean:+.3f} std~{std:.3f} | cosmo_norm "
          f"mean {cosmo.mean(0)} std {cosmo.std(0)} | wrote {out_path}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=["IllustrisTNG", "Astrid", "SIMBA"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--no_nbody", action="store_true", help="skip Nbody overdensity cache")
    ap.add_argument("--no_mgas", action="store_true", help="skip Mgas cache (Nbody only)")
    args = ap.parse_args()

    norm = np.load(NORM3D, allow_pickle=True).item()
    mg_m, mg_s = float(norm["gas"]["mean"]), float(norm["gas"]["std"])
    cos_m = np.asarray(norm["param"]["mean"][:2], dtype=np.float32)
    cos_s = np.asarray(norm["param"]["std"][:2], dtype=np.float32)
    print(f"norm3d gas mean/std = {mg_m:.4f}/{mg_s:.4f}")
    print(f"norm3d param[:2] (Om,s8) mean {cos_m} std {cos_s}")

    os.makedirs(args.out, exist_ok=True)

    # Nbody: shared norm3d 'nbody' stats (Astrid unit-corrected by (128/25)^3 first)
    nb_m = nb_s = None
    if not args.no_nbody:
        nb_m, nb_s = float(norm["nbody"]["mean"]), float(norm["nbody"]["std"])
        print(f"norm3d nbody mean/std = {nb_m:.4f}/{nb_s:.4f} | Astrid div = {NBODY_SUITE_DIV['Astrid']:.6f}")

    counts = {}
    for s in args.suites:
        if not args.no_mgas:
            counts[s] = process_suite(s, mg_m, mg_s, cos_m, cos_s, args.out, args.log_every)
        if not args.no_nbody:
            counts[s] = process_suite_nbody(s, nb_m, nb_s, args.out, args.log_every)

    meta = dict(
        mgas_mean=np.float32(mg_m), mgas_std=np.float32(mg_s),
        cosmo_mean=cos_m, cosmo_std=cos_s,
        source=np.array(NORM3D),
        transform=np.array("Mgas: log1p+zscore(norm3d gas); Nbody: Astrid/(128/25)^3 then log1p+zscore(norm3d nbody); cosmo: zscore(param[:2])"),
        field=np.array("Mgas + Nbody(scalar-corrected norm3d) both cached"),
        astrid_nbody_div=np.float32(NBODY_SUITE_DIV["Astrid"]),
        suites=np.array(args.suites), counts=np.array([counts[s] for s in args.suites]))
    if nb_m is not None:
        meta["nbody_mean"] = np.float32(nb_m)
        meta["nbody_std"] = np.float32(nb_s)
    np.savez(os.path.join(args.out, "norm_meta.npz"), **meta)

    # Mirror stats into the repo-local norm_latent.npz so infer.py picks up the
    # SAME normalisation (nbody overdensity, mgas norm3d gas, cosmo param[:2]).
    if nb_m is not None:
        local = "cached/norm_latent.npz"
        os.makedirs(os.path.dirname(local), exist_ok=True)
        np.savez(local,
                 nbody_mean=np.float32(nb_m), nbody_std=np.float32(nb_s),
                 mgas_mean=np.float32(mg_m), mgas_std=np.float32(mg_s),
                 cosmo_mean=cos_m, cosmo_std=cos_s,
                 nbody_transform=np.array("scalar-corrected-norm3d"))
        print(f"Wrote {local} (nbody scalar-corrected norm3d stats for infer)")
    print(f"\nDone. cache -> {args.out}")


if __name__ == "__main__":
    main()
