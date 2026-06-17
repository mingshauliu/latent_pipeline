"""Pre-normalise the L25 (LH 128^3) Mgas (baryon) cubes for IllustrisTNG, Astrid,
SIMBA and write them to a systematic ceph cache.

Only the BARYON field (Mgas) is cached — the Nbody (DM-only) field is left alone
in its source dir (/mnt/home/camels/ceph/.../Nbody/), shared across suites.

Normalization (per /mnt/home/mliu1/ceph/norm3d.npy, log1p):
    Mgas_norm = (log1p(Mgas) - norm['gas']['mean']) / norm['gas']['std']

Streamed cube-by-cube via np.lib.format.open_memmap (low RAM). Output layout:

    <OUT>/
      Mgas_norm_IllustrisTNG_LH_128_z=0.0.npy   (1000,128,128,128) f32
      Mgas_norm_Astrid_LH_128_z=0.0.npy         (1000,128,128,128) f32
      Mgas_norm_SIMBA_LH_128_z=0.0.npy          (1000,128,128,128) f32
      cosmo_{suite}.npy                          (1000,2)  Omega_m, sigma_8
      norm_meta.npz                              gas stats + provenance

    python prep_cache.py
    python prep_cache.py --suites SIMBA
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


def overdensity_log1p(raw):
    """Nbody Mtot -> log1p(rho / rho_bar_cube). Per-cube mean division makes the
    transform UNIT-FREE and suite-invariant (Astrid Nbody is ~192x off-scale vs
    TNG/SIMBA purely from particle-mass units; overdensity removes it). See CLAUDE.md."""
    c = raw.astype(np.float32)
    m = np.float32(c.mean()) + np.float32(1e-8)
    return np.log1p(c / m).astype(np.float32)


def normalise_nbody(raw, mean, std):
    """overdensity log1p then global z-score. float32 out."""
    return ((overdensity_log1p(raw) - np.float32(mean)) / np.float32(std)).astype(np.float32)


def estimate_nbody_od_stats(suites, n_per_suite=48, seed=0):
    """Global (mean,std) of log1p(1+delta) over a subset across suites. Pooled std."""
    rng = np.random.RandomState(seed)
    ms, vs = [], []
    for suite in suites:
        a = np.load(NBODY_TMPL.format(suite=suite), mmap_mode="r")
        pick = rng.choice(len(a), size=min(n_per_suite, len(a)), replace=False)
        for i in pick:
            d = overdensity_log1p(np.asarray(a[i]))
            ms.append(float(d.mean())); vs.append(float(d.var()))
    mean = float(np.mean(ms))
    std = float(np.sqrt(np.mean(vs) + np.var(ms)) + 1e-8)
    return mean, std


def process_suite_nbody(suite, nb_m, nb_s, out_dir, log_every=100):
    nb_in = np.load(NBODY_TMPL.format(suite=suite), mmap_mode="r")
    n = len(nb_in)
    out_path = os.path.join(out_dir, f"Nbody_norm_{suite}_LH_128_z=0.0.npy")
    nb_out = open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, GRID, GRID, GRID))
    print(f"\n[{suite}] {n} Nbody cubes (overdensity) -> {os.path.basename(out_path)}")
    s = q = 0.0
    for i in range(n):
        nbn = normalise_nbody(np.asarray(nb_in[i]), nb_m, nb_s)
        nb_out[i] = nbn
        s += float(nbn.mean()); q += float((nbn ** 2).mean())
        if (i + 1) % log_every == 0:
            print(f"  {suite} nbody {i+1}/{n}", flush=True)
    nb_out.flush(); del nb_out
    mean, std = s / n, (q / n - (s / n) ** 2) ** 0.5
    print(f"  [{suite}] Nbody_norm pool mean~{mean:+.3f} std~{std:.3f} | wrote {out_path}")
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

    # Nbody overdensity stats (computed once over the pool; suite-invariant)
    nb_m = nb_s = None
    if not args.no_nbody:
        nb_m, nb_s = estimate_nbody_od_stats(args.suites)
        print(f"Nbody overdensity log1p(1+delta) global mean/std = {nb_m:.4f}/{nb_s:.4f}")

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
        transform=np.array("Mgas: log1p+zscore(norm3d gas); Nbody: overdensity log1p(1+delta)+global zscore; cosmo: zscore(param[:2])"),
        field=np.array("Mgas + Nbody(overdensity) both cached"),
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
                 nbody_transform=np.array("overdensity"))
        print(f"Wrote {local} (nbody overdensity stats for infer)")
    print(f"\nDone. cache -> {args.out}")


if __name__ == "__main__":
    main()
