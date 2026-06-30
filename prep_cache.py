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
# N-body DM peculiar-velocity grids voxelised by voxelise/voxelise_nbody_vel.py
# (mass-weighted mean |v| per voxel, km/s; --stack output). Extra FM input channel.
NBODYVEL_TMPL = "/mnt/home/mliu1/ceph/CAMELS-L25n256/Nbody/Grids_Vcdm_Nbody_{suite}_LH_128_z=0.0.npy"
# Electron density — already public in the CMD release (NO voxelisation needed; same
# dir as Mtot/Mgas). Extra FM OUTPUT (target) channel for the multi-task (Mgas + ne)
# variant. Values are tiny number densities (~1e-7 h^2/cm^3) -> norm = LOG10 (NOT log1p,
# which is degenerate at that scale) + shared norm3d 'ne' z-score (mean -6.96/std 0.57;
# pool log10 mean -6.94/std 0.56 matches; zscored peak ~8.8 < clamp 10).
NE_TMPL = "/mnt/home/camels/ceph/PUBLIC_RELEASE/CMD/3D_grids/data/{suite}/Grids_ne_{suite}_LH_128_z=0.0.npy"
# Temperature — also public in the CMD release (NO voxelisation). Extra FM OUTPUT
# (target) channel for the full multi-task [Mgas, ne, T] (encoder3D parity). Raw T is
# Kelvin (1e3-1e7) -> norm = LOG1P + z-score. The norm3d 'T' stat (10.08/3.12) is a
# DIFFERENT field (mass-weighted) -> using it leaves pool mean -0.16/std 0.90, so we
# compute a FRESH pool stat over the public Grids_T (pool log1p ~9.57/2.80). NO suite
# div (T intensive), NO overdensity.
T_TMPL = "/mnt/home/camels/ceph/PUBLIC_RELEASE/CMD/3D_grids/data/{suite}/Grids_T_{suite}_LH_128_z=0.0.npy"
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


# N-body velocity is a mass-weighted MEAN |v| (intensive, km/s) -> NO suite volume/mass
# correction (unlike the Mtot SUM which needed the Astrid (128/25)^3 div). One global
# norm3d 'nbodyVel' stat (log1p mean 4.814 / std 0.617) covers all suites.
def normalise_vel(raw, mean, std):
    """log1p -> shared norm3d nbodyVel z-score. float32 out."""
    return ((np.log1p(raw.astype(np.float32)) - np.float32(mean)) / np.float32(std)).astype(np.float32)


def process_suite_vel(suite, vn_m, vn_s, out_dir, log_every=100):
    vn_in = np.load(NBODYVEL_TMPL.format(suite=suite), mmap_mode="r")
    n = len(vn_in)
    out_path = os.path.join(out_dir, f"NbodyVel_norm_{suite}_LH_128_z=0.0.npy")
    vn_out = open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, GRID, GRID, GRID))
    print(f"\n[{suite}] {n} NbodyVel cubes (log1p+norm3d nbodyVel) -> {os.path.basename(out_path)}")
    s = q = 0.0
    for i in range(n):
        vnn = normalise_vel(np.asarray(vn_in[i]), vn_m, vn_s)
        vn_out[i] = vnn
        s += float(vnn.mean()); q += float((vnn ** 2).mean())
        if (i + 1) % log_every == 0:
            print(f"  {suite} nbodyvel {i+1}/{n}", flush=True)
    vn_out.flush(); del vn_out
    mean, std = s / n, (q / n - (s / n) ** 2) ** 0.5
    print(f"  [{suite}] NbodyVel_norm pool mean~{mean:+.3f} std~{std:.3f} | wrote {out_path}")
    return n


# ── electron density (ne) — FM 2nd output channel ───────────────────────────────
# USER DECISION (2026-06-23): the public Grids_ne are tiny number densities (~1e-7),
# so log1p(ne)~ne~0 (std 0 -> dead target). Use LOG10 + the shared norm3d 'ne' z-score
# (mean -6.96/std 0.57; encoder3D-proven, matches the public-grid scale). NO suite div
# (ne is intensive). NE_FLOOR guards log10(0) at empty/SF voxels (set ne=0 -> floor).
NE_FLOOR = 1e-12


def normalise_ne(raw, mean, std):
    """log10 (clip to NE_FLOOR) then z-score with shared norm3d 'ne' stats. float32 out."""
    v = np.clip(raw.astype(np.float32), NE_FLOOR, None)
    return ((np.log10(v) - np.float32(mean)) / np.float32(std)).astype(np.float32)


# ── temperature (T) — FM 3rd output channel ─────────────────────────────────────
def normalise_temp(raw, mean, std):
    """log1p then z-score with the FRESH pool T stat. float32 out."""
    return ((np.log1p(raw.astype(np.float32)) - np.float32(mean)) / np.float32(std)).astype(np.float32)


def compute_temp_stat(suites, every=200):
    """Fresh pool log1p mean/std over the public Grids_T (norm3d T is a mismatched
    field). Subsamples cubes per suite for speed; stats are stable."""
    ms, vs = [], []
    for s in suites:
        a = np.load(T_TMPL.format(suite=s), mmap_mode="r")
        for i in range(0, len(a), every):
            l = np.log1p(np.asarray(a[i], dtype=np.float32))
            ms.append(float(l.mean())); vs.append(float(l.var()))
    m = float(np.mean(ms))
    sd = float(np.sqrt(np.mean(vs) + np.var(ms)))
    print(f"fresh pool T (log1p) mean/std = {m:.4f}/{sd:.4f} (no suite div; norm3d T NOT used)")
    return m, sd


def process_suite_temp(suite, t_m, t_s, out_dir, log_every=100):
    t_in = np.load(T_TMPL.format(suite=suite), mmap_mode="r")
    n = len(t_in)
    out_path = os.path.join(out_dir, f"T_norm_{suite}_LH_128_z=0.0.npy")
    t_out = open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, GRID, GRID, GRID))
    print(f"\n[{suite}] {n} T cubes (log1p+fresh pool z-score) -> {os.path.basename(out_path)}")
    s = q = 0.0
    amax = 0.0
    for i in range(n):
        tn = normalise_temp(np.asarray(t_in[i]), t_m, t_s)
        t_out[i] = tn
        s += float(tn.mean()); q += float((tn ** 2).mean())
        amax = max(amax, float(np.abs(tn).max()))
        if (i + 1) % log_every == 0:
            print(f"  {suite} T {i+1}/{n}", flush=True)
    t_out.flush(); del t_out
    mean, std = s / n, (q / n - (s / n) ** 2) ** 0.5
    print(f"  [{suite}] T_norm pool mean~{mean:+.3f} std~{std:.3f} max|val|~{amax:.2f} "
          f"| wrote {out_path}")


def process_suite_ne(suite, ne_m, ne_s, out_dir, log_every=100):
    ne_in = np.load(NE_TMPL.format(suite=suite), mmap_mode="r")
    n = len(ne_in)
    out_path = os.path.join(out_dir, f"Ne_norm_{suite}_LH_128_z=0.0.npy")
    ne_out = open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, GRID, GRID, GRID))
    print(f"\n[{suite}] {n} ne cubes (log10+norm3d ne) -> {os.path.basename(out_path)}")
    s = q = 0.0
    amax = 0.0
    for i in range(n):
        nen = normalise_ne(np.asarray(ne_in[i]), ne_m, ne_s)
        ne_out[i] = nen
        s += float(nen.mean()); q += float((nen ** 2).mean())
        amax = max(amax, float(np.abs(nen).max()))
        if (i + 1) % log_every == 0:
            print(f"  {suite} ne {i+1}/{n}", flush=True)
    ne_out.flush(); del ne_out
    mean, std = s / n, (q / n - (s / n) ** 2) ** 0.5
    # max|val| reported so we can check whether clamp_val=10 would truncate the ne
    # peak (cluster cores) -> would impede the encoder. If amax > clamp_val, raise the
    # target clamp or use a per-field clamp (tight on Nbody input, loose on targets).
    print(f"  [{suite}] Ne_norm pool mean~{mean:+.3f} std~{std:.3f} max|val|~{amax:.2f} "
          f"| wrote {out_path}")
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
    ap.add_argument("--no_nbody", action="store_true",
                    help="skip Nbody cache (log1p+norm3d z-score; not overdensity)")
    ap.add_argument("--no_mgas", action="store_true", help="skip Mgas cache (Nbody only)")
    ap.add_argument("--with_vel", action="store_true",
                    help="ALSO cache the N-body velocity channel (requires voxelise/"
                         "voxelise_nbody_vel.py --stack output to exist first)")
    ap.add_argument("--with_ne", action="store_true",
                    help="ALSO cache the electron-density (ne) target channel from the "
                         "public CMD Grids_ne (no voxelise). log10 + norm3d 'ne' z-score.")
    ap.add_argument("--with_temp", action="store_true",
                    help="ALSO cache the temperature (T) target channel from the public "
                         "CMD Grids_T (no voxelise). log1p + FRESH pool z-score.")
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

    # N-body velocity channel (opt-in): shared norm3d 'nbodyVel' stats, no suite div.
    vn_m = vn_s = None
    if args.with_vel:
        vn_m, vn_s = float(norm["nbodyVel"]["mean"]), float(norm["nbodyVel"]["std"])
        print(f"norm3d nbodyVel mean/std = {vn_m:.4f}/{vn_s:.4f} (no suite div)")

    # ne (opt-in): log10 + shared norm3d 'ne' z-score (public Grids_ne, no voxelise).
    ne_m = ne_s = None
    if args.with_ne:
        ne_m, ne_s = float(norm["ne"]["mean"]), float(norm["ne"]["std"])
        print(f"norm3d ne mean/std = {ne_m:.4f}/{ne_s:.4f} (log10, no suite div)")

    # T (opt-in): log1p + FRESH pool z-score (public Grids_T; norm3d T is a mismatched field).
    t_m = t_s = None
    if args.with_temp:
        t_m, t_s = compute_temp_stat(args.suites)

    counts = {}
    for s in args.suites:
        if not args.no_mgas:
            counts[s] = process_suite(s, mg_m, mg_s, cos_m, cos_s, args.out, args.log_every)
        if not args.no_nbody:
            counts[s] = process_suite_nbody(s, nb_m, nb_s, args.out, args.log_every)
        if args.with_vel:
            counts[s] = process_suite_vel(s, vn_m, vn_s, args.out, args.log_every)
        if args.with_ne:
            counts[s] = process_suite_ne(s, ne_m, ne_s, args.out, args.log_every)
        if args.with_temp:
            counts[s] = process_suite_temp(s, t_m, t_s, args.out, args.log_every)

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
    if vn_m is not None:
        meta["nbodyvel_mean"] = np.float32(vn_m)
        meta["nbodyvel_std"] = np.float32(vn_s)
        meta["nbodyvel_transform"] = np.array("log1p+zscore(norm3d nbodyVel); no suite div (mean |v|)")
    if ne_m is not None:
        meta["ne_mean"] = np.float32(ne_m)
        meta["ne_std"] = np.float32(ne_s)
        meta["ne_transform"] = np.array("log10+zscore(norm3d ne); no suite div; public Grids_ne")
    if t_m is not None:
        meta["t_mean"] = np.float32(t_m)
        meta["t_std"] = np.float32(t_s)
        meta["t_transform"] = np.array("log1p+zscore(FRESH pool stat); no suite div; public Grids_T")
    np.savez(os.path.join(args.out, "norm_meta.npz"), **meta)

    # Mirror stats into the repo-local norm_latent.npz so infer.py picks up the
    # SAME normalisation. MERGE-and-update (don't clobber): preserve existing keys and
    # overlay whatever this run computed -> e.g. `--with_vel --no_nbody` keeps the prior
    # nbody/mgas stats while adding nbodyvel.
    if nb_m is not None or vn_m is not None or ne_m is not None or t_m is not None:
        local = "cached/norm_latent.npz"
        os.makedirs(os.path.dirname(local), exist_ok=True)
        local_kw = {}
        if os.path.exists(local):
            local_kw = {k: v for k, v in np.load(local, allow_pickle=True).items()}
        # mgas/cosmo always available this run
        local_kw.update(mgas_mean=np.float32(mg_m), mgas_std=np.float32(mg_s),
                        cosmo_mean=cos_m, cosmo_std=cos_s)
        if nb_m is not None:
            local_kw.update(nbody_mean=np.float32(nb_m), nbody_std=np.float32(nb_s),
                            nbody_transform=np.array("scalar-corrected-norm3d"))
        if vn_m is not None:
            local_kw.update(nbodyvel_mean=np.float32(vn_m), nbodyvel_std=np.float32(vn_s))
        if ne_m is not None:
            local_kw.update(ne_mean=np.float32(ne_m), ne_std=np.float32(ne_s))
        if t_m is not None:
            local_kw.update(t_mean=np.float32(t_m), t_std=np.float32(t_s))
        np.savez(local, **local_kw)
        print(f"Wrote {local} (infer norm stats"
              + (" + nbodyvel" if vn_m is not None else "")
              + (" + ne" if ne_m is not None else "")
              + (" + T" if t_m is not None else "") + ")")
    print(f"\nDone. cache -> {args.out}")


if __name__ == "__main__":
    main()
