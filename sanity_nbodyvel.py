"""Sanity for NbodyVel stack + cache. Run after voxelise --stack and prep_cache --with_vel."""
import numpy as np
ST = "/mnt/home/mliu1/ceph/CAMELS-L25n256/Nbody/Grids_Vcdm_Nbody_{}_LH_128_z=0.0.npy"
CA = "/mnt/ceph/users/mliu1/latent_pipeline_cache/L25_LH_128_norm/NbodyVel_norm_{}_LH_128_z=0.0.npy"
CLAMP = 10.0
for s in ["IllustrisTNG", "Astrid", "SIMBA"]:
    raw = np.load(ST.format(s), mmap_mode="r")
    nrm = np.load(CA.format(s), mmap_mode="r")
    idx = [0, 500, 999]
    r = np.asarray(raw[idx], np.float32); n = np.asarray(nrm[idx], np.float32)
    print(f"[{s}] raw shape {raw.shape} dtype {raw.dtype} | norm shape {nrm.shape}")
    print(f"   raw |v| km/s: min {r.min():.3f} max {r.max():.2f} mean {r.mean():.3f} "
          f"nan {np.isnan(r).any()} neg {(r<0).any()}")
    print(f"   norm: mean {n.mean():+.3f} std {n.std():.3f} max|val| {np.abs(n).max():.3f} "
          f"(clamp {CLAMP}) -> {'OK' if np.abs(n).max()<=CLAMP else 'CLIPPED!'}")
meta = np.load("cached/norm_latent.npz")
print("norm_latent nbodyvel stats:",
      "mean", float(meta["nbodyvel_mean"]) if "nbodyvel_mean" in meta else "MISSING",
      "std", float(meta["nbodyvel_std"]) if "nbodyvel_std" in meta else "MISSING")
