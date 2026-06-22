"""Multi-suite CAMELS data for the FM branch.

Concatenates Nbody(Mtot) -> Mgas pairs over several hydro suites (IllustrisTNG,
Astrid, SIMBA), conditioned on (Omega_m, sigma_8).

Normalization is done LAZILY in __getitem__ (log1p + global z-score) using stats
computed once over a subset of the pool — see `compute_field_stats`. (Writing
pre-normalised cubes to a ceph cache is a deferred I/O optimisation; see CLAUDE.md.)
"""

import glob
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_cube(entry):
    """Read one cube. `entry` is either an mmap-slice/array or a path string
    (synth sample_*.npy). Returns float32 ndarray."""
    if isinstance(entry, (str, Path)):
        return np.load(entry).astype(np.float32)
    return np.asarray(entry, dtype=np.float32)


# ── suite path resolution ─────────────────────────────────────────────────────

def resolve_suite_paths(d, suite):
    """Format the per-suite nbody/mgas/param paths from config templates.

    Config provides `nbody_path_tmpl`, `mgas_path_tmpl`, `param_path_tmpl` with a
    `{suite}` placeholder. Returns (nbody_path, mgas_path, param_path).
    """
    return (
        d["nbody_path_tmpl"].format(suite=suite),
        d["mgas_path_tmpl"].format(suite=suite),
        d["param_path_tmpl"].format(suite=suite),
    )


def load_suite_pool(d):
    """Open mmaps + params for every suite. Returns:

    nbody_arrs : list[np.memmap]   per-suite Nbody cubes
    mgas_arrs  : list[np.memmap]   per-suite Mgas cubes
    cosmo_all  : (N_total, 2) float32   stacked (Omega_m, sigma_8)
    flat       : list[(suite_idx, local_idx)]   global -> (suite, local)
    """
    suites = d["suites"]
    n_cosmo = d.get("n_cosmo", 2)
    nbody_arrs, mgas_arrs, cosmo_parts, flat = [], [], [], []
    for si, suite in enumerate(suites):
        nb_p, mg_p, pa_p = resolve_suite_paths(d, suite)
        nb = np.load(nb_p, mmap_mode="r")
        mg = np.load(mg_p, mmap_mode="r")
        pa = np.loadtxt(pa_p).astype(np.float32)
        n = min(len(nb), len(mg), len(pa))
        nbody_arrs.append(nb)
        mgas_arrs.append(mg)
        cosmo_parts.append(pa[:n, :n_cosmo])
        flat.extend([(si, j) for j in range(n)])
        print(f"  [{suite}] nbody {nb.shape} mgas {mg.shape} params {pa.shape} -> use {n}")
    cosmo_all = np.concatenate(cosmo_parts, axis=0).astype(np.float32)
    print(f"  pool total = {len(flat)} pairs over {len(suites)} suites")
    return nbody_arrs, mgas_arrs, cosmo_all, flat


# ── normalization stats ───────────────────────────────────────────────────────

def compute_field_stats(arrs, flat, n_sample=64, seed=0):
    """log1p + global (mean,std) over a random subset of cubes across suites."""
    rng = np.random.RandomState(seed)
    pick = rng.choice(len(flat), size=min(n_sample, len(flat)), replace=False)
    vals = []
    for k in pick:
        si, li = flat[k]
        v = np.log1p(read_cube(arrs[si][li]))
        vals.append((float(v.mean()), float(v.var())))
    means = np.array([m for m, _ in vals])
    vars = np.array([v for _, v in vals])
    mean = float(means.mean())
    # pooled std: E[var] + var(means)
    std = float(np.sqrt(vars.mean() + means.var()) + 1e-8)
    return mean, std


def load_nf_pool(nfd):
    """Load the NF 128^3 Mgas pool for `mode` in {real, synth}.

    real  : mmap per-suite Grids_Mgas_{suite}_LH_128 cubes.
    synth : per-suite dir of FM-synth sample_*.npy (numeric-sorted).
    Returns (mgas_src, cosmo_all, flat) where mgas_src[si] is mmap (real) or a
    list of paths (synth); flat maps global -> (suite, local).
    """
    mode = nfd.get("mode", "real")
    suites = nfd["suites"]
    n_cosmo = nfd.get("n_cosmo", 2)
    mgas_src, cosmo_parts, flat = [], [], []
    for si, suite in enumerate(suites):
        pa = np.loadtxt(nfd["param_path_tmpl"].format(suite=suite)).astype(np.float32)
        if mode == "real":
            src = np.load(nfd["mgas_path_tmpl"].format(suite=suite), mmap_mode="r")
            n = min(len(src), len(pa))
        elif mode == "synth":
            sdir = nfd["synth_dir_tmpl"].format(suite=suite)
            src = sorted(glob.glob(os.path.join(sdir, "sample_*.npy")),
                         key=lambda p: int(re.search(r"sample_(\d+)", os.path.basename(p)).group(1)))
            if not src:
                raise FileNotFoundError(f"no sample_*.npy in {sdir}")
            n = min(len(src), len(pa))
        else:
            raise ValueError(f"unknown nf data mode {mode}")
        mgas_src.append(src)
        cosmo_parts.append(pa[:n, :n_cosmo])
        flat.extend([(si, j) for j in range(n)])
        print(f"  [NF/{mode} {suite}] {n} cubes")
    cosmo_all = np.concatenate(cosmo_parts, axis=0).astype(np.float32)
    print(f"  NF pool total = {len(flat)} ({mode})")
    return mgas_src, cosmo_all, flat


def nf_mgas_stats(nfd):
    """NF Mgas normalization stats. 128^3 paradigm: reuse the SHARED norm3d 'gas'
    stats (mean 22.003, std 1.309) so the FM cache (prep_cache.py), the Magneticum
    held-out cache (prep_magneticum_cache.py), and the NF input all log1p+z-score
    identically. (256^3 used pool-computed stats; dropped with the 128^3 switch.)"""
    path = nfd.get("norm3d_path", "/mnt/home/mliu1/ceph/norm3d.npy")
    norm = np.load(path, allow_pickle=True).item()
    return float(norm["gas"]["mean"]), float(norm["gas"]["std"])


def compute_velocity_stats(nbody_arrs, mgas_arrs, flat, n_sample=64, seed=0,
                           clamp_val=10.0):
    """Pooled (mean,std) of the FM 'velocity' field v = Mgas_norm - Nbody_norm over a
    random subset of cubes. Both fields are read from the PRE-NORMED FM cache (already
    log1p+z-scored) and clamped to +/-clamp_val BEFORE differencing, matching exactly
    what the FM sees (data.CachedFMDataset clamps both fields). NO log1p here -- v is a
    difference of z-scored fields and is signed (can be negative), so log1p is invalid.
    """
    rng = np.random.RandomState(seed)
    pick = rng.choice(len(flat), size=min(n_sample, len(flat)), replace=False)
    means, vars_ = [], []
    for k in pick:
        si, li = flat[k]
        nb = np.array(nbody_arrs[si][li], dtype=np.float32)
        mg = np.array(mgas_arrs[si][li], dtype=np.float32)
        if clamp_val is not None:
            c = float(clamp_val)
            np.clip(nb, -c, c, out=nb)
            np.clip(mg, -c, c, out=mg)
        v = mg - nb
        means.append(float(v.mean()))
        vars_.append(float(v.var()))
    means = np.array(means); vars_ = np.array(vars_)
    mean = float(means.mean())
    std = float(np.sqrt(vars_.mean() + means.var()) + 1e-8)  # pooled std
    return mean, std


def velocity_stats(d, cache_path="cached/norm_velocity.npz", clamp_val=10.0):
    """Load cached velocity (mean,std) from `cache_path`, else compute from the FM cache
    pool (load_cache_pool) and write it. SINGLE source of truth shared by the standalone
    velocity-NF training and the FM aux critic -- never recompute ad hoc."""
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return float(z["vel_mean"]), float(z["vel_std"])
    nbody_arrs, mgas_arrs, _, flat = load_cache_pool(d)
    mean, std = compute_velocity_stats(nbody_arrs, mgas_arrs, flat, clamp_val=clamp_val)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    # atomic write (tmp + rename) so concurrent DDP ranks can't truncate each other's
    # file -- last rename wins, but every reader sees a complete npz.
    tmp = f"{cache_path}.{os.getpid()}.tmp.npz"  # np.savez keeps a .npz suffix as-is
    np.savez(tmp, vel_mean=np.float32(mean), vel_std=np.float32(std),
             clamp_val=np.float32(clamp_val if clamp_val is not None else 0.0))
    os.replace(tmp, cache_path)
    print(f"  [velocity stats] mean {mean:.4f} std {std:.4f} -> {cache_path}")
    return mean, std


def compute_cosmo_stats(cosmo_all):
    mean = cosmo_all.mean(0).astype(np.float32)
    std = cosmo_all.std(0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


# ── cache-backed pool (prep_cache.py output) ──────────────────────────────────

def load_cache_pool(d):
    """Load the fully pre-normalised FM cache (Nbody + Mgas + cosmo) from `cache_dir`.

    Nbody: `Nbody_norm_{suite}_LH_128_z=0.0.npy` — overdensity log1p(1+delta) + global
           z-score (suite-invariant; particle-mass units removed — see CLAUDE.md).
    Mgas : `Mgas_norm_{suite}_LH_128_z=0.0.npy` — log1p + norm3d gas z-score.
    cosmo: `cosmo_{suite}.npy` — (Omega_m, sigma_8) z-scored with norm3d param[:2].

    Everything is read straight from the cache (no lazy normalisation).
    Returns (nbody_arrs, mgas_arrs, cosmo_all, flat).
    """
    cache = d["cache_dir"]
    suites = d["suites"]
    nbody_arrs, mgas_arrs, cosmo_parts, flat = [], [], [], []
    for si, suite in enumerate(suites):
        nb = np.load(os.path.join(cache, f"Nbody_norm_{suite}_LH_128_z=0.0.npy"), mmap_mode="r")
        mg = np.load(os.path.join(cache, f"Mgas_norm_{suite}_LH_128_z=0.0.npy"), mmap_mode="r")
        co = np.load(os.path.join(cache, f"cosmo_{suite}.npy"))
        n = min(len(nb), len(mg), len(co))
        nbody_arrs.append(nb)
        mgas_arrs.append(mg)
        cosmo_parts.append(co[:n])
        flat.extend([(si, j) for j in range(n)])
        print(f"  [cache {suite}] nbody(norm) {nb.shape} mgas(norm) {mg.shape} -> {n}")
    cosmo_all = np.concatenate(cosmo_parts, axis=0).astype(np.float32)
    print(f"  cache pool total = {len(flat)} | Nbody+Mgas fully pre-normalised")
    return nbody_arrs, mgas_arrs, cosmo_all, flat


class CachedFMDataset(Dataset):
    """Pre-normed Nbody -> pre-normed Mgas + pre-normed cosmo (all cached). PBC roll aug."""

    def __init__(self, nbody_arrs, mgas_arrs, cosmo_all, flat, indices,
                 crop_size=None, augment=True, clamp_val=10.0):
        self.nbody_arrs = nbody_arrs
        self.mgas_arrs = mgas_arrs
        self.cosmo_all = cosmo_all
        self.flat = flat
        self.indices = np.asarray(indices)
        self.crop_size = crop_size
        self.augment = augment
        # Clamp the heavy-tailed Nbody overdensity z-score (max ~+21 = 45 sigma at
        # DM halo cores) to +/- clamp_val. Caps the rare outlier voxels that drive
        # per-batch MSE to ~1e4 and blow up the converged FM (see CLAUDE.md). Mgas
        # range is ~+/-9 so clamp_val=10 leaves it untouched. None disables.
        self.clamp_val = clamp_val

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        g = int(self.indices[idx])
        si, li = self.flat[g]
        nb = np.array(self.nbody_arrs[si][li], dtype=np.float32)   # already normalised; copy (mmap is read-only)
        mg = np.array(self.mgas_arrs[si][li], dtype=np.float32)   # already normalised; copy (mmap is read-only)
        if self.clamp_val is not None:
            c = float(self.clamp_val)
            np.clip(nb, -c, c, out=nb)
            np.clip(mg, -c, c, out=mg)
        nb = torch.from_numpy(nb)
        mg = torch.from_numpy(mg)
        D = nb.shape[0]
        if self.crop_size and D > self.crop_size:
            for ax in range(3):
                s = random.randint(0, D - 1)
                ix = torch.arange(s, s + self.crop_size) % D
                nb = nb.index_select(ax, ix)
                mg = mg.index_select(ax, ix)
        elif self.augment:
            shifts = (random.randint(0, D - 1), random.randint(0, D - 1), random.randint(0, D - 1))
            nb = torch.roll(nb, shifts, dims=(0, 1, 2))
            mg = torch.roll(mg, shifts, dims=(0, 1, 2))
        cosmo = torch.from_numpy(self.cosmo_all[g].astype(np.float32))
        return nb.unsqueeze(0), mg.unsqueeze(0), cosmo


# ── dataset ───────────────────────────────────────────────────────────────────

class MultiSuiteFMDataset(Dataset):
    """Nbody -> Mgas pairs across suites with lazy log1p+z-score + PBC roll aug.

    norm: dict with keys nbody_mean, nbody_std, mgas_mean, mgas_std,
          cosmo_mean (2,), cosmo_std (2,).
    """

    def __init__(self, nbody_arrs, mgas_arrs, cosmo_all, flat, indices, norm,
                 crop_size=None, augment=True):
        self.nbody_arrs = nbody_arrs
        self.mgas_arrs = mgas_arrs
        self.cosmo_all = cosmo_all
        self.flat = flat
        self.indices = np.asarray(indices)
        self.norm = norm
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def _norm_field(self, v, mean, std):
        return (np.log1p(v) - mean) / std

    def __getitem__(self, idx):
        g = int(self.indices[idx])
        si, li = self.flat[g]
        nb = self._norm_field(np.asarray(self.nbody_arrs[si][li], dtype=np.float32),
                              self.norm["nbody_mean"], self.norm["nbody_std"])
        mg = self._norm_field(np.asarray(self.mgas_arrs[si][li], dtype=np.float32),
                              self.norm["mgas_mean"], self.norm["mgas_std"])
        nb = torch.from_numpy(nb)
        mg = torch.from_numpy(mg)
        D = nb.shape[0]
        if self.crop_size and D > self.crop_size:
            for ax in range(3):
                s = random.randint(0, D - 1)
                ix = torch.arange(s, s + self.crop_size) % D
                nb = nb.index_select(ax, ix)
                mg = mg.index_select(ax, ix)
        elif self.augment:
            shifts = (random.randint(0, D - 1), random.randint(0, D - 1), random.randint(0, D - 1))
            nb = torch.roll(nb, shifts, dims=(0, 1, 2))
            mg = torch.roll(mg, shifts, dims=(0, 1, 2))
        cosmo = (self.cosmo_all[g] - self.norm["cosmo_mean"]) / self.norm["cosmo_std"]
        return nb.unsqueeze(0), mg.unsqueeze(0), torch.from_numpy(cosmo.astype(np.float32))
