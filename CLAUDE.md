# latent_pipeline

Check and alter this CLAUDE.md whenever necessary after a session.

Two-branch pipeline:

1. **FM branch** — conditional flow matching `p(Mgas | Nbody, Ω_m, σ8, z_latent)`,
   N-body total-matter density → gas-mass map, conditioned on cosmology + an
   **8-dim latent encoded from Mgas**. Trained jointly over three CAMELS hydro
   suites (IllustrisTNG + Astrid + SIMBA), **LH 128³, z=0**.
2. **NF branch** — 3D SE-ResNet encoder + conditional normalizing flow,
   **Mgas (128³) → (Ω_m, σ8)**. Used as the evaluation harness for FM synth
   quality. **Two trainings**: on real CAMELS 128³ Mgas, and on FM-synth 128³ Mgas.
   (2026-06-16: 256³ → 128³ — NF trains on the SAME 128³ pool as FM; shared norm3d
   `gas` normalisation across FM cache / NF / Magneticum held-out.)

Lineage: structure from `../encoder3D` (multi-suite + latent encoder); optimised
model + training recipe + NF package from `../upscaling` (PixelNorm, zero-init
FiLM/out_conv, EMA, stability-tuned schedule, xcorr/pk metrics, SE-ResNet + NSF).

## Goal

FM generalises across feedback models by training on 3 suites at once. NF judges
whether FM-synth Mgas retains the per-cube cosmology signal: compare NF posteriors
(real-trained vs synth-trained, and cross-eval). Acceptance: per-suite P(k)
transport ratio within ±10% over k ≤ 15 (h/Mpc), xcorr ≥ 0.9.

## Data

| field | source | shape |
|-------|--------|-------|
| Nbody (FM input, 128) | `…/Nbody/Grids_Mtot_Nbody_{suite}_LH_128_z=0.0.npy` | (1000,128³) |
| Mgas (FM target, 128) | `…/{suite}/Grids_Mgas_{suite}_LH_128_z=0.0.npy` | (1000,128³) |
| Mgas (NF real, 128) | `…/{suite}/Grids_Mgas_{suite}_LH_128_z=0.0.npy` | (1000,128³) |
| cosmo | `…/{suite}/params_LH_{suite}.txt` cols `[:, :2]` = (Ω_m, σ8) | (1000,2) |

- Nbody dir: `/mnt/home/camels/ceph/PUBLIC_RELEASE/CMD/3D_grids/data/Nbody/`.
  Hydro dir: `/mnt/ceph/users/camels/PUBLIC_RELEASE/CMD/3D_grids/data/{suite}/`.
- Suites: IllustrisTNG, Astrid, SIMBA. 1000 LH each → **3000 pairs**, concatenated
  (`data.load_suite_pool` for FM 128³, `data.load_nf_pool` for NF 128³).
- **Index alignment verified**: Nbody cube i ↔ Mgas cube i ↔ cosmo i per suite
  (Nbody `params[:, :2]` ≡ hydro `params[:, :2]`, exact).
- Box: **25 Mpc/h**, z=0. Source = N-body (DM-only) `Grids_Mtot_Nbody_*`, NOT hydro `Mcdm`.

### Normalization — lazy in the dataloader (offline cache deferred)

`log1p` + global z-score applied per-sample in `__getitem__` (`data.MultiSuiteFMDataset`,
`nf.module.MultiSuiteMgasDataset`). Stats computed once over a subset of the pool
(`data.compute_field_stats`) and cached:
- `cached/norm_latent.npz` — FM 128³ stats (`nbody_mean/std`, `mgas_mean/std`, `cosmo_mean/std`).
- **NF (128³) does NOT compute its own stats** — `data.nf_mgas_stats` returns the SHARED
  norm3d `gas` stats (mean 22.003, std 1.309), identical to the FM cache + Magneticum
  held-out cache. (Old `cached/norm_nf256.npz` pool-stats path removed with the 128³ switch.)

Cosmo (Ω_m, σ8) standardised by `cosmo_mean/std`. FM-synth output is written in
**physical** Mgas (`expm1` denorm) so the NF re-applies the same log1p+z-score.
**Done (2026-06-16)**: both fields pre-normalised to the ceph cache (see below);
`use_cache: true` is the canonical path. The lazy `MultiSuiteFMDataset`/`load_suite_pool`
path remains for `use_cache: false` but normalises Nbody as plain `log1p` (NOT
overdensity) — Astrid would be off-scale there; treat that path as legacy/non-Astrid.

### Pre-normalised FM cache (`prep_cache.py`) — Nbody + Mgas both cached

**Both** FM fields are pre-normalised + cached (2026-06-16: Nbody added; everything
now reads the ceph cache, no raw + lazy norm). Output:
`/mnt/ceph/users/mliu1/latent_pipeline_cache/L25_LH_128_norm/`
  - `Mgas_norm_{suite}_LH_128_z=0.0.npy` (1000,128³) f32 — `log1p` + norm3d `gas`
    z-score (mean 22.003, std 1.309).
  - `Nbody_norm_{suite}_LH_128_z=0.0.npy` (1000,128³) f32 — **overdensity**
    `log1p(ρ/ρ̄_cube)` + a single **global** z-score (`nbody_mean/std` ≈ 0.30/0.46).
  - `cosmo_{suite}.npy` (1000,2), `norm_meta.npz` (all stats + provenance).

**Why overdensity for Nbody**: raw Mtot scale is particle-mass-dependent — Astrid
is ~192× TNG/SIMBA (raw mean 1.2e11 vs 6.4e8/3.0e8). Dividing each cube by its own
mean is unit-free → all three suites collapse to the same scale (log1p(1+δ) mean
0.27/0.28/0.35), so one global z-score works and held-out suites normalise the same
way. Ω_m amplitude is already supplied via FiLM conditioning. `infer.norm_nbody`
applies the IDENTICAL transform at sampling time (train/infer parity). Stats also
mirrored to `cached/norm_latent.npz` so `infer.py` picks them up.
Rebuild: `python prep_cache.py` (both) | `--no_mgas` (Nbody only) | `--no_nbody`.

### Held-out Magneticum cache (`prep_magneticum_cache.py`)

Magneticum = **4th feedback model, NOT in training pool** (TNG/Astrid/SIMBA) → OOD
held-out test. Source `…/CAMELS-L25n256/Grids_Mgas_Magneticum_LH_128_z=0.0.npy`
(48 cubes, 128³, raw float64) + `params_LH_Magneticum.txt` (48,7) cols[:2]=(Ω_m,σ8).
Cached with the **same recipe as prep_cache.py** (log1p + norm3d `gas`, cosmo z-score
norm3d `param`[:2]) → comparable to training distribution (cache pool mean +0.019
std 1.002). Output:
`/mnt/ceph/users/mliu1/latent_pipeline_cache/L25_LH_128_norm_heldout/`
  - `Mgas_norm_Magneticum_LH_128_z=0.0.npy` (48,128³) f32, `cosmo_Magneticum.npy`, `norm_meta.npz`.
Note: Magneticum is 128³ and **NF now trains at 128³ too** → resolution-matched,
clean OOD-feedback held-out test (same norm3d gas normalisation, no resolution shift).

## Architecture

- **FM net** (`model.py::ClassicUNet`, ported from `upscaling/model_classic.py`):
  3-level UNet, `in_channels=2` ([x_t, nbody]), `out_channels=1`, `base_channels=128`,
  **PixelNorm**, **circular padding**, FiLM per block with **zero-init** FiLM + out_conv.
  Conditioning fuses time(64) + cosmo(2→64) + **latent(8)** → FiLM context.
- **Latent encoder** (`model.py::GasEncoder`): SE-ResNet3D (pre-act blocks, `SEBlock3D`,
  circular pad, dropout) → AdaptiveAvgPool → LayerNorm → Linear→8 → tanh. Encodes Mgas.
- **NF** (`nf/`, ported from `upscaling/nf`): `encoder.ResNet3D` (SE, GroupNorm, circular pad,
  AdaptiveAvgPool → resolution-agnostic), `flow.ConditionalFlow` (zuko NSF), aux MSE head,
  info-bottleneck `summary_dim=6`, flow warm-up. `num_params=2`. **128³ paradigm**:
  `base_channels=4`, `summary_dim=6`, `flow_hidden=128`, `flow_transforms=4` (upscaling
  values — tuned against encoder→flow memorization on the 2400-cube train pool).

## Flow matching

`x0 = nbody + noise_std·ε`, `x1 = Mgas`, `x_t = (1-t)·x0 + t·x1`, `t ~ U(0,1)`.
`latent = GasEncoder(Mgas)`. Net predicts `v ≈ x1 − x0`; loss = `MSE(v_pred, x1 − x0)`.
EMA covers BOTH the UNet and the encoder.

### Inference latent (train/test gap)

Mgas unknown at sampling → latent supplied by `inference.latent_mode`:
- `mean` (default): `latent = 0` (marginal over feedback realisations).
- `sample`: `latent ~ N(0, I)` per cube (stochastic feedback draw).
- `encode`: encode the reference true Mgas (oracle / sanity). Train-time xcorr uses this.

## NF — two trainings (128³)

One config, switch with `--data_mode` (or `nf.data.mode`):
- **real**  : NF on real CAMELS LH 128³ Mgas (`mgas_path_tmpl`, same pool as FM).
- **synth** : NF on FM-synth 128³ Mgas dirs (`synth_dir_tmpl` → `{suite}_LH128`;
  generate 128³ Mgas from the LH 128 Nbody first, see Usage).

Both encode 128³ and predict (Ω_m, σ8). Targets standardised per-dim (flow buffers).
Compare real-NF vs synth-NF (and cross-apply) to quantify FM signal retention.

## Layout

```
latent_pipeline/
  CLAUDE.md
  config/config.yaml     # data (128³ FM + NF), model, training, inference, nf
  model.py               # ClassicUNet (latent-conditioned) + GasEncoder (SE-ResNet3D)
  data.py                # pools (load_suite_pool / load_nf_pool), stats, MultiSuiteFMDataset
  module.py              # FlowMatchingModel (FM Lightning + latent + EMA + xcorr/pk)
  train.py               # FM training (multi-suite, DDP)
  infer.py               # FM sampling (latent_mode), writes physical Mgas sample_*.npy
  nf/
    encoder.py flow.py   # ResNet3D + ConditionalFlow (vendored)
    module.py            # MultiSuiteMgasDataset/DataModule + LitNFRegressor
    train.py infer.py    # NF train/infer (--data_mode real|synth)
    predict.py __init__.py
  run.sh / submit.sh     # SLURM (4×A100, ports from upscaling)
  cached/                # norm_latent.npz (FM 128); NF reuses shared norm3d gas
  plot_nf_2param.py      # NF in-distribution truth-vs-pred (val split, multi-suite)
  prep_magneticum_cache.py  # held-out Magneticum 128³ cache (norm3d gas)
  logs/                  # slurm stdout/err
```

## Usage

```bash
# FM: train (computes+caches cached/norm_latent.npz on first run)
./submit.sh train config/config.yaml
python train.py --fast_dev_run            # 1-batch CPU/1-GPU smoke

# FM: generate synth 128^3 Mgas per suite (feeds synth-NF) -> output_dir/<suite>_LH128/
./submit.sh infer config/config.yaml      # set inference.checkpoint first; gen at 128³

# NF: TWO trainings
./submit.sh nf_train config/config.yaml --data_mode real  --checkpoint_dir nf_ck_real
./submit.sh nf_train config/config.yaml --data_mode synth --checkpoint_dir nf_ck_synth

# NF: eval (truth-vs-pred PNG + predictions_{mode}.npz)
./submit.sh nf_infer config/config.yaml --data_mode real  --checkpoint nf_ck_real/best.ckpt
./submit.sh nf_infer config/config.yaml --data_mode synth --checkpoint nf_ck_synth/best.ckpt
```

## Sizing review — all-128³ paradigm (2026-06-16)

Ported hyperparameters were upscaling's, tuned for 128³ + non-zero-mean data.
Fixes:

- **Data is now zero-mean/unit-var** (data.py log1p+z-scores both fields), so FM
  init loss ~2 (not ~16). upscaling's blow-up brakes relaxed: `lr 3e-4→6e-4`,
  `warmup 30→15`, `gradient_clip 0.5→1.0`. GPU smoke gave `train_loss 0.516`
  (correlated fields → v=x1−x0 var <2); a spike to ~16 means revert.
- **NF reverted to 128³** (2026-06-16, was a 256³ detour): `mgas_path_tmpl LH_256→LH_128`,
  `summary_dim 16→6`, `batch_size 2→8`, `accumulate_grad_batches 8→2` (eff batch 64
  unchanged → flow_warmup 50/early_stop 80 schedule intact). NF normalises with the
  SHARED norm3d `gas` stats via `data.nf_mgas_stats` (no per-pool stats; matches FM
  cache + Magneticum). `synth_dir_tmpl → {suite}_LH128`.
- **FM synth-gen now 128³**: `inference` block rewired to LH_128 Nbody sources +
  `method euler→dopri5` (fits at 128³) + source names `{suite}_LH128` (match
  `nf.data.synth_dir_tmpl`). Train res = gen res = NF res, fully consistent.

Unchanged: FM trains at 128³ (base_channels=128 fine), NF base=4 (tiny by design),
EMA decay 0.9999/warmup 5000. NF bottleneck 16³ at 128³ (upscaling-native depth, no
4th EncoderBlock needed).

## In-distribution NF eval (`plot_nf_2param.py`)

Ported from `../upscaling/plot_nf_2param.py`, adapted to the multi-suite pool.
Encodes the **held-out val split** (same split/seed as training via
`MultiSuiteNFDataModule`) and plots truth-vs-pred with per-param RMSE/R². Distinct
from `nf/infer.py` (which runs the WHOLE pool incl. train). Usage:
```bash
python plot_nf_2param.py --config config/config.yaml \
    --checkpoint nf_ck_real/best.ckpt --data_mode real --out nf_indist_real.png
```

## Status

- 2026-06-16: scaffolded + cache-wired. All modules byte-compile; FM
  (ClassicUNet+GasEncoder) and NF (ResNet3D+NSF) pass CPU forward smoke tests.
- **2026-06-16 GPU smoke ✓** (1×A100-80GB, job 6519580): FM `--fast_dev_run` passed.
  19.8M params; cache pool 3000; `train_loss=0.516` (no blow-up; nbody↔mgas correlated
  → v=x1−x0 var <2, so <"≈2" expectation is expected, not a bug); val xcorr 0.644,
  pk_ratio 1.73 at init.
- **Bug fixed**: `nf/module.py` val_ds built from undefined `self.mgas_arrs` →
  `self.mgas_src` (would crash every NF train at first val).
- **Fully pre-normalised FM cache** at `…/latent_pipeline_cache/L25_LH_128_norm/`
  (3 suites × 1000). Mgas: norm3d gas. **Nbody: overdensity log1p(1+δ) + global
  z-score** (2026-06-16; particle-mass units removed — Astrid was 192× off). Cosmo:
  norm3d `param[:2]`. `data.use_cache: true` → FM reads Nbody+Mgas+cosmo all from
  cache (no lazy norm). `infer.norm_nbody` mirrors the overdensity transform.

### Current state (2026-06-17)

GPU wiring ✓. **NF-real DONE** (R² 0.966/0.886, see 2026-06-17 session). **FM blow-up
diagnosed + fixed** (clamp ±10 + spike guard 5.0 + resume from best-059) and **FM
resubmitted by user**. `sbatch`/`scancel` are user-run only (FI Slurm policy).
- `config/_smoke.yaml` = throwaway small/precision-32 dev-run config (safe to delete).

### Session — 2026-06-17

**NF real DONE** (job 6519619): trained, ckpts in `nf_ck_real/` (best
`nf-752--5.4560.ckpt`). In-distribution truth-vs-pred (`plot_nf_2param.py --data_mode
real`) → `nf_indist_real.png`: **Ω_m R²=0.966 (RMSE 0.0215), σ8 R²=0.886 (RMSE
0.0395)**. Strong real-NF reference for the later synth-NF comparison.

**FM blow-up diagnosed + fixed.** FM train (job 6519618) reached val **0.0194**
(best-epoch=059, <0.02 target) then **spiked at epoch 73**: `train_loss=7.89e+3`,
knocking weights out of the minimum (val stuck ~0.028 after). Root cause = **heavy-tailed
Nbody overdensity cache**: z-scored (mean 0.30/std 0.46) but **max ~+21 (≈45σ)** at DM
halo cores (real log1p(1+δ), not a bug). Mgas bounded ~±9. After convergence one batch
(size 2) drawing an extreme cube → target `mgas−nbody` ≈ −12 at those voxels → per-batch
MSE 7890. `gradient_clip=1.0` bounds the step but direction is far off; `skip_nan_loss`
only catches non-finite (7890 passes). lr/clip were NOT the cause (stable 72 epochs);
kept lr 6e-4/clip 1.0 (user: high lr needed to leave the basin on resume).

Fix (all code-validated, no cache rebuild):
- `data.CachedFMDataset` clamps pre-normed cubes to ±`clamp_val` (default 10 →
  `data.clamp_val: 10`). Caps the >10σ Nbody tail; Mgas (±9) untouched. Verified:
  clamp=10 → max|val|=10.0; no-clamp → 20.4.
- `module.training_step` loss-spike guard: skip finite-but-huge batches
  (`training.loss_spike_thresh: 5.0`; normal loss 0.02–0.7). Verified: 0.02/0.7 KEEP,
  7890 SKIP, inf SKIP. `train.py` threads `clamp_val` into `CachedFMDataset`.
- `training.resume_from` = `latent-pipeline/d72nm6ol/checkpoints/best-epoch=059-val_loss=0.019434.ckpt`
  (lr still ~6e-4 there). Resume restore verified (fast_dev_run can't test it — forces
  max_epochs=1 vs epoch 59; irrelevant to the 2000-epoch job).

**FM RESUBMITTED** (user, 2026-06-17) — `scancel 6519618` + `./submit.sh train` (resumes
from best-059 with clamp+guard). Watch: train_loss re-settles ~0.02, no `spike_skip`
storm, val back <0.02, xcorr climbing from ~0.6. If it re-spikes despite clamp → lower
`clamp_val` to ~6, resubmit. Note: dev node has no GPU → full smoke can't run locally;
clamp+guard validated by direct unit test instead.

### Next

1. **FM (running)** — confirm it clears epoch 73 cleanly (val <0.02, no spike). Pick the
   new best ckpt when converged.
2. **Synth-gen** — set `inference.checkpoint` = new FM best → `./submit.sh infer`; verify
   it emits 128³ into `{suite}_LH128` dirs.
3. **NF synth** — `./submit.sh nf_train … --data_mode synth --checkpoint_dir nf_ck_synth`;
   then `python plot_nf_2param.py --data_mode synth` for the in-distribution check.
4. **Compare + OOD** — real-NF vs synth-NF (and cross-eval) + Magneticum held-out (OOD).
   Acceptance (Goal): per-suite P(k) ratio ±10% over k≤15, xcorr ≥0.9.

### Session close — 2026-06-16 (historical)

Wiring + smoke-verified. Built this session: 128³ NF paradigm (summary_dim 6, batch
8/accum 2, shared norm3d-gas norm), all-128³ synth-gen (FM infer → dopri5/LH_128),
`plot_nf_2param.py` (val-split in-distribution), Magneticum held-out cache, **Nbody
overdensity cache** (suite-invariant, train/infer parity exact), NF val_ds `mgas_arrs`
bug fix. (FM + NF-real batch jobs were submitted next session — see 2026-06-17.)

## Open assumptions / things to verify

1. **Latent train/test gap**: training conditions on a latent encoded from the true
   Mgas; inference uses `latent_mode=mean` (zeros). If synth quality is latent-starved,
   add the stretch prior `p(latent | nbody, cosmo)` (see infer.py modes).
2. **FM synth-gen resolution**: all-128³ paradigm → FM emits 128³ synth (train res =
   gen res, no resolution transfer). `inference` block now targets LH_128 + dopri5.
   The original 256³ FM resolution-transfer test is intentionally dropped.
3. **Cross-suite feedback params**: only Ω_m, σ8 are conditioned (universal). Feedback
   axes differ across suites and are intentionally marginalised.
4. **NF field stats sharing**: real-NF and synth-NF BOTH normalise with the shared
   norm3d gas stats (`data.nf_mgas_stats`) → automatically comparable, no ordering
   dependency (the old norm_nf256-from-real-pool requirement is gone).
```
