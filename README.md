# latent_pipeline

Nbody→Mgas conditional flow matching (FM) + Mgas→(Ω_m, σ8) normalizing flow (NF),
multi-suite CAMELS 128³ (IllustrisTNG + Astrid + SIMBA, LH, z=0).

- **FM** generates gas-mass maps from N-body density, conditioned on cosmology + an
  8-dim latent encoded from Mgas. Generalises across feedback models by training on 3
  suites at once.
- **NF** infers (Ω_m, σ8) from Mgas — the eval harness judging whether FM-synth Mgas
  keeps the per-cube cosmology signal (real-NF vs synth-NF, plus OOD held-out feedback).


## Structure

```
config/config.yaml   data + FM model/training + inference + NF + heldout
model.py             FM net (ClassicUNet) + latent GasEncoder (SE-ResNet3D)
data.py              suite pools, norm stats, FM datasets (cache + lazy)
module.py            FlowMatchingModel (FM Lightning: latent, EMA, xcorr/pk, vel-aux)
train.py             FM training (multi-suite, DDP) + warm_load_partial
infer.py             FM sampling → physical Mgas sample_*.npy

nf/
  encoder.py flow.py ResNet3D encoder + ConditionalFlow (NSF)
  module.py          NF dataset/datamodule + LitNFRegressor
  train.py infer.py  NF train / whole-pool eval (--data_mode real|synth|velocity)
  ood.py             OOD test: NF on held-out feedback (Magneticum / Swift-EAGLE)
  predict.py         posterior prediction API

prep_cache.py             pre-normalise FM cache (Nbody + Mgas [+ --with_vel/--with_ne])
prep_magneticum_cache.py  held-out Magneticum 128³ cache
extract_latents.py        encode training Mgas → latent Gaussian fit (1-ch encoder)
extract_latents_ne.py     ne-aware (2-ch Mgas+ne) latent extract + eff-dim + t-SNE
tsne_latent.py            t-SNE of an extracted latent, coloured by suite
plot_nf_2param.py         NF in-distribution (val-split) truth-vs-pred
eval_indist.py            FM in-distribution sanity (synth vs true, imshow + P(k))
eval_nf_synth.py          synth-fidelity judge: real-NF on FM-synth val cubes
sanity_nbodyvel.py        check NbodyVel stack + normed cache
run.sh submit.sh          SLURM wrappers
*.sbatch                  per-task batch scripts (train, eval, smoke, sweeps)
experiments/              isolated variants (ne, ne_vel, velocity, sweeps)
```

## Model variants (config-gated, backward-compatible)

| flag | effect |
|------|--------|
| `data.use_velocity` | + N-body VELOCITY conditioning input (`in_channels=3`) |
| `data.use_ne` | + ne 2nd OUTPUT target + 2-ch (Mgas,ne) encoder (`out_channels=2`) |
| `model.latent_head` | `tanh` (default) \| `raw` \| `mlp` (encoder3D no-tanh head; smoother latent) |
| `model.variational` | β-VAE latent (mu/logvar + KL); else deterministic |
| `training.velocity_nf_ckpt` | enable velocity-NF cosmo-consistency aux loss |

The headline combined run = `experiments/ne_vel/` : `[nbody, nbody_vel] → [Mgas, ne]`,
encoder3D-sized encoder + `latent_head: mlp`.

## Usage

```bash
# --- caches (prereq; USER runs heavy/SLURM steps) ---
python prep_cache.py                       # FM cache: Nbody + Mgas (+ cosmo)
python prep_cache.py --with_ne  --no_nbody --no_mgas   # add Ne_norm (log10+norm3d ne)
python prep_cache.py --with_vel --no_nbody --no_mgas   # add NbodyVel_norm (needs --stack first)

# --- FM train / sample ---
./submit.sh train config/config.yaml       # multi-suite FM (DDP); cache norm on first run
./submit.sh infer config/config.yaml       # set inference.checkpoint → synth 128³ Mgas

# --- experiment variants (USER submits) ---
sbatch experiments/ne_vel/fm_ne_vel.sbatch # [nbody,vel]->[Mgas,ne], mlp latent head

# --- NF: two trainings + eval ---
./submit.sh nf_train config/config.yaml --data_mode real  --checkpoint_dir nf_ck_real
./submit.sh nf_train config/config.yaml --data_mode synth --checkpoint_dir nf_ck_synth
python plot_nf_2param.py --data_mode real  --checkpoint nf_ck_real/best.ckpt
./submit.sh nf_ood config/config.yaml --checkpoint nf_ck_real/best.ckpt --tag real \
    --heldout_name Magneticum --heldout_mgas <…> --heldout_param <…>

# --- latent inspection ---
python extract_latents_ne.py --config experiments/ne/fm_ne.yaml --checkpoint <ne.ckpt> \
    --out cached/latent_stats_ne.npz --tsne_out cached/latent_tsne_ne.png
```

Smoke tests: `smoke_ne.py` (ne / ne_vel / latent_head) via `sbatch gpu_smoke_ne.sbatch`.

## Acceptance (goal)

Per-suite P(k) transport ratio within ±10% over k ≤ 15 (h/Mpc), xcorr ≥ 0.9.
