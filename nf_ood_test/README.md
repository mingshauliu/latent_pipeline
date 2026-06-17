# NF OOD test — train IllustrisTNG-only, test Magneticum

Isolated single-suite OOD generalisation test. Does NOT touch the main pipeline
(own config, own `ckpt/`, own `outputs/`). Reuses the existing `nf.train` /
`nf.ood` code unchanged — only the config differs (single training suite).

**Question:** train the NF cosmo regressor on ONE feedback model (IllustrisTNG)
and test on an UNSEEN feedback model (Magneticum, the 4th suite) → how well does
(Ω_m, σ8) inference from Mgas transfer across feedback physics?

## Files
- `config_tng.yaml` — copy of `config/config.yaml` with `nf.data.suites:
  [IllustrisTNG]` and `nf.training.checkpoint_dir: nf_ood_test/ckpt`. All norm /
  model / heldout (Magneticum) settings unchanged.
- `eval.sh` — auto-picks best (lowest val/nll) ckpt, runs `nf.ood` → OOD plot.
- `ckpt/` — checkpoints. `outputs/` — predictions + OOD plot.

## Run (from `latent_pipeline/`)
```bash
# 1. Train NF on IllustrisTNG only (4x A100; SLURM submitted by user per FI policy)
./submit.sh nf_train nf_ood_test/config_tng.yaml --run_name ood_tng_only

# 2. After training finishes -> Magneticum OOD truth-vs-pred plot
bash nf_ood_test/eval.sh
#   or via SLURM:
#   CK=$(ls nf_ood_test/ckpt/nf-*.ckpt | ...pick lowest val/nll...)
#   ./submit.sh nf_ood nf_ood_test/config_tng.yaml \
#       --checkpoint "$CK" --tag tng_only --output_dir nf_ood_test/outputs
```

## Output
`nf_ood_test/outputs/ood_Magneticum_tng_only.png` — truth-vs-pred scatter
(Ω_m, σ8) with per-param RMSE / R², plus `predictions_ood_Magneticum_tng_only.npz`.

## Data path (validated on CPU)
- Train pool: 1000 IllustrisTNG LH 128³ Mgas cubes, Om[0.1,0.5] s8[0.6,1.0].
- OOD pool: 48 Magneticum LH 128³ cubes (raw float64), physical cosmo same ranges.
- Both normalise via the SHARED norm3d 'gas' log1p+z-score (mean 22.003, std
  1.309) inside `MultiSuiteMgasDataset` — Magneticum normed cube comes out
  mean≈0 std≈1 (single norm, no double-norm). Targets = physical (Ω_m, σ8); the
  flow un-standardises its draws, so predictions are in physical units.

## Read the result
This TNG-only OOD R² is the **cross-feedback baseline**. Compare against:
- the 3-suite NF (main `nf_ck_real`) on Magneticum — does training on more
  feedback models improve OOD transfer?
- in-distribution TNG val R² (run `plot_nf_2param.py` style on this ckpt) — the
  gap = pure OOD (feedback) generalisation loss.
