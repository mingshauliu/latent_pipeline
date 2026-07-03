#!/bin/bash
# FULL PARITY BUNDLE: gn (GroupNorm) + the remaining un-ported encoder3D differences.
#
# WHY (2026-07-02): fresh live-FiLM runs STILL collapse (ne_vel 1.35/8, ne_vel_T 1.27/8
# at ep31; gn 1.56/8 at ep10 but confounded — norm AND noise_std both differ from the
# pixel runs). Remaining encoder3D deltas found by diff, all collapse-pressure on the
# latent path, bundled here:
#   weight_decay 1e-3 -> 1e-4       (10x less shrink on FiLM proj + encoder)
#   encoder_dropout 0.1 -> 0.0      (their ResNetBlock has NO dropout; dropout noise on
#                                    the latent teaches the decoder to ignore it)
#   ema.include_encoder: false      (they have NO EMA; also un-confounds the t-SNE —
#                                    eff-dim now reads the LIVE encoder)
#   warmup_epochs 5 -> 20           (their LambdaLR warmup)
# Plus gn's: norm_type group, noise_std 0.2, fresh train, live FiLM/out, mlp head,
# base40 encoder, [nbody,vel] -> [Mgas,ne,T]. tsne_n_batches 64 (N=128 latents).
#
#   in_channels = 5 = x_t(3 = Mgas,ne,T) + nbody(1) + vel(1);  out_channels = 3.
#   wandb latent-pipeline-nevelT-gn2. Watch: latent_eff_dim >> 1.5 by ~ep30 (success
#   metric), xcorr vs 0.686/0.729 reference, pk_ratio -> >0.9.
#
# PREREQS (all already built): Ne_norm, T_norm, velocity cache.
#
#   sbatch experiments/ne_vel_T_gn2/fm_ne_vel_T_gn2.sbatch
#
#SBATCH -J fm_ne_vel_T_gn2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH -t 1-12:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load python
source /mnt/home/mliu1/env/bin/activate
mkdir -p logs

echo "Node: $(hostname) | FM [nbody,vel] -> [Mgas,ne,T], FULL PARITY BUNDLE (gn2)"
nvidia-smi

srun python train.py --config experiments/ne_vel_T_gn2/fm_ne_vel_T_gn2.yaml
