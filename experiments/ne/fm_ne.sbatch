#!/bin/bash
# SEPARATE FM model: MULTI-TASK output (Mgas + ne electron density) -> regularise the
# backbone + de-collapse the latent (encoder3D uses out_channels=3 gas/gas_t/ne; we add
# just ne -> out_channels=2). The flow runs both targets jointly (x0 = nbody repeated to
# 2ch + noise, x1 = cat([Mgas, ne])); the latent encoder takes the stacked (Mgas, ne).
#
# Warm-starts (PARTIAL, init_strict=false) from the dim8 best FM ep486 (val 0.012231):
# full UNet backbone + encoder SE-ResNet copy verbatim; net.enc1.conv1/skip expand input
# 2->3 (role-remapped: Mgas x_t + nbody cols kept, new ne x_t col fresh); out_conv expands
# 1->2 (Mgas out row kept, ne row at zero-init); encoder stem 1->2 (ne col fresh). Fresh
# optimiser, cosine T_max=150 lr 2e-4->1e-6 / 150 ep, noise_std 0.1 (matches ep486).
# wandb latent-pipeline-ne.
#
# PREREQ (run first — ne is PUBLIC, no voxelise):
#   python prep_cache.py --with_ne   -> Ne_norm_{suite} in the FM cache (log10 + norm3d
#   'ne' z-score) + ne stats in norm_meta/norm_latent. Check Ne_norm pool mean~0 std~1
#   and max|val| (~8.8 expected < clamp_val=10, so ne survives clamp untouched).
#
#   sbatch experiments/ne/fm_ne.sbatch
#
#SBATCH -J fm_ne
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

echo "Node: $(hostname) | FM + ne multi-task output channel"
nvidia-smi

srun python train.py --config experiments/ne/fm_ne.yaml
