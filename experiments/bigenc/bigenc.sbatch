#!/bin/bash
# bigenc — match encoder3D's encoder capacity (its ResNetBranch ~8.3M vs our
# GasEncoder 1.47M). encoder_base 16 -> 40 -> ~9.2M encoder (UNet 18.35M unchanged,
# total ~27.5M ~ encoder3D 26.7M). Hypothesis: a richer Mgas->latent encoder resists
# the latent collapse (eff dim 1.42/8) + feeds the flow a more informative latent,
# lifting xcorr past the 0.686 density-only oracle.
#
# Warm-starts the UNet backbone (strict-off -> warm_load_partial) from the dim8 best
# ffdsq458 ep486 (val 0.012231, noise_std 0.1); the ENCODER is wider -> retrains FRESH
# (warm_load_partial leaves gas_encoder.proj + SE-ResNet convs freshly initialised).
# Recipe: logitnormal time + noise_std 0.2, fresh optimiser, cosine T_max=150
# lr 2e-4->1e-6 over 150 ep.
#
#   sbatch experiments/bigenc/bigenc.sbatch
#
#SBATCH -J bigenc
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

echo "Node: $(hostname)"
nvidia-smi

srun python train.py --config experiments/bigenc/bigenc.yaml
