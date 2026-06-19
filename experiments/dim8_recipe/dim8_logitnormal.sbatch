#!/bin/bash
# dim8 + encoder3D recipe — push xcorr toward 0.9 (current oracle 0.686, P(k) 0.84).
# Warm-starts (strict, EMA-baked + ema_shadow) from the dim8 best (sv6mp9wt ep269,
# val 0.013353), fresh optimiser, cosine T_max=150 lr 2e-4->1e-6 over 150 ep.
# Recipe changes vs that ckpt (both from encoder3D, which reaches xcorr ~0.9):
#   - time_sampling=logitnormal : t=sigmoid(N(0,1)), focus the hard t~0.5 (SD3/EDM).
#   - noise_std 0.1 -> 0.2       : more stochastic budget to paint small-scale Mgas.
# NOT ported (need extra data we don't load): N-body VELOCITY conditioning
# (encoder3D in_channels=5) + multi-task gas/gas_t/ne — the bigger physical levers.
#
#   sbatch experiments/dim8_recipe/dim8_logitnormal.sbatch
#
#SBATCH -J dim8_recipe
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

srun python train.py --config experiments/dim8_recipe/dim8_logitnormal.yaml
