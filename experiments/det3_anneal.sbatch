#!/bin/bash
# det3 + aggressive anneal — compare vs the constant-lr det3 run (vmve93wg).
# Warm-starts (strict, EMA-baked + ema_shadow) from the det3 best ckpt
# (ep008 val 0.014682), fresh optimiser, cosine T_max=40 lr 1e-4->1e-6 over 50 ep.
# Hypothesis: annealing lets it SETTLE below 0.0154 instead of bouncing in the
# ~0.018 noise band (constant lr 9.4e-5).
#
#   sbatch experiments/det3_anneal.sbatch
#
#SBATCH -J det3_anneal
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

srun python train.py --config experiments/latent_sweep/det_dim3_anneal.yaml
